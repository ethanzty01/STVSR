"""Measure standalone tiny inference and bounded streaming at a specified resolution.

Run each precision in a fresh process, with the GPU otherwise idle. The default
960x540 input maps to 3840x2160 output. At a 60 Hz input cadence each completed
pair contributes two new output frames to the 120 Hz output timeline.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch

from models.RSTT_tiny import RSTTTiny
from models.layers import WindowAttention3D


MIB = 2 ** 20


def tensor_record(tensor):
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'logical_mib': tensor.numel() * tensor.element_size() / MIB}


def memory_record():
    return {'allocated_mib': torch.cuda.memory_allocated() / MIB,
            'reserved_mib': torch.cuda.memory_reserved() / MIB,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / MIB,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / MIB}


def peak_summary(phases):
    return {key: max((phase[key] for phase in phases if phase.get(key) is not None), default=None)
            for key in ['peak_allocated_mib', 'peak_reserved_mib', 'sampled_process_peak_mib']}


class ProcessMemorySampler:
    """Sample this PID's driver-visible usage, including allocator and context."""
    def __init__(self):
        self.samples = []
        self.error = None
        self.stopping = threading.Event()

    def sample(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                 '--format=csv,noheader,nounits'], capture_output=True,
                text=True, check=True, timeout=3)
            values = [float(line.split(',')[1]) for line in result.stdout.splitlines()
                      if line.split(',')[0].strip() == str(os.getpid())]
            if values:
                self.samples.append(sum(values))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.error = str(error)

    def run(self):
        while not self.stopping.wait(.1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopping.set()
        self.thread.join()
        self.sample()

    def report(self):
        return {'sampled_process_peak_mib': max(self.samples) if self.samples else None,
                'sample_count': len(self.samples), 'sample_interval_seconds': .1,
                'error': self.error}


def shape_hooks(model, records):
    """Record metadata only: hooks never retain GPU tensors."""
    handles = []

    def hook_for(name):
        def hook(module, inputs, output):
            record = {'module': name}
            if isinstance(module, WindowAttention3D):
                record['query_tokens'] = tensor_record(inputs[0])
                record['attention_matrix'] = tensor_record(output[1])
                record['feature'] = tensor_record(output[0])
            else:
                record['feature'] = tensor_record(output)
            record['memory_after_module'] = memory_record()
            records.append(record)
        return hook

    for name, module in model.named_modules():
        if (name in {'input_proj', 'conv_last', 'pixel_shuffle'} or
                name.startswith(('encoder_layers.', 'decoder_layers.', 'downsample.', 'upsample.'))
                and name.count('.') == 1 or isinstance(module, WindowAttention3D)):
            handles.append(module.register_forward_hook(hook_for(name)))
    return handles


def storage_inventory(height, width, channels, weight_bytes, buffer_bytes):
    padded_h, padded_w = max(32, (height + 7) // 8 * 8), max(32, (width + 7) // 8 * 8)

    def item(name, shape, dtype_bytes=4, **extra):
        elements = 1
        for size in shape:
            elements *= size
        return {'name': name, 'shape': shape, 'fp32_mib': elements * dtype_bytes / MIB, **extra}

    return {
        'parameters_mib_fp32': weight_bytes / MIB,
        'registered_buffers_mib': buffer_bytes / MIB,
        'input_pair': item('two RGB input frames', [1, 2, 3, height, width]),
        'encoder_skip_features': [item(f'encoder_{i}', [1, 2, channels, padded_h // 2**i, padded_w // 2**i])
                                  for i in range(4)],
        'decoder_features': [item(f'decoder_{i}', [1, 3, channels, padded_h // 2**(3-i), padded_w // 2**(3-i)])
                             for i in range(4)],
        'reconstruction_projection_per_frame': item('48-channel direct head', [1, 48, height, width]),
        'output_one_frame': item('one HR RGB frame', [1, 3, height*4, width*4]),
        'output_three_frames': item('returned three HR RGB frames', [1, 3, 3, height*4, width*4]),
        'host_rgb8_frame_mib': 3 * height * 4 * width * 4 / MIB,
        'nv12_8bit_surface_mib': height * 4 * width * 4 * 1.5 / MIB,
        'notes': [
            'Logical tensor sizes are not additive peaks: lifetimes and allocator reuse matter.',
            'All four encoder features stay live for decoder cross-attention; decoder advances stage by stage.',
            'Attention Q/K/V, logits, probabilities, shifted-window masks, MLP activations, convolution workspaces, and residuals are temporary.',
            'The framewise head creates three output tensors, then torch.stack allocates the returned three-frame tensor.',
            'The streaming generator releases the previous three-frame output before computing the next pair; the consumer must also release yielded GPU views.',
            'RGB8 host and NV12 surface entries are size formulas. Codec surface pools depend on the selected codec and encoder and are outside this inference measurement.'
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp32', 'amp-fp16'], default='fp32')
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--stream-frames', type=int, default=6)
    args = parser.parse_args()
    if min(args.height, args.width, args.iterations) < 1 or args.stream_frames < 2 or args.warmup < 0:
        parser.error('positive dimensions/iterations, nonnegative warmup and at least two stream frames required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for measured GPU memory')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model_config = checkpoint.get('model_config', {'embed_dim': 32, 'depth': 2, 'head': 'direct', 'drop_path_rate': 0})
    model = RSTTTiny(**model_config).eval().requires_grad_(False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    del checkpoint
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(p.numel() * p.element_size() for p in model.buffers())
    model.cuda()
    torch.cuda.synchronize()
    model_only_memory = memory_record()
    inputs = torch.rand(1, 2, 3, args.height, args.width, device='cuda')
    initialization_memory = memory_record()

    def precision_context():
        return torch.autocast('cuda', dtype=torch.float16) if args.precision == 'amp-fp16' else nullcontext()

    @torch.inference_mode()
    def forward():
        with precision_context():
            return model(inputs)

    result = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'model_config': model_config, 'parameters': sum(p.numel() for p in model.parameters()),
        'precision': args.precision, 'weight_dtype': str(next(model.parameters()).dtype),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'pid': os.getpid(), 'input': tensor_record(inputs), 'model_only_memory': model_only_memory,
        'target_input_fps': 60, 'target_output_fps': 120,
        'protocol': 'Standalone tiny only; CUDA inference_mode; batch 1; output deleted between pairs; fresh process per precision.',
        'stage_trace': [],
    }
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() / MIB
    hooks = shape_hooks(model, result['stage_trace'])
    with ProcessMemorySampler() as sampler:
        start = time.perf_counter()
        output = forward()
        torch.cuda.synchronize()
        result['cold_forward'] = {**memory_record(), 'wall_seconds': time.perf_counter() - start,
                                  'baseline_allocated_mib': baseline,
                                  'peak_increment_mib': torch.cuda.max_memory_allocated()/MIB-baseline}
    result['cold_forward'].update(sampler.report())
    for hook in hooks:
        hook.remove()
    result['output'] = tensor_record(output)
    # Keep allocator peaks accumulated since the cold-forward reset through the
    # finite-output check and every warmup; preserve the earlier initialization
    # peak too. This closes the gaps between the separately timed phases.
    with ProcessMemorySampler() as sampler:
        result['all_output_finite'] = bool(torch.isfinite(output).all())
        del output
        for _ in range(args.warmup):
            output = forward()
            del output
        torch.cuda.synchronize()
        initialization_and_warmup = memory_record()
    warmup_samples = sampler.report()
    initialization_and_warmup.update(warmup_samples)
    initialization_and_warmup.update(peak_summary([
        initialization_memory, result['cold_forward'], initialization_and_warmup]))
    initialization_and_warmup.update({
        'warmup_iterations': args.warmup,
        'includes_cold_forward_and_output_check': True,
        'sample_count': result['cold_forward']['sample_count'] + warmup_samples['sample_count']})
    result['initialization_and_warmup'] = initialization_and_warmup
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() / MIB
    with ProcessMemorySampler() as sampler:
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        start = time.perf_counter()
        for _ in range(args.iterations):
            output = forward()
            del output
        end_event.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        milliseconds = start_event.elapsed_time(end_event) / args.iterations
        result['steady_pair_forward'] = {
            **memory_record(), 'baseline_allocated_mib': baseline,
            'peak_increment_mib': torch.cuda.max_memory_allocated()/MIB-baseline,
            'iterations': args.iterations, 'cuda_milliseconds_per_pair': milliseconds,
            'wall_milliseconds_per_pair': elapsed * 1000 / args.iterations,
            'unique_stream_output_fps_from_pair_timing': 2000 / milliseconds,
            'required_milliseconds_per_pair_for_120fps_output': 1000 / 60}
    result['steady_pair_forward'].update(sampler.report())

    # Reuse a host frame as a deterministic synthetic source; every yielded GPU
    # frame is a fresh transfer. Consume each HR result on the CPU immediately.
    del inputs
    torch.cuda.empty_cache()
    host_source = torch.rand(1, 3, args.height, args.width)

    def source():
        for _ in range(args.stream_frames):
            yield host_source.to('cuda')

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() / MIB
    with ProcessMemorySampler() as sampler:
        start = time.perf_counter()
        count = 0
        with torch.inference_mode(), precision_context():
            for frame in model.forward_sequence(source()):
                # Conversion happens on CPU so the output format does not add
                # another full-resolution floating-point CUDA tensor.
                host_output = frame.cpu().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                count += 1
                del host_output, frame
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result['bounded_stream_to_host_rgb8'] = {
            **memory_record(), 'baseline_allocated_mib': baseline,
            'peak_increment_mib': torch.cuda.max_memory_allocated()/MIB-baseline,
            'input_frames': args.stream_frames, 'output_frames': count,
            'seconds': elapsed, 'emitted_frames_per_second_including_final_endpoint': count/elapsed,
            'sustained_unique_frames_per_second': 2*(args.stream_frames-1)/elapsed,
            'input_transfer_and_host_quantization_included': True,
            'codec_included': False}
    result['bounded_stream_to_host_rgb8'].update(sampler.report())
    result['overall_measured_peak'] = peak_summary([result[name] for name in [
        'initialization_and_warmup', 'cold_forward', 'steady_pair_forward', 'bounded_stream_to_host_rgb8']])
    result['storage_inventory'] = storage_inventory(args.height, args.width, model.embed_dim, weight_bytes, buffer_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['precision', 'parameters', 'all_output_finite',
                     'initialization_and_warmup', 'cold_forward', 'steady_pair_forward',
                     'bounded_stream_to_host_rgb8', 'overall_measured_peak']}, indent=2))


if __name__ == '__main__':
    main()
