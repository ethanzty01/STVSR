"""Stream a video through standalone tiny: spatial x4 and temporal x2.

OpenCV decodes/encodes on the CPU. GPU inference keeps a pair of source frames
and consumes outputs immediately. The final predicted frame is held once so
N input frames produce 2N encoded frames at twice the input frame rate.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import torch

from benchmark_tiny_4k import ProcessMemorySampler, memory_record
from models.RSTT_tiny import RSTTTiny


def open_capture(path):
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])


def inspect_video(path):
    capture = open_capture(path)
    if not capture.isOpened():
        raise RuntimeError(f'Could not reopen encoded video: {path}')
    metadata = {'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': float(capture.get(cv2.CAP_PROP_FPS)),
                'container_frame_count': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                'decoded_frame_count': 0}
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[:2] != (metadata['height'], metadata['width']):
                raise RuntimeError('Encoded video contains inconsistent frame dimensions')
            metadata['decoded_frame_count'] += 1
            del frame
    finally:
        capture.release()
    metadata['duration_seconds'] = metadata['decoded_frame_count'] / metadata['fps']
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='Defaults to output path with .json suffix')
    parser.add_argument('--precision', choices=['fp32', 'amp-fp16'], default='fp32')
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix('.json')
    if args.input.resolve() == args.output.resolve():
        parser.error('Choose separate input and output video paths')
    if args.output.exists() or report_path.exists():
        parser.error('Choose new output/report paths; a requested destination already exists')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    capture = open_capture(args.input)
    if not capture.isOpened():
        raise RuntimeError(f'Could not open input video: {args.input}')
    writer = None
    try:
        input_fps = float(capture.get(cv2.CAP_PROP_FPS))
        success, first_frame = capture.read()
        if not success:
            raise ValueError('Input video contains no decodable frames')
        if not math.isfinite(input_fps) or input_fps <= 0:
            raise ValueError('Input video must contain a positive frame rate')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for GPU inference and memory measurement')
        height, width = first_frame.shape[:2]
        output_fps = 2 * input_fps
        output_size = (4 * width, 4 * height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.CAP_FFMPEG,
                                 cv2.VideoWriter_fourcc(*'mp4v'), output_fps, output_size)
        if not writer.isOpened():
            raise RuntimeError('OpenCV FFmpeg could not initialize the mp4v encoder')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', {
            'embed_dim': 32, 'depth': 2, 'head': 'direct', 'drop_path_rate': 0})
        model = RSTTTiny(**model_config).eval().requires_grad_(False)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
        del checkpoint
        model.cuda()
        input_count = 0

        def source():
            nonlocal input_count, first_frame
            frame = first_frame
            first_frame = None
            while frame is not None:
                if frame.shape[:2] != (height, width):
                    raise ValueError('Every input frame must have the same dimensions')
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_count += 1
                yield torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
                    device='cuda', dtype=torch.float32).div_(255)
                success, frame = capture.read()
                if not success:
                    frame = None

        # Reset before the first forward so cuDNN's initial algorithm search is
        # included along with every later decode/infer/encode iteration.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() / 2**20
        predicted_count = 0
        precision = torch.autocast('cuda', dtype=torch.float16) if args.precision == 'amp-fp16' else nullcontext()
        with ProcessMemorySampler() as sampler:
            start = time.perf_counter()
            with torch.inference_mode(), precision:
                for gpu_frame in model.forward_sequence(source()):
                    cpu_frame = gpu_frame[0].cpu().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                    bgr_frame = cv2.cvtColor(cpu_frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
                    writer.write(bgr_frame)
                    predicted_count += 1
                    # Keep only the latest CPU BGR frame for the final hold.
                    del cpu_frame, gpu_frame
            writer.write(bgr_frame)
            encoded_count = predicted_count + 1
            del bgr_frame
            torch.cuda.synchronize()
            writer.release()
            writer = None
            elapsed = time.perf_counter() - start
            measured_memory = memory_record()
        measured_memory.update(sampler.report())
        measured_memory['baseline_allocated_mib'] = baseline
        measured_memory['peak_increment_mib'] = measured_memory['peak_allocated_mib'] - baseline
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    verified = inspect_video(args.output)
    if (predicted_count != 2 * input_count - 1 or encoded_count != 2 * input_count or
            verified['decoded_frame_count'] != encoded_count or
            (verified['width'], verified['height']) != output_size or
            not math.isclose(verified['fps'], output_fps, rel_tol=1e-6)):
        raise RuntimeError(f'Encoded video verification failed: {verified}')
    report = {
        'input': str(args.input.resolve()), 'output': str(args.output.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'model_config': model_config, 'parameters': sum(p.numel() for p in model.parameters()),
        'precision': args.precision, 'torch': torch.__version__, 'opencv': cv2.__version__,
        'gpu': torch.cuda.get_device_name(), 'cudnn_benchmark': True,
        'input_width': width, 'input_height': height, 'input_fps': input_fps,
        'input_frames': input_count, 'model_predicted_frames': predicted_count,
        'final_hold_frames': 1, 'encoded_frames': encoded_count,
        'input_duration_seconds': input_count / input_fps,
        'end_to_end_seconds': elapsed, 'encoded_frames_per_second': encoded_count / elapsed,
        'verified_output': verified, 'overall_measured_peak': measured_memory,
        'codec': 'mp4v', 'codec_backend': 'OpenCV FFmpeg CPU',
        'protocol': 'Standalone tiny, batch 1, inference_mode, bounded pair streaming; CPU decode, GPU forward, immediate CPU RGB8 conversion and CPU encoding. Includes cold-forward peak. Final CPU frame held once to preserve duration.'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
