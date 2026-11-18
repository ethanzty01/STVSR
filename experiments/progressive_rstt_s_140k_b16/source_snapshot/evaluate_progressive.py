"""Compare the exported progressive tiny, previous tiny and fixed RSTT-S."""
import argparse
import csv
import hashlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distill_data import REDSDistillationDataset
from distill_metrics import compute_metrics
from eval_paper_tiny import VimeoSeven
from models.RSTT import RSTT
from models.RSTT_tiny import RSTTTiny
from utils.eval_utils import index_generation

METRICS = ('psnr_rgb', 'psnr_y', 'ssim_y')
KINDS = ('all', 'endpoints', 'center')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def checkpoint_info(path, checkpoint=None):
    result = {'path': str(path.resolve()),
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if checkpoint is not None:
        for key in ('step', 'stage', 'selected_stage', 'source_checkpoint', 'cumulative_step',
                    'validation_psnr_y', 'model_config'):
            if key in checkpoint:
                result[key] = checkpoint[key]
    return result


def load_models(args, device):
    models, info = {}, {}
    for name, path in [('progressive', args.run_dir / 'final.pth'),
                       ('previous_tiny', args.previous)]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = RSTTTiny(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        models[name] = model.to(device).eval().requires_grad_(False)
        info[name] = checkpoint_info(path, checkpoint)
    teacher = RSTT(depths=[4] * 8)
    teacher.load_state_dict(torch.load(args.teacher, map_location='cpu', weights_only=True), strict=True)
    models['teacher'] = teacher.to(device).eval().requires_grad_(False)
    info['teacher'] = checkpoint_info(args.teacher)
    for name, model in models.items():
        info[name]['parameters'] = sum(p.numel() for p in model.parameters())
    return models, info


def stitch_pairs(pairs):
    """Keep the first prediction of each duplicated endpoint: 3+2+2 frames."""
    if len(pairs) != 3 or any(pair.shape[1] != 3 for pair in pairs):
        raise ValueError('Expected three predictions containing three frames each')
    return torch.cat([pairs[0], pairs[1][:, 1:], pairs[2][:, 1:]], dim=1)


@torch.inference_mode()
def predict_seven(models, lr):
    outputs = {}
    for name, model in models.items():
        if name == 'teacher':
            outputs[name] = model(lr).cpu()
        else:
            outputs[name] = stitch_pairs([model(lr[:, i:i + 2]).cpu() for i in range(3)])
    return outputs


def unique_vid4_windows(length, limit=None):
    """Return source indices and local indices retained on their first occurrence."""
    seen = set()
    for number, indices in enumerate(index_generation(7, length)):
        if limit is not None and number >= limit:
            break
        keep = [i for i, frame in enumerate(indices) if frame not in seen]
        seen.update(indices)
        if keep:
            yield indices, keep


def score(outputs, targets, keys, groups, frame_indices=None, keep=None):
    rows = []
    for name, prediction in outputs.items():
        for b, frames in enumerate(compute_metrics(prediction, targets)['per_frame']):
            indices = list(range(len(frames))) if frame_indices is None else frame_indices[b]
            for local, metrics in enumerate(frames):
                if keep is not None and local not in keep:
                    continue
                rows.append({'model': name, 'key': keys[b], 'group': groups[b],
                             'frame': indices[local], 'local_frame': local,
                             'window_start': indices[0],
                             # Observed/interpolated is relative to this window's input times.
                             'kind': 'center' if local % 2 else 'endpoints', **metrics})
    return rows


def mean_metrics(rows):
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def kind_metrics(rows):
    return {kind: mean_metrics([r for r in rows if kind == 'all' or r['kind'] == kind])
            for kind in KINDS if any(kind == 'all' or r['kind'] == kind for r in rows)}


def summarize(rows, equal_sequence=False):
    result = {}
    for model in sorted({row['model'] for row in rows}):
        model_rows = [row for row in rows if row['model'] == model]
        groups = {group: kind_metrics([r for r in model_rows if r['group'] == group])
                  for group in sorted({r['group'] for r in model_rows})}
        result[model] = {'all': kind_metrics(model_rows), 'groups': groups}
        if equal_sequence:
            result[model]['equal_sequence_mean'] = {
                kind: {metric: float(np.mean([group[kind][metric] for group in groups.values()
                                             if kind in group])) for metric in METRICS}
                for kind in KINDS}
    return result


def comparisons(metrics, equal_sequence=False):
    field = 'equal_sequence_mean' if equal_sequence else 'all'
    result = {}
    groups = {'all': {name: value[field] for name, value in metrics.items()}}
    for group in metrics['progressive']['groups']:
        groups[group] = {name: value['groups'][group] for name, value in metrics.items()}
    for group, values in groups.items():
        result[group] = {}
        for kind in values['progressive']:
            new, old, teacher = [values[name][kind]
                                 for name in ('progressive', 'previous_tiny', 'teacher')]
            result[group][kind] = {
                'psnr_y_gain_over_previous': new['psnr_y'] - old['psnr_y'],
                'psnr_y_gap_to_teacher': teacher['psnr_y'] - new['psnr_y'],
                'ssim_y_gain_over_previous': new['ssim_y'] - old['ssim_y'],
                'ssim_y_gap_to_teacher': teacher['ssim_y'] - new['ssim_y']}
    return result


def save_dataset(directory, rows, metadata, equal_sequence=False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = summarize(rows, equal_sequence)
    result = {**metadata, 'metrics': metrics,
              'comparison': comparisons(metrics, equal_sequence)}
    save_json(directory / 'results.json', result)
    return result


def evaluate_vimeo(models, args, device, directory):
    dataset = VimeoSeven(args.data_root, 'test', patch_size=None, limit=args.limit)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / 'test_ids.json', [r['id'] for r in dataset.records])
    rows, started = [], time.perf_counter()
    loader = DataLoader(dataset, batch_size=args.vimeo_batch_size, num_workers=args.workers)
    with ThreadPoolExecutor(max_workers=args.metric_workers) as pool:
        pending = deque()
        for i, batch in enumerate(loader):
            outputs = predict_seven(models, batch['lr'].to(device))
            pending.append(pool.submit(score, outputs, batch['hr'], batch['key'], batch['group']))
            if len(pending) >= args.metric_workers:
                rows.extend(pending.popleft().result())
            if i % 100 == 0:
                print(json.dumps({'dataset': 'vimeo', 'clips': min((i + 1) * args.vimeo_batch_size, len(dataset)),
                                  'total': len(dataset), 'seconds': time.perf_counter() - started}), flush=True)
        for future in pending:
            rows.extend(future.result())
    return save_dataset(directory, rows, {'clips': len(dataset), 'frames_per_model': len(dataset) * 7,
                        'batch_size': args.vimeo_batch_size, 'seconds': time.perf_counter() - started})


def evaluate_vid4(models, args, device, directory):
    manifest = json.loads((args.data_root / 'test/Vid4/manifest.json').read_text())
    rows, frame_counts, started = [], {}, time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.metric_workers) as pool:
        pending = deque()
        for sequence, paths in sorted(manifest['sequences'].items()):
            frames = torch.cat([torch.load(args.data_root / path, map_location='cpu', weights_only=True)['hr']
                                for path in sorted(paths)]).float().div_(255)
            height, width = frames.shape[-2:]
            frames = frames[..., :height - height % 4, :width - width % 4]
            lr = F.interpolate(frames, scale_factor=.25, mode='bicubic', align_corners=False,
                               antialias=True).clamp_(0, 1).mul_(255).round_().div_(255)
            count = 0
            for indices, keep in unique_vid4_windows(len(frames), args.limit):
                count += len(keep)
                outputs = predict_seven(models, lr[indices[::2]][None].to(device))
                pending.append(pool.submit(score, outputs, frames[indices][None], [sequence], [sequence],
                                           [indices], keep))
                if len(pending) >= args.metric_workers:
                    rows.extend(pending.popleft().result())
            frame_counts[sequence] = count
            print(json.dumps({'dataset': 'vid4', 'sequence': sequence, 'frames': count}), flush=True)
        for future in pending:
            rows.extend(future.result())
    for model in models:
        identities = [(r['key'], r['frame']) for r in rows if r['model'] == model]
        if len(set(identities)) != len(identities) or len(identities) != sum(frame_counts.values()):
            raise RuntimeError(f'Duplicate or missing Vid4 frame labels for {model}')
    return save_dataset(directory, rows, {'frames_per_sequence': frame_counts,
                        'frames_per_model': sum(frame_counts.values()),
                        'seconds': time.perf_counter() - started}, equal_sequence=True)


@torch.inference_mode()
def predict_reds(models, batch, device):
    outputs = {}
    for name, model in models.items():
        if name == 'teacher':
            prediction = model(batch['teacher_lr'].to(device))
            starts = batch['pair_index'].to(device) * 2
            time_indices = starts[:, None] + torch.arange(3, device=device)[None]
            outputs[name] = prediction[torch.arange(len(starts), device=device)[:, None], time_indices].cpu()
        else:
            outputs[name] = model(batch['student_lr'].to(device)).cpu()
    return outputs


def evaluate_reds(models, args, device, directory):
    dataset = REDSDistillationDataset(args.data_root, 'test', patch_size=None, limit=args.limit)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / 'test_ids.json', [r['id'] for r in dataset.records])
    rows, started = [], time.perf_counter()
    loader = DataLoader(dataset, batch_size=args.reds_batch_size, num_workers=args.workers)
    with ThreadPoolExecutor(max_workers=args.metric_workers) as pool:
        pending = deque()
        for i, batch in enumerate(loader):
            outputs = predict_reds(models, batch, device)
            indices = [[int(key.rsplit('/', 1)[1]) + int(pair) * 2 + t for t in range(3)]
                       for key, pair in zip(batch['key'], batch['pair_index'])]
            pending.append(pool.submit(score, outputs, batch['gt'], batch['key'], batch['group'], indices))
            if len(pending) >= args.metric_workers:
                rows.extend(pending.popleft().result())
            if i % 25 == 0:
                print(json.dumps({'dataset': 'reds', 'clips': min((i + 1) * args.reds_batch_size, len(dataset)),
                                  'total': len(dataset), 'seconds': time.perf_counter() - started}), flush=True)
        for future in pending:
            rows.extend(future.result())
    return save_dataset(directory, rows, {'windows': len(dataset), 'frames_per_model': len(dataset) * 3,
                        'batch_size': args.reds_batch_size, 'seconds': time.perf_counter() - started})


@torch.inference_mode()
def benchmark_fps(models, device):
    results = {}
    for name, model in models.items():
        frames = 4 if name == 'teacher' else 2
        x = torch.rand(1, frames, 3, 144, 180, device=device)
        for _ in range(10):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(100):
            prediction = model(x)
            del prediction
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - start) / 100
        results[name] = {'input_shape': list(x.shape), 'output_frames_per_call': frames * 2 - 1,
                         'milliseconds_per_call': seconds * 1000,
                         'returned_frames_per_second': (frames * 2 - 1) / seconds,
                         'unique_stream_frames_per_second': (frames * 2 - 2) / seconds}
    return {'protocol': 'FP32; LR 144x180; batch 1; 10 warmups, 100 timed calls; model forward only; '
                        'unique streaming FPS excludes the shared endpoint between calls.',
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'models': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../Datasets_pt'))
    parser.add_argument('--previous', type=Path, default=Path('experiments/tiny_mixed_200k_b16/distilled_best.pth'))
    parser.add_argument('--teacher', type=Path, default=Path('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth'))
    parser.add_argument('--datasets', nargs='+', choices=('vimeo', 'vid4', 'reds'), default=['vimeo', 'vid4', 'reds'])
    parser.add_argument('--vimeo-batch-size', type=int, default=4)
    parser.add_argument('--reds-batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--metric-workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='Smoke clips per dataset; Vid4 windows per sequence')
    parser.add_argument('--fps', action='store_true', help='Time model forward on an otherwise idle device')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if min(args.vimeo_batch_size, args.reds_batch_size, args.metric_workers, args.threads) < 1 or args.workers < 0:
        parser.error('Batch sizes and thread counts must be positive; workers must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    directory = args.run_dir / ('evaluation_smoke' if args.limit else 'evaluation')
    directory.mkdir(parents=True, exist_ok=True)
    models, info = load_models(args, device)
    result = {'checkpoints': info, 'smoke_limit': args.limit, 'datasets': {},
              'protocol': {'shared': 'FP32; RGB rounded to uint8 before RGB/Y PSNR and Y SSIM; no border crop.',
                           'vimeo': 'Seven output frames, three adjacent tiny pairs, first occurrence of shared endpoint; '
                                    'x4 PyTorch antialiased bicubic LR rounded to 8-bit; arithmetic frame means.',
                           'vid4': 'Original index_generation(7, N); each global frame scored once, first occurrence; '
                                   'LR as Vimeo; equally weighted sequence means; endpoint/center follows the retained '
                                   'window input timestamps, which can change parity in the final window.',
                           'reds': 'Every seven-frame test window, middle three targets; x4 antialiased bicubic '
                                   'float LR; arithmetic frame-observation means including overlapping windows.'}}
    evaluators = {'vimeo': evaluate_vimeo, 'vid4': evaluate_vid4, 'reds': evaluate_reds}
    for dataset in args.datasets:
        result['datasets'][dataset] = evaluators[dataset](models, args, device, directory / dataset)
        save_json(directory / 'results.json', result)
        print(json.dumps({'event': 'dataset_complete', 'dataset': dataset,
                          'comparison': result['datasets'][dataset]['comparison']}), flush=True)
    if args.fps:
        result['fps'] = benchmark_fps(models, device)
        save_json(directory / 'fps.json', result['fps'])
    save_json(directory / 'results.json', result)
    print(json.dumps({'event': 'evaluation_complete', 'path': str(directory / 'results.json')}), flush=True)
    return result


if __name__ == '__main__':
    main()
