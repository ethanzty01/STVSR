"""Evaluate mixed-data tiny checkpoints on full Vimeo and REDS test windows."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from distill_data import REDSDistillationDataset, VimeoDistillationDataset
from train_tiny import evaluate, save_json, seed_everything
from models.RSTT import RSTT
from models.RSTT_tiny import RSTTTiny


def load_student(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = RSTTTiny(**checkpoint['model_config']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, {
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'step': checkpoint.get('step'),
        'base_step': checkpoint.get('base_step'),
        'cumulative_step': checkpoint.get('cumulative_step', checkpoint.get('step')),
        'validation_psnr_y': checkpoint.get('validation_psnr_y'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../Datasets_pt'))
    parser.add_argument('--teacher', type=Path,
                        default=Path('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth'))
    parser.add_argument('--vimeo-batch-size', type=int, default=4)
    parser.add_argument('--reds-batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Optional smoke-test limit per dataset')
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(2026)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    initial, initial_info = load_student(args.run_dir / 'initial.pth', device)
    best, best_info = load_student(args.run_dir / 'distilled_best.pth', device)
    last, last_info = load_student(args.run_dir / 'distilled_last.pth', device)
    teacher = RSTT(depths=[4] * 8).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True), strict=True)
    models = {'trained_best': best, 'trained_last': last}
    result = {
        'protocol': 'FP32; full-frame middle triple of each 7-frame window; '
                    'bicubic antialiased x4 degradation; rounded uint8 RGB/Y PSNR and Y SSIM; '
                    'no border crop; arithmetic frame mean; initial denotes previous 20k tiny',
        'checkpoints': {'initial': initial_info, 'trained_best': best_info, 'trained_last': last_info},
        'teacher_sha256': hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        'datasets': {},
    }
    for name, cls, batch_size in [
        ('Vimeo-90k', VimeoDistillationDataset, args.vimeo_batch_size),
        ('REDS', REDSDistillationDataset, args.reds_batch_size),
    ]:
        dataset = cls(args.data_root, 'test', patch_size=None, limit=args.limit)
        directory = args.run_dir / ('smoke_test' if args.limit else 'test') / name
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / 'test_ids.json', [record['id'] for record in dataset.records])
        metrics, seconds = evaluate(models, teacher, initial, dataset, device, directory, batch_size)
        comparison = {}
        for label in models:
            comparison[label] = {
                kind: {
                    'psnr_y_gain_over_previous': metrics[label]['all'][kind]['psnr_y'] - metrics['initial']['all'][kind]['psnr_y'],
                    'psnr_y_gap_to_teacher': metrics['teacher']['all'][kind]['psnr_y'] - metrics[label]['all'][kind]['psnr_y'],
                    'ssim_y_gain_over_previous': metrics[label]['all'][kind]['ssim_y'] - metrics['initial']['all'][kind]['ssim_y'],
                }
                for kind in ['all', 'endpoints', 'center']
            }
        entry = {'windows': len(dataset), 'scored_frames_per_model': len(dataset) * 3,
                 'batch_size': batch_size, 'seconds': seconds, 'metrics': metrics, 'comparison': comparison}
        save_json(directory / 'results.json', entry)
        result['datasets'][name] = entry
        print(json.dumps({'event': 'dataset_evaluated', 'dataset': name, **entry}), flush=True)
    destination = args.run_dir / ('smoke_test_results.json' if args.limit else 'results.json')
    save_json(destination, result)
    print(json.dumps({'event': 'evaluation_complete', 'results': str(destination)}), flush=True)


if __name__ == '__main__':
    main()
