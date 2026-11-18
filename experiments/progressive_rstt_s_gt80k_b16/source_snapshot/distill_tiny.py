"""Train paired supervised/KD students and evaluate held-out Vimeo clips."""
import argparse
import copy
import csv
import hashlib
import json
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from distill_data import VimeoDistillationDataset
from distill_metrics import compute_metrics
from models.RSTT import RSTT
from models.RSTT_tiny import RSTTTiny, interpolate_time


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def charbonnier(prediction, target):
    return ((prediction - target).square() + 1e-6).sqrt().mean()


def teacher_targets(teacher, batch, device):
    output = teacher(batch['teacher_lr'].to(device))
    starts = batch['pair_index'].to(device) * 2
    indices = starts[:, None] + torch.arange(3, device=device)[None]
    return output[torch.arange(len(starts), device=device)[:, None], indices]


def baseline(x):
    temporal = interpolate_time(x)
    b, t, c, h, w = temporal.shape
    return F.interpolate(temporal.reshape(b*t, c, h, w), scale_factor=4,
                         mode='bilinear', align_corners=False).reshape(b, t, c, h*4, w*4)


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n')


def save_checkpoint(path, model, step, score, optimizer=None):
    content = {'state_dict': model.state_dict(), 'step': step, 'validation_psnr_y': score,
               'model_config': {'embed_dim': model.embed_dim, 'depth': model.depth,
                                'drop_path_rate': model.drop_path_rate, 'head': model.head}}
    if optimizer is not None:
        content['optimizer'] = optimizer.state_dict()
    torch.save(content, path)


@torch.no_grad()
def validate(models, loader, device):
    totals = {name: [] for name in models}
    for model in models.values():
        model.eval()
    for batch in loader:
        inputs, target = batch['student_lr'].to(device), batch['gt']
        for name, model in models.items():
            metrics = compute_metrics(model(inputs).cpu(), target)
            totals[name].extend(frame['psnr_y'] for clip in metrics['per_frame'] for frame in clip)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def aggregate(rows):
    result = {}
    for model in sorted({r['model'] for r in rows}):
        selected = [r for r in rows if r['model'] == model]
        result[model] = {}
        for group in ['all', 'slow', 'medium', 'fast']:
            group_rows = [r for r in selected if group == 'all' or r['group'] == group]
            if not group_rows:
                continue
            result[model][group] = {}
            for kind in ['all', 'endpoints', 'center']:
                frames = [r for r in group_rows if kind == 'all' or
                          (r['frame'] % 2 == (1 if kind == 'center' else 0))]
                result[model][group][kind] = {
                    metric: float(np.mean([r[metric] for r in frames]))
                    for metric in ['psnr_rgb', 'psnr_y', 'ssim_y']}
    return result


def score_batch(predictions, targets, keys, groups):
    rows = []
    for name, prediction in predictions.items():
        measurements = compute_metrics(prediction, targets)
        for b, frames in enumerate(measurements['per_frame']):
            for f, metrics in enumerate(frames):
                rows.append({'key': keys[b], 'group': groups[b],
                             'model': name, 'frame': f, **metrics})
    return rows


@torch.no_grad()
def evaluate(models, teacher, initial, dataset, device, output_dir, batch_size):
    for model in [teacher, initial, *models.values()]:
        if model is not None:
            model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    start_time = time.perf_counter()
    # Bound the pending host predictions while overlapping CPU scoring and GPU inference.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = deque()
        for i, batch in enumerate(loader):
            inputs, targets = batch['student_lr'].to(device), batch['gt']
            predictions = {}
            for name in (['teacher'] if teacher is not None else []) + ['initial', 'bilinear', *models]:
                if name == 'teacher':
                    prediction = teacher_targets(teacher, batch, device)
                elif name == 'initial':
                    prediction = initial(inputs)
                elif name == 'bilinear':
                    prediction = baseline(inputs)
                else:
                    prediction = models[name](inputs)
                predictions[name] = prediction.cpu()
                del prediction
            pending.append(pool.submit(score_batch, predictions, targets, batch['key'], batch['group']))
            if len(pending) >= 4:
                rows.extend(pending.popleft().result())
            if i % 25 == 0:
                print(json.dumps({'event': 'test', 'clips': min((i+1)*batch_size, len(dataset)),
                                  'total': len(dataset), 'seconds': time.perf_counter()-start_time}), flush=True)
        for future in pending:
            rows.extend(future.result())
    with (output_dir / 'test_per_frame.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return aggregate(rows), time.perf_counter()-start_time


@torch.no_grad()
def benchmark(teacher, student, device):
    result = {}
    for name, model, frames in [('teacher', teacher, 4), ('student', student, 2)]:
        if model is None:
            continue
        x = torch.rand(1, frames, 3, 64, 112, device=device)
        model.eval()
        for _ in range(3):
            model(x)
        synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        for _ in range(20):
            y = model(x)
            del y
        synchronize(device)
        result[name] = {'input_shape': list(x.shape),
                        'output_frames': 7 if frames == 4 else 3,
                        'milliseconds_per_call': (time.perf_counter()-start)*1000/20,
                        'parameters': sum(p.numel() for p in model.parameters())}
        result[name]['milliseconds_per_output_frame'] = result[name]['milliseconds_per_call'] / result[name]['output_frames']
        if device.type == 'cuda':
            result[name]['peak_increment_mib'] = (torch.cuda.max_memory_allocated(device)-before)/2**20
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('../Datasets_pt'))
    parser.add_argument('--teacher', type=Path, default=Path('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--depth', type=int, default=2, help='Individual blocks per stage')
    parser.add_argument('--head', choices=['direct', 'legacy'], default='direct')
    parser.add_argument('--mode', choices=['paired', 'supervised', 'distill'], default='paired')
    parser.add_argument('--init', type=Path, help='Start a new training schedule from saved student weights')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--kd-weight', type=float, default=.5)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--validation-count', type=int, default=128)
    parser.add_argument('--validation-limit', type=int, default=128)
    parser.add_argument('--test-per-group', type=int, default=0, help='0 uses the entire official test split')
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--evaluate-only', action='store_true', help='Evaluate existing best checkpoints in output')
    args = parser.parse_args()
    if args.evaluate_only:
        saved = json.loads((args.output / 'config.json').read_text())
        for key in ['seed', 'validation_count', 'validation_limit']:
            setattr(args, key, saved[key])
        args.data_root = Path(saved['data_root'])
        args.teacher = Path(saved['teacher'])
        args.depth = saved['student']['depth']
        args.head = saved['student'].get('head', 'legacy')
        args.mode = saved.get('mode', 'paired')
        args.init = None
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.steps < 1 or args.batch_size < 1 or args.kd_weight < 0 or args.val_every < 1:
        parser.error('steps/batch-size must be positive and kd-weight nonnegative')
    if not args.evaluate_only:
        args.output.mkdir(parents=True, exist_ok=False)
    teacher = None
    if args.mode != 'supervised':
        teacher = RSTT(depths=[4]*8).to(device).eval()
        teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True), strict=True)
        teacher.requires_grad_(False)
    student = RSTTTiny(depth=args.depth, head=args.head, drop_path_rate=0.).to(device)
    if args.init is not None:
        checkpoint = torch.load(args.init, map_location=device, weights_only=True)
        student.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    supervised = copy.deepcopy(student)
    initial = copy.deepcopy(student).eval().requires_grad_(False)
    models = ({'supervised': student} if args.mode == 'supervised' else
              {'distilled': student} if args.mode == 'distill' else
              {'distilled': student, 'supervised': supervised})
    config = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    config.update({'torch': torch.__version__, 'device_name': torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                   'teacher_sha256': hashlib.sha256(args.teacher.read_bytes()).hexdigest() if teacher is not None else None,
                   'init_sha256': hashlib.sha256(args.init.read_bytes()).hexdigest() if args.init is not None else None,
                   'precision': 'FP32', 'student': {'embed_dim':32, 'depth':args.depth, 'head':args.head, 'num_frames':2, 'drop_path_rate':0},
                   'protocol': 'HR128 train patches; held-out train validation; full-frame test middle triple; bicubic antialias=True LR; rounded uint8 Y metrics, no border crop'})
    train = VimeoDistillationDataset(args.data_root, 'train', validation_count=args.validation_count, seed=args.seed)
    val = VimeoDistillationDataset(args.data_root, 'val', validation_count=args.validation_count,
                                  seed=args.seed, limit=args.validation_limit)
    if not args.evaluate_only:
        save_json(args.output / 'split_ids.json', {'train':[r['id'] for r in train.records], 'val':[r['id'] for r in val.records]})
    config['training_clips'] = len(train)
    config['validation_clips'] = len(val)
    best = {name: -float('inf') for name in models}
    start = time.perf_counter()
    if not args.evaluate_only:
        save_json(args.output / 'config.json', config)
        save_checkpoint(args.output / 'initial.pth', initial, 0, None)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if args.init is not None:
            # Keep the starting weights eligible if additional training degrades validation.
            best = validate(models, val_loader, device)
            save_json(args.output / 'initial_validation.json', best)
            for name, model in models.items():
                save_checkpoint(args.output / f'{name}_best.pth', model, 0, best[name])
        iterator = iter(loader)
        optimizers = {name:torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(.9,.99), weight_decay=1e-4)
                      for name,m in models.items()}
        schedulers = {name:torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=1e-6)
                      for name,opt in optimizers.items()}
        with (args.output / 'train.jsonl').open('w', buffering=1) as log:
            for step in range(1,args.steps+1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                x,gt = batch['student_lr'].to(device),batch['gt'].to(device)
                with torch.no_grad():
                    soft = teacher_targets(teacher,batch,device) if teacher is not None else None
                losses = {}
                for name,model in models.items():
                    model.train()
                    optimizers[name].zero_grad(set_to_none=True)
                    pred = model(x)
                    hard = charbonnier(pred,gt)
                    kd = charbonnier(pred,soft) if name=='distilled' else pred.new_zeros(())
                    loss = hard + (args.kd_weight*kd if name=='distilled' else 0.)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss at step {step}: {name}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizers[name].step()
                    schedulers[name].step()
                    losses[name] = {'gt':hard.item(),'kd':kd.item(),'total':loss.item()}
                record = {'step':step,'elapsed_seconds':time.perf_counter()-start,'losses':losses,
                          'lr':next(iter(optimizers.values())).param_groups[0]['lr']}
                if step%args.val_every==0 or step==args.steps:
                    scores = validate(models,val_loader,device)
                    record['validation_psnr_y'] = scores
                    for name in models:
                        if scores[name]>best[name]:
                            best[name] = scores[name]
                            save_checkpoint(args.output/f'{name}_best.pth',models[name],step,scores[name])
                        save_checkpoint(args.output/f'{name}_last.pth',models[name],step,scores[name],optimizers[name])
                log.write(json.dumps(record)+'\n')
                if step%25==0 or step==1:
                    print(json.dumps(record),flush=True)
        training_seconds = time.perf_counter()-start
        save_json(args.output/'training_summary.json', {'seconds':training_seconds, 'steps':args.steps,
                   'examples_per_arm':args.steps*args.batch_size,'best_validation_psnr_y':best})
    else:
        training_seconds = json.loads((args.output/'training_summary.json').read_text())['seconds']
    initial.load_state_dict(torch.load(args.output/'initial.pth', map_location=device,weights_only=True)['state_dict'])
    chosen = {}
    for name,model in models.items():
        checkpoint=torch.load(args.output/f'{name}_best.pth',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        chosen[name] = {'step':checkpoint['step'],'validation_psnr_y':checkpoint['validation_psnr_y']}
    test = VimeoDistillationDataset(args.data_root,'test',patch_size=None,seed=args.seed)
    if args.test_per_group:
        indices=[]
        for group in ['slow','medium','fast']:
            candidates=[i for i,r in enumerate(test.records) if r['motion_group']==group]
            random.Random(args.seed).shuffle(candidates)
            indices.extend(candidates[:args.test_per_group])
        test_ids=[test.records[i]['id'] for i in indices]
        test=Subset(test,indices)
    else:
        test_ids=[r['id'] for r in test.records]
    save_json(args.output/'test_ids.json',test_ids)
    results,test_seconds=evaluate(models,teacher,initial,test,device,args.output,args.test_batch_size)
    timings=benchmark(teacher,student,device)
    summary={'config':json.loads((args.output/'config.json').read_text()),'test_clips':len(test),
             'evaluation': {'batch_size': args.test_batch_size, 'test_per_group': args.test_per_group,
                            'target_frames_1based': [3, 4, 5], 'device': str(device)},
             'test_frames_per_model':len(test)*3,'selected_checkpoints':chosen,'training_seconds':training_seconds,
             'test_seconds':test_seconds,'metrics':results,'benchmark':timings}
    save_json(args.output/'results.json',summary)
    print(json.dumps({'event':'complete','results':str(args.output/'results.json'),'metrics':results,'benchmark':timings}),flush=True)


if __name__=='__main__':
    main()
