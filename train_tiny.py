"""Train RSTTTiny directly, continue mixed-data training, or reduce RSTT-S in stages."""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from distill_data import VimeoDistillationDataset
from distill_metrics import compute_metrics
from models.RSTT import RSTT
from models.RSTT_progressive import ProgressiveRSTT, migrate_model, select_channels
from models.RSTT_tiny import RSTTTiny, interpolate_time


# Direct supervised training and distillation

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


def run_direct(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(sys.argv[0]).name} direct', description=__doc__)
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
    args = parser.parse_args(argv)
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


# Mixed Vimeo/REDS training and continuation

class SeededMixedDataset(Dataset):
    """Sample keys carry augmentation seeds, independent of loader prefetch state."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, key):
        source, index, seed = key
        random.seed(seed)
        return self.datasets[source][index]


class BalancedStepSampler(Sampler):
    """Half of each batch comes from each dataset, sampled with replacement."""

    def __init__(self, lengths, batch_size, steps, seed, start_step=0):
        if len(lengths) != 2 or min(lengths) < 1 or batch_size < 2 or batch_size % 2:
            raise ValueError('Two nonempty datasets and an even batch size are required')
        self.lengths, self.batch_size = lengths, batch_size
        self.steps, self.seed, self.start_step = steps, seed, start_step

    def __len__(self):
        return max(0, self.steps - self.start_step)

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            rng = random.Random(self.seed + step * 1_000_003)
            batch = [(source, rng.randrange(self.lengths[source]), rng.randrange(2**63))
                     for source in (0, 1) for _ in range(self.batch_size // 2)]
            rng.shuffle(batch)
            yield batch


def make_datasets(args):
    from distill_data import REDSDistillationDataset, VimeoDistillationDataset
    validation_patch = getattr(args, 'validation_patch_size', None) or args.patch_size
    train = [VimeoDistillationDataset(args.data_root, 'train', patch_size=args.patch_size,
                                     validation_count=128, seed=args.seed),
             REDSDistillationDataset(args.data_root, 'train', patch_size=args.patch_size,
                                     seed=args.seed)]
    validation = {
        'vimeo': VimeoDistillationDataset(args.data_root, 'val', patch_size=validation_patch,
                                         validation_count=128, seed=args.seed,
                                         limit=args.validation_limit),
        'reds': REDSDistillationDataset(args.data_root, 'val', patch_size=validation_patch,
                                       seed=args.seed, limit=args.validation_limit),
    }
    return train, validation


def make_train_loader(args, datasets, start_step=0):
    sampling_seed = getattr(args, 'sampling_seed', None)
    sampling_seed = args.seed if sampling_seed is None else sampling_seed
    kwargs = {'num_workers': args.workers, 'pin_memory': args.device.startswith('cuda')}
    if args.workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    # A dedicated generator keeps worker startup from changing the model RNG stream.
    return DataLoader(SeededMixedDataset(datasets),
                      batch_sampler=BalancedStepSampler(list(map(len, datasets)), args.batch_size,
                                                        args.steps, sampling_seed, start_step),
                      generator=torch.Generator().manual_seed(sampling_seed), **kwargs)


def build_components(args, device):
    state = torch.load(args.init, map_location='cpu', weights_only=True)
    model_config = state.get('model_config', {'embed_dim': 32, 'depth': 2, 'head': 'direct',
                                             'drop_path_rate': 0.})
    student = RSTTTiny(**model_config).to(device)
    student.load_state_dict(state.get('state_dict', state), strict=True)
    teacher = None
    if args.mode == 'distill':
        teacher = RSTT(depths=[4] * 8).to(device).eval().requires_grad_(False)
        teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(.9, .99),
                                  weight_decay=1e-4, fused=device.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    forward_student = torch.compile(student, mode=args.compile_mode) if args.compile else student
    forward_teacher = torch.compile(teacher, mode=args.compile_mode) if args.compile and teacher else teacher
    return student, teacher, forward_student, forward_teacher, optimizer, scheduler, scaler, state


def temporal_charbonnier(prediction, target, center_weight=1.):
    """Mean RGB Charbonnier with normalized weights [1, center_weight, 1]."""
    if center_weight == 1.:
        return charbonnier(prediction, target)
    if prediction.ndim != 5 or prediction.shape[1] != 3 or prediction.shape != target.shape:
        raise ValueError('Temporal weighting expects matching [B,3,C,H,W] tensors')
    frames = ((prediction - target).square() + 1e-6).sqrt().mean(dim=(0, 2, 3, 4))
    return (frames[0] + center_weight * frames[1] + frames[2]) / (2. + center_weight)


def mixed_train_step(args, batch, student, teacher, optimizer, scaler, device):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = batch['student_lr'].to(device, non_blocking=True)
    target = batch['gt'].to(device, non_blocking=True)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    with torch.autocast(device.type, dtype=dtype, enabled=args.precision != 'fp32'):
        with torch.no_grad():
            soft = teacher_targets(teacher, batch, device) if teacher is not None else None
        prediction = student(inputs)
        # Accumulate these reductions in float32 for stable small residuals.
        hard = temporal_charbonnier(prediction.float(), target.float(),
                                   getattr(args, 'center_weight', 1.))
        kd = charbonnier(prediction.float(), soft.float()) if soft is not None else hard.new_zeros(())
        loss = hard + args.kd_weight * kd
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    return {'gt': hard.item(), 'kd': kd.item(), 'total': loss.item()}


def rng_state():
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [numpy[0], numpy[1].tolist(), *numpy[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def atomic_checkpoint(path, student, step, base_step, validation, best_score, optimizer=None,
                      scheduler=None, scaler=None, elapsed_seconds=0.):
    checkpoint = {'state_dict': student.state_dict(), 'step': step,
                  'base_step': base_step, 'cumulative_step': base_step + step,
                  'validation_psnr_y': validation, 'best_validation_psnr_y': best_score,
                  'model_config': {'embed_dim': student.embed_dim, 'depth': student.depth,
                                   'drop_path_rate': student.drop_path_rate, 'head': student.head},
                  'elapsed_seconds': elapsed_seconds}
    if optimizer is not None:
        checkpoint.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          scaler=scaler.state_dict(), rng=rng_state())
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate_mixed(student, loaders, device, return_details=False):
    student.eval()
    details = {}
    for name, loader in loaders.items():
        rows = {'all': [], 'endpoints': [], 'center': []}
        for batch in loader:
            prediction = student(batch['student_lr'].to(device)).cpu()
            for clip in compute_metrics(prediction, batch['gt'])['per_frame']:
                for frame, metrics in enumerate(clip):
                    rows['all'].append(metrics)
                    rows['center' if frame == 1 else 'endpoints'].append(metrics)
        details[name] = {kind: {metric: float(np.mean([r[metric] for r in records]))
                               for metric in ('psnr_rgb', 'psnr_y', 'ssim_y')}
                         for kind, records in rows.items()}
    scores = {name: value['all']['psnr_y'] for name, value in details.items()}
    scores['balanced_mean'] = float(np.mean(list(scores.values())))
    return (scores, details) if return_details else scores


def evaluate_after_training(args):
    """Reuse the initialized CUDA context after releasing training allocations."""
    # Clear compiled graphs as well as Python cycles before returning cached blocks.
    if args.compile:
        torch.compiler.reset()
    gc.collect()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()
    from evaluate_tiny_mixed import main as evaluate_main
    evaluate_main(['--run-dir', str(args.output), '--data-root', str(args.data_root),
                   '--teacher', str(args.teacher), '--vimeo-batch-size', str(args.vimeo_batch_size),
                   '--reds-batch-size', str(args.reds_batch_size),
                   '--device', args.device, '--threads', str(args.threads)])


def parse_mixed_args(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(sys.argv[0]).name} mixed', description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('../Datasets_pt'))
    parser.add_argument('--init', type=Path, default=Path('experiments/tiny_direct_d2_20k_b8/distilled_best.pth'))
    parser.add_argument('--teacher', type=Path, default=Path('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200000, help='Additional optimizer steps')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=128, help='HR training patch side')
    parser.add_argument('--validation-patch-size', type=int,
                        help='Fixed HR validation patch side; defaults to training patch size')
    parser.add_argument('--center-weight', type=float, default=1.,
                        help='GT frame weights [1, center_weight, 1], normalized by their sum')
    parser.add_argument('--mode', choices=['supervised', 'distill'], default='distill')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--kd-weight', type=float, default=.5)
    parser.add_argument('--val-every', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=1000, help='Console interval; every step is written to train.jsonl')
    parser.add_argument('--validation-limit', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--sampling-seed', type=int,
                        help='Training sample/augmentation seed, independent of the fixed data split')
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune'], default='default')
    parser.add_argument('--resume', action='store_true', help='Restore output/distilled_last.pth and saved configuration')
    parser.add_argument('--profile-steps', type=int, default=0, help='Benchmark data loading and training, then exit')
    parser.add_argument('--profile-warmup', type=int, default=10)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--vimeo-batch-size', type=int, default=4)
    parser.add_argument('--reds-batch-size', type=int, default=1)
    args = parser.parse_args(argv)
    if args.resume:
        saved = json.loads((args.output / 'config.json').read_text())
        for name in ('data_root', 'init', 'teacher', 'steps', 'batch_size', 'patch_size', 'mode',
                     'lr', 'kd_weight', 'val_every', 'validation_limit', 'seed', 'precision',
                     'compile', 'compile_mode'):
            setattr(args, name, Path(saved[name]) if name in ('data_root', 'init', 'teacher') else saved[name])
        # Legacy checkpoints retain their original crop, loss and sampling behavior.
        args.validation_patch_size = saved.get('validation_patch_size', saved['patch_size'])
        args.center_weight = saved.get('center_weight', 1.)
        args.sampling_seed = saved.get('sampling_seed', saved['seed'])
    if args.validation_patch_size is None:
        args.validation_patch_size = args.patch_size
    if args.sampling_seed is None:
        args.sampling_seed = args.seed
    if any(side < 16 or side % 4 for side in (args.patch_size, args.validation_patch_size)):
        parser.error('Training/validation patch sides must be multiples of four, at least 16')
    if not math.isfinite(args.center_weight) or args.center_weight <= 0:
        parser.error('center-weight must be finite and positive')
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2 or min(args.val_every, args.log_every) < 1:
        parser.error('steps/val-every/log-every must be positive; batch-size must be positive and even')
    if args.validation_limit < 1 or args.validation_limit > 128:
        parser.error('validation-limit must be in [1,128]')
    if args.workers < 0 or args.profile_steps < 0 or args.profile_warmup < 0:
        parser.error('workers/profile-steps/profile-warmup must be nonnegative')
    if args.profile_steps:
        args.steps = args.profile_steps + args.profile_warmup
    return args


def run_mixed(argv=None):
    args = parse_mixed_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=args.resume)
    train, validation = make_datasets(args)
    (student, teacher, forward_student, forward_teacher, optimizer,
     scheduler, scaler, initial_state) = build_components(args, device)
    base_step = initial_state.get('cumulative_step', initial_state.get('step', 0))
    start_step, previous_seconds, best_score = 0, 0., -float('inf')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(torch=str(torch.__version__), device_name=torch.cuda.get_device_name(device)
                  if device.type == 'cuda' else 'CPU', base_step=base_step,
                  model_config=initial_state.get('model_config'),
                  parameters=sum(p.numel() for p in student.parameters()),
                  training_clips={'vimeo': len(train[0]), 'reds': len(train[1])},
                  validation_clips={name: len(data) for name, data in validation.items()},
                  init_sha256=hashlib.sha256(args.init.read_bytes()).hexdigest(),
                  teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest() if teacher else None,
                  dataset_ratio={'vimeo': .5, 'reds': .5},
                  protocol=f'HR{args.patch_size} training / HR{args.validation_patch_size} validation patches; '
                           'equal dataset contribution per batch; '
                           f'GT Charbonnier normalized frame weights [1,{args.center_weight},1]; '
                           f'output KD weight {args.kd_weight if teacher is not None else 0.}; '
                           'FP32 validation, rounded uint8 Y metrics, no border crop; '
                           'best checkpoint maximizes equally weighted Vimeo/REDS validation PSNR-Y',
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if args.resume:
        restored = torch.load(args.output / 'distilled_last.pth', map_location=device, weights_only=True)
        student.load_state_dict(restored['state_dict'])
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        scaler.load_state_dict(restored['scaler'])
        start_step, base_step = restored['step'], restored['base_step']
        previous_seconds, best_score = restored['elapsed_seconds'], restored['best_validation_psnr_y']
        restore_rng(restored['rng'])
        log_path = args.output / 'train.jsonl'
        if log_path.exists():
            # Discard updates after the last durable optimizer state before replaying them.
            lines = [line for line in log_path.read_text().splitlines()
                     if json.loads(line)['step'] <= start_step]
            log_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
    else:
        save_json(args.output / 'config.json', config)
        save_json(args.output / 'split_ids.json', {
            'train': {name: [r['id'] for r in data.records]
                      for name, data in zip(('vimeo', 'reds'), train)},
            'val': {name: [r['id'] for r in data.records] for name, data in validation.items()}})
        atomic_checkpoint(args.output / 'initial.pth', student, 0, base_step, None, None)
    loader = make_train_loader(args, train, start_step)
    val_loaders = {name: DataLoader(data, batch_size=args.batch_size, num_workers=min(args.workers, 2),
                                    pin_memory=device.type == 'cuda')
                   for name, data in validation.items()}
    if not args.resume and not args.profile_steps:
        initial_scores, initial_details = validate_mixed(student, val_loaders, device, return_details=True)
        best_score = initial_scores['balanced_mean']
        save_json(args.output / 'initial_validation.json', initial_scores)
        save_json(args.output / 'initial_validation_details.json', initial_details)
        atomic_checkpoint(args.output / 'distilled_best.pth', student, 0, base_step, initial_scores, best_score)
        atomic_checkpoint(args.output / 'distilled_last.pth', student, 0, base_step, initial_scores, best_score,
                          optimizer, scheduler, scaler)
        print(json.dumps({'event': 'initial_validation', 'scores': initial_scores}), flush=True)
    run_start = time.perf_counter()
    if device.type == 'cuda' and not args.profile_steps:
        torch.cuda.reset_peak_memory_stats(device)
    measured_start = run_start
    if args.profile_steps and device.type == 'cuda' and args.profile_warmup == 0:
        torch.cuda.reset_peak_memory_stats(device)
    with (args.output / 'train.jsonl').open('a' if args.resume else 'w', buffering=1) as log:
        for step, batch in enumerate(loader, start=start_step + 1):
            losses = mixed_train_step(args, batch, forward_student, forward_teacher, optimizer, scaler, device)
            scheduler.step()
            elapsed = previous_seconds + time.perf_counter() - run_start
            record = {'step': step, 'cumulative_step': base_step + step, 'elapsed_seconds': elapsed,
                      'losses': losses, 'lr': optimizer.param_groups[0]['lr']}
            if args.profile_steps:
                if step == args.profile_warmup:
                    synchronize(device)
                    measured_start = time.perf_counter()
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
            elif step % args.val_every == 0 or step == args.steps:
                scores, details = validate_mixed(student, val_loaders, device, return_details=True)
                record['validation_psnr_y'] = scores
                record['validation_frame_metrics'] = details
                if scores['balanced_mean'] > best_score:
                    best_score = scores['balanced_mean']
                    atomic_checkpoint(args.output / 'distilled_best.pth', student, step, base_step, scores, best_score)
                atomic_checkpoint(args.output / 'distilled_last.pth', student, step, base_step, scores, best_score,
                                  optimizer, scheduler, scaler,
                                  previous_seconds + time.perf_counter() - run_start)
            log.write(json.dumps(record) + '\n')
            if step == 1 or step % args.log_every == 0 or step == args.steps or 'validation_psnr_y' in record:
                print(json.dumps(record), flush=True)
    synchronize(device)
    if args.profile_steps:
        seconds = time.perf_counter() - measured_start
        result = {'batch_size': args.batch_size, 'precision': args.precision, 'compile': args.compile,
                  'warmup_steps': args.profile_warmup, 'measured_steps': args.profile_steps,
                  'seconds': seconds, 'seconds_per_step': seconds / args.profile_steps,
                  'samples_per_second': args.profile_steps * args.batch_size / seconds,
                  'estimated_200k_hours': seconds / args.profile_steps * 200000 / 3600}
        if device.type == 'cuda':
            result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
        save_json(args.output / 'profile.json', result)
        print(json.dumps({'event': 'profile_complete', **result}), flush=True)
        return
    summary = {'additional_steps': args.steps, 'base_step': base_step,
               'cumulative_step': base_step + args.steps,
               'examples': args.steps * args.batch_size,
               'examples_per_dataset': args.steps * args.batch_size // 2,
               'seconds': previous_seconds + time.perf_counter() - run_start,
               'best_validation_psnr_y': best_score}
    if device.type == 'cuda':
        summary.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    save_json(args.output / 'training_summary.json', summary)
    print(json.dumps({'event': 'training_complete', **summary}), flush=True)
    if not args.skip_test:
        # Releasing the loader also shuts down its exhausted persistent worker iterator.
        del loader, val_loaders, student, teacher, forward_student, forward_teacher
        del optimizer, scheduler, scaler, initial_state
        if args.resume:
            del restored
        evaluate_after_training(args)


# Progressive RSTT-S reduction

STAGES = [
    {'name': '01_direct_head', 'steps': 10000, 'freeze_steps': 2000,
     'embed_dim': 96, 'depths': [4] * 8, 'num_frames': 4,
     'head': 'direct', 'baseline_mode': 'original'},
    {'name': '02_two_frames', 'steps': 20000, 'embed_dim': 96,
     'depths': [4] * 8, 'num_frames': 2, 'head': 'direct', 'baseline_mode': 'separable'},
    {'name': '03_encoder_depth2', 'steps': 10000, 'embed_dim': 96,
     'depths': [2] * 4 + [4] * 4, 'num_frames': 2,
     'head': 'direct', 'baseline_mode': 'separable'},
    {'name': '04_decoder_depth2', 'steps': 10000, 'embed_dim': 96,
     'depths': [2] * 8, 'num_frames': 2, 'head': 'direct', 'baseline_mode': 'separable'},
    {'name': '05_width64', 'steps': 20000, 'embed_dim': 64,
     'depths': [2] * 8, 'num_frames': 2, 'head': 'direct', 'baseline_mode': 'separable'},
    {'name': '06_width48', 'steps': 20000, 'embed_dim': 48,
     'depths': [2] * 8, 'num_frames': 2, 'head': 'direct', 'baseline_mode': 'separable'},
    {'name': '07_width32', 'steps': 30000, 'embed_dim': 32,
     'depths': [2] * 8, 'num_frames': 2, 'head': 'direct', 'baseline_mode': 'separable'},
    {'name': '08_gt_finetune', 'steps': 20000, 'embed_dim': 32,
     'depths': [2] * 8, 'num_frames': 2, 'head': 'direct',
     'baseline_mode': 'separable', 'supervised': True},
]
MODEL_KEYS = ('embed_dim', 'depths', 'num_frames', 'head', 'baseline_mode')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def stage_config(stage):
    return {**{key: stage[key] for key in MODEL_KEYS}, 'drop_path_rate': 0.}


def take_temporal(tensor, pair_index, count, stride):
    """Select a per-example consecutive temporal window from BTCHW features."""
    indices = pair_index.to(tensor.device)[:, None] * stride
    indices = indices + torch.arange(count, device=tensor.device)[None]
    return tensor[torch.arange(tensor.shape[0], device=tensor.device)[:, None], indices]


def triplet(output, batch):
    return output if output.shape[1] == 3 else take_temporal(output, batch['pair_index'], 3, 2)


def model_inputs(config, batch, device):
    key = 'teacher_lr' if config['num_frames'] == 4 else 'student_lr'
    return batch[key].to(device, non_blocking=True)


def normalized_feature_loss(student, teacher, pair_index, channels=None):
    """Match layer outputs after the same temporal/channel surgery as the weights."""
    losses = []
    for kind in ('encoder', 'decoder'):
        for current, previous in zip(student[kind], teacher[kind]):
            if previous.shape[1] != current.shape[1]:
                previous = take_temporal(previous, pair_index, current.shape[1],
                                         1 if kind == 'encoder' else 2)
            if channels is not None and previous.shape[2] != current.shape[2]:
                previous = previous.index_select(2, channels.to(previous.device))
            if current.shape != previous.shape:
                raise ValueError(f'Feature mismatch at {kind}: {current.shape} vs {previous.shape}')
            # Each channel is normalized with the fixed teacher's RMS. The
            # floor limits the influence of nearly silent channels.
            previous = previous.float()
            scale = previous.square().mean(dim=(0, 1, 3, 4), keepdim=True).sqrt().clamp_min(.1)
            losses.append(((current.float() - previous) / scale).square().mean())
    return torch.stack(losses).mean()


def clone_teacher(result):
    """Own outputs that must survive another torch.compile CUDA-graph call."""
    if isinstance(result, tuple):
        output, features = result
        return output.clone(), {key: [value.clone() for value in values]
                                for key, values in features.items()}
    return result.clone()


@torch.no_grad()
def validate_model(model, loaders, device):
    """FP32, rounded uint8, limited-range Y, arithmetic frame PSNR mean."""
    model.eval()
    config = model.get_config()
    scores = {}
    coefficients = torch.tensor([65.481, 128.553, 24.966], device=device,
                                dtype=torch.float64).view(1, 1, 3, 1, 1) / 255.
    for name, loader in loaders.items():
        total, count = 0., 0
        for batch in loader:
            prediction = triplet(model(model_inputs(config, batch, device)), batch)
            target = batch['gt'].to(device, non_blocking=True)
            prediction = prediction.float().clamp(0, 1).mul(255).round().to(torch.float64)
            target = target.float().clamp(0, 1).mul(255).round().to(torch.float64)
            # The +16 Y offset cancels in the difference.
            difference = ((prediction - target) * coefficients).sum(dim=2)
            mse = difference.square().mean(dim=(-1, -2))
            psnr = 10 * torch.log10(255. ** 2 / mse)
            total += psnr.sum().item()
            count += psnr.numel()
        scores[name] = total / count
    scores['balanced_mean'] = float(np.mean(list(scores.values())))
    return scores


def close_loader(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def calibration_inputs(args, train, source, device, stage_index):
    calibration_args = copy.copy(args)
    calibration_args.steps = args.calibration_batches
    calibration_args.seed = args.seed + 700000001 + stage_index * 10000019
    loader = make_train_loader(calibration_args, train)
    try:
        for batch in loader:
            yield model_inputs(source.get_config(), batch, device)
    finally:
        close_loader(loader)


def make_surgery(args, source, stage, loaders, train, device, stage_index):
    config = stage_config(stage)
    channels = None
    if config['embed_dim'] < source.get_config()['embed_dim']:
        calibration = calibration_inputs(args, train, source, device, stage_index)
        try:
            channels = select_channels(source, calibration, config['embed_dim'],
                                       max_batches=args.calibration_batches)
        finally:
            calibration.close()
    changed = [i for i, (old, new) in enumerate(zip(source.get_config()['depths'], config['depths']))
               if new < old]
    candidates = []
    choices = ('first', 'last') if changed and args.depth_selection == 'validation' else ('first',)
    for choice in choices:
        block_pairs = {}
        for i in changed:
            family, local = ('encoder_layers', i) if i < 4 else ('decoder_layers', i - 4)
            start = 0 if choice == 'first' else source.get_config()['depths'][i] - config['depths'][i]
            block_pairs[f'{family}.{local}'] = list(range(start, start + config['depths'][i]))
        target, report = migrate_model(source, config, channel_indices=channels,
                                       block_pairs=block_pairs or None)
        target = target.to(device)
        scores = validate_model(target, loaders, device)
        candidates.append({'choice': choice, 'scores': scores, 'report': report,
                           'state_dict': {key: value.detach().cpu().clone()
                                          for key, value in target.state_dict().items()}})
        del target
    selected = max(candidates, key=lambda value: value['scores']['balanced_mean'])
    student = ProgressiveRSTT(**config).to(device)
    student.load_state_dict(selected['state_dict'], strict=True)
    report = selected['report']
    report['depth_candidates'] = [{key: value[key] for key in ('choice', 'scores')}
                                  for value in candidates]
    report['selected_depth_choice'] = selected['choice']
    return student, report, selected['scores']


def freeze_for_step(student, stage, completed_steps):
    frozen = completed_steps < stage.get('freeze_steps', 0)
    student.requires_grad_(not frozen)
    if frozen:
        student.conv_last.requires_grad_(True)
    return frozen


def optimizer_for_stage(student, stage, device, args):
    width = stage['name'].split('_', 1)[1].startswith('width')
    if width:
        groups = [{'params': list(student.parameters()), 'lr': args.width_lr}]
    else:
        head, backbone = [], []
        for name, parameter in student.named_parameters():
            (head if name.startswith('conv_last.') else backbone).append(parameter)
        groups = [{'params': head, 'lr': args.head_lr},
                  {'params': backbone, 'lr': args.backbone_lr}]
    optimizer = torch.optim.AdamW(groups, betas=(.9, .99), weight_decay=1e-4,
                                  fused=device.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, stage['steps'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    return optimizer, scheduler, scaler


def checkpoint(student, stage, step, scores, best_score, report, elapsed, base_step,
               optimizer=None, scheduler=None, scaler=None):
    state = {'state_dict': student.state_dict(), 'model_config': student.get_config(),
             'stage': stage, 'step': step, 'cumulative_step': base_step + step,
             'validation_psnr_y': scores, 'best_validation_psnr_y': best_score,
             'migration': report, 'elapsed_seconds': elapsed}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), rng=rng_state(),
                     optimizer_state_tensors=len(optimizer.state),
                     parameter_tensors=sum(1 for _ in student.parameters()))
    return state


def gradient_coverage(model):
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        raise RuntimeError(f'Missing gradients for {len(missing)}/{len(trainable)} trainable parameter tensors: '
                           + ', '.join(missing[:12]))
    return {'trainable_parameter_tensors': len(trainable), 'parameter_tensors_with_gradients': len(trainable)}


def progressive_train_step(args, batch, student, previous, original, student_config, previous_config,
               optimizer, scaler, device, channels, supervised=False, check_gradients=False):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    if args.compile and device.type == 'cuda':
        torch.compiler.cudagraph_mark_step_begin()
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    with torch.autocast(device.type, dtype=dtype, enabled=args.precision != 'fp32'):
        previous_output, previous_features, original_output = None, None, None
        if not supervised:
            with torch.no_grad():
                previous_output, previous_features = clone_teacher(
                    previous(model_inputs(previous_config, batch, device), return_features=True))
                if previous is original:
                    original_output = previous_output
                elif args.anchor_weight:
                    original_output = clone_teacher(original(batch['teacher_lr'].to(device, non_blocking=True)))
        prediction, features = student(model_inputs(student_config, batch, device), return_features=True)
        selected = triplet(prediction, batch).float()
        hard = charbonnier(selected, batch['gt'].to(device, non_blocking=True).float())
        kd, anchor, feature = (hard.new_zeros(()) for _ in range(3))
        if previous_output is not None:
            if previous_output.shape[1] == prediction.shape[1]:
                kd = charbonnier(prediction.float(), previous_output.float())
            else:
                kd = charbonnier(selected, triplet(previous_output, batch).float())
            if args.feature_weight:
                feature = normalized_feature_loss(features, previous_features, batch['pair_index'], channels)
        if original_output is not None:
            anchor = charbonnier(selected, triplet(original_output, batch).float())
        loss = hard + args.kd_weight * kd + args.anchor_weight * anchor + args.feature_weight * feature
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    coverage = gradient_coverage(student) if check_gradients else None
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    result = {'gt': hard.item(), 'previous_kd': kd.item(), 'original_anchor': anchor.item(),
              'feature_kd': feature.item(), 'total': loss.item()}
    if coverage is not None:
        result['gradient_coverage'] = coverage
    return result


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = ProgressiveRSTT(**state['model_config']).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    return model, state


def compile_model(model, args):
    # Each training stage has fixed batch/patch shapes and a fixed channel width.
    # Keep distinct static graphs as width changes across stages.
    return torch.compile(model, mode=args.compile_mode, dynamic=False) if args.compile else model


def trim_log(path, step):
    if path.exists():
        rows = [line for line in path.read_text().splitlines() if json.loads(line)['step'] <= step]
        path.write_text('\n'.join(rows) + ('\n' if rows else ''))


def run_stage(args, stage, index, source_path, original, loaders, train, device, base_step):
    directory = args.output / stage['name']
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(json.dumps({'event': 'stage_already_complete', **summary}), flush=True)
        return directory / 'best.pth', summary
    # A new stage has the same RNG starting point whether reached continuously
    # or after restarting between stages. Partial-stage resumes restore below.
    seed_everything(args.seed + index * 10000019)
    directory.mkdir(exist_ok=True)
    previous, source_state = (original, None) if source_path is None else load_model(source_path, device)
    previous.eval().requires_grad_(False)
    restore_path = directory / 'last.pth'
    before_path = directory / 'before_validation.json'
    if before_path.exists():
        before = json.loads(before_path.read_text())
    elif source_state is not None:
        before = source_state['validation_psnr_y']
        atomic_json(before_path, before)
    else:
        before = validate_model(previous, loaders, device)
        atomic_json(before_path, before)
    print(json.dumps({'event': 'stage_before_surgery', 'stage': stage['name'], 'scores': before}), flush=True)
    restored = None
    if restore_path.exists():
        student, restored = load_model(restore_path, device)
        report, scores = restored['migration'], restored['validation_psnr_y']
        start_step, best_score = restored['step'], restored['best_validation_psnr_y']
        previous_seconds = restored['elapsed_seconds']
        immediate = json.loads((directory / 'after_surgery_validation.json').read_text())
    else:
        student, report, immediate = make_surgery(args, previous, stage, loaders, train, device, index)
        start_step, previous_seconds = 0, 0.
        scores, best_score = immediate, immediate['balanced_mean']
        atomic_json(directory / 'migration.json', report)
        atomic_json(directory / 'after_surgery_validation.json', immediate)
        initial = checkpoint(student, stage, 0, scores, best_score, report, 0., base_step)
        atomic_save(directory / 'initial.pth', initial)
        atomic_save(directory / 'best.pth', initial)
        print(json.dumps({'event': 'stage_after_surgery', 'stage': stage['name'],
                          'scores': immediate, 'parameters': sum(p.numel() for p in student.parameters())}), flush=True)
    optimizer, scheduler, scaler = optimizer_for_stage(student, stage, device, args)
    freeze_for_step(student, stage, start_step)
    if restored is not None:
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        scaler.load_state_dict(restored['scaler'])
        restore_rng(restored['rng'])
        trim_log(directory / 'train.jsonl', start_step)
    else:
        atomic_save(restore_path, checkpoint(student, stage, 0, scores, best_score, report, 0.,
                                           base_step, optimizer, scheduler, scaler))
    source_config, student_config = previous.get_config(), student.get_config()
    channels = torch.tensor(report.get('channel_indices', list(range(student_config['embed_dim']))),
                            dtype=torch.long, device=device)
    forward_student = compile_model(student, args)
    forward_previous = compile_model(previous, args) if not stage.get('supervised') else None
    forward_original = (forward_previous if previous is original else compile_model(original, args)) \
        if not stage.get('supervised') else None
    loader_args = copy.copy(args)
    loader_args.steps = stage['steps']
    loader_args.seed = args.seed + index * 10000019
    loader = make_train_loader(loader_args, train, start_step)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()
    measured_start = run_start
    profile_start = args.profile_warmup
    if args.profile_steps and not profile_start and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with (directory / 'train.jsonl').open('a', buffering=1) as log:
            for step, batch in enumerate(loader, start=start_step + 1):
                unfrozen_now = stage.get('freeze_steps', 0) > 0 and step - 1 == stage['freeze_steps']
                if unfrozen_now:
                    freeze_for_step(student, stage, step - 1)
                    if args.compile:
                        # AOTAutograd can reuse the frozen backbone's graph even
                        # after requires_grad changes. Drop that graph explicitly.
                        del forward_student, forward_previous, forward_original
                        torch.compiler.reset()
                        forward_student = compile_model(student, args)
                        forward_previous = compile_model(previous, args)
                        forward_original = forward_previous if previous is original else compile_model(original, args)
                    print(json.dumps({'event': 'unfreeze_backbone', 'stage': stage['name'], 'step': step,
                                      'recompiled': args.compile}), flush=True)
                check_gradients = step > stage.get('freeze_steps', 0) and (unfrozen_now or step == start_step + 1)
                losses = progressive_train_step(args, batch, forward_student, forward_previous, forward_original,
                                    student_config, source_config, optimizer, scaler, device, channels,
                                    supervised=stage.get('supervised', False), check_gradients=check_gradients)
                scheduler.step()
                elapsed = previous_seconds + time.perf_counter() - run_start
                coverage = losses.pop('gradient_coverage', None)
                record = {'stage': stage['name'], 'step': step, 'cumulative_step': base_step + step,
                          'elapsed_seconds': elapsed, 'losses': losses,
                          'lr': [group['lr'] for group in optimizer.param_groups]}
                if coverage is not None:
                    record['gradient_coverage'] = coverage
                if args.profile_steps and step == profile_start:
                    synchronize(device)
                    measured_start = time.perf_counter()
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                if not args.profile_steps and (step % args.val_every == 0 or step == stage['steps']):
                    scores = validate_model(student, loaders, device)
                    record['validation_psnr_y'] = scores
                    if scores['balanced_mean'] > best_score:
                        best_score = scores['balanced_mean']
                        atomic_save(directory / 'best.pth', checkpoint(student, stage, step, scores,
                                                                      best_score, report, elapsed, base_step))
                    atomic_save(restore_path, checkpoint(student, stage, step, scores, best_score, report,
                                                       previous_seconds + time.perf_counter() - run_start,
                                                       base_step, optimizer, scheduler, scaler))
                log.write(json.dumps(record) + '\n')
                if coverage is not None or step == start_step + 1 or step % args.log_every == 0 or step == stage['steps'] \
                        or 'validation_psnr_y' in record:
                    print(json.dumps(record), flush=True)
    finally:
        close_loader(loader)
    synchronize(device)
    if args.profile_steps:
        seconds = time.perf_counter() - measured_start
        summary = {'stage': stage['name'], 'steps': args.profile_steps,
                   'warmup_steps': args.profile_warmup, 'seconds_per_step': seconds / args.profile_steps,
                   'batch_size': args.batch_size, 'precision': args.precision,
                   'compile': args.compile, 'parameters': sum(p.numel() for p in student.parameters())}
        if device.type == 'cuda':
            summary.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
        atomic_json(args.output / 'profile.json', summary)
    else:
        best = torch.load(directory / 'best.pth', map_location='cpu', weights_only=True)
        summary = {'stage': stage['name'], 'steps': stage['steps'],
                   'cumulative_optimizer_steps': base_step + stage['steps'], 'best_step': best['step'],
                   'parameters': sum(p.numel() for p in student.parameters()),
                   'before': before, 'after_surgery': immediate, 'recovered_best': best['validation_psnr_y'],
                   'surgery_delta_psnr_y': immediate['balanced_mean'] - before['balanced_mean'],
                   'recovery_delta_psnr_y': best_score - immediate['balanced_mean'],
                   'net_delta_psnr_y': best_score - before['balanced_mean'],
                   'seconds': previous_seconds + time.perf_counter() - run_start,
                   'checkpoint': str((directory / 'best.pth').resolve()), 'model_config': student_config,
                   'batch_size': args.batch_size, 'precision': args.precision,
                   'compile': args.compile, 'compile_mode': args.compile_mode,
                   'optimizer_state_tensors': len(optimizer.state),
                   'parameter_tensors': sum(1 for _ in student.parameters()),
                   'trainable_parameter_tensors': sum(parameter.requires_grad for parameter in student.parameters())}
        if device.type == 'cuda':
            summary.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                           memory_scope='Current process since stage training began, including validation')
        atomic_json(summary_path, summary)
    print(json.dumps({'event': 'profile_complete' if args.profile_steps else 'stage_complete', **summary}), flush=True)
    del forward_student, forward_previous, forward_original, student, previous, source_state
    del optimizer, scheduler, scaler, loader, restored
    if args.compile:
        torch.compiler.reset()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return directory / 'best.pth', summary


def export_final(args, summaries):
    candidates = [row for row in summaries if row['model_config']['embed_dim'] == 32]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda row: row['recovered_best']['balanced_mean'])
    state = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=True)
    config = {'embed_dim': 32, 'depth': 2, 'head': 'direct', 'drop_path_rate': 0.}
    tiny = RSTTTiny(**config)
    tiny.load_state_dict(state['state_dict'], strict=True)
    final = {'state_dict': tiny.state_dict(), 'model_config': config,
             'step': state['step'], 'cumulative_step': state['cumulative_step'],
             'selected_stage': chosen['stage'], 'source_checkpoint': chosen['checkpoint'],
             'validation_psnr_y': state['validation_psnr_y'],
             'parameters': sum(p.numel() for p in tiny.parameters())}
    destination = args.output / 'final.pth'
    atomic_save(destination, final)
    result = {key: value for key, value in final.items() if key != 'state_dict'}
    result.update(path=str(destination.resolve()), sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    atomic_json(args.output / 'final_manifest.json', result)
    return result


def parse_progressive_args(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(sys.argv[0]).name} progressive', description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../Datasets_pt'))
    parser.add_argument('--teacher', type=Path,
                        default=Path('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth'))
    parser.add_argument('--batch-size', type=int, choices=(8, 16), default=16)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--stages', type=int, default=8, help='Execute this many initial stages (1..8)')
    parser.add_argument('--stage-steps', type=str, help='Comma-separated optimizer steps for all eight stages')
    parser.add_argument('--step-scale', type=float, default=1., help='Scale all stage lengths for smoke tests')
    parser.add_argument('--val-every', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=500)
    parser.add_argument('--validation-limit', type=int, default=128)
    parser.add_argument('--calibration-batches', type=int, default=32)
    parser.add_argument('--depth-selection', choices=('validation', 'first'), default='validation')
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--width-lr', type=float, default=5e-5)
    parser.add_argument('--kd-weight', type=float, default=.3)
    parser.add_argument('--anchor-weight', type=float, default=.1)
    parser.add_argument('--feature-weight', type=float, default=.05)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--precision', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile-mode', choices=('default', 'reduce-overhead', 'max-autotune'), default='reduce-overhead')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile-steps', type=int, default=0)
    parser.add_argument('--profile-warmup', type=int, default=10)
    args = parser.parse_args(argv)
    if args.resume:
        saved = json.loads((args.output / 'config.json').read_text())
        operational = {'output', 'resume', 'workers', 'threads', 'device', 'log_every'}
        for name in vars(args).keys() - operational:
            if name in saved:
                setattr(args, name, Path(saved[name]) if name in ('data_root', 'teacher') else saved[name])
    if not 1 <= args.stages <= 8 or args.step_scale <= 0:
        parser.error('stages must be in [1,8] and step-scale must be positive')
    if min(args.val_every, args.log_every, args.calibration_batches, args.threads) < 1:
        parser.error('validation/log/calibration/thread intervals must be positive')
    if not 1 <= args.validation_limit <= 128 or args.patch_size < 128 or args.patch_size % 128:
        parser.error('validation-limit must be [1,128], patch-size must be a positive multiple of 128')
    if min(args.workers, args.profile_steps, args.profile_warmup) < 0:
        parser.error('workers/profile arguments must be nonnegative')
    if min(args.kd_weight, args.anchor_weight, args.feature_weight) < 0:
        parser.error('loss weights must be nonnegative')
    if min(args.head_lr, args.backbone_lr, args.width_lr) <= 1e-6:
        parser.error('learning rates must exceed cosine eta_min=1e-6')
    stages = copy.deepcopy(STAGES)
    if args.stage_steps:
        steps = [int(value) for value in args.stage_steps.split(',')]
        if len(steps) != 8 or min(steps) < 1:
            parser.error('stage-steps requires eight positive integers')
        for stage, length in zip(stages, steps):
            stage['steps'] = length
        stages[0]['freeze_steps'] = min(2000, steps[0] // 5)
    for stage in stages:
        stage['steps'] = max(1, round(stage['steps'] * args.step_scale))
        if 'freeze_steps' in stage:
            stage['freeze_steps'] = min(stage['steps'] - 1, max(0, round(stage['freeze_steps'] * args.step_scale)))
    stages = stages[:args.stages]
    if args.profile_steps:
        stages = stages[:1]
        stages[0]['steps'] = args.profile_steps + args.profile_warmup
        stages[0]['freeze_steps'] = 0
    return args, stages


def run_progressive(argv=None):
    args, stages = parse_progressive_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=args.resume)
    train, validation = make_datasets(args)
    loaders = {name: DataLoader(data, batch_size=args.batch_size, num_workers=min(args.workers, 2),
                               pin_memory=device.type == 'cuda',
                               generator=torch.Generator().manual_seed(args.seed + 901))
               for name, data in validation.items()}
    original = ProgressiveRSTT.from_original(args.teacher, map_location='cpu').to(device).eval().requires_grad_(False)
    if not args.resume:
        config = {key: str(value.resolve()) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update(stage_schedule=stages, torch=str(torch.__version__),
                      cuda_version=torch.version.cuda,
                      cudnn_version=torch.backends.cudnn.version(),
                      cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      compile_dynamic=False,
                      device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                      teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                      training_clips=dict(zip(('vimeo', 'reds'), map(len, train))),
                      validation_clips={name: len(data) for name, data in validation.items()},
                      protocol=f'Balanced Vimeo/REDS minibatches; fixed {len(validation["vimeo"])}+'
                               f'{len(validation["reds"])} validation clips; '
                               'FP32 rounded uint8 Y PSNR, no border crop; stage best by equal dataset mean; '
                               '4-frame predictions evaluated on the same selected GT triplets as 2-frame models')
        atomic_json(args.output / 'config.json', config)
        atomic_json(args.output / 'split_ids.json', {
            'train': {name: [record['id'] for record in data.records]
                      for name, data in zip(('vimeo', 'reds'), train)},
            'validation': {name: [record['id'] for record in data.records]
                           for name, data in validation.items()}})
    summaries, source_path, base_step = [], None, 0
    try:
        for index, stage in enumerate(stages):
            source_path, summary = run_stage(args, stage, index, source_path, original,
                                             loaders, train, device, base_step)
            if args.profile_steps:
                return
            summaries.append(summary)
            base_step += stage['steps']
            atomic_json(args.output / 'stage_results.json', summaries)
        final = export_final(args, summaries)
        atomic_json(args.output / 'training_summary.json', {'optimizer_steps': base_step,
                    'stages': summaries, 'final': final})
        print(json.dumps({'event': 'progressive_training_complete', 'optimizer_steps': base_step,
                          'final': final}), flush=True)
    finally:
        for loader in loaders.values():
            close_loader(loader)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runners = {'direct': run_direct, 'mixed': run_mixed, 'progressive': run_progressive}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', choices=runners, help='Use WORKFLOW --help for training options')
    workflow = parser.parse_args(argv[:1]).workflow
    return runners[workflow](argv[1:])


if __name__ == '__main__':
    main()
