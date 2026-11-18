"""Recover RSTT-S after successive head, temporal, depth, and width reductions.

Each stage starts from the preceding stage's best validation checkpoint. The
original RSTT-S and the preceding model provide fixed distillation targets.
The final checkpoint contains only a standalone RSTTTiny state dictionary.
"""
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from distill_tiny import charbonnier, seed_everything, synchronize
from models.RSTT_progressive import ProgressiveRSTT, migrate_model, select_channels
from models.RSTT_tiny import RSTTTiny
from train_tiny_mixed import make_datasets, make_train_loader, restore_rng, rng_state


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


def train_step(args, batch, student, previous, original, student_config, previous_config,
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
                losses = train_step(args, batch, forward_student, forward_previous, forward_original,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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


def main(argv=None):
    args, stages = parse_args(argv)
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


if __name__ == '__main__':
    main()
