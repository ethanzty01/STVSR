"""Continue RSTT-tiny training with balanced Vimeo/REDS batches and resumable state."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from distill_tiny import charbonnier, save_json, seed_everything, synchronize, teacher_targets, validate
from models.RSTT import RSTT
from models.RSTT_tiny import RSTTTiny


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
    train = [VimeoDistillationDataset(args.data_root, 'train', patch_size=args.patch_size,
                                     validation_count=128, seed=args.seed),
             REDSDistillationDataset(args.data_root, 'train', patch_size=args.patch_size,
                                     seed=args.seed)]
    validation = {
        'vimeo': VimeoDistillationDataset(args.data_root, 'val', patch_size=args.patch_size,
                                         validation_count=128, seed=args.seed,
                                         limit=args.validation_limit),
        'reds': REDSDistillationDataset(args.data_root, 'val', patch_size=args.patch_size,
                                       seed=args.seed, limit=args.validation_limit),
    }
    return train, validation


def make_train_loader(args, datasets, start_step=0):
    kwargs = {'num_workers': args.workers, 'pin_memory': args.device.startswith('cuda')}
    if args.workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    # A dedicated generator keeps worker startup from changing the model RNG stream.
    return DataLoader(SeededMixedDataset(datasets),
                      batch_sampler=BalancedStepSampler(list(map(len, datasets)), args.batch_size,
                                                        args.steps, args.seed, start_step),
                      generator=torch.Generator().manual_seed(args.seed), **kwargs)


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


def train_step(args, batch, student, teacher, optimizer, scaler, device):
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
        hard = charbonnier(prediction.float(), target.float())
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


def validate_mixed(student, loaders, device):
    scores = {name: validate({'tiny': student}, loader, device)['tiny']
              for name, loader in loaders.items()}
    scores['balanced_mean'] = float(np.mean(list(scores.values())))
    return scores


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('../Datasets_pt'))
    parser.add_argument('--init', type=Path, default=Path('experiments/tiny_direct_d2_20k_b8/distilled_best.pth'))
    parser.add_argument('--teacher', type=Path, default=Path('checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200000, help='Additional optimizer steps')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=128, help='HR training/validation patch side')
    parser.add_argument('--mode', choices=['supervised', 'distill'], default='distill')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--kd-weight', type=float, default=.5)
    parser.add_argument('--val-every', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=1000, help='Console interval; every step is written to train.jsonl')
    parser.add_argument('--validation-limit', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
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
    args = parser.parse_args()
    if args.resume:
        saved = json.loads((args.output / 'config.json').read_text())
        for name in ('data_root', 'init', 'teacher', 'steps', 'batch_size', 'patch_size', 'mode',
                     'lr', 'kd_weight', 'val_every', 'validation_limit', 'seed', 'precision',
                     'compile', 'compile_mode'):
            setattr(args, name, Path(saved[name]) if name in ('data_root', 'init', 'teacher') else saved[name])
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2 or min(args.val_every, args.log_every) < 1:
        parser.error('steps/val-every/log-every must be positive; batch-size must be positive and even')
    if args.validation_limit < 1 or args.validation_limit > 128:
        parser.error('validation-limit must be in [1,128]')
    if args.workers < 0 or args.profile_steps < 0 or args.profile_warmup < 0:
        parser.error('workers/profile-steps/profile-warmup must be nonnegative')
    if args.profile_steps:
        args.steps = args.profile_steps + args.profile_warmup
    return args


def main():
    args = parse_args()
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
                  protocol=f'HR{args.patch_size} patches; equal dataset contribution per batch; '
                           f'GT Charbonnier + {args.kd_weight} KD; '
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
        initial_scores = validate_mixed(student, val_loaders, device)
        best_score = initial_scores['balanced_mean']
        save_json(args.output / 'initial_validation.json', initial_scores)
        atomic_checkpoint(args.output / 'distilled_best.pth', student, 0, base_step, initial_scores, best_score)
        atomic_checkpoint(args.output / 'distilled_last.pth', student, 0, base_step, initial_scores, best_score,
                          optimizer, scheduler, scaler)
        print(json.dumps({'event': 'initial_validation', 'scores': initial_scores}), flush=True)
    run_start = time.perf_counter()
    measured_start = run_start
    if args.profile_steps and device.type == 'cuda' and args.profile_warmup == 0:
        torch.cuda.reset_peak_memory_stats(device)
    with (args.output / 'train.jsonl').open('a' if args.resume else 'w', buffering=1) as log:
        for step, batch in enumerate(loader, start=start_step + 1):
            losses = train_step(args, batch, forward_student, forward_teacher, optimizer, scaler, device)
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
                scores = validate_mixed(student, val_loaders, device)
                record['validation_psnr_y'] = scores
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
    save_json(args.output / 'training_summary.json', summary)
    print(json.dumps({'event': 'training_complete', **summary}), flush=True)
    if not args.skip_test:
        # Releasing the loader also shuts down its exhausted persistent worker iterator.
        del loader, val_loaders, student, teacher, forward_student, forward_teacher
        del optimizer, scheduler, scaler, initial_state
        if args.resume:
            del restored
        evaluate_after_training(args)


if __name__ == '__main__':
    main()
