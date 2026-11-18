"""Run fixed-validation patch/loss trials, refine the winner, and evaluate it.

Every trial starts from the same checkpoint. Selection uses validation only;
full test sets are evaluated after the 60k refinement phase is finished.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
ARMS = [('a_patch128_w1', 128, 1.), ('b_patch256_w1', 256, 1.), ('c_patch256_w2', 256, 2.)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def audit_run(directory, steps, expected_init, reference_split=None):
    """Verify executed updates, selected checkpoint, and fixed data membership."""
    config = read_json(directory / 'config.json')
    summary = read_json(directory / 'training_summary.json')
    rows = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
    assert len(rows) == steps and [r['step'] for r in rows] == list(range(1, steps + 1))
    assert all(torch.isfinite(torch.tensor(list(r['losses'].values()))).all().item() for r in rows)
    assert all(r['losses']['kd'] == 0 for r in rows)
    assert config['init_sha256'] == sha256(expected_init)
    assert config['validation_patch_size'] == 128 and config['seed'] == 2026
    assert config['parameters'] == 286140 and summary['additional_steps'] == steps
    initial = torch.load(directory / 'initial.pth', map_location='cpu', weights_only=True)
    origin = torch.load(expected_init, map_location='cpu', weights_only=True)
    assert initial['state_dict'].keys() == origin['state_dict'].keys()
    assert all(torch.equal(value, origin['state_dict'][key]) for key, value in initial['state_dict'].items())
    last = torch.load(directory / 'distilled_last.pth', map_location='cpu', weights_only=True)
    best = torch.load(directory / 'distilled_best.pth', map_location='cpu', weights_only=True)
    states = last['optimizer']['state']
    histogram = Counter(int(value['step'].item()) for value in states.values())
    assert len(states) == 344 and histogram == {steps: 344}
    assert last['step'] == steps and last['scheduler']['last_epoch'] == steps
    assert all(torch.isfinite(state[field]).all().item()
               for state in states.values() for field in ('exp_avg', 'exp_avg_sq'))
    initial_score = read_json(directory / 'initial_validation.json')['balanced_mean']
    expected_best = max([initial_score] + [r['validation_psnr_y']['balanced_mean'] for r in rows
                                          if 'validation_psnr_y' in r])
    assert abs(best['best_validation_psnr_y'] - expected_best) < 1e-10
    split = read_json(directory / 'split_ids.json')
    if reference_split is not None:
        assert split == reference_split
    return {'pass': True, 'steps': steps, 'examples': steps * config['batch_size'],
            'optimizer_step_histogram': dict(histogram), 'best_step': best['step'],
            'best_validation': best['validation_psnr_y'], 'initial_validation_mean': initial_score,
            'patch_size': config['patch_size'], 'center_weight': config['center_weight'],
            'sampling_seed': config['sampling_seed'], 'seconds': summary['seconds'],
            'best_sha256': sha256(directory / 'distilled_best.pth')}, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/tiny_patch_center_20261117'))
    parser.add_argument('--init', type=Path, default=Path('experiments/progressive_rstt_s_gt80k_b16/final.pth'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--screen-steps', type=int, default=20000)
    parser.add_argument('--refine-steps', type=int, default=60000)
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output, initial_path = args.output.resolve(), args.init.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'experiment_plan.json'
    plan = {'initial_checkpoint': str(initial_path), 'initial_sha256': sha256(initial_path),
            'arms': [{'name': name, 'patch_size': patch, 'center_weight': weight} for name, patch, weight in ARMS],
            'screen_steps': args.screen_steps, 'refine_steps': args.refine_steps,
            'batch_size': 16, 'validation_patch_size': 128, 'validation_clips_per_dataset': 128,
            'split_seed': 2026, 'screen_sampling_seed': 2027, 'refine_sampling_seed': 2028,
            'screen_lr': 2e-5, 'refine_lr': 1e-5, 'eta_min': 1e-6,
            'selection': 'Highest equally weighted Vimeo/REDS validation PSNR-Y; ties keep earlier arm.',
            'mode': 'supervised', 'precision': 'bf16', 'compile_mode': 'reduce-overhead',
            'total_planned_optimizer_updates': 3 * args.screen_steps + args.refine_steps}
    if plan_path.exists():
        if not args.resume:
            raise RuntimeError('Experiment exists; use --resume to continue its saved plan')
        assert read_json(plan_path) == plan, 'Resume configuration differs from the saved plan'
    else:
        save_json(plan_path, plan)
    state_path = output / 'experiment_status.json'
    state = {'status': 'running', 'phase': 'initializing', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'runs': {}, 'pid': os.getpid()}
    started = time.perf_counter()
    reference_split = read_json(initial_path.parent / 'split_ids.json')

    def execute(command, log_path, phase):
        state.update(status='running', phase=phase, command=command, log=str(log_path))
        with log_path.open('a' if args.resume else 'w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            save_json(state_path, state)
            code = process.wait()
        state['child_returncode'] = code
        save_json(state_path, state)
        if code:
            raise RuntimeError(f'{phase} exited with {code}; see {log_path}')

    def train(name, patch, weight, steps, lr, sampling_seed, init):
        directory = output / name
        if not (directory / 'training_summary.json').exists():
            command = [sys.executable, '-B', str(ROOT / 'train_tiny.py'), 'mixed',
                       '--output', str(directory), '--init', str(init), '--steps', str(steps),
                       '--patch-size', str(patch), '--validation-patch-size', '128',
                       '--center-weight', str(weight), '--batch-size', '16', '--mode', 'supervised',
                       '--lr', str(lr), '--seed', '2026', '--sampling-seed', str(sampling_seed),
                       '--compile', '--compile-mode', 'reduce-overhead', '--skip-test']
            if directory.exists():
                if not args.resume or not (directory / 'distilled_last.pth').exists():
                    raise RuntimeError(f'Existing incomplete run lacks a durable checkpoint: {directory}')
                command.append('--resume')
            execute(command, output / f'{name}.log', name)
        audit, _ = audit_run(directory, steps, init, reference_split)
        config = read_json(directory / 'config.json')
        assert (config['patch_size'], config['center_weight'], config['sampling_seed']) == (patch, weight, sampling_seed)
        state['runs'][name] = audit
        save_json(directory / 'training_audit.json', audit)
        save_json(state_path, state)
        print(json.dumps({'event': 'run_complete', 'run': name, **audit}), flush=True)
        return audit

    try:
        for name, patch, weight in ARMS:
            train(name, patch, weight, args.screen_steps, 2e-5, 2027, initial_path)
        means = [state['runs'][name]['initial_validation_mean'] for name, _, _ in ARMS]
        assert max(means) - min(means) < 1e-4, 'Initial validation differs across trials'
        winner, patch, weight = max(ARMS, key=lambda arm: state['runs'][arm[0]]['best_validation']['balanced_mean'])
        selection = {'selected_arm': winner, 'patch_size': patch, 'center_weight': weight,
                     'arms': state['runs'].copy(), 'selection_uses_test_data': False}
        save_json(output / 'selection.json', selection)
        print(json.dumps({'event': 'selected_arm', 'arm': winner, 'patch_size': patch,
                          'center_weight': weight}), flush=True)
        name = 'd_selected_refine60k'
        train(name, patch, weight, args.refine_steps, 1e-5, 2028,
              output / winner / 'distilled_best.pth')
        final_source = output / name / 'distilled_best.pth'
        final = output / 'final.pth'
        shutil.copyfile(final_source, final)
        checkpoint = torch.load(final, map_location='cpu', weights_only=True)
        manifest = {key: checkpoint[key] for key in ('step', 'cumulative_step', 'model_config', 'validation_psnr_y')}
        manifest.update(path=str(final), sha256=sha256(final), parameters=286140,
                        selected_arm=winner, source_checkpoint=str(final_source),
                        total_executed_updates=sum(r['steps'] for r in state['runs'].values()),
                        total_training_sample_visits=sum(r['examples'] for r in state['runs'].values()))
        save_json(output / 'final_manifest.json', manifest)
        save_json(output / 'training_audit.json', {'pass': True, 'runs': state['runs'],
                  'total_steps': manifest['total_executed_updates'],
                  'all_splits_match_original': True, 'common_start': str(initial_path)})
        if not args.skip_evaluation:
            execute([sys.executable, '-B', str(ROOT / 'evaluate_progressive.py'),
                     '--run-dir', str(output), '--previous', str(initial_path),
                     '--intermediate-checkpoint', str(ROOT / 'experiments/tiny_mixed_200k_b16/distilled_best.pth'),
                     '--workers', '2', '--fps'], output / 'evaluation.log', 'full_evaluation')
        state.update(status='complete', phase='complete', selected_arm=winner,
                     seconds_this_invocation=time.perf_counter() - started,
                     completed_utc=datetime.now(timezone.utc).isoformat())
        save_json(state_path, state)
        print(json.dumps({'event': 'experiment_complete', 'manifest': manifest}), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        save_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
