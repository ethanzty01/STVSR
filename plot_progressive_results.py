"""Render CPU-only progressive-stage and optional GT-extension diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = ('#0B5CAD', '#B64040', '#0F766E')
STAGE_LABELS = {'direct_head': 'Direct head', 'two_frames': 'Two frames',
                'encoder_depth2': 'Encoder depth 2', 'decoder_depth2': 'Decoder depth 2',
                'width64': 'Width 64', 'width48': 'Width 48', 'width32': 'Width 32',
                'gt_finetune': 'GT finetune'}


def stage_label(stage):
    number, name = stage.split('_', 1)
    return f'{int(number)}. ' + STAGE_LABELS.get(name, name.replace('_', ' '))


def save_figure(fig, directory, name, note):
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def read_training_records(path):
    """Read complete JSONL rows while an optional extension is still running."""
    text = path.read_text()
    lines = text.splitlines()
    if text and not text.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def validation_curves(run_dir, stages, directory, suffix):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    end = stages[-1]['cumulative_optimizer_steps']
    unit = 1000 if end >= 1000 else 1
    series = [('vimeo', 'Vimeo validation', '-'), ('reds', 'REDS validation', '--'),
              ('balanced_mean', 'Balanced mean', '-.')]
    for stage_index, stage in enumerate(stages):
        start = stage['cumulative_optimizer_steps'] - stage['steps']
        records = [row for row in read_training_records(run_dir / stage['stage'] / 'train.jsonl')
                   if 'validation_psnr_y' in row]
        x = [start / unit] + [r['cumulative_step'] / unit for r in records]
        for (key, label, style), color in zip(series, COLORS):
            y = [stage['after_surgery'][key]] + [r['validation_psnr_y'][key] for r in records]
            ax.plot(x, y, style, color=color, marker='.', markersize=3,
                    label=label if stage_index == 0 else None)
            ax.plot([start / unit] * 2, [stage['before'][key], y[0]], ':', color=color, linewidth=1.8)
            ax.scatter(start / unit, stage['before'][key], s=20, facecolors='white', edgecolors=color, zorder=4)
            ax.scatter((start + stage['best_step']) / unit, stage['recovered_best'][key],
                       marker='*', s=75, color=color, edgecolors='white', linewidths=.4, zorder=5)
        ax.axvline(start / unit, color='.85', linewidth=.7, zorder=0)
        ax.text((start + stage['steps'] / 2) / unit, .98, str(stage_index + 1),
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker='*', linestyle='', color='.3', markersize=9),
                Line2D([], [], linestyle=':', color='.3')]
    labels += ['Selected stage best', 'Surgery change']
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=5, fontsize=8)
    ax.set(xlabel=f'Executed optimizer steps{ " (thousands)" if unit == 1000 else ""}',
           ylabel='Validation PSNR-Y (dB)', title='Progressive training and surgery' + suffix,
           xticks=[0] + [s['cumulative_optimizer_steps'] / unit for s in stages])
    ax.set_title(ax.get_title(), pad=34)
    ax.margins(x=.025, y=.15)
    ax.grid(axis='y', color='.9')
    stage_map = ' | '.join(stage_label(s['stage']) for s in stages)
    save_figure(fig, directory, 'validation_curves',
                'Stage best selected by equal Vimeo/REDS mean; each stage starts from the preceding best.\n' + stage_map)


def extension_validation_curves(extension_dir, stages, directory):
    config = json.loads((extension_dir / 'config.json').read_text())
    initial = json.loads((extension_dir / 'initial_validation.json').read_text())
    records = [row for row in read_training_records(extension_dir / 'train.jsonl')
               if 'validation_psnr_y' in row]
    # Use executed stage budgets as the horizontal coordinate. A preceding
    # stage may hand off its best checkpoint from an earlier local step.
    base = stages[-1]['cumulative_optimizer_steps']
    planned = config['steps']
    local_steps = [0] + [row['step'] for row in records]
    scores = [initial] + [row['validation_psnr_y'] for row in records]
    selected = max(range(len(scores)), key=lambda i: scores[i]['balanced_mean'])
    unit = 1000 if base + planned >= 1000 else 1
    x = [(base + step) / unit for step in local_steps]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.65))
    panels = [('vimeo', '(a) Vimeo validation'), ('reds', '(b) REDS validation'),
              ('balanced_mean', '(c) Equal-dataset mean')]
    for ax, (key, title), color in zip(axes, panels, COLORS):
        values = [score[key] for score in scores]
        ax.plot(x, values, '-o', color=color, linewidth=1.5, markersize=3)
        ax.axhline(initial[key], linestyle='--', color='.45', linewidth=.9)
        ax.scatter(x[selected], values[selected], marker='*', s=95, color=color,
                   edgecolor='white', linewidth=.5, zorder=4)
        ax.set(title=title, xlim=(base / unit, (base + planned) / unit),
               xlabel=f'Total steps{ " (thousands)" if unit == 1000 else ""}',
               ylabel='PSNR-Y (dB)' if key == 'vimeo' else '')
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=9)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(axis='y', color='.9')
        ax.margins(y=.18)
    complete = local_steps[-1] == planned
    status = 'complete' if complete else 'in progress'
    fig.suptitle(f'Stage 9: additional GT training ({status})', fontsize=11, y=.98)
    note = (f'Stars: checkpoint with the highest equal-dataset mean; dashed: {base:,}-step initialization.\n'
            f'Latest validation: {local_steps[-1]:,} / {planned:,} extension steps. '
            'Panel y-axis ranges differ.')
    # Preserve readable final-size labels on this two-column, three-panel plot.
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=7.5)
    fig.tight_layout(rect=(0, .12, 1, .94), w_pad=1.4)
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'extension_validation_curves.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return {'run_dir': str(extension_dir), 'planned_steps': planned,
            'last_validated_step': local_steps[-1], 'selected_step': local_steps[selected],
            'selected_scores': scores[selected], 'complete': complete}


def surgery_recovery(stages, directory, suffix):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(stages))
    for offset, key, label, color, marker in zip((-.2, 0, .2),
            ('before', 'after_surgery', 'recovered_best'),
            ('Before surgery', 'After surgery', 'Recovered best'), COLORS, ('o', 'x', '*')):
        ax.scatter(x + offset, [s[key]['balanced_mean'] for s in stages], color=color,
                   marker=marker, s=70 if marker == '*' else 35, label=label, zorder=3)
    ax.set(xticks=x, xticklabels=[stage_label(s['stage']).replace(' ', '\n', 1) for s in stages],
           ylabel='Balanced validation PSNR-Y (dB)', title='Immediate surgery loss and training recovery' + suffix)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3)
    ax.set_title(ax.get_title(), pad=34)
    ax.margins(y=.15)
    ax.grid(axis='y', color='.9')
    save_figure(fig, directory, 'surgery_recovery',
                'Balanced = equal mean of Vimeo and REDS validation PSNR; the selected checkpoint may be at step 0.')


def final_comparison(result, directory, suffix):
    datasets = result['datasets']
    categories = [('vid4', None, 'Vid4'), ('vimeo', 'fast', 'Vimeo\nFast'),
                  ('vimeo', 'medium', 'Vimeo\nMedium'), ('vimeo', 'slow', 'Vimeo\nSlow'),
                  ('reds', None, 'REDS')]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    models = [('progressive', 'Progressive tiny', COLORS[0], 'o'),
              ('previous_tiny', 'Previous tiny', COLORS[1], 's'),
              ('teacher', 'RSTT-S teacher', COLORS[2], '^')]
    if any('intermediate' in dataset.get('metrics', {}) for dataset in datasets.values()):
        models.insert(1, ('intermediate', 'Earlier progressive tiny', '#7755AA', 'D'))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * .18
    for offset, (model, label, color, marker) in zip(offsets, models):
        values = []
        for dataset, group, _ in categories:
            metrics = datasets.get(dataset, {}).get('metrics', {}).get(model, {})
            if group:
                metrics = metrics.get('groups', {}).get(group, {})
            else:
                metrics = metrics.get('equal_sequence_mean' if dataset == 'vid4' else 'all', {})
            values.append(metrics.get('all', {}).get('psnr_y', np.nan))
        ax.scatter(np.arange(len(categories)) + offset, values, label=label, color=color, marker=marker, s=50)
    ax.set(xticks=np.arange(len(categories)), xticklabels=[r[2] for r in categories],
           ylabel='Test PSNR-Y (dB)', title='Final test-set comparison' + suffix)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=len(models),
              fontsize=9 if len(models) == 4 else 10)
    ax.set_title(ax.get_title(), pad=34)
    ax.margins(x=.1, y=.15)
    ax.grid(axis='y', color='.9')
    save_figure(fig, directory, 'final_test_comparison',
                'Vimeo/Vid4: seven-frame evaluation, LR rounded to 8-bit; Vid4 sequences weighted equally.\n'
                'REDS: middle triples of seven-frame windows, float LR; overlapping observations retained.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--extension-run-dir', type=Path,
                        help='Optional GT extension; plot its validation and read its final evaluation')
    parser.add_argument('--evaluation-dir', default='evaluation', help='Use evaluation_smoke to preview smoke test metrics')
    args = parser.parse_args(argv)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    stages = json.loads((args.run_dir / 'stage_results.json').read_text())
    directory = args.run_dir / 'figures'
    directory.mkdir(exist_ok=True)
    suffix = ' (smoke data)' if 'smoke' in args.run_dir.name else ''
    total_steps = stages[-1]['cumulative_optimizer_steps']
    stage_suffix = f' (stages 1-{len(stages)}, {total_steps:,} updates)' + suffix
    validation_curves(args.run_dir, stages, directory, stage_suffix)
    surgery_recovery(stages, directory, stage_suffix)
    extension = extension_validation_curves(args.extension_run_dir, stages, directory) \
        if args.extension_run_dir else None
    evaluation_root = args.extension_run_dir or args.run_dir
    evaluation = evaluation_root / args.evaluation_dir / 'results.json'
    if evaluation.exists():
        result = json.loads(evaluation.read_text())
        final_comparison(result, directory, ' (smoke data)' if result.get('smoke_limit') else suffix)
    print(json.dumps({'figures': str(directory), 'final_comparison': evaluation.exists(),
                      'evaluation_source': str(evaluation), 'extension': extension}))


if __name__ == '__main__':
    main()
