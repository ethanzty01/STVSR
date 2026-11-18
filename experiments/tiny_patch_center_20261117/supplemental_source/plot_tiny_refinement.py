"""Plot the fixed-validation screening and subsequent Tiny refinement."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_curve(directory):
    initial = json.loads((directory / 'initial_validation.json').read_text())
    rows = []
    for line in (directory / 'train.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'validation_psnr_y' in row:
            rows.append(row)
    return np.array([0] + [r['step'] for r in rows]) / 1000, [initial] + [r['validation_psnr_y'] for r in rows]


def finish(fig, directory, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('experiments/tiny_patch_center_20261117'))
    args = parser.parse_args()
    root = args.run_dir
    destination = root / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    arms = [('a_patch128_w1', 'A: HR128, 1:1:1', '#0072B2', '-'),
            ('b_patch256_w1', 'B: HR256, 1:1:1', '#D55E00', '--'),
            ('c_patch256_w2', 'C: HR256, 1:2:1', '#009E73', '-.')]
    fields = [('vimeo', 'Vimeo'), ('reds', 'REDS'), ('balanced_mean', 'Equal dataset mean')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, label, color, style in arms:
        x, values = load_curve(root / name)
        selected = int(np.argmax([v['balanced_mean'] for v in values]))
        for ax, (field, title) in zip(axes, fields):
            y = [v[field] for v in values]
            ax.plot(x, y, style, color=color, label=label, linewidth=1.7)
            ax.plot(x[selected], y[selected], '*', color=color, markersize=10)
            ax.set(title=title, xlabel='Updates (thousands)', ylabel='Validation PSNR-Y (dB)')
            ax.grid(axis='y', alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.text(.5, .01, 'Fixed HR128 validation crops. Stars select the same checkpoint by equal dataset mean; axes are zoomed.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .88))
    finish(fig, destination, 'screening_validation')

    refine = root / 'd_selected_refine60k'
    if (refine / 'initial_validation.json').exists():
        x, values = load_curve(refine)
        selected = int(np.argmax([v['balanced_mean'] for v in values]))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, (field, title) in zip(axes, fields):
            y = [v[field] for v in values]
            ax.plot(x, y, color='#0072B2', linewidth=1.7)
            ax.axhline(y[0], color='#777777', linestyle='--', linewidth=1)
            ax.plot(x[selected], y[selected], '*', color='#D55E00', markersize=11)
            ax.set(title=title, xlabel='Refinement updates (thousands)', ylabel='Validation PSNR-Y (dB)')
            ax.grid(axis='y', alpha=.25)
        fig.suptitle('Selected configuration: 60k refinement', y=1.02)
        fig.text(.5, .01, 'Dashed lines: refinement initialization. Stars: best equal dataset mean. Only recorded validation points are shown.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .98))
        finish(fig, destination, 'refinement_validation')
    print(json.dumps({'figures': str(destination)}))


if __name__ == '__main__':
    main()
