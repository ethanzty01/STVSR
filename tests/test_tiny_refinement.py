"""Check temporal supervision, a fixed validation target, and resume compatibility."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill_metrics import compute_metrics
from train_tiny import (
    BalancedStepSampler, charbonnier, make_datasets, make_train_loader, parse_mixed_args,
    temporal_charbonnier, validate, validate_mixed,
)


def test_equal_temporal_weights_preserve_loss_and_gradient_exactly():
    torch.manual_seed(3)
    prediction = torch.rand(2, 3, 3, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction)
    expected = charbonnier(prediction, target)
    actual = temporal_charbonnier(prediction, target, 1.)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, prediction)[0],
                               torch.autograd.grad(expected, prediction)[0], rtol=0, atol=0)


def test_center_weight_is_normalized_and_changes_only_time_contribution():
    prediction = torch.ones(2, 3, 3, 4, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = temporal_charbonnier(prediction, target, 2.)
    torch.testing.assert_close(loss, torch.tensor((1 + 1e-6) ** .5))
    gradient = torch.autograd.grad(loss, prediction)[0]
    torch.testing.assert_close(gradient[:, 1], 2 * gradient[:, 0])
    torch.testing.assert_close(gradient[:, 2], gradient[:, 0])
    # Unequal errors expose accidental averaging before temporal weighting.
    x = torch.tensor([1., 2., 4.]).reshape(1, 3, 1, 1, 1)
    expected = (((x.flatten().square() + 1e-6).sqrt()) * torch.tensor([1., 2., 1.])).sum() / 4
    torch.testing.assert_close(temporal_charbonnier(x, torch.zeros_like(x), 2.), expected)


def test_larger_training_crop_keeps_validation_crop_and_split_seed(monkeypatch):
    import distill_data
    class Dataset:
        def __init__(self, root, split, **kwargs):
            self.root, self.split, self.options = root, split, kwargs
    monkeypatch.setattr(distill_data, 'VimeoDistillationDataset', Dataset)
    monkeypatch.setattr(distill_data, 'REDSDistillationDataset', Dataset)
    args = SimpleNamespace(data_root=Path('unused'), patch_size=256,
                           validation_patch_size=128, seed=2026, validation_limit=128)
    train, val = make_datasets(args)
    assert all(d.options['patch_size'] == 256 for d in train)
    assert all(d.options['patch_size'] == 128 for d in val.values())
    assert all(d.options['seed'] == 2026 for d in train + list(val.values()))


def test_sampling_seed_can_change_without_changing_split_seed():
    args = SimpleNamespace(workers=0, device='cpu', batch_size=4, steps=5,
                           seed=2026, sampling_seed=2027)
    loader = make_train_loader(args, [list(range(9)), list(range(6))], start_step=2)
    assert list(loader.batch_sampler) == list(BalancedStepSampler([9, 6], 4, 5, 2027, start_step=2))
    assert args.seed == 2026


def test_validation_matches_legacy_score_and_separates_frame_kinds():
    torch.manual_seed(1)
    target = torch.rand(2, 3, 3, 13, 13) * .5
    prediction = target + torch.tensor([.05, .15, .1]).reshape(1, 3, 1, 1, 1)
    loader = [{'student_lr': prediction, 'gt': target}]
    model = torch.nn.Identity()
    expected = validate({'tiny': model}, loader, torch.device('cpu'))['tiny']
    scores, details = validate_mixed(model, {'vimeo': loader, 'reds': loader},
                                    torch.device('cpu'), return_details=True)
    assert scores == {'vimeo': expected, 'reds': expected, 'balanced_mean': expected}
    rows = compute_metrics(prediction, target)['per_frame']
    for kind, indices in [('endpoints', [0, 2]), ('center', [1])]:
        mean = float(np.mean([clip[i]['psnr_y'] for clip in rows for i in indices]))
        assert details['vimeo'][kind]['psnr_y'] == mean


def test_resume_restores_new_controls_and_legacy_defaults(tmp_path, monkeypatch):
    saved = dict(data_root='data', init='initial.pth', teacher='teacher.pth', steps=10,
                 batch_size=4, patch_size=64, mode='supervised', lr=2e-5, kd_weight=.5,
                 val_every=2, validation_limit=2, seed=2026, precision='bf16',
                 compile=False, compile_mode='default')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(sys, 'argv', ['train', '--output', str(tmp_path), '--resume',
                                    '--center-weight', '9', '--validation-patch-size', '256'])
    legacy = parse_mixed_args()
    assert (legacy.center_weight, legacy.validation_patch_size, legacy.sampling_seed) == (1., 64, 2026)
    saved.update(center_weight=2., validation_patch_size=128, sampling_seed=2027)
    path.write_text(json.dumps(saved))
    restored = parse_mixed_args()
    assert (restored.center_weight, restored.validation_patch_size, restored.sampling_seed) == (2., 128, 2027)
