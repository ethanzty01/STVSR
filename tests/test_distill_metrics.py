import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill_metrics import compute_metrics, _luminance


def test_identical_images_and_clamping():
    target = torch.rand(2, 3, 3, 16, 20)
    result = compute_metrics(target, target)
    assert len(result['per_frame']) == 2
    assert math.isinf(result['all']['psnr_rgb'])
    assert math.isinf(result['all']['psnr_y'])
    assert result['all']['ssim_y'] == pytest.approx(1.0)
    result = compute_metrics(torch.full_like(target, 2), torch.ones_like(target))
    assert math.isinf(result['all']['psnr_rgb'])


def test_known_error_and_temporal_groups():
    target = torch.full((1, 3, 3, 16, 16), 100 / 255)
    prediction = target.clone()
    prediction[:, ::2] = 110 / 255
    prediction[:, 1] = 120 / 255
    result = compute_metrics(prediction, target)
    assert result['endpoints']['psnr_rgb'] == pytest.approx(20 * math.log10(255 / 10))
    assert result['center']['psnr_rgb'] == pytest.approx(20 * math.log10(255 / 20))
    assert result['center']['psnr_y'] == pytest.approx(20 * math.log10(255 / (20 * 219 / 255)))


def test_matches_official_ssim_and_y_coefficients():
    spec = importlib.util.spec_from_file_location(
        'official_metrics', Path(__file__).resolve().parents[1] / 'utils/metrics.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    generator = torch.Generator().manual_seed(10)
    target = torch.rand(1, 3, 3, 24, 28, generator=generator)
    prediction = target + 0.05 * torch.randn(target.shape, generator=generator)
    result = compute_metrics(prediction, target, crop_border=2)
    rgb_pred = prediction[0, 1].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()[2:-2, 2:-2]
    rgb_target = target[0, 1].mul(255).round().byte().permute(1, 2, 0).numpy()[2:-2, 2:-2]
    expected = official.structural_similarity(_luminance(rgb_pred), _luminance(rgb_target))
    assert result['center']['ssim_y'] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(_luminance(np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)), [16, 235])


def test_rejects_invalid_inputs():
    image = torch.zeros(1, 3, 3, 16, 16)
    with pytest.raises(ValueError):
        compute_metrics(image, image, crop_border=3)
    image[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        compute_metrics(image, image)
