"""Frame metrics using RSTT's rounded uint8, limited-range Y convention."""

import cv2
import numpy as np
import torch


def _psnr(first, second):
    mse = np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def _luminance(rgb):
    # Equivalent to skimage.color.rgb2ycbcr(uint8_rgb)[..., 0].
    return rgb.astype(np.float64) @ np.array([65.481, 128.553, 24.966]) / 255.0 + 16.0


def _ssim_y(first, second):
    kernel = cv2.getGaussianKernel(11, 1.5)

    def smooth(value):
        return cv2.sepFilter2D(value, -1, kernel, kernel)[5:-5, 5:-5]

    mean_first, mean_second = smooth(first), smooth(second)
    variance_first = smooth(first ** 2) - mean_first ** 2
    variance_second = smooth(second ** 2) - mean_second ** 2
    covariance = smooth(first * second) - mean_first * mean_second
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first ** 2 + mean_second ** 2 + c1) * (
        variance_first + variance_second + c2
    )
    return float(np.mean(numerator / denominator))


def _mean(records):
    return {key: float(np.mean([record[key] for record in records]))
            for key in ("psnr_rgb", "psnr_y", "ssim_y")}


@torch.no_grad()
def compute_metrics(prediction, target, crop_border=0):
    """Return per-frame metrics and arithmetic frame averages.

    Inputs use [B,T,3,H,W] and RGB in [0,1]. Both are clamped, multiplied
    by 255, and rounded to uint8 before scoring, matching the official
    evaluator. Y uses BT.601 limited-range coefficients with data_range=255.
    SSIM uses an 11x11 Gaussian (sigma=1.5), evaluated on the valid region.

    ``endpoints`` averages even-indexed frames (observed time stamps), and
    ``center`` averages odd-indexed frames (interpolated time stamps).
    ``per_frame`` is a nested [batch][time] list of metric dictionaries.
    Perfect predictions have infinite PSNR. Callers writing strict JSON
    should encode those values explicitly if encountered.
    """
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("Expected matching [B,T,3,H,W] tensors.")
    b, t, c, h, w = prediction.shape
    if b < 1 or t < 3 or t % 2 != 1 or c != 3:
        raise ValueError("Expected a nonempty batch and an odd T >= 3 with RGB channels.")
    if not isinstance(crop_border, int) or crop_border < 0:
        raise ValueError("crop_border must be a nonnegative integer.")
    if min(h, w) - 2 * crop_border < 11:
        raise ValueError("SSIM requires at least 11x11 pixels after border cropping.")

    arrays = []
    for tensor in (prediction, target):
        if not torch.isfinite(tensor).all():
            raise ValueError("Metric inputs must contain finite values.")
        array = tensor.float().clamp(0, 1).mul(255).round().to(torch.uint8)
        array = array.permute(0, 1, 3, 4, 2).cpu().numpy()
        if crop_border:
            array = array[:, :, crop_border:-crop_border, crop_border:-crop_border]
        arrays.append(array)

    per_frame = []
    for batch in range(b):
        records = []
        for frame in range(t):
            first, second = arrays[0][batch, frame], arrays[1][batch, frame]
            first_y, second_y = _luminance(first), _luminance(second)
            records.append({"psnr_rgb": _psnr(first, second),
                            "psnr_y": _psnr(first_y, second_y),
                            "ssim_y": _ssim_y(first_y, second_y)})
        per_frame.append(records)
    return {"per_frame": per_frame,
            "all": _mean([item for batch in per_frame for item in batch]),
            "endpoints": _mean([item for batch in per_frame for item in batch[::2]]),
            "center": _mean([item for batch in per_frame for item in batch[1::2]])}
