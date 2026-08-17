"""
metrics.py
----------
Image-quality and restoration evaluation metrics: PSNR, SSIM, and
(optionally) LPIPS perceptual similarity.
"""

import torch
import torch.nn.functional as F
import numpy as np


def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio. pred/target: torch.Tensor in [0, max_val]."""
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def _gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.T @ g
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window.to(device)


def ssim(pred, target, window_size=11, max_val=1.0):
    """
    Structural Similarity Index. pred/target: (B, C, H, W) tensors in [0, max_val].
    """
    channels = pred.shape[1]
    window = _gaussian_window(window_size, channels=channels, device=pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean().item()


def evaluate_batch(pred, target):
    """Returns dict of {psnr, ssim} for a batch, averaged."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }


def evaluate_lpips(pred, target, device="cpu"):
    """
    Optional perceptual metric. Requires `pip install lpips`.
    pred/target: (B, C, H, W) in [0, 1]; expands to 3 channels if grayscale.
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("Install lpips first: pip install lpips")

    loss_fn = lpips.LPIPS(net="alex").to(device)

    def to_rgb(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x * 2 - 1  # lpips expects [-1, 1]

    with torch.no_grad():
        d = loss_fn(to_rgb(pred).to(device), to_rgb(target).to(device))
    return d.mean().item()
