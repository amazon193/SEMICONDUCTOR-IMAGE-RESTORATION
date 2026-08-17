"""
degradation.py
---------------
Physics-informed degradation pipeline to synthetically generate
(degraded, clean) training pairs from clean SEM images when a
paired dataset isn't already available:

    Clean SEM image -> add noise -> downsample -> Degraded image

Includes SEM-realistic noise models (Poisson shot noise + Gaussian
read noise) rather than generic i.i.d. Gaussian noise.
"""

import numpy as np
import torch
import torch.nn.functional as F


def add_poisson_gaussian_noise(img, poisson_scale=30.0, gaussian_sigma=0.02, seed=None):
    """
    Simulate SEM sensor noise: Poisson (shot) noise dominates at low signal,
    Gaussian (read) noise is additive on top. This is closer to real
    electron-detector noise than plain Gaussian noise.

    Args:
        img: torch.Tensor or np.ndarray, values in [0, 1]
        poisson_scale: higher = less shot noise (simulates electron dose)
        gaussian_sigma: std-dev of additive Gaussian read noise
    """
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        arr = img.detach().cpu().numpy()
    else:
        arr = img

    rng = np.random.default_rng(seed)

    arr = np.clip(arr, 0, 1)
    # Poisson shot noise: scale up, sample, scale back down
    noisy = rng.poisson(arr * poisson_scale) / poisson_scale
    # Gaussian read noise
    noisy = noisy + rng.normal(0, gaussian_sigma, size=arr.shape)
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return torch.from_numpy(noisy) if is_tensor else noisy


def add_charging_artifact(img, strength=0.15, seed=None):
    """
    Simulate SEM charging artifacts: a smooth low-frequency brightness
    gradient/blob caused by electron accumulation on non-conductive
    regions of the sample.
    """
    rng = np.random.default_rng(seed)
    is_tensor = isinstance(img, torch.Tensor)
    arr = img.detach().cpu().numpy() if is_tensor else img

    h, w = arr.shape[-2], arr.shape[-1]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    sigma = rng.uniform(0.2, 0.5) * min(h, w)
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    blob = blob * strength * rng.uniform(-1, 1)

    out = np.clip(arr + blob, 0, 1).astype(np.float32)
    return torch.from_numpy(out) if is_tensor else out


def downsample(img, scale=2, mode="bicubic"):
    """
    Downsample a (C,H,W) or (H,W) image by `scale`, simulating fast
    low-resolution inspection scans.
    """
    is_tensor = isinstance(img, torch.Tensor)
    arr = img if is_tensor else torch.from_numpy(img)

    squeeze_channel = False
    if arr.ndim == 2:
        arr = arr[None, None, ...]
        squeeze_channel = True
    elif arr.ndim == 3:
        arr = arr[None, ...]

    h, w = arr.shape[-2], arr.shape[-1]
    small = F.interpolate(arr, size=(h // scale, w // scale), mode=mode, align_corners=False)

    if squeeze_channel:
        small = small[0, 0]
    else:
        small = small[0]

    return small if is_tensor else small.numpy()


def degrade(clean_img, downsample_scale=2, poisson_scale=30.0,
            gaussian_sigma=0.02, add_charging=True, seed=None):
    """
    Full degradation pipeline matching the project diagram:

        Clean SEM image -> add noise -> downsample -> Degraded image

    Returns the degraded image at the SAME resolution as the input
    (upsampled back via bicubic after downsampling), so it can be
    paired directly with the clean image for training.
    """
    img = clean_img
    orig_size = img.shape[-2:]

    noisy = add_poisson_gaussian_noise(img, poisson_scale, gaussian_sigma, seed=seed)
    if add_charging:
        noisy = add_charging_artifact(noisy, seed=seed)

    small = downsample(noisy, scale=downsample_scale)

    is_tensor = isinstance(small, torch.Tensor)
    small_t = small if is_tensor else torch.from_numpy(small)
    if small_t.ndim == 2:
        small_t = small_t[None, None, ...]
    elif small_t.ndim == 3:
        small_t = small_t[None, ...]

    degraded = F.interpolate(small_t, size=orig_size, mode="bicubic", align_corners=False)
    degraded = degraded.clamp(0, 1)[0]

    return degraded if is_tensor else degraded.numpy()


class SyntheticDegradationDataset(torch.utils.data.Dataset):
    """
    Wraps a directory of CLEAN-only .npy images and generates a fresh
    synthetic degraded version on the fly for each sample. Use this
    when you do NOT already have a paired NoisyLR/CleanHR dataset.
    """

    def __init__(self, clean_dir, downsample_scale=2, poisson_scale=30.0,
                 gaussian_sigma=0.02, add_charging=True):
        import os
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(clean_dir))
        self.downsample_scale = downsample_scale
        self.poisson_scale = poisson_scale
        self.gaussian_sigma = gaussian_sigma
        self.add_charging = add_charging

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import os
        arr = np.load(os.path.join(self.clean_dir, self.files[idx])).astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = arr[None, ...]
        clean = torch.from_numpy(arr)

        degraded = degrade(
            clean,
            downsample_scale=self.downsample_scale,
            poisson_scale=self.poisson_scale,
            gaussian_sigma=self.gaussian_sigma,
            add_charging=self.add_charging,
        )
        return degraded, clean
