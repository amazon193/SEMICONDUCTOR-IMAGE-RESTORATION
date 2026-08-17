"""
preprocessing.py
-----------------
Dataset loading, verification, and PyTorch Dataset class for paired
clean / degraded SEM image patches stored as .npy files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def verify_dataset(root_dir, noisy_folder="NoisyLR", clean_folder="CleanHR"):
    """
    Sanity-check that noisy/clean folders exist and filenames match.

    Args:
        root_dir: path containing noisy_folder and clean_folder
        noisy_folder: name of the degraded-image subfolder
        clean_folder: name of the clean-image subfolder

    Returns:
        dict with counts and any mismatches found.
    """
    noisy_dir = os.path.join(root_dir, noisy_folder)
    clean_dir = os.path.join(root_dir, clean_folder)

    if not os.path.isdir(noisy_dir):
        raise FileNotFoundError(f"Noisy folder not found: {noisy_dir}")
    if not os.path.isdir(clean_dir):
        raise FileNotFoundError(f"Clean folder not found: {clean_dir}")

    noisy_files = set(os.listdir(noisy_dir))
    clean_files = set(os.listdir(clean_dir))

    missing_in_clean = noisy_files - clean_files
    missing_in_noisy = clean_files - noisy_files

    result = {
        "noisy_count": len(noisy_files),
        "clean_count": len(clean_files),
        "matched": len(noisy_files & clean_files),
        "missing_in_clean": sorted(missing_in_clean)[:10],
        "missing_in_noisy": sorted(missing_in_noisy)[:10],
    }

    print(f"Noisy files: {result['noisy_count']}")
    print(f"Clean files: {result['clean_count']}")
    print(f"Matched pairs: {result['matched']}")
    if missing_in_clean:
        print(f"WARNING: {len(missing_in_clean)} files in noisy missing from clean")
    if missing_in_noisy:
        print(f"WARNING: {len(missing_in_noisy)} files in clean missing from noisy")

    return result


def load_npy_image(path, normalize=True):
    """Load a single .npy image patch and return as (C, H, W) float32 tensor."""
    arr = np.load(path).astype(np.float32)

    if normalize and arr.max() > 1.5:
        arr = arr / 255.0

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr.transpose(2, 0, 1)

    return torch.from_numpy(arr)


def load_array(path, normalize_output=True):
    """
    Load any supported image file (.npy, .png, .jpg, .tif) as a float32
    numpy array in [0, 1], shape (H, W) or (H, W, C). Used by inference/app
    code that wants a plain numpy array rather than a torch tensor.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32)

    if normalize_output and arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def normalize(arr):
    """Scale a numpy array to [0, 1] float32. Handles uint8 (0-255) input."""
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_chw_tensor(arr):
    """Convert a HxW or HxWxC numpy array in [0,1] to a (C,H,W) torch tensor."""
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr.copy())


def tensor_to_uint8(t):
    """Convert a (C,H,W) or (H,W) tensor in [0,1] back to a uint8 numpy image."""
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


class SEMPairDataset(Dataset):
    """
    Paired dataset of degraded (noisy/low-res) and clean SEM image patches.

    Assumes matching filenames between noisy_dir and clean_dir.
    Automatically upsamples the degraded image to the clean image's
    resolution (bicubic) if a resolution mismatch is present.
    """

    def __init__(self, noisy_dir, clean_dir, file_list=None, augment=False):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.augment = augment
        self.files = file_list if file_list is not None else sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.files)

    def _augment_pair(self, noisy, clean):
        if torch.rand(1).item() < 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            clean = torch.flip(clean, dims=[-1])
        if torch.rand(1).item() < 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            clean = torch.flip(clean, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
            clean = torch.rot90(clean, k, dims=[-2, -1])
        return noisy, clean

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy = load_npy_image(os.path.join(self.noisy_dir, fname))
        clean = load_npy_image(os.path.join(self.clean_dir, fname))

        if noisy.shape[-1] != clean.shape[-1] or noisy.shape[-2] != clean.shape[-2]:
            noisy = torch.nn.functional.interpolate(
                noisy.unsqueeze(0),
                size=clean.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            noisy = noisy.clamp(0, 1)

        if self.augment:
            noisy, clean = self._augment_pair(noisy, clean)

        return noisy, clean


def train_val_split(root_dir, noisy_folder="NoisyLR", clean_folder="CleanHR",
                     val_fraction=0.1, seed=42):
    """Build train/val SEMPairDataset objects with a reproducible split."""
    import random

    noisy_dir = os.path.join(root_dir, noisy_folder)
    clean_dir = os.path.join(root_dir, clean_folder)

    files = sorted(os.listdir(noisy_dir))
    rng = random.Random(seed)
    rng.shuffle(files)

    n_val = max(1, int(len(files) * val_fraction))
    val_files = files[:n_val]
    train_files = files[n_val:]

    train_ds = SEMPairDataset(noisy_dir, clean_dir, file_list=train_files, augment=True)
    val_ds = SEMPairDataset(noisy_dir, clean_dir, file_list=val_files, augment=False)

    return train_ds, val_ds
