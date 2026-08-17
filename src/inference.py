"""
inference.py
------------
Run a trained NAFNet checkpoint on a single degraded image (.npy or
standard image file) and save the restored output.

Usage:
    python src/inference.py --checkpoint models/nafnet_best.pth \
        --input path/to/degraded.npy --output outputs/restored/result.png
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model


def load_input_image(path, in_channels=1):
    """Load .npy, .png, or .jpg as a (1, C, H, W) tensor in [0, 1]."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = arr.transpose(2, 0, 1)
    else:
        img = Image.open(path).convert("L" if in_channels == 1 else "RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, ...]
        else:
            arr = arr.transpose(2, 0, 1)

    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
    return tensor


def save_output_image(tensor, path):
    """Save a (1, C, H, W) or (C, H, W) tensor in [0,1] as a PNG."""
    arr = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    if arr.shape[0] == 1:
        arr = arr[0]
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    else:
        arr = arr.transpose(1, 2, 0)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def load_model(checkpoint_path, in_channels=1, model_size="small", device="cpu"):
    model = build_model(in_channels=in_channels, size=model_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def restore_image(model, input_tensor, device="cpu"):
    with torch.no_grad():
        restored = model(input_tensor.to(device))
    return restored.clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description="Run NAFNet inference on a degraded SEM image")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "base"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, args.in_channels, args.model_size, device)
    input_tensor = load_input_image(args.input, args.in_channels)
    restored = restore_image(model, input_tensor, device)
    save_output_image(restored, args.output)

    print(f"Restored image saved to {args.output}")


if __name__ == "__main__":
    main()
