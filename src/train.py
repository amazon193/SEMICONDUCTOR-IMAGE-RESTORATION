"""
train.py
--------
Training script for NAFNet-based SEM image restoration.

Usage:
    python src/train.py --data_dir data/train --epochs 50 --batch_size 16

Expects data_dir to contain NoisyLR/ and CleanHR/ subfolders with
matching .npy filenames (see data/README.md).
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from preprocessing import train_val_split
from metrics import evaluate_batch


def parse_args():
    p = argparse.ArgumentParser(description="Train NAFNet for SEM image restoration")
    p.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing NoisyLR/ and CleanHR/ subfolders")
    p.add_argument("--noisy_folder", type=str, default="NoisyLR")
    p.add_argument("--clean_folder", type=str, default="CleanHR")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "base"])
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--checkpoint_dir", type=str, default="models")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds, val_ds = train_val_split(
        args.data_dir, args.noisy_folder, args.clean_folder,
        val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = build_model(in_channels=args.in_channels, size=args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")

    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            restored = model(noisy)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_metrics = {"psnr": 0.0, "ssim": 0.0}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                restored = model(noisy)
                m = evaluate_batch(restored, clean)
                val_metrics["psnr"] += m["psnr"]
                val_metrics["ssim"] += m["ssim"]
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_psnr={val_metrics['psnr']:.2f}dB | "
              f"val_ssim={val_metrics['ssim']:.4f} | "
              f"{elapsed:.1f}s")

        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "nafnet_last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "nafnet_best.pth"))
            print(f"  -> new best checkpoint saved (PSNR={best_psnr:.2f}dB)")

    print("Training complete.")


if __name__ == "__main__":
    main()
