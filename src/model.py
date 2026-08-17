"""
model.py
--------
NAFNet (Nonlinear Activation Free Network) for image restoration.

Reference: "Simple Baselines for Image Restoration" (Chen et al., 2022)
This is a compact, from-scratch re-implementation suited for
single-channel SEM image patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them (replaces nonlinear activations)."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core NAFNet block: depthwise conv + SimpleGate + simplified channel attention + FFN."""

    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_path=0.0):
        super().__init__()
        dw_channels = channels * dw_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3,
                                padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    """
    U-Net-shaped restoration network built from NAFBlocks.

    Args:
        in_channels: input/output image channels (1 for grayscale SEM patches)
        width: base channel width
        enc_blk_nums: number of NAFBlocks per encoder stage
        middle_blk_num: number of NAFBlocks at the bottleneck
        dec_blk_nums: number of NAFBlocks per decoder stage
    """

    def __init__(self, in_channels=1, width=32,
                 enc_blk_nums=(2, 2, 4), middle_blk_num=4,
                 dec_blk_nums=(2, 2, 2)):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, in_channels, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, x):
        b, c, h, w = x.shape
        inp = self._pad_to_multiple(x)

        feat = self.intro(inp)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle_blks(feat)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)

        out = self.ending(feat) + inp
        return out[:, :, :h, :w]


def build_model(in_channels=1, width=32, size="small"):
    """Convenience factory. size in {'tiny', 'small', 'base'}."""
    configs = {
        "tiny": dict(width=16, enc_blk_nums=(1, 1, 2), middle_blk_num=2, dec_blk_nums=(1, 1, 1)),
        "small": dict(width=32, enc_blk_nums=(2, 2, 4), middle_blk_num=4, dec_blk_nums=(2, 2, 2)),
        "base": dict(width=64, enc_blk_nums=(2, 2, 4, 8), middle_blk_num=12, dec_blk_nums=(2, 2, 2, 2)),
    }
    cfg = configs[size]
    if size == "base":
        return NAFNet(in_channels=in_channels, **cfg)
    return NAFNet(in_channels=in_channels, **cfg)


if __name__ == "__main__":
    model = build_model(in_channels=1, size="small")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet-small params: {n_params / 1e6:.2f}M")

    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print("input:", x.shape, "-> output:", y.shape)
