"""
test_pipeline.py
------------------
Basic sanity tests for the restoration pipeline: degradation, model
forward pass, and metrics. Run with:

    pytest tests/test_pipeline.py
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import build_model                     # noqa: E402
from degradation import degrade                    # noqa: E402
from preprocessing import normalize, to_chw_tensor  # noqa: E402
from metrics import psnr, ssim                      # noqa: E402


def test_degradation_preserves_shape_and_range():
    # degrade() downsamples internally then upsamples back to original
    # resolution, so output shape matches input shape.
    clean = torch.rand(1, 128, 128)
    degraded = degrade(clean, downsample_scale=2)
    assert degraded.shape == clean.shape
    assert degraded.min() >= 0.0 and degraded.max() <= 1.0


def test_nafnet_forward_pass_shape_preserved():
    model = build_model(in_channels=1, size="tiny")
    x = torch.randn(1, 1, 96, 96)
    out = model(x)
    assert out.shape == x.shape


def test_nafnet_handles_non_multiple_of_8_input():
    model = build_model(in_channels=1, size="tiny")
    x = torch.randn(1, 1, 65, 71)  # not divisible by padder size
    out = model(x)
    assert out.shape == x.shape


def test_metrics_perfect_match_gives_high_psnr():
    x = torch.rand(1, 1, 32, 32)
    p = psnr(x, x)
    s = ssim(x, x)
    assert p > 40  # identical tensors -> very high PSNR (capped by float precision)
    assert abs(s - 1.0) < 1e-3


def test_preprocessing_normalize_uint8_range():
    arr = (np.random.rand(64, 64) * 255).astype(np.float32)
    norm = normalize(arr)
    assert norm.max() <= 1.0 and norm.min() >= 0.0

    tensor = to_chw_tensor(norm)
    assert tensor.shape == (1, 64, 64)


if __name__ == "__main__":
    test_degradation_preserves_shape_and_range()
    test_nafnet_forward_pass_shape_preserved()
    test_nafnet_handles_non_multiple_of_8_input()
    test_metrics_perfect_match_gives_high_psnr()
    test_preprocessing_normalize_uint8_range()
    print("All tests passed.")
