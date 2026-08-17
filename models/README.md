# Models

Trained checkpoints (`.pth`) are saved here by `src/train.py`:

- `nafnet_last.pth` — most recent epoch checkpoint
- `nafnet_best.pth` — checkpoint with best validation PSNR

Checkpoints are **gitignored** (too large for git) — download them from
the release/artifact link below once training completes, or retrain
using `notebooks/04_nafnet_training.ipynb`.

| Model | Params | Val PSNR | Val SSIM | Download |
|---|---|---|---|---|
| NAFNet-small | ~2.5M | TBD | TBD | TBD |

To use a checkpoint for inference:

```bash
python src/inference.py --checkpoint models/nafnet_best.pth \
    --input path/to/degraded.npy --output outputs/restored/result.png
```
