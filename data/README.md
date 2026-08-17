# Data

This project expects paired clean / degraded SEM image patches as `.npy`
files (float32, single-channel, e.g. 128x128).

Expected structure (matching the hackathon-provided dataset):

```
data/
├── train/
│   ├── NoisyLR/       # degraded (noisy + downsampled) images
│   │   ├── 000000.npy
│   │   ├── 000001.npy
│   │   └── ...
│   └── CleanHR/       # clean ground-truth images (matching filenames)
│       ├── 000000.npy
│       ├── 000001.npy
│       └── ...
└── val/                # optional, same structure
```

The full dataset (3,200+ pairs, ~200MB+) is **not** committed to this repo
(see `.gitignore`). Only a handful of sample `.npy` files are kept under
`data/sample/` for smoke-testing the pipeline (`tests/test_pipeline.py`).

## Getting the full dataset

1. Download / receive the hackathon-provided dataset (`train.zip`).
2. Extract it locally or in Google Colab into `data/train/`.
3. Verify structure with:
   ```bash
   python -c "from src.preprocessing import verify_dataset; verify_dataset('data/train')"
   ```
