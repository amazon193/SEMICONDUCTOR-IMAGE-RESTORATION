# AI-Based Restoration of Degraded Images for Semiconductor Inspection

Restoring noisy / low-resolution SEM (Scanning Electron Microscope) inspection
images using a **NAFNet** (Nonlinear Activation Free Network) deep learning
model, to improve downstream defect-detection accuracy in semiconductor
fabrication pipelines.

```
Clean SEM image
      ↓
Add noise
      ↓
Downsample
      ↓
Degraded image
      ↓
   NAFNet
      ↓
Restored image
```

## Problem Statement

Semiconductor inspection images (SEM / optical) are frequently degraded by
sensor noise, electron-beam scan noise, defocus blur, charging artifacts, and
low-resolution fast-scan acquisition. Degraded images cause missed defects
(false negatives) or false alarms (false positives) during automated
inspection, directly impacting fab yield and cost. This project builds an
AI-based restoration pipeline that recovers high-fidelity images from
degraded inputs without hallucinating structures that don't exist.

## Repository Structure

```
semiconductor-image-restoration/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                     # dataset (small samples only; full data via .gitignore)
│   ├── README.md
│   └── sample/
│
├── notebooks/                 # exploratory / Colab-friendly notebooks
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_degradation_generation.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_nafnet_training.ipynb
│   └── 05_evaluation.ipynb
│
├── src/                        # production-quality reusable code
│   ├── preprocessing.py
│   ├── degradation.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── metrics.py
│
├── models/                    # saved checkpoints (.pth) — gitignored, README only
│
├── app/
│   └── streamlit_app.py       # interactive demo UI
│
├── outputs/
│   ├── restored/
│   ├── comparisons/
│   └── metrics/
│
├── tests/
│   └── test_pipeline.py
│
└── docs/
    ├── architecture.png
    └── results.md
```

## Quick Start

```bash
git clone https://github.com/<your-username>/semiconductor-image-restoration.git
cd semiconductor-image-restoration
pip install -r requirements.txt

# Train
python src/train.py --data_dir data/sample --epochs 50

# Run inference on a single image
python src/inference.py --checkpoint models/nafnet_best.pth --input path/to/degraded.npy --output outputs/restored/result.png

# Launch demo app
streamlit run app/streamlit_app.py
```

## Training on Google Colab

See `notebooks/04_nafnet_training.ipynb` — mount Drive, unzip dataset,
and run training with GPU acceleration.

## Model

**NAFNet** (Nonlinear Activation Free Network) — an efficient encoder-decoder
architecture using SimpleGate activations and Simplified Channel Attention,
achieving strong restoration quality without expensive nonlinear activations
or attention mechanisms. See `src/model.py`.

## Evaluation Metrics

- PSNR, SSIM (image fidelity)
- LPIPS (perceptual similarity)
- Downstream defect-detection accuracy delta (task-level metric)

See `docs/results.md` for results once training completes.

## License

MIT (adjust as needed for your hackathon submission rules).
