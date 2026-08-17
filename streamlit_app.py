"""
streamlit_app.py
-----------------
Interactive demo: upload a degraded SEM image (.npy/.png/.jpg) and
view the NAFNet-restored output side-by-side, with PSNR/SSIM if a
ground-truth clean image is also provided.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import build_model  # noqa: E402
from metrics import evaluate_batch  # noqa: E402


st.set_page_config(page_title="SEM Image Restoration", layout="wide")
st.title("SEM Image Restoration (NAFNet)")
st.caption("Upload a degraded semiconductor inspection image to restore it.")

CHECKPOINT_PATH = st.sidebar.text_input("Checkpoint path", "models/nafnet_best.pth")
IN_CHANNELS = st.sidebar.selectbox("Input channels", [1, 3], index=0)
MODEL_SIZE = st.sidebar.selectbox("Model size", ["tiny", "small", "base"], index=1)


def ensure_checkpoint(local_path: str) -> str:
    """
    Trained .pth files are too large to commit to git (see .gitignore), so a
    fresh deploy (e.g. Streamlit Community Cloud) won't have one on disk.
    If CHECKPOINT_URL is set in .streamlit/secrets.toml or as an env var
    (a direct download link - Hugging Face, GitHub Release asset, Drive
    direct-download link, etc.), download it once and cache locally.
    """
    if os.path.exists(local_path):
        return local_path

    url = ""
    try:
        url = st.secrets.get("CHECKPOINT_URL", "")
    except Exception:
        pass
    url = url or os.environ.get("CHECKPOINT_URL", "")
    if not url:
        return local_path  # nothing to download; load_model() will warn and use random weights

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with st.spinner("Downloading model checkpoint (first run only)..."):
        import requests
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path


CHECKPOINT_PATH = ensure_checkpoint(CHECKPOINT_PATH)


@st.cache_resource
def load_model(checkpoint_path, in_channels, model_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(in_channels=in_channels, size=model_size).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        st.sidebar.success("Checkpoint loaded.")
    else:
        st.sidebar.warning("Checkpoint not found - using randomly initialized weights.")
    model.eval()
    return model, device


def read_uploaded(file, in_channels):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".npy":
        arr = np.load(file).astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = arr.transpose(2, 0, 1)
    else:
        img = Image.open(file).convert("L" if in_channels == 1 else "RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[None, ...] if arr.ndim == 2 else arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


def to_display(tensor):
    arr = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    if arr.shape[0] == 1:
        return arr[0]
    return arr.transpose(1, 2, 0)


model, device = load_model(CHECKPOINT_PATH, IN_CHANNELS, MODEL_SIZE)

col1, col2 = st.columns(2)
with col1:
    degraded_file = st.file_uploader("Degraded image (.npy/.png/.jpg)", type=["npy", "png", "jpg", "jpeg"])
with col2:
    clean_file = st.file_uploader("Optional: clean ground truth (for metrics)", type=["npy", "png", "jpg", "jpeg"])

if degraded_file is not None:
    degraded_tensor = read_uploaded(degraded_file, IN_CHANNELS)

    with torch.no_grad():
        restored_tensor = model(degraded_tensor.to(device)).clamp(0, 1).cpu()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Degraded Input")
        st.image(to_display(degraded_tensor), clamp=True, use_container_width=True)
    with c2:
        st.subheader("Restored Output")
        st.image(to_display(restored_tensor), clamp=True, use_container_width=True)

    if clean_file is not None:
        clean_tensor = read_uploaded(clean_file, IN_CHANNELS)
        if clean_tensor.shape == restored_tensor.shape:
            metrics = evaluate_batch(restored_tensor, clean_tensor)
            st.subheader("Metrics vs. Ground Truth")
            m1, m2 = st.columns(2)
            m1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            m2.metric("SSIM", f"{metrics['ssim']:.4f}")
        else:
            st.warning("Ground-truth image shape doesn't match restored output - skipping metrics.")
else:
    st.info("Upload a degraded image to get started.")
