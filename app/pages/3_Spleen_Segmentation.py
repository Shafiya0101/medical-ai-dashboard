"""
3D Spleen Segmentation — MONAI U-Net inference
================================================
Loads a trained 3D U-Net from models/, runs sliding-window inference on a
sample CT volume from data/sample_volumes/, lets the user scroll through
axial slices with a predicted-mask overlay.

To train the model, run scripts/train_segmentation.py first.
"""
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_loader import get_model_path

st.set_page_config(page_title="Spleen Segmentation", page_icon="🔬", layout="wide")
st.title("🔬 3D Spleen Segmentation — MONAI U-Net")

st.markdown(
    "Browse a sample 3D abdominal CT volume, scroll through axial slices, "
    "and see the predicted spleen mask overlaid on the image. Inference uses "
    "**sliding-window** so the model works on arbitrary-size volumes."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample_volumes"


# =============================================================================
# Lazy imports (heavy)
# =============================================================================
@st.cache_resource(show_spinner="Loading MONAI U-Net...")
def load_unet():
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    trained = False
    try:
        path = get_model_path("unet_spleen.pt")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        trained = True
    except Exception as e:
        print(f"Could not load U-Net: {e}")
    return model.to(DEVICE).eval(), trained


@st.cache_data(show_spinner="Loading & preprocessing volume...")
def load_volume(volume_path, label_path):
    """Load + preprocess a single Spleen volume the same way the trainer does."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, CropForegroundd, EnsureTyped,
    )
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    data = transforms({"image": str(volume_path), "label": str(label_path)})
    img = data["image"][0].cpu().numpy()
    lbl = data["label"][0].cpu().numpy().astype(np.uint8)
    return img, lbl


@st.cache_data(show_spinner="Running sliding-window inference...")
def run_inference(volume_path, label_path, _model_id):
    """Return predicted mask (same shape as the preprocessed volume)."""
    from monai.inferers import sliding_window_inference
    img, lbl = load_volume(volume_path, label_path)
    model, _ = load_unet()
    with torch.no_grad():
        x = torch.from_numpy(img)[None, None].to(DEVICE)
        logits = sliding_window_inference(x, roi_size=(96, 96, 96), sw_batch_size=2, predictor=model)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.uint8)
    return img, lbl, pred


def dice_score(pred, label):
    pred, label = pred.astype(bool), label.astype(bool)
    inter = (pred & label).sum()
    s = pred.sum() + label.sum()
    return float(2 * inter / s) if s > 0 else 1.0


def iou_score(pred, label):
    pred, label = pred.astype(bool), label.astype(bool)
    inter = (pred & label).sum()
    union = (pred | label).sum()
    return float(inter / union) if union > 0 else 1.0


# =============================================================================
# UI
# =============================================================================
model, trained = load_unet()
if not trained:
    st.error(
        "**Could not load the trained U-Net weights** (`unet_spleen.pt`).\n\n"
        f"Train it first:\n```bash\npython scripts/train_segmentation.py\n```\n\n"
        f"This downloads the Decathlon Task09 Spleen dataset (~1.5 GB) and "
        f"trains for ~10 epochs (20–40 min on a T4 GPU)."
    )
    st.stop()

# Find available sample volumes
if not SAMPLE_DIR.exists():
    st.error(
        f"**No sample volumes found in `{SAMPLE_DIR}`.**\n\n"
        f"Run `python scripts/prepare_sample_volumes.py` to copy 2–3 volumes "
        f"from the Decathlon dataset into the repo (small enough for demo)."
    )
    st.stop()

available = sorted(SAMPLE_DIR.glob("*_image.nii.gz"))
if not available:
    st.error(f"No `*_image.nii.gz` files in `{SAMPLE_DIR}`.")
    st.stop()

st.sidebar.header("Volume")
vol_name = st.sidebar.selectbox("Sample CT volume", [p.name for p in available])
vol_path = SAMPLE_DIR / vol_name
lbl_path = SAMPLE_DIR / vol_name.replace("_image.nii.gz", "_label.nii.gz")

if not lbl_path.exists():
    st.error(f"Missing label file: `{lbl_path.name}`")
    st.stop()

# Run inference (cached)
img, lbl, pred = run_inference(str(vol_path), str(lbl_path), "unet_spleen")

# ---------- KPIs ----------
dice = dice_score(pred, lbl)
iou = iou_score(pred, lbl)
spleen_vox = int(pred.sum())
gt_vox = int(lbl.sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dice coefficient", f"{dice:.3f}")
c2.metric("IoU", f"{iou:.3f}")
c3.metric("Predicted spleen voxels", f"{spleen_vox:,}")
c4.metric("Ground-truth voxels", f"{gt_vox:,}")

# ---------- Slice viewer ----------
st.subheader("Axial slice viewer")
n_slices = img.shape[2]

# Default to the slice with the largest GT footprint
slice_areas = (lbl > 0).sum(axis=(0, 1))
default_z = int(np.argmax(slice_areas)) if slice_areas.max() > 0 else n_slices // 2

z = st.slider("Slice index (axial)", 0, n_slices - 1, default_z)

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img[:, :, z], cmap="gray")
    ax.contour(lbl[:, :, z], colors="lime", linewidths=1.5)
    ax.set_title(f"Ground truth (green) — z={z}")
    ax.axis("off")
    st.pyplot(fig)
with col_b:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img[:, :, z], cmap="gray")
    ax.contour(lbl[:, :, z], colors="lime", linewidths=1.5)
    ax.contour(pred[:, :, z], colors="red", linewidths=1.5, linestyles="--")
    ax.set_title("GT (green) vs Prediction (red)")
    ax.axis("off")
    st.pyplot(fig)

# ---------- Spleen-area-per-slice curve ----------
st.subheader("Spleen footprint by slice")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(np.arange(n_slices), (lbl > 0).sum(axis=(0, 1)), color="green", label="ground truth")
ax.plot(np.arange(n_slices), (pred > 0).sum(axis=(0, 1)), color="red", linestyle="--", label="prediction")
ax.axvline(z, color="black", alpha=0.4, linestyle=":")
ax.set_xlabel("Axial slice (z)")
ax.set_ylabel("Spleen voxels in slice")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.divider()
with st.expander("How to read this"):
    st.markdown("""
- **Dice coefficient** — twice the overlap divided by the sum of the two
  volumes. 1.0 = perfect, 0.0 = no overlap. The standard metric for medical
  segmentation. State-of-the-art on Spleen is ~0.96; a 10-epoch class run
  typically lands at 0.70–0.85.
- **IoU (Jaccard)** — overlap divided by union. Always lower than Dice for
  the same prediction. Use whichever the convention in your field is.
- **Why the contour view?** A filled mask hides boundary errors; contours
  show exactly where ground truth and prediction diverge. Boundary jitter is
  usually fixable with more training; missed lobes need more data or
  stronger augmentation.
    """)

st.caption(
    "Dataset: Medical Segmentation Decathlon Task09 Spleen — "
    "[medicaldecathlon.com](http://medicaldecathlon.com/)"
)
