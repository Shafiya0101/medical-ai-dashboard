"""
AI for Health — Interactive Dashboard
======================================
Three demos in one app:
  1. DICOM Metadata Explorer
  2. Chest X-ray Classification (ResNet50 vs ViT) with Grad-CAM
  3. 3D Spleen Segmentation (MONAI U-Net) with interactive slice viewer

Run with:
    streamlit run app/main.py
"""
import streamlit as st

st.set_page_config(
    page_title="AI for Health Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Landing page content ----
st.title("🏥 AI for Health — Interactive Dashboard")

st.markdown("""
A teaching dashboard that walks through three foundational tasks in medical
AI, built for the **PGE5 AI for Health** course.

#### What this app demonstrates

| Page | Task | Modality | Method |
|---|---|---|---|
| **DICOM Explorer** | Metadata extraction & cohort analysis | Chest X-ray (DICOM) | Pandas + pydicom |
| **X-ray Classification** | Pneumonia detection | Chest X-ray (PNG) | ResNet50 + ViT, Grad-CAM |
| **Spleen Segmentation** | Organ delineation | Abdominal CT (NIfTI) | 3D U-Net (MONAI) |

Use the sidebar to navigate.
""")

st.divider()

# =============== PROBLEM FRAMING ===============
st.header("🎯 What problem are we solving?")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
Medical AI sits at the intersection of three hard problems:

**1. The data itself is hard.**
Medical images are not normal images. A chest X-ray is a 16-bit grayscale
DICOM with 50+ metadata tags carrying patient demographics, scanner settings,
and acquisition parameters — all of which a model needs to reason about (or
*not* leak from). A CT scan is a 3D volume of Hounsfield units, not pixels.
A whole-slide pathology image is 4 GB. Each modality demands its own tooling,
its own preprocessing, and its own ways of going wrong.

**2. The stakes are high.**
A false negative on a chest X-ray means a missed pneumothorax. A misplaced
segmentation boundary means cutting into the wrong tissue. Models that work
"on average" can fail catastrophically on the small subgroup where it matters
most. This is why we evaluate with ROC curves and Dice coefficients, not
just accuracy.

**3. The deployment context is messy.**
Models trained on one hospital's data often fail on another's because of
scanner differences, demographics, label-extraction methods, and reporting
conventions. Generalisation is the hard part, not training.

This dashboard isolates the three foundational skills you need before you can
even *think* about deploying a clinical model:
""")

with col2:
    st.info("""
**The three foundational skills**

📊 **Wrangle the data** — read DICOM/NIfTI, extract metadata, build cohorts, spot quality issues.

🧠 **Train a model** — fine-tune a pretrained CNN/ViT, evaluate honestly with ROC-AUC.

🔍 **Inspect what the model learned** — Grad-CAM, attention maps, segmentation overlays. Without this, you have a black box you can't defend.
    """)

st.divider()

# =============== APPROACH ===============
st.header("🛠️ Our approach, per page")

with st.expander("📊 **DICOM Metadata Explorer** — Why metadata first?", expanded=False):
    st.markdown("""
Before you train any model, you need to know what data you have.

A DICOM file's pixels are only half the story — the **header** carries everything
that drives a real ML project: which scanner produced the image, what body part,
how old the patient was, what acquisition protocol was used. This page:

- Walks any folder of `.dcm` files recursively
- Extracts the four metadata fields the assignment specifies
  (modality, age, acquisition date, body part) plus useful extras
- Surfaces missing-value patterns — real DICOM archives are messy
- Produces summary stats and the charts you need for a data section

**Why this matters in practice.** Cohort selection ("all chest CTs from
patients aged 60–80 in 2024") is the first step of every clinical-ML project,
and the SQL-like query you actually run is over DICOM tags.

**Dataset:** SIIM-ACR Pneumothorax (250 chest X-rays, ~30 MB, downloads via fastai).
    """)

with st.expander("🧠 **X-ray Classification** — ResNet50 vs Vision Transformer", expanded=False):
    st.markdown("""
The classic medical-AI starter task: given a chest X-ray, predict whether
pneumonia is present. We compare two architectures head-to-head:

- **ResNet50** — convolutional, ~25M params, ImageNet pretrained. The
  workhorse of medical imaging for years.
- **ViT-B/16** — transformer, ~86M params, ImageNet pretrained. Treats the
  image as a sequence of patches.

Both are fine-tuned on **PneumoniaMNIST** (5,856 pediatric chest X-rays)
and evaluated with **ROC-AUC** (the metric of choice when class balance
or operating threshold matters).

**Why two models?** Architectural choice has measurable trade-offs:
ConvNets bake in a strong locality prior (great with limited data); ViTs
need more data but model long-range relationships better. On a small dataset
like PneumoniaMNIST, ResNet typically edges out — which is itself a finding.

**Why attention maps?** A model that picks up on an image-corner annotation
("Hospital A — emergency dept") instead of the lungs will have great
validation accuracy and be useless in production. Grad-CAM and attention
rollout let you check what the model is actually looking at.
    """)

with st.expander("🔬 **3D Spleen Segmentation** — Why MONAI, why 3D?", expanded=False):
    st.markdown("""
Segmentation = "label every voxel". The output isn't a single number, it's a
3D mask the same shape as the input. The clinical use: volumetry (organ size
over time), surgical planning, radiotherapy target definition.

We use:
- **Decathlon Task09 Spleen** — 41 abdominal CT volumes with expert
  per-voxel spleen masks. The Medical Segmentation Decathlon is the standard
  benchmark suite.
- **MONAI U-Net** — the PyTorch-based medical-imaging framework from
  Nvidia + KCL. Provides the U-Net architecture, the Dice+CE loss, the
  sliding-window inference, and a preprocessing pipeline tuned for 3D
  medical volumes.
- **Dice coefficient** + **IoU** — the standard segmentation metrics. Dice
  is roughly "twice the overlap divided by the union"; IoU is "overlap
  divided by union". They always agree on ordering.

**Why 3D?** A 2D slice-by-slice approach ignores anatomical context that
exists in the through-plane direction. A 3D U-Net sees the whole organ at
once.

**Realistic expectation.** State-of-the-art on Spleen is Dice ~0.96 after
~600 epochs. Our 10-epoch training (for time reasons) lands at Dice ~0.70–0.85,
which is still enough to see visibly correct masks.
    """)

st.divider()

# =============== NAVIGATION ===============
st.header("📍 Get started")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 📊 DICOM Explorer")
    st.markdown("Auto-downloads the dataset on first run. Takes ~30 seconds.")
    st.page_link("pages/1_📊_DICOM_Explorer.py", label="Open →", icon="📊")
with c2:
    st.markdown("### 🧠 X-ray Classification")
    st.markdown("Upload a chest X-ray (or use a built-in sample). Inference takes ~2 seconds.")
    st.page_link("pages/2_🧠_Xray_Classification.py", label="Open →", icon="🧠")
with c3:
    st.markdown("### 🔬 Spleen Segmentation")
    st.markdown("Browse a sample 3D CT volume, scroll through slices, see predicted spleen mask.")
    st.page_link("pages/3_🔬_Spleen_Segmentation.py", label="Open →", icon="🔬")

st.divider()
st.caption("Built for the PGE5 AI for Health course. Code: https://github.com/yourusername/ai-for-health")
