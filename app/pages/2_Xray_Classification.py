"""
Chest X-ray Classification — ResNet50 vs ViT-B/16
==================================================
Inference page. Loads a fine-tuned model from models/, runs inference on
an uploaded X-ray (or built-in sample), shows probability + Grad-CAM (for
ResNet) or attention rollout (for ViT).

The first time the page is opened without a trained checkpoint, it offers to
run scripts/train_classifier.py (which fine-tunes ResNet50 + ViT on
PneumoniaMNIST). On a T4 this is ~10 min; afterwards inference is instant.
"""
from pathlib import Path
import io

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.set_page_config(page_title="X-ray Classification", page_icon="🧠", layout="wide")
st.title("🧠 Chest X-ray Classification — ResNet50 vs ViT")
st.markdown(
    "Upload a chest X-ray (or use a built-in sample). The page runs inference "
    "with both a fine-tuned **ResNet50** and a fine-tuned **ViT-B/16**, shows "
    "the predicted probability of pneumonia, and visualises what each model "
    "attended to."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "data" / "sample_xrays"

CLASS_NAMES = ["normal", "pneumonia"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
# Model loading (cached)
# =============================================================================
@st.cache_resource(show_spinner="Loading ResNet50...")
def load_resnet():
    path = MODEL_DIR / "resnet50_pneumonia.pt"
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        trained = True
    else:
        trained = False
    model = model.to(DEVICE).eval()
    return model, trained


@st.cache_resource(show_spinner="Loading ViT-B/16...")
def load_vit():
    import timm
    path = MODEL_DIR / "vit_pneumonia.pt"
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        trained = True
    else:
        trained = False
    model = model.to(DEVICE).eval()
    return model, trained


# =============================================================================
# Preprocess
# =============================================================================
def preprocess(pil_img):
    """PIL.Image -> (3,224,224) normalised tensor + (3,224,224) [0,1] tensor for display."""
    tx_norm = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tx_disp = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    return tx_norm(pil_img).unsqueeze(0), tx_disp(pil_img)


# =============================================================================
# Attention visualisations
# =============================================================================
def gradcam_heatmap(model, input_tensor, target_class=None):
    """Compute Grad-CAM on the last ResNet50 layer4 block. Returns (224,224) [0,1]."""
    activations, gradients = {}, {}

    target_layer = model.layer4[-1]

    def fwd_hook(_m, _i, o):
        activations["v"] = o

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    input_tensor = input_tensor.to(DEVICE)
    model.zero_grad()
    out = model(input_tensor)
    cls = int(out.argmax(dim=1).item()) if target_class is None else int(target_class)
    out[0, cls].backward()

    A = activations["v"][0].detach()           # (C, h, w)
    G = gradients["v"][0].detach()              # (C, h, w)
    weights = G.mean(dim=(1, 2))                # (C,)
    cam = torch.einsum("c,chw->hw", weights, A)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)
    h1.remove()
    h2.remove()
    return cam[0, 0].cpu().numpy()


def vit_attention_rollout(model, input_tensor):
    """Cumulative attention rollout for a timm ViT. Returns (224,224) [0,1]."""
    attentions = []
    handles = []
    def make_hook(mod):
        def hook(_m, inp, _out):
            B, N, C = inp[0].shape
            qkv = mod.qkv(inp[0]).reshape(B, N, 3, mod.num_heads, mod.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * mod.scale
            attn = attn.softmax(dim=-1)
            attentions.append(attn.detach().cpu())
        return hook
    for blk in model.blocks:
        handles.append(blk.attn.register_forward_hook(make_hook(blk.attn)))
    with torch.no_grad():
        _ = model(input_tensor.to(DEVICE))
    for h in handles:
        h.remove()

    result = torch.eye(attentions[0].size(-1))
    for a in attentions:
        a = a.mean(dim=1)
        I = torch.eye(a.size(-1))
        a = (a + I) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        result = a[0] @ result
    mask = result[0, 1:]                            # CLS attention to each patch
    side = int(mask.shape[-1] ** 0.5)
    mask = mask.reshape(side, side).numpy()
    mask = mask / (mask.max() + 1e-8)
    mask = F.interpolate(
        torch.tensor(mask).float()[None, None], size=(224, 224),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    return mask


def overlay_heatmap(rgb01, heatmap, alpha=0.5):
    """Overlay a [0,1] heatmap on a [0,1] RGB image with jet colormap."""
    color = cm.jet(heatmap)[..., :3]
    return np.clip((1 - alpha) * rgb01 + alpha * color, 0, 1)


# =============================================================================
# UI
# =============================================================================
resnet, resnet_trained = load_resnet()
vit, vit_trained = load_vit()

# Warn if no trained weights
if not (resnet_trained and vit_trained):
    st.warning(
        f"**No fine-tuned weights found** in `models/`. The page will run with "
        f"randomly-initialised classifier heads, so predictions are meaningless. "
        f"Train first with:\n\n```bash\npython scripts/train_classifier.py\n```\n\n"
        f"Status: ResNet50 trained = `{resnet_trained}`, ViT trained = `{vit_trained}`"
    )

# Sidebar: pick input
st.sidebar.header("Input image")
src = st.sidebar.radio("Source", ["Upload an X-ray", "Built-in sample"])

img_pil = None
if src == "Upload an X-ray":
    uploaded = st.sidebar.file_uploader("PNG, JPG, or DICOM", type=["png", "jpg", "jpeg", "dcm"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".dcm"):
                import pydicom
                ds = pydicom.dcmread(uploaded)
                arr = ds.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                img_pil = Image.fromarray(arr.astype(np.uint8)).convert("L")
            else:
                img_pil = Image.open(uploaded).convert("L")
        except Exception as e:
            st.sidebar.error(f"Could not open file: {e}")
else:
    if SAMPLE_DIR.exists():
        samples = sorted([p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if samples:
            pick = st.sidebar.selectbox("Sample", [p.name for p in samples])
            img_pil = Image.open(SAMPLE_DIR / pick).convert("L")
        else:
            st.sidebar.warning(f"No sample images found in `{SAMPLE_DIR}`")
    else:
        st.sidebar.warning(f"Sample directory not found: `{SAMPLE_DIR}`")

if img_pil is None:
    st.info("👈 Upload an X-ray or pick a built-in sample in the sidebar.")
    st.stop()

# Preprocess
x_norm, x_disp = preprocess(img_pil)

# Inference
with torch.no_grad():
    resnet_logits = resnet(x_norm.to(DEVICE))
    vit_logits = vit(x_norm.to(DEVICE))
resnet_probs = torch.softmax(resnet_logits, dim=1)[0].cpu().numpy()
vit_probs = torch.softmax(vit_logits, dim=1)[0].cpu().numpy()

# ---------- Top row: image + predictions ----------
col_img, col_pred = st.columns([1, 1])
with col_img:
    st.subheader("Input")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(x_disp.permute(1, 2, 0).numpy())
    ax.axis("off")
    st.pyplot(fig)

with col_pred:
    st.subheader("Predictions")
    pred_df_data = {
        "Class":    CLASS_NAMES,
        "ResNet50": [f"{p:.1%}" for p in resnet_probs],
        "ViT-B/16": [f"{p:.1%}" for p in vit_probs],
    }
    import pandas as pd
    st.table(pd.DataFrame(pred_df_data))

    resnet_pred = CLASS_NAMES[int(resnet_probs.argmax())]
    vit_pred = CLASS_NAMES[int(vit_probs.argmax())]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("ResNet50 prediction", resnet_pred,
                  delta=f"{resnet_probs.max():.0%} confidence", delta_color="off")
    with c2:
        st.metric("ViT-B/16 prediction", vit_pred,
                  delta=f"{vit_probs.max():.0%} confidence", delta_color="off")

    if resnet_pred != vit_pred:
        st.warning("⚠ The two models disagree on this image.")

# ---------- Attention maps ----------
st.subheader("What did the models look at?")
st.caption("Heatmaps highlight regions the model attended to when making the prediction.")

col_r, col_v = st.columns(2)
disp_np = x_disp.permute(1, 2, 0).numpy()

with col_r:
    st.markdown("**ResNet50 — Grad-CAM**")
    with st.spinner("Computing Grad-CAM..."):
        try:
            cam = gradcam_heatmap(resnet, x_norm)
            overlay = overlay_heatmap(disp_np, cam, alpha=0.5)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

with col_v:
    st.markdown("**ViT-B/16 — Attention rollout**")
    with st.spinner("Computing attention rollout..."):
        try:
            roll = vit_attention_rollout(vit, x_norm)
            overlay = overlay_heatmap(disp_np, roll, alpha=0.5)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Attention rollout failed: {e}")

with st.expander("How to read these maps"):
    st.markdown("""
- **Grad-CAM** weights the last conv feature map by the gradient of the
  predicted-class logit and ReLU's it. Hot regions are areas whose activation
  most increased the prediction. ResNet-style heatmaps tend to be focal.
- **Attention rollout** multiplies attention matrices across all 12 transformer
  blocks (adding the identity for the residual connection) and extracts the
  CLS token's attention to each 16×16 patch. ViT heatmaps are usually more
  diffuse because every patch can attend to every other patch.
- **Caveat.** A model can have a perfectly correct prediction with a heatmap
  that focuses on the wrong region (shortcut learning). These maps are a
  sanity check, not a guarantee of clinical reasoning.
    """)

st.divider()
st.caption(
    "Training dataset: PneumoniaMNIST — "
    "[medmnist.com](https://medmnist.com/). "
    "Models: ResNet50 (ImageNet → PneumoniaMNIST), ViT-B/16 (ImageNet → PneumoniaMNIST)."
)
