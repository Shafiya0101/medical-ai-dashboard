# AI for Health — Interactive Dashboard

> A teaching dashboard that walks through three foundational tasks in medical AI: **DICOM metadata extraction**, **chest X-ray classification (ResNet50 vs ViT)**, and **3D organ segmentation (MONAI U-Net)**. Built as a Streamlit app for the PGE5 AI for Health course.

![status](https://img.shields.io/badge/status-educational-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![framework](https://img.shields.io/badge/built%20with-PyTorch%20%2B%20MONAI%20%2B%20Streamlit-orange)

---

## 🎯 What problem are we solving?

Medical AI sits at the intersection of three hard problems:

**1. The data is hard.** Medical images aren't natural images. A chest X-ray is a 16-bit grayscale DICOM with 50+ metadata tags. A CT scan is a 3D volume of Hounsfield units, not pixels. A pathology slide is 4 GB. Each modality demands its own tooling, its own preprocessing, and its own failure modes.

**2. The stakes are high.** A false negative on a chest X-ray means a missed pneumothorax. A misplaced segmentation boundary means cutting into the wrong tissue. Models that "work on average" can fail catastrophically on the subgroup that matters. This is why we evaluate with ROC curves and Dice coefficients, not accuracy.

**3. The deployment context is messy.** Models trained on one hospital's data routinely fail on another's because of scanner differences, demographics, and reporting conventions. **Generalisation is the hard part.**

This dashboard isolates the three foundational skills you need before you can deploy anything clinical:

| Skill | This dashboard's page | The deeper question |
|---|---|---|
| 📊 Wrangle the data | DICOM Explorer | "Which subset of my data should I train on?" |
| 🧠 Train a model | X-ray Classification | "Which architecture, and is it actually learning the right thing?" |
| 🔍 Inspect what the model learned | Grad-CAM / attention maps / mask overlays | "Can I trust this prediction enough to act on it?" |

---

## 🖥️ The three pages

### 📊 Page 1 — DICOM Metadata Explorer

Walks a folder of DICOM files recursively, extracts metadata (modality, patient age, acquisition date, body part, and more), builds a `pandas` DataFrame, and reports summary statistics + charts. Auto-downloads the **SIIM_SMALL** teaching set (~30 MB, 250 chest X-rays) on first run.

**Why this first?** Cohort selection ("all chest CTs from patients aged 60–80 in 2024") is the first step of every clinical-ML project, and the SQL-like query you actually run is over DICOM tags.

### 🧠 Page 2 — Chest X-ray Classification: ResNet50 vs ViT

Upload a chest X-ray (or use a built-in sample). The page runs inference with both a fine-tuned **ResNet50** and a fine-tuned **ViT-B/16** on the same input, shows the predicted probability of pneumonia, and overlays **Grad-CAM** (for ResNet) and **attention rollout** (for ViT) heatmaps.

**Why two models?** Architectural choice has measurable trade-offs:
- ResNet50 (~25M params) — convolutional, strong locality prior, fast.
- ViT-B/16 (~86M params) — transformer, models long-range relationships, needs more data.

On a small dataset like PneumoniaMNIST, ResNet50 typically edges out — which is itself a finding worth reporting.

**Why attention maps?** A model that picks up on an image-corner annotation ("Hospital A — emergency dept") instead of the lungs will have great validation accuracy and be useless in production. Heatmaps are how you catch this.

### 🔬 Page 3 — 3D Spleen Segmentation: MONAI U-Net

Browse a sample 3D abdominal CT volume from the **Medical Segmentation Decathlon Task09 (Spleen)** dataset, scroll through axial slices, and see the predicted spleen mask overlaid on the image. Inference uses **sliding-window** so the model handles arbitrary-size volumes.

**Why MONAI?** It's the medical-imaging PyTorch framework (Nvidia + KCL): U-Net architectures, DiceCE loss, sliding-window inference, and 3D-aware augmentation built in.

**Why 3D?** A 2D slice-by-slice approach ignores through-plane anatomical context. A 3D U-Net sees the whole organ at once.

---

## 🚀 Quick start

### Local install

```bash
git clone https://github.com/<your-username>/ai-for-health.git
cd ai-for-health
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train the models (one time, ~30 min on a T4 GPU)

```bash
# Fine-tune ResNet50 and ViT-B/16 on PneumoniaMNIST  (~10 min on T4)
python scripts/train_classifier.py --epochs 3

# Train the 3D U-Net on Spleen — downloads 1.5 GB on first run  (~20-40 min on T4)
python scripts/train_segmentation.py --epochs 10

# Copy 2 small CT volumes + a few sample X-rays into data/ for the demo pages
python scripts/prepare_sample_volumes.py
python scripts/prepare_sample_xrays.py
```

Each script saves model weights to `models/` and a JSON training log next to them.

### Run the dashboard

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in a browser. The DICOM Explorer works immediately; the other two pages need the trained checkpoints from the step above.

### Run on Colab (no local GPU needed)

```python
# In a Colab notebook
!git clone https://github.com/<your-username>/ai-for-health.git
%cd ai-for-health
!pip install -r requirements.txt
!python scripts/train_classifier.py --epochs 3
!python scripts/train_segmentation.py --epochs 10
!python scripts/prepare_sample_volumes.py
!python scripts/prepare_sample_xrays.py

# To preview the dashboard from Colab, expose it via localtunnel:
!npm install -g localtunnel
!streamlit run app/main.py & npx localtunnel --port 8501
```

---

## 📁 Repo structure

```
ai-for-health/
├── app/
│   ├── main.py                      # Landing page (problem framing)
│   └── pages/
│       ├── 1_📊_DICOM_Explorer.py
│       ├── 2_🧠_Xray_Classification.py
│       └── 3_🔬_Spleen_Segmentation.py
├── scripts/
│   ├── train_classifier.py          # Fine-tune ResNet50 + ViT
│   ├── train_segmentation.py        # Train 3D U-Net
│   ├── prepare_sample_volumes.py    # Copy 2 CTs into data/sample_volumes/
│   └── prepare_sample_xrays.py      # Export 6 X-rays into data/sample_xrays/
├── notebooks/                       # Standalone Jupyter notebooks
│   ├── 01_dicom_explorer.ipynb
│   └── 02_classification_and_segmentation.ipynb
├── data/
│   ├── sample_xrays/                # Generated by prepare_sample_xrays.py
│   ├── sample_volumes/              # Generated by prepare_sample_volumes.py
│   ├── medmnist/                    # Downloaded by training (gitignored)
│   └── monai/                       # Downloaded by training (gitignored)
├── models/                          # Trained checkpoints (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📚 Datasets used

All datasets used in this project are publicly available and free.

| # | Dataset | Modality | Size | Source | License |
|---|---|---|---|---|---|
| 1 | **SIIM_SMALL** (teaching subset of SIIM-ACR Pneumothorax) | Chest X-ray (DICOM) | ~30 MB, 250 files | [SIIM-ACR](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) | CC BY-SA 4.0 |
| 2 | **PneumoniaMNIST** (MedMNIST v3) | Chest X-ray (PNG-equivalent) | 5,856 images @ 224×224 | [medmnist.com](https://medmnist.com/) | CC BY 4.0 |
| 3 | **Decathlon Task09 Spleen** | Abdominal CT (NIfTI) | ~1.5 GB, 41 volumes | [medicaldecathlon.com](http://medicaldecathlon.com/) | CC BY-SA 4.0 |

### Other datasets surveyed (not used in code but cited in the report)

| Dataset | Modality | URL |
|---|---|---|
| BraTS 2020 | Brain MRI (NIfTI) | https://www.kaggle.com/datasets/awsaf49/brats2020-training-data |
| OpenSlide test data | Pathology WSI | https://openslide.cs.cmu.edu/download/openslide-testdata/ |
| MIMIC-IV | ICU EHR + clinical notes | https://physionet.org/content/mimiciv/ |
| Augmented Clinical Notes | Synthetic clinical text | https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes |
| NIH Chest X-ray 14 | Chest X-ray (PNG) | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| ISIC Archive | Dermoscopy | https://www.isic-archive.com/ |

---

## 🧪 What you'll see when it works

After running both training scripts, the dashboard pages report:

- **DICOM Explorer** — 250 DICOMs loaded, modality=CR, body-part=CHEST, patient ages ranging ~1–95 years, ~5 charts including modality bar, body-part bar, age histogram with mean/median lines, sex pie chart, acquisitions per year.
- **X-ray Classification** — ResNet50 test AUC typically ~0.95, ViT-B/16 test AUC typically ~0.93 (after 3 epochs). Grad-CAM heatmaps focal on lungs; ViT attention rollout more diffuse.
- **Spleen Segmentation** — Dice ~0.70–0.85 after 10 epochs (state-of-the-art is ~0.96 after 600 epochs). Predicted contour visibly overlapping ground truth; spleen-footprint-per-slice curves matching.

---

## 🛠️ Tech stack

- **PyTorch** for model training
- **torchvision** for ResNet50 + ImageNet weights
- **timm** for ViT-B/16
- **MONAI** for the 3D U-Net + medical-imaging transforms
- **medmnist** for PneumoniaMNIST
- **pydicom** for DICOM I/O
- **nibabel** for NIfTI I/O (via MONAI)
- **fastai** for the SIIM_SMALL download
- **grad-cam** + custom attention rollout for interpretability
- **Streamlit** for the UI
- **pandas / matplotlib / scikit-learn** for analysis

---

## ⚠️ Caveats and limitations

- Models trained for ~3 epochs (classifier) and ~10 epochs (segmenter) are **demo-quality**, not clinical-quality. State-of-the-art needs an order of magnitude more compute.
- Inference on raw user-uploaded X-rays goes through the same preprocessing as training (224×224, ImageNet normalisation). Out-of-distribution images (e.g. lateral views, peds, very different scanners) will give unreliable predictions.
- The Grad-CAM and attention-rollout heatmaps highlight regions the model relied on; they are **not** a guarantee that the model is reasoning clinically. Shortcut learning is real — a model can have correct outputs with wrong attention.
- This is a **teaching repo**. Do not use it on real patient data.

---

## 🙏 Acknowledgements

- **SIIM** + **ACR** + **FISABIO** + **RSNA** for the pneumothorax + COVID-19 chest X-ray datasets
- **MedMNIST** team for the PneumoniaMNIST benchmark
- **Medical Segmentation Decathlon** organisers for the Spleen task
- **MONAI** consortium (NVIDIA + KCL + others) for the medical-imaging PyTorch framework
- **fastai** for the SIIM_SMALL teaching subset distribution

Built for the **PGE5 AI for Health** course (Aivancity, 2026).
