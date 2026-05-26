"""
Export a handful of PneumoniaMNIST test images as PNG so the Streamlit
classifier page has built-in samples to demo without requiring the user
to upload anything.

Run AFTER training (medmnist npz must be cached locally):
    python scripts/prepare_sample_xrays.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

import medmnist
from medmnist import INFO

OUT_DIR = Path("data/sample_xrays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

info = INFO["pneumoniamnist"]
DataClass = getattr(medmnist, info["python_class"])
ds = DataClass(split="test", download=True, size=224, root="data/medmnist")

# Pick a few of each class
n_per_class = 3
class_idx = {0: [], 1: []}
for i in range(len(ds)):
    _, lbl = ds[i]
    c = int(lbl[0])
    if len(class_idx[c]) < n_per_class:
        class_idx[c].append(i)
    if all(len(v) == n_per_class for v in class_idx.values()):
        break

for c, idxs in class_idx.items():
    label = info["label"][str(c)]
    for k, i in enumerate(idxs):
        img, _ = ds[i]
        out = OUT_DIR / f"{label}_{k+1:02d}_idx{i}.png"
        img.save(out)
        print(f"  saved {out}")

print(f"\nSamples written to {OUT_DIR}/ — the Streamlit classifier page will see these in the dropdown.")
