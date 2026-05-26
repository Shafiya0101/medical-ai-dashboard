"""
Copy 2 sample CT volumes from the downloaded Decathlon Task09 Spleen dataset
into data/sample_volumes/ so the Streamlit segmentation page has something to
load without re-downloading 1.5 GB on every fresh clone.

Run AFTER training (which downloads the dataset):
    python scripts/prepare_sample_volumes.py
"""
import shutil
from pathlib import Path

DATA_ROOT = Path("data/monai/Task09_Spleen")
OUT_DIR = Path("data/sample_volumes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Decathlon layout:  Task09_Spleen/imagesTr/spleen_*.nii.gz  +  labelsTr/spleen_*.nii.gz
img_dir = DATA_ROOT / "imagesTr"
lbl_dir = DATA_ROOT / "labelsTr"

if not img_dir.exists():
    raise SystemExit(
        f"Could not find {img_dir}. Run scripts/train_segmentation.py first "
        f"so the Decathlon dataset is downloaded."
    )

# Pick the two smallest volumes (so the repo stays light)
images = sorted(img_dir.glob("spleen_*.nii.gz"), key=lambda p: p.stat().st_size)
take = [p for p in images if not p.name.startswith("._")][:2]
print(f"Copying {len(take)} smallest volumes into {OUT_DIR}/")

for img in take:
    case = img.name.replace(".nii.gz", "")
    lbl = lbl_dir / img.name
    if not lbl.exists():
        print(f"  skip {case}: no matching label")
        continue
    shutil.copy(img, OUT_DIR / f"{case}_image.nii.gz")
    shutil.copy(lbl, OUT_DIR / f"{case}_label.nii.gz")
    print(f"  {case}: {img.stat().st_size / 1e6:.1f} MB image + {lbl.stat().st_size / 1e6:.1f} MB label")

print("\nDone. The Streamlit segmentation page will now see these in the dropdown.")
