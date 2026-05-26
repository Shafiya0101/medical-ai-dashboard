# Data directory

This directory is mostly populated by the training scripts. It contains:

- `medmnist/` — PneumoniaMNIST cache (downloaded by `train_classifier.py`). Gitignored.
- `monai/` — Medical Decathlon Task09 Spleen cache (downloaded by `train_segmentation.py`). Gitignored.
- `sample_xrays/` — A few PNG X-rays for the classification demo. Created by `scripts/prepare_sample_xrays.py`.
- `sample_volumes/` — 2 sample CT volumes for the segmentation demo. Created by `scripts/prepare_sample_volumes.py`.

You don't need to populate any of these manually — running the training scripts handles it.
