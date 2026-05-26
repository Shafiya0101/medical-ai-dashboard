# Models directory

Trained model checkpoints land here:

- `resnet50_pneumonia.pt` — ~100 MB, produced by `scripts/train_classifier.py`
- `vit_pneumonia.pt` — ~330 MB, produced by `scripts/train_classifier.py`
- `unet_spleen.pt` — ~20 MB, produced by `scripts/train_segmentation.py`
- `*_training_log.json` — training history (kept in git)

The `.pt` files are gitignored (too large for git). To get them, run the training scripts.
