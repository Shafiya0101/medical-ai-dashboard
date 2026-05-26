"""
Train a 3D U-Net on the Decathlon Task09 Spleen dataset.

Outputs:
    models/unet_spleen.pt
    models/segmentation_training_log.json

Run:
    python scripts/train_segmentation.py --epochs 10
On a Colab T4: ~20-40 minutes for 10 epochs (plus ~5 min download).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped, AsDiscrete,
)
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference


def build_transforms():
    win_min, win_max = -57, 164
    spacing = (1.5, 1.5, 2.0)
    train_tx = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=win_min, a_max=win_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_tx = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=win_min, a_max=win_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_tx, val_tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--data-root", default="data/monai")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("⚠ Training a 3D U-Net on CPU is impractical (~hours per epoch). Use a GPU.")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_root).mkdir(parents=True, exist_ok=True)

    train_tx, val_tx = build_transforms()

    print("Downloading Decathlon Task09 Spleen (≈1.5 GB)...")
    train_ds = DecathlonDataset(
        root_dir=args.data_root, task="Task09_Spleen", section="training",
        transform=train_tx, download=True, num_workers=2,
        val_frac=0.2, seed=0, cache_num=24,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_root, task="Task09_Spleen", section="validation",
        transform=val_tx, download=False, num_workers=2,
        val_frac=0.2, seed=0, cache_num=8,
    )
    print(f"Train volumes: {len(train_ds)}, Val volumes: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    dice_m = DiceMetric(include_background=False, reduction="mean")
    iou_m = MeanIoU(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    log = {"epoch": [], "loss": [], "val_dice": [], "val_iou": []}
    best_dice = 0.0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / n
        entry = {"epoch": ep, "loss": train_loss, "time_s": time.time() - t0}

        if ep % args.val_every == 0 or ep == args.epochs:
            model.eval()
            dice_m.reset(); iou_m.reset()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    logits = sliding_window_inference(x, roi_size=(96, 96, 96), sw_batch_size=2, predictor=model)
                    preds = [post_pred(p) for p in decollate_batch(logits)]
                    labels = [post_label(l) for l in decollate_batch(y)]
                    dice_m(y_pred=preds, y=labels)
                    iou_m(y_pred=preds, y=labels)
            v_dice = dice_m.aggregate().item()
            v_iou = iou_m.aggregate().item()
            entry["val_dice"] = v_dice
            entry["val_iou"] = v_iou
            if v_dice > best_dice:
                best_dice = v_dice
                torch.save(model.state_dict(), Path(args.models_dir) / "unet_spleen.pt")
            print(f"Epoch {ep}/{args.epochs} | loss={train_loss:.4f} | val_dice={v_dice:.4f} | val_iou={v_iou:.4f} | {entry['time_s']:.1f}s")
        else:
            print(f"Epoch {ep}/{args.epochs} | loss={train_loss:.4f} | {entry['time_s']:.1f}s")

        log["epoch"].append(ep)
        log["loss"].append(train_loss)
        if "val_dice" in entry:
            log["val_dice"].append([ep, entry["val_dice"]])
            log["val_iou"].append([ep, entry["val_iou"]])

    print(f"\nTotal training time: {time.time() - t_start:.0f}s | best val Dice: {best_dice:.4f}")

    with open(Path(args.models_dir) / "segmentation_training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log → {args.models_dir}/segmentation_training_log.json")


if __name__ == "__main__":
    main()
