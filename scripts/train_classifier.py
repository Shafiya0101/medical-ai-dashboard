"""
Train ResNet50 and ViT-B/16 on PneumoniaMNIST.

Outputs:
    models/resnet50_pneumonia.pt
    models/vit_pneumonia.pt
    models/classifier_training_log.json

Run:
    python scripts/train_classifier.py --epochs 3
On a Colab T4: ~10 minutes total.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import roc_auc_score, accuracy_score

import medmnist
from medmnist import INFO
import timm


def get_loaders(batch_size=64, data_root="data/medmnist"):
    """Return (train, val, test) DataLoaders for PneumoniaMNIST at 224x224."""
    info = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tx = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tx = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    Path(data_root).mkdir(parents=True, exist_ok=True)
    train_ds = DataClass(split="train", download=True, size=224, transform=train_tx, root=data_root)
    val_ds   = DataClass(split="val",   download=True, size=224, transform=eval_tx,  root=data_root)
    test_ds  = DataClass(split="test",  download=True, size=224, transform=eval_tx,  root=data_root)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2),
        info,
    )


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.squeeze().long()
            p = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(yb.numpy().tolist())
    auc = roc_auc_score(labels, probs)
    preds = [int(p > 0.5) for p in probs]
    acc = accuracy_score(labels, preds)
    return auc, acc, probs, labels


def train_one(model, name, train_loader, val_loader, test_loader, epochs, lr, device, log_dict):
    print(f"\n=== Training {name} ===")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.squeeze().long().to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / n
        val_auc, val_acc, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {ep}/{epochs} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")
        history.append({"epoch": ep, "loss": train_loss, "val_auc": val_auc, "val_acc": val_acc, "time_s": dt})

    test_auc, test_acc, _, _ = evaluate(model, test_loader, device)
    print(f"{name} TEST  AUC={test_auc:.4f}  ACC={test_acc:.4f}")
    log_dict[name] = {"history": history, "test_auc": test_auc, "test_acc": test_acc}
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--skip-resnet", action="store_true")
    ap.add_argument("--skip-vit", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("⚠ Training on CPU — this will be very slow. Use a GPU.")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, info = get_loaders(batch_size=args.batch_size)
    print(f"Dataset: PneumoniaMNIST, classes={list(info['label'].values())}")
    print(f"Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    log = {}

    if not args.skip_resnet:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)
        resnet = train_one(resnet, "ResNet50", train_loader, val_loader, test_loader,
                           args.epochs, args.lr, device, log)
        torch.save(resnet.state_dict(), models_dir / "resnet50_pneumonia.pt")
        print(f"Saved → {models_dir / 'resnet50_pneumonia.pt'}")

    if not args.skip_vit:
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
        vit = train_one(vit, "ViT-B/16", train_loader, val_loader, test_loader,
                        args.epochs, args.lr, device, log)
        torch.save(vit.state_dict(), models_dir / "vit_pneumonia.pt")
        print(f"Saved → {models_dir / 'vit_pneumonia.pt'}")

    with open(models_dir / "classifier_training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log → {models_dir / 'classifier_training_log.json'}")


if __name__ == "__main__":
    main()
