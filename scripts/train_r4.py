#!/usr/bin/env python3
"""Review 4 — Train production DenseNet-121 with ablation and Optuna HPO.

Usage:
    python scripts/train_r4.py --data_root /path/to/nih-data --output_dir ./outputs/r4
    python scripts/train_r4.py --data_root /path/to/nih-data --optuna_trials 20
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from src.data.dataset import AdvancedXrayDataset, get_dataloaders, get_transforms
from src.models.densenet import DenseNetCXR, EfficientNetCXR
from src.utils.metrics import evaluate_model, per_class_auc
from src.utils.visualization import plot_learning_curves, plot_roc_curves
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train DenseNet-121 production model (R4)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="./outputs/r4")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--subset_frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--optuna_trials", type=int, default=0,
                   help="Number of Optuna HPO trials (0 = skip)")
    return p.parse_args()


def train_model(model, tr_loader, vl_loader, name, epochs, lr, pos_weight, device, save_dir):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 10,
        steps_per_epoch=len(tr_loader), epochs=epochs,
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, no_improve = 0.0, 0
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss += loss.item() * imgs.size(0)

        model.eval()
        vl_loss, all_y, all_p = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in vl_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                vl_loss += criterion(out, lbls).item() * imgs.size(0)
                all_y.append(lbls.cpu().numpy())
                all_p.append(torch.sigmoid(out).cpu().numpy())

        y = np.concatenate(all_y)
        p_arr = np.concatenate(all_p)
        try:
            auc = float(roc_auc_score(y, p_arr, average="macro"))
        except ValueError:
            auc = 0.5

        history["train_loss"].append(tr_loss / len(tr_loader.dataset))
        history["val_loss"].append(vl_loss / len(vl_loader.dataset))
        history["val_auc"].append(auc)
        print(f"  Ep {ep:02d}  tr={history['train_loss'][-1]:.4f}  vl={history['val_loss'][-1]:.4f}  AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "auc": auc, "epoch": ep,
                 "history": history, "classes": CLASSES},
                save_path,
            )
            print(f"    Saved (AUC={best_auc:.4f}) -> {save_path}")
        else:
            no_improve += 1
            if no_improve >= 4:
                print(f"  Early stop ep {ep}")
                break

    return model, history, best_auc, save_path


def run_optuna(train_loader, val_loader, pos_weight, device, n_trials):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Install optuna: pip install optuna")
        return {}

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        model = DenseNetCXR(dropout=dropout).to(device)
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for _ in range(2):
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                criterion(model(imgs), lbls).backward()
                opt.step()
        y, p = evaluate_model(model, val_loader, device)
        try:
            return float(roc_auc_score(y, p, average="macro"))
        except ValueError:
            return 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best Optuna params: {study.best_params}  AUC={study.best_value:.4f}")
    return study.best_params


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
        args.data_root,
        subset_frac=args.subset_frac,
        batch_size=args.batch_size,
        aug_level="strong",
        device=device,
    )

    if args.optuna_trials > 0:
        best_params = run_optuna(train_loader, val_loader, pos_weight, device, args.optuna_trials)
        lr = best_params.get("lr", args.lr)
        dropout = best_params.get("dropout", args.dropout)
    else:
        lr, dropout = args.lr, args.dropout

    # ── DenseNet-121 ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Training DenseNet-121 (production)")
    print("=" * 50)
    densenet = DenseNetCXR(dropout=dropout).to(device)
    dn, dn_hist, dn_auc, _ = train_model(
        densenet, train_loader, val_loader,
        "DenseNet121", args.epochs, lr, pos_weight, device, args.output_dir,
    )
    y, p = evaluate_model(dn, test_loader, device)
    print("\nDenseNet-121 per-class AUC:")
    for cls, auc in per_class_auc(y, p).items():
        print(f"  {cls:<22} {auc:.4f}")
    plot_learning_curves(dn_hist, title="R4-DenseNet121", save_dir=args.output_dir)
    plot_roc_curves(y, p, title="R4-DenseNet121", save_dir=args.output_dir)

    # ── EfficientNet-B2 baseline ─────────────────────────────
    print("\n" + "=" * 50)
    print("Training EfficientNet-B2 (baseline)")
    print("=" * 50)
    effnet = EfficientNetCXR(dropout=dropout).to(device)
    en, en_hist, en_auc, _ = train_model(
        effnet, train_loader, val_loader,
        "EfficientNetB2", args.epochs, lr, pos_weight, device, args.output_dir,
    )
    y2, p2 = evaluate_model(en, test_loader, device)
    print("\nEfficientNet-B2 per-class AUC:")
    for cls, auc in per_class_auc(y2, p2).items():
        print(f"  {cls:<22} {auc:.4f}")
    plot_learning_curves(en_hist, title="R4-EfficientNetB2", save_dir=args.output_dir)
    plot_roc_curves(y2, p2, title="R4-EfficientNetB2", save_dir=args.output_dir)

    print(f"\nDenseNet AUC={dn_auc:.4f}  EfficientNet AUC={en_auc:.4f}")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
