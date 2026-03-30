#!/usr/bin/env python3
"""Review 2 — Train FineTunedResNet and CNN-RNN-Hybrid (temporal) models.

Usage:
    python scripts/train_r2.py --data_root /path/to/nih-data --output_dir ./outputs/r2
"""

import argparse
import os

import torch
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.models.pretrained import FineTunedResNet
from src.models.temporal import CNN_RNN_Hybrid
from src.training.trainer import Trainer
from src.utils.metrics import evaluate_model, per_class_auc
from src.utils.visualization import plot_learning_curves, plot_roc_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train pretrained and temporal models (R2)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="./outputs/r2")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--subset_frac", type=float, default=0.10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "densenet121", "efficientnet_b0"])
    p.add_argument("--rnn_type", default="LSTM", choices=["LSTM", "GRU", "RNN"])
    p.add_argument("--attn", default="bahdanau", choices=["bahdanau", "self", "multihead"])
    return p.parse_args()


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

    # ── FineTunedResNet ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("Training FineTunedResNet-50")
    print("=" * 50)
    resnet = FineTunedResNet().to(device)
    opt_r = optim.AdamW(
        [p for p in resnet.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5,
    )
    trainer_r = Trainer(resnet, train_loader, val_loader,
                        optimizer=opt_r, pos_weight=pos_weight,
                        patience=args.patience, device=device)
    hist_r = trainer_r.fit(
        epochs=args.epochs,
        save_path=os.path.join(args.output_dir, "resnet_checkpoint.pth"),
    )
    y, p = evaluate_model(resnet, test_loader, device)
    aucs = per_class_auc(y, p)
    print("\nResNet-50 per-class AUC:")
    for cls, auc in aucs.items():
        print(f"  {cls:<22} {auc:.4f}")
    plot_learning_curves(hist_r, title="R2-ResNet", save_dir=args.output_dir)
    plot_roc_curves(y, p, title="R2-ResNet", save_dir=args.output_dir)

    # ── CNN-RNN Hybrid ───────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Training CNN-{args.rnn_type}-{args.attn} Hybrid")
    print("=" * 50)
    hybrid = CNN_RNN_Hybrid(
        backbone=args.backbone,
        rnn_type=args.rnn_type,
        attn=args.attn,
    ).to(device)
    opt_h = optim.AdamW(
        [p for p in hybrid.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5,
    )
    trainer_h = Trainer(hybrid, train_loader, val_loader,
                        optimizer=opt_h, pos_weight=pos_weight,
                        patience=args.patience, device=device)
    hist_h = trainer_h.fit(
        epochs=args.epochs,
        save_path=os.path.join(args.output_dir, "hybrid_checkpoint.pth"),
    )
    y2, p2 = evaluate_model(hybrid, test_loader, device)
    aucs2 = per_class_auc(y2, p2)
    print("\nCNN-RNN per-class AUC:")
    for cls, auc in aucs2.items():
        print(f"  {cls:<22} {auc:.4f}")
    plot_learning_curves(hist_h, title="R2-Hybrid", save_dir=args.output_dir)
    plot_roc_curves(y2, p2, title="R2-Hybrid", save_dir=args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
