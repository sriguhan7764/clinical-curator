#!/usr/bin/env python3
"""Review 1 — Train ChestMLP and ChestCNN with hyperparameter grid search.

Usage:
    python scripts/train_r1.py --data_root /path/to/nih-data --output_dir ./outputs/r1
"""

import argparse
import os

import torch
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.models.cnn import ChestCNN
from src.models.mlp import ChestMLP
from src.training.trainer import Trainer
from src.utils.metrics import evaluate_model, per_class_auc
from src.utils.visualization import plot_f1_heatmap, plot_learning_curves, plot_roc_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train MLP and CNN on NIH ChestX-ray14 (R1)")
    p.add_argument("--data_root", required=True, help="Path to NIH dataset root")
    p.add_argument("--output_dir", default="./outputs/r1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--subset_frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--grid_search", action="store_true", help="Run quick 2-epoch grid search first")
    return p.parse_args()


def run_grid_search(model_cls, grid, train_loader, val_loader, pos_weight, device):
    """Quick 2-epoch grid search. Returns best params dict."""
    best = {"auc": 0.0, "params": grid[0]}
    for params in grid:
        model = model_cls(dropout_rate=params["dropout"]).to(device)
        opt = optim.Adam(model.parameters(), lr=params["lr"])
        trainer = Trainer(model, train_loader, val_loader, optimizer=opt,
                          pos_weight=pos_weight, patience=5, device=device)
        trainer.fit(epochs=2)
        if trainer.best_auc > best["auc"]:
            best = {"auc": trainer.best_auc, "params": params}
    print(f"  Best params: {best['params']}  AUC={best['auc']:.4f}")
    return best["params"]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
        args.data_root,
        subset_frac=args.subset_frac,
        batch_size=args.batch_size,
        device=device,
    )

    grid = [
        {"lr": 1e-3,  "dropout": 0.4},
        {"lr": 1e-4,  "dropout": 0.4},
        {"lr": 1e-4,  "dropout": 0.3},
    ]

    for model_cls, name in [(ChestMLP, "MLP"), (ChestCNN, "CNN")]:
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print("=" * 50)

        best_params = {"lr": args.lr, "dropout": args.dropout}
        if args.grid_search:
            best_params = run_grid_search(model_cls, grid, train_loader, val_loader, pos_weight, device)

        model = model_cls(dropout_rate=best_params["dropout"]).to(device)
        opt = optim.Adam(model.parameters(), lr=best_params["lr"])
        save_path = os.path.join(args.output_dir, f"{name.lower()}_checkpoint.pth")

        trainer = Trainer(model, train_loader, val_loader,
                          optimizer=opt, pos_weight=pos_weight,
                          patience=args.patience, device=device)
        history = trainer.fit(epochs=args.epochs, save_path=save_path)

        # Evaluate on test set
        y, p = evaluate_model(model, test_loader, device)
        aucs = per_class_auc(y, p)
        print("\nPer-class AUC:")
        for cls, auc in aucs.items():
            print(f"  {cls:<22} {auc:.4f}")

        plot_learning_curves(history, title=f"R1-{name}", save_dir=args.output_dir)
        plot_roc_curves(y, p, title=f"R1-{name}", save_dir=args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
