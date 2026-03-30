"""Plotting utilities for training curves, ROC curves, and per-class F1 heatmaps."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def plot_learning_curves(
    history: dict,
    title: str = "",
    save_dir: Optional[str] = None,
) -> None:
    """Plot train/val loss and val AUC curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["val_loss"], label="Val Loss", marker="s")
    ax1.set_title(f"{title} — Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["val_auc"], color="green", marker="^", label="Val AUC")
    ax2.set_title(f"{title} — AUC Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"curves_{title.replace(' ', '_')}.png"), dpi=100
        )
    plt.show()
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "",
    save_dir: Optional[str] = None,
) -> list[float]:
    """Plot per-class ROC curves and return list of per-class AUC values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    aucs = []
    for i, cls in enumerate(CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            ax.plot(fpr, tpr, lw=1.5, label=f"{cls} ({auc:.2f})")
            aucs.append(auc)
        except ValueError:
            aucs.append(0.5)

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{title} — ROC Curves | Mean AUC={np.mean(aucs):.4f}")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"roc_{title.replace(' ', '_')}.png"), dpi=100
        )
    plt.show()
    plt.close()
    return aucs


def plot_f1_heatmap(
    f1_scores: dict,
    title: str = "",
    save_dir: Optional[str] = None,
) -> None:
    """Plot per-class F1 score as a horizontal heatmap."""
    values = [f1_scores.get(c, 0.0) for c in CLASSES]
    plt.figure(figsize=(14, 3))
    sns.heatmap(
        np.array(values).reshape(1, -1),
        annot=True,
        fmt=".2f",
        xticklabels=CLASSES,
        yticklabels=["F1"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    plt.title(f"{title} — Per-Class F1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"f1_{title.replace(' ', '_')}.png"), dpi=100
        )
    plt.show()
    plt.close()
