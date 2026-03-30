"""Evaluation metrics for multi-label chest X-ray classification."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
):
    """Run model on loader and return (y_true, y_prob) numpy arrays.

    Returns:
        y_true: (N, 14) ground-truth binary labels
        y_prob: (N, 14) sigmoid probabilities
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch[0], batch[1]
            p = torch.sigmoid(model(imgs.to(dev)))
            all_y.append(lbls.numpy())
            all_p.append(p.cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Return {class_name: AUC} for each of the 14 NIH classes."""
    result = {}
    for i, cls in enumerate(CLASSES):
        try:
            result[cls] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            result[cls] = 0.5
    return result


def per_class_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return {class_name: F1} using a fixed probability threshold."""
    y_pred = (y_prob > threshold).astype(int)
    return {
        cls: float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        for i, cls in enumerate(CLASSES)
    }


def mean_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged AUC across all 14 classes."""
    try:
        return float(roc_auc_score(y_true, y_prob, average="macro"))
    except ValueError:
        return 0.5
