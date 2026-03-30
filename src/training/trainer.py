"""Generic supervised training loop for multi-label chest X-ray classification.

Supports any ``nn.Module`` that accepts ``(B, C, H, W)`` image tensors and
returns raw logits of shape ``(B, n_classes)``.

Features
--------
- ``BCEWithLogitsLoss`` with optional class-frequency-based ``pos_weight``
- Cosine annealing learning-rate schedule
- Per-epoch macro-AUC validation
- Early stopping with configurable patience
- Best-checkpoint saving (epoch, state_dict, AUC, history)

Usage
-----
    trainer = Trainer(model, train_loader, val_loader, pos_weight=pos_weight)
    history = trainer.fit(epochs=10, save_path="outputs/checkpoint.pth")
    print(f"Best val AUC: {trainer.best_auc:.4f}")
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

_DEFAULT_LR = 1e-4
_DEFAULT_WD = 1e-5
_DEFAULT_EPOCHS = 10


class Trainer:
    """Training loop with early stopping, cosine LR, and checkpoint saving.

    Parameters
    ----------
    model : nn.Module
        The model to train. Must return raw logits of shape ``(B, n_classes)``.
    train_loader : DataLoader
        Training set loader. Batches must be ``(images, labels)`` tuples.
    val_loader : DataLoader
        Validation set loader. Used for AUC computation and early stopping.
    optimizer : torch.optim.Optimizer, optional
        Defaults to ``AdamW(lr=1e-4, weight_decay=1e-5)``.
    pos_weight : torch.Tensor, optional
        Per-class positive weights for ``BCEWithLogitsLoss``. Shape ``(n_classes,)``.
        Typically computed as ``(n_neg / n_pos)`` per class to handle class imbalance.
    patience : int
        Number of epochs with no AUC improvement before early stopping. Default 3.
    device : torch.device, optional
        Inference device. Auto-detected (CUDA → CPU) if not provided.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        pos_weight: Optional[torch.Tensor] = None,
        patience: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), lr=_DEFAULT_LR, weight_decay=_DEFAULT_WD
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=_DEFAULT_EPOCHS
        )
        self.patience = patience
        self.best_auc: float = 0.0
        self.no_improve: int = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, train: bool = True
    ) -> tuple[float, float]:
        """Run one full pass over *loader*.

        Parameters
        ----------
        loader : DataLoader
        train : bool
            If ``True``, computes gradients and updates parameters.

        Returns
        -------
        (avg_loss, macro_auc) : tuple[float, float]
        """
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        labels_all: list[np.ndarray] = []
        probs_all: list[np.ndarray] = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, lbls)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                labels_all.append(lbls.cpu().numpy())
                probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())

        y = np.concatenate(labels_all)
        p = np.concatenate(probs_all)
        try:
            auc = float(roc_auc_score(y, p, average="macro"))
        except ValueError:
            auc = 0.5  # undefined when a class has only one label in the batch
        return total_loss / len(loader.dataset), auc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int = _DEFAULT_EPOCHS,
        save_path: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train.
        save_path : str, optional
            Where to write the best checkpoint as a ``.pth`` file.
            Parent directories are created automatically.

        Returns
        -------
        dict
            ``{ "train_loss": [...], "val_loss": [...], "val_auc": [...] }``
        """
        print(
            f"Training {type(self.model).__name__} | "
            f"device={self.device} | epochs={epochs} | patience={self.patience}"
        )
        t0 = time.time()

        for ep in range(1, epochs + 1):
            tr_loss, _ = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_auc = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["val_auc"].append(vl_auc)

            print(
                f"  ep {ep:02d}/{epochs}  "
                f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  AUC={vl_auc:.4f}"
            )

            if vl_auc > self.best_auc:
                self.best_auc = vl_auc
                self.no_improve = 0
                if save_path:
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    torch.save(
                        {
                            "epoch": ep,
                            "state_dict": self.model.state_dict(),
                            "auc": vl_auc,
                            "history": self.history,
                        },
                        save_path,
                    )
                    print(f"    [checkpoint] saved → {save_path}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"  [early stop] no improvement for {self.patience} epochs")
                    break

        elapsed = (time.time() - t0) / 60
        print(f"Finished in {elapsed:.1f} min | best AUC = {self.best_auc:.4f}")
        return self.history
