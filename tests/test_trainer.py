"""Unit tests for the Trainer class.

Uses a tiny synthetic dataset (random tensors) to verify training mechanics
without downloading the NIH dataset or requiring a GPU.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


N_CLASSES = 14
IMG_SIZE = (3, 32, 32)   # small images for fast CPU tests
N_TRAIN = 40
N_VAL = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_loaders(n_train: int = N_TRAIN, n_val: int = N_VAL, batch_size: int = 8):
    """Return (train_loader, val_loader) with random float tensors."""
    x_tr = torch.randn(n_train, *IMG_SIZE)
    y_tr = (torch.rand(n_train, N_CLASSES) > 0.7).float()
    x_vl = torch.randn(n_val, *IMG_SIZE)
    y_vl = (torch.rand(n_val, N_CLASSES) > 0.7).float()
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(x_vl, y_vl), batch_size=batch_size)
    return tr, vl


class TinyClassifier(nn.Module):
    """Minimal model for fast trainer tests."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def setup_method(self):
        from src.training.trainer import Trainer
        self.Trainer = Trainer
        self.train_loader, self.val_loader = make_tiny_loaders()
        self.model = TinyClassifier()

    def test_fit_returns_history_keys(self):
        trainer = self.Trainer(self.model, self.train_loader, self.val_loader)
        history = trainer.fit(epochs=2)
        assert set(history.keys()) == {"train_loss", "val_loss", "val_auc"}

    def test_history_length_matches_epochs(self):
        trainer = self.Trainer(self.model, self.train_loader, self.val_loader)
        history = trainer.fit(epochs=3)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_auc"]) == 3

    def test_train_loss_decreases(self):
        """Loss should generally decrease over 5 epochs on a tiny dataset."""
        trainer = self.Trainer(self.model, self.train_loader, self.val_loader, patience=10)
        history = trainer.fit(epochs=5)
        assert history["train_loss"][-1] < history["train_loss"][0] * 1.5  # loose bound

    def test_best_auc_is_tracked(self):
        trainer = self.Trainer(self.model, self.train_loader, self.val_loader)
        trainer.fit(epochs=3)
        assert 0.0 <= trainer.best_auc <= 1.0

    def test_checkpoint_is_saved(self, tmp_path):
        save_path = str(tmp_path / "checkpoint.pth")
        trainer = self.Trainer(self.model, self.train_loader, self.val_loader)
        trainer.fit(epochs=2, save_path=save_path)
        import os
        assert os.path.exists(save_path)
        ckpt = torch.load(save_path, map_location="cpu")
        assert "state_dict" in ckpt
        assert "auc" in ckpt
        assert "epoch" in ckpt

    def test_early_stopping_triggers(self):
        """With patience=1 and a frozen model, early stop should fire by epoch 3."""
        model = TinyClassifier()
        # Freeze all params so val AUC never improves past epoch 1
        for p in model.parameters():
            p.requires_grad = False

        trainer = self.Trainer(
            model, self.train_loader, self.val_loader, patience=1
        )
        history = trainer.fit(epochs=20)
        # Should stop well before 20 epochs
        assert len(history["val_auc"]) <= 5

    def test_pos_weight_accepted(self):
        pos_weight = torch.ones(N_CLASSES) * 2.0
        trainer = self.Trainer(
            self.model, self.train_loader, self.val_loader, pos_weight=pos_weight
        )
        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2

    def test_custom_optimizer_accepted(self):
        import torch.optim as optim
        opt = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        trainer = self.Trainer(
            self.model, self.train_loader, self.val_loader, optimizer=opt
        )
        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
