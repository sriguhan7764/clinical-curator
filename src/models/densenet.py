"""Production DenseNet-121 and EfficientNet-B2 models for CXR classification (Review 4)."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NUM_CLASSES = 14


class DenseNetCXR(nn.Module):
    """DenseNet-121 fine-tuned for multi-label chest X-ray classification.

    Fine-tunes denseblock4 + norm5 while keeping earlier layers frozen.
    Output: raw logits of shape (batch, 14).
    """

    def __init__(
        self,
        n_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        base = models.densenet121(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.features = base.features

        for name, param in self.features.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        f = F.relu(self.features(x))
        f = self.pool(f)
        return self.classifier(f)

    def get_cam_layer(self):
        """Return the last normalization layer for GradCAM targeting."""
        return self.features.norm5


class EfficientNetCXR(nn.Module):
    """EfficientNet-B2 baseline with last feature block unfrozen.

    Output: raw logits of shape (batch, 14).
    """

    def __init__(self, n_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        base = models.efficientnet_b2(weights="IMAGENET1K_V1")
        for p in base.parameters():
            p.requires_grad = False
        for p in base.features[7].parameters():
            p.requires_grad = True
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1408, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.backbone(x)))
