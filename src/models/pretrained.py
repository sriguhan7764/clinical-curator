"""Transfer-learning models: FeatureExtractor backbone + FineTunedResNet head (Review 2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NUM_CLASSES = 14


class FeatureExtractor(nn.Module):
    """Frozen/partially-frozen backbone that outputs spatial feature sequences.

    Returns:
        seq    — (B, 49, C)  spatial tokens for RNN/attention input
        pooled — (B, C)      global-average-pooled feature vector
    """

    BACKBONES: dict = {
        "resnet50":        (models.resnet50,        models.ResNet50_Weights.DEFAULT,        2048),
        "densenet121":     (models.densenet121,     models.DenseNet121_Weights.DEFAULT,     1024),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
    }

    def __init__(self, backbone: str = "resnet50", freeze: bool = True):
        super().__init__()
        fn, w, self.feat_dim = self.BACKBONES[backbone]
        self.backbone_name = backbone
        base = fn(weights=w)

        if "resnet" in backbone:
            self.features = nn.Sequential(*list(base.children())[:-2])
        else:
            self.features = base.features

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        else:
            all_p = list(self.features.parameters())
            for i, p in enumerate(all_p):
                p.requires_grad = i >= int(len(all_p) * 0.75)

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        f = self.features(x)
        if "densenet" in self.backbone_name:
            f = F.relu(f)
        f = self.pool(f)
        B, C, H, W = f.shape
        seq = f.view(B, C, H * W).permute(0, 2, 1)  # B x 49 x C
        pooled = f.mean(dim=[2, 3])                  # B x C
        return seq, pooled


class FineTunedResNet(nn.Module):
    """ResNet-50 with layer3 + layer4 unfrozen and a new multi-label head.

    Output: raw logits of shape (batch, 14).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for name, p in base.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                p.requires_grad = False
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
