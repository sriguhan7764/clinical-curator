"""Custom 4-block CNN for NIH ChestX-ray14 multi-label classification.

Review 1 — first convolutional baseline. Four ``ConvBlock`` units progressively
downsample the spatial resolution (224 → 112 → 56 → 28 → 14) while doubling
the channel count (3 → 32 → 64 → 128 → 256). An ``AdaptiveAvgPool2d(4, 4)``
makes the classifier head resolution-agnostic.

Architecture
------------
Input (3×224×224)
  ConvBlock(3  → 32 ) → 32 ×112×112
  ConvBlock(32 → 64 ) → 64 × 56× 56
  ConvBlock(64 → 128) → 128× 28× 28
  ConvBlock(128→ 256) → 256× 14× 14
  AdaptiveAvgPool2d(4, 4)
  FC(4096→512) → BN → ReLU → Dropout
  FC(512→128)  → ReLU
  FC(128→14)   [raw logits]

Usage
-----
    model = ChestCNN(dropout_rate=0.3)
    logits = model(x)            # x: (B, 3, 224, 224)
    probs  = torch.sigmoid(logits)
"""

import torch.nn as nn

NUM_CLASSES = 14


class ConvBlock(nn.Module):
    """Dual-convolution building block used in ChestCNN.

    Each block applies two 3×3 convolutions separated by BatchNorm and ReLU,
    then halves the spatial dimensions via MaxPool.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class ChestCNN(nn.Module):
    """Four-block custom CNN for multi-label chest X-ray classification.

    Parameters
    ----------
    n_classes : int
        Number of output classes (default 14).
    dropout_rate : float
        Dropout probability in the classifier head.
    """

    def __init__(self, n_classes: int = NUM_CLASSES, dropout_rate: float = 0.3) -> None:
        super().__init__()
        # Feature extractor: 4 ConvBlocks
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        # Resolution-agnostic spatial pooling → fixed 4×4 feature map
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, n_classes)``.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
