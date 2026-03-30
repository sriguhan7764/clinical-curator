"""Deep MLP baseline for NIH ChestX-ray14 multi-label classification.

Review 1 baseline. Demonstrates that even a flattened-pixel MLP can learn weak
correlations in chest X-ray data, providing a lower-bound reference AUC against
which convolutional architectures are compared.

Architecture
------------
Flatten → Linear(1024) → BN → ReLU → Dropout
        → Linear(512)  → BN → ReLU → Dropout
        → Linear(256)  → BN → ReLU → Dropout/2
        → Linear(14)   [raw logits — apply sigmoid for probabilities]

Usage
-----
    model = ChestMLP(dropout_rate=0.4)
    logits = model(x)                        # x: (B, 3, 224, 224)
    probs  = torch.sigmoid(logits)           # (B, 14) in [0, 1]
"""

import torch.nn as nn

NUM_CLASSES = 14
IMG_SIZE = (224, 224)
_DEFAULT_INPUT_SIZE = IMG_SIZE[0] * IMG_SIZE[1] * 3  # 150 528


class ChestMLP(nn.Module):
    """Multi-layer perceptron for 14-class multi-label chest X-ray classification.

    Parameters
    ----------
    input_size : int
        Flattened input dimension. Default is ``224 * 224 * 3 = 150 528``.
    n_classes : int
        Number of output classes. Default is 14 (NIH ChestX-ray14).
    dropout_rate : float
        Dropout probability applied after each hidden layer.
        The final hidden layer uses ``dropout_rate / 2``.
    """

    def __init__(
        self,
        input_size: int = _DEFAULT_INPUT_SIZE,
        n_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            # --- hidden block 1 ---
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # --- hidden block 2 ---
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # --- hidden block 3 ---
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            # --- output ---
            nn.Linear(256, n_classes),
        )

    def forward(self, x):  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, C, H, W)``. The ``Flatten`` layer
            inside the network handles the reshape automatically.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, n_classes)``.
            Apply ``torch.sigmoid`` for per-class probabilities.
        """
        return self.net(x)
