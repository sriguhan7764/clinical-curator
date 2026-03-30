"""DCGAN Generator and Discriminator for synthetic X-ray generation (Review 3)."""

import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128


class Generator(nn.Module):
    """DCGAN generator: latent noise z → synthetic grayscale X-ray (1×64×64).

    Uses DCGAN weight initialization (normal μ=0, σ=0.02).
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            # 256×4×4 → 128×8×8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128×8×8 → 64×16×16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64×16×16 → 32×32×32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32×32×32 → 1×64×64
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, z):
        x = F.relu(self.fc(z)).view(-1, 256, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    """DCGAN discriminator with SpectralNorm for training stability.

    Input:  1×64×64 image (real or generated)
    Output: scalar logit (real > 0, fake < 0)
    """

    def __init__(self):
        super().__init__()
        SN = nn.utils.spectral_norm
        self.net = nn.Sequential(
            SN(nn.Conv2d(1, 32, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            SN(nn.Linear(256 * 4 * 4, 1)),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        return self.net(x)
