"""Convolutional Autoencoder and Variational Autoencoder for X-ray representation learning (Review 3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128


class Encoder(nn.Module):
    """Convolutional encoder: 1×64×64 → latent_dim."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            # 1×64×64 → 32×32×32
            nn.Conv2d(1, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            # 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            # 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            # 128×8×8 → 256×4×4
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    """Transposed-convolutional decoder: latent_dim → 1×64×64."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            # 256×4×4 → 128×8×8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            # 128×8×8 → 64×16×16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            # 64×16×16 → 32×32×32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            # 32×32×32 → 1×64×64
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z):
        x = F.relu(self.fc(z)).view(-1, 256, 4, 4)
        return self.deconv(x)


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder for unsupervised X-ray representation learning."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """Variational Autoencoder with reparameterization trick.

    Encodes input to (mu, logvar), samples z ~ N(mu, exp(logvar)),
    decodes z back to image space.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim * 2)  # outputs concatenated [mu | logvar]
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, : self.latent_dim], h[:, self.latent_dim :]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, : self.latent_dim], h[:, self.latent_dim :]
        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n: int = 16, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decode(z)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """VAE ELBO loss = MSE reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(recon, target, reduction="sum") / target.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + beta * kld, recon_loss.item(), kld.item()
