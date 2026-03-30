#!/usr/bin/env python3
"""Review 3 — Train ConvAutoencoder, VAE, and DCGAN on grayscale X-rays.

Usage:
    python scripts/train_r3.py --data_root /path/to/nih-data --output_dir ./outputs/r3
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from src.data.dataset import XrayDataset, get_ae_transforms
from src.models.autoencoder import ConvAutoencoder, VAE, vae_loss
from src.models.gan import Discriminator, Generator
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train AE, VAE, and GAN on NIH X-rays (R3)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="./outputs/r3")
    p.add_argument("--ae_epochs", type=int, default=30)
    p.add_argument("--gan_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--lr_ae", type=float, default=1e-3)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--subset_frac", type=float, default=0.10)
    return p.parse_args()


def get_loaders(args):
    import os as _os
    csv_path = None
    for root, _, files in _os.walk(args.data_root):
        if "Data_Entry_2017.csv" in files:
            csv_path = _os.path.join(root, "Data_Entry_2017.csv")
            break
    df = pd.read_csv(csv_path)
    for lbl in CLASSES:
        df[lbl] = df["Finding Labels"].map(lambda x: 1.0 if lbl in x else 0.0)
    df = df.sample(frac=args.subset_frac, random_state=SEED).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.1, random_state=SEED)
    tf = get_ae_transforms(args.img_size)
    mk = lambda d, sh: DataLoader(
        XrayDataset(d.reset_index(drop=True), args.data_root, tf),
        batch_size=args.batch_size, shuffle=sh, num_workers=2,
    )
    return mk(tr, True), mk(te, False)


def train_ae(model, loader, epochs, lr, device, save_path, is_vae=False):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    best_loss = float("inf")
    label = "vae" if is_vae else "ae"

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            with autocast(enabled=use_amp):
                if is_vae:
                    recon, mu, logvar, _ = model(imgs)
                    loss, _, _ = vae_loss(recon, imgs, mu, logvar)
                else:
                    recon, _ = model(imgs)
                    loss = F.mse_loss(recon, imgs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        sched.step()
        avg = ep_loss / len(loader)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Ep {ep:03d}/{epochs}  loss={avg:.5f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)

    print(f"  Best {label} loss={best_loss:.5f} -> {save_path}")


def train_gan(G, D, loader, epochs, lr_g, lr_d, latent_dim, device, out_dir):
    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        g_losses, d_losses = [], []
        for imgs, _ in loader:
            bs = imgs.size(0)
            real = imgs.to(device)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # Discriminator step
            opt_d.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z).detach()
            d_loss = (
                criterion(D(real), real_labels)
                + criterion(D(fake), fake_labels)
            ) / 2
            d_loss.backward()
            opt_d.step()

            # Generator step
            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            g_loss = criterion(D(G(z)), real_labels)
            g_loss.backward()
            opt_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if ep % 10 == 0 or ep == epochs:
            print(
                f"  Ep {ep:03d}/{epochs}  "
                f"G={sum(g_losses)/len(g_losses):.4f}  "
                f"D={sum(d_losses)/len(d_losses):.4f}"
            )

    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(out_dir, "discriminator.pth"))
    print(f"  GAN weights saved to {out_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  AMP={device.type == 'cuda'}")

    train_loader, test_loader = get_loaders(args)
    print(f"Train={len(train_loader.dataset)}  Test={len(test_loader.dataset)}")

    # ── Autoencoder ──────────────────────────────────────────
    print("\nTraining Convolutional Autoencoder...")
    ae = ConvAutoencoder(args.latent_dim).to(device)
    train_ae(ae, train_loader, args.ae_epochs, args.lr_ae, device,
             os.path.join(args.output_dir, "ae_best.pth"), is_vae=False)

    # ── VAE ──────────────────────────────────────────────────
    print("\nTraining VAE...")
    vae = VAE(args.latent_dim).to(device)
    train_ae(vae, train_loader, args.ae_epochs, args.lr_ae, device,
             os.path.join(args.output_dir, "vae_best.pth"), is_vae=True)

    # ── DCGAN ────────────────────────────────────────────────
    print("\nTraining DCGAN...")
    G = Generator(args.latent_dim).to(device)
    D = Discriminator().to(device)
    train_gan(G, D, train_loader, args.gan_epochs,
              args.lr_g, args.lr_d, args.latent_dim, device, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
