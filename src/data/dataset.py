"""NIH ChestX-ray14 dataset classes and data-loading utilities."""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224)
SEED = 42


class NIHChestXrayDataset(Dataset):
    """Standard NIH ChestX-ray14 dataset for supervised classification (R1/R2)."""

    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.image_map: dict[str, str] = {}
        needed = set(df["Image Index"].values)
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f in needed:
                    self.image_map[f] = os.path.join(root, f)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_map.get(row["Image Index"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", IMG_SIZE)
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[CLASSES].values.astype(np.float32))
        return img, labels


class AdvancedXrayDataset(Dataset):
    """Enhanced dataset with patient metadata (age, gender) for R4 models."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        transform=None,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.return_meta = return_meta
        self.paths: dict[str, str] = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith((".png", ".jpg")):
                    self.paths[f] = os.path.join(root, f)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.paths.get(row["Image Index"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE[0], IMG_SIZE[0]))
        if self.transform:
            img = self.transform(img)
        lbl = torch.tensor(row[CLASSES].values.astype(np.float32))
        if self.return_meta:
            age = float(row.get("Patient Age", 50)) / 100.0
            gender = 1.0 if str(row.get("Patient Gender", "M")) == "M" else 0.0
            return img, lbl, torch.tensor([age, gender])
        return img, lbl


class XrayDataset(Dataset):
    """Grayscale dataset for autoencoder / GAN training (64×64, R3)."""

    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.paths: dict[str, str] = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith((".png", ".jpg")):
                    self.paths[f] = os.path.join(root, f)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.paths.get(row["Image Index"])
        try:
            img = Image.open(path).convert("L")  # grayscale
        except Exception:
            img = Image.new("L", (64, 64))
        if self.transform:
            img = self.transform(img)
        lbl = torch.tensor(row[CLASSES].values.astype(np.float32))
        return img, lbl


class TemporalPatientDataset(Dataset):
    """Groups X-rays by Patient ID ordered by follow-up number for RNN training (R2)."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform=None,
        max_seq: int = 5,
        min_seq: int = 2,
    ):
        self.transform = transform
        self.max_seq = max_seq
        self.sequences: list[dict] = []
        self.image_map: dict[str, str] = {}

        for root, _, files in os.walk(data_root):
            for f in files:
                if f.endswith(".png"):
                    self.image_map[f] = os.path.join(root, f)

        df = df.copy()
        if "Patient ID" not in df.columns:
            df["Patient ID"] = df["Image Index"].apply(
                lambda x: int(x.split("_")[0])
            )
        if "Follow-up #" not in df.columns:
            df["Follow-up #"] = df["Image Index"].apply(
                lambda x: int(x.split("_")[1].split(".")[0])
            )

        for _, grp in df.groupby("Patient ID"):
            grp_s = grp.sort_values("Follow-up #")
            if len(grp_s) >= min_seq:
                seq_df = grp_s.tail(max_seq)
                self.sequences.append(
                    {
                        "images": seq_df["Image Index"].tolist(),
                        "labels": seq_df.iloc[-1][CLASSES].values.astype(np.float32),
                        "seq_len": len(seq_df),
                    }
                )
        print(f"  Temporal sequences: {len(self.sequences)}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        s = self.sequences[idx]
        imgs = []
        for name in s["images"]:
            path = self.image_map.get(name)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", IMG_SIZE)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        while len(imgs) < self.max_seq:
            imgs.insert(0, torch.zeros_like(imgs[0]))
        seq_tensor = torch.stack(imgs)  # T x C x H x W
        return seq_tensor, torch.tensor(s["labels"]), torch.tensor(s["seq_len"])


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(aug_level: str = "standard") -> Tuple:
    """Return (train_transform, val_transform) for 224×224 RGB images."""
    if aug_level == "strong":
        train_tf = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE[0] + 20, IMG_SIZE[1] + 20)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAffine(degrees=0, shear=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    val_tf = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def get_ae_transforms(img_size: int = 64):
    """Return grayscale transform for autoencoder/GAN (normalized to [-1, 1])."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


# ---------------------------------------------------------------------------
# High-level data loader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str,
    subset_frac: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 2,
    aug_level: str = "standard",
    device: Optional[torch.device] = None,
):
    """Build train/val/test DataLoaders and compute pos_weight for BCEWithLogitsLoss."""
    csv_path = None
    for root, _, files in os.walk(data_root):
        if "Data_Entry_2017.csv" in files:
            csv_path = os.path.join(root, "Data_Entry_2017.csv")
            break
    if csv_path is None:
        raise FileNotFoundError("Data_Entry_2017.csv not found under data_root")

    df = pd.read_csv(csv_path)
    for lbl in CLASSES:
        df[lbl] = df["Finding Labels"].map(lambda x: 1.0 if lbl in x else 0.0)
    df = df.sample(frac=subset_frac, random_state=SEED).reset_index(drop=True)

    tr, tmp = train_test_split(df, test_size=0.2, random_state=SEED)
    vl, te = train_test_split(tmp, test_size=0.5, random_state=SEED)

    train_tf, val_tf = get_transforms(aug_level)

    def make_loader(d: pd.DataFrame, tf, shuffle: bool) -> DataLoader:
        ds = NIHChestXrayDataset(d.reset_index(drop=True), data_root, tf)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    dev = device or torch.device("cpu")
    pos_weight = torch.tensor(
        [(len(tr) - tr[c].sum()) / max(tr[c].sum(), 1) for c in CLASSES],
        dtype=torch.float32,
    ).to(dev)

    return make_loader(tr, train_tf, True), make_loader(vl, val_tf, False), make_loader(te, val_tf, False), pos_weight
