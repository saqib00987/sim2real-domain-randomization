"""Datasets, transforms and the stratified train/val split."""
from __future__ import annotations

import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REAL_IMAGE_SUFFIXES = ("*.jpg", "*.JPG", "*.jpeg")


def build_transforms(image_size: int):
    train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, test


class SyntheticDataset(Dataset):
    """Rendered training images for one diversity level. Label = 0..4."""

    def __init__(self, cfg, level: int, transform=None):
        self.transform = transform
        self.samples: list[tuple] = []
        level_dir = cfg.synthetic / cfg.level_name(level)
        for class_idx, obj_name in enumerate(cfg.objects):
            obj_dir = level_dir / obj_name
            if not obj_dir.exists():
                print(f"  WARNING - missing: {obj_dir}")
                continue
            for img_path in sorted(obj_dir.glob("*.png")):
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class RealYCBDataset(Dataset):
    """Real YCB photographs. Never seen during training.

    Only photographic suffixes are collected. YCB ships per-object masks/ and
    poses/ subdirectories alongside the RGB images; globbing for *.png here
    would silently pull mask images into the test set.
    """

    def __init__(self, cfg, transform=None):
        self.transform = transform
        self.samples: list[tuple] = []
        for class_idx, obj_name in enumerate(cfg.objects):
            obj_dir = cfg.real_test / obj_name
            if not obj_dir.exists():
                print(f"  WARNING - missing: {obj_dir}")
                continue
            imgs: set = set()
            for pattern in REAL_IMAGE_SUFFIXES:
                imgs.update(obj_dir.rglob(pattern))
            # Exclude anything living under a masks/ or poses/ subdirectory.
            imgs = {p for p in imgs
                    if not {"masks", "poses"} & set(p.relative_to(obj_dir).parts)}
            for path in sorted(imgs):
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def stratified_split(dataset: SyntheticDataset, val_fraction: float, seed: int):
    """Split holding val_fraction out of *each class* independently."""
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in sorted(class_indices):
        idxs = class_indices[label].copy()
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_fraction)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")