"""
src/dataset.py
━━━━━━━━━━━━━━
PyTorch Dataset classes that consume pre-saved .npy arrays.

OPTIMISATIONS vs original
─────────────────────────
1. Pre-caching: preprocess + stain-norm is applied ONCE to the whole split
   at construction time (vectorised NumPy/PIL loop with tqdm), not per
   __getitem__ call.  This eliminates the biggest CPU bottleneck.
2. Tensors are stored as float32 (normalised) so __getitem__ only needs to
   apply augmentation — no PIL, no cv2, no Resize per call.
3. Balanced oversampling: dataset length reduced to
   num_classes × max_count (1× pass, not 3×).  More passes come from
   augmentation diversity, not redundant index repetition.
4. WeightedRandomSampler support: build_loaders accepts
   use_sampler=True to replace manual oversampling with PyTorch's sampler.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────
def _preprocess_split(
    images: np.ndarray,
    preprocess,
    desc: str = "Caching",
) -> torch.Tensor:
    """
    Apply preprocess ONLY to every image ONCE.
    Returns uint8 tensor (N, C, H, W) — NOT normalised.

    ★ FIX: Do NOT apply normalise here. Augmentations (AugMix, flips, affine)
    must run on uint8 tensors. Normalise is applied in __getitem__ AFTER augment.
    """
    result = []
    for i in tqdm(range(len(images)), desc=desc, leave=False, ncols=80):
        img = Image.fromarray(images[i])
        img = preprocess(img)   # Resize + stain-norm → uint8 tensor
        result.append(img)
    return torch.stack(result)   # (N, C, H, W) uint8


# ── Balanced oversampled training dataset ─────────────────────────────────────
class NpyTrainDataset(Dataset):
    """
    Stores pre-cached uint8 tensors.
    __getitem__ applies augmentation on uint8, THEN normalises to float32.

    ★ FIXES:
    1. Augmentation runs on uint8 (not float32 normalised) — matching notebook.
    2. __len__ returns 3× oversampling — matching paper Algorithm 1.
    """

    def __init__(
        self,
        images: np.ndarray,        # (N, H, W, 3) uint8  — raw from .npy
        labels: np.ndarray,        # (N,) int64
        preprocess=None,
        augment=None,
        balance_augment=None,
        normalise=None,
    ):
        self.augment         = augment
        self.balance_augment = balance_augment
        self.normalise       = normalise        # ★ stored, applied per-item in __getitem__

        # ── Pre-cache: preprocess only, stay as uint8 ─────────────────────
        print("  Pre-caching training images (uint8) …", flush=True)
        self.tensors = _preprocess_split(
            images, preprocess, desc="  train"
        )                                      # ★ uint8 (N, C, H, W)
        self.labels   = torch.from_numpy(labels.astype(np.int64))

        counts = Counter(labels.tolist())
        self.max_count   = max(counts.values())
        self.num_classes = len(counts)

        # Group indices by class
        self.by_class: list[list[int]] = [[] for _ in range(self.num_classes)]
        for idx, lbl in enumerate(labels):
            self.by_class[int(lbl)].append(idx)

    def __len__(self) -> int:
        # ★ FIX: Restore paper's 3× oversampling (Algorithm 1, line 4)
        return self.num_classes * self.max_count * 3

    def __getitem__(self, index: int):
        cls_idx  = index % self.num_classes
        img_list = self.by_class[cls_idx]
        real_idx = img_list[(index // self.num_classes) % len(img_list)]

        # ★ FIX: tensor is uint8 — augmentations work correctly on uint8
        image = self.tensors[real_idx].clone()

        # Minority-class extra augmentation (AugMix on uint8 — correct)
        if len(img_list) < self.max_count and self.balance_augment is not None:
            image = self.balance_augment(image)

        if self.augment is not None:
            image = self.augment(image)

        # ★ FIX: Normalise to float32 LAST (matching notebook's pipeline)
        if self.normalise is not None:
            image = self.normalise(image)

        return image, cls_idx


# ── Val / Test dataset ────────────────────────────────────────────────────────
class NpyValTestDataset(Dataset):
    """Pre-caches all images at construction — zero per-sample I/O cost."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        vtpreprocess=None,
        desc: str = "val/test",
    ):
        print(f"  Pre-caching {desc} images …", flush=True)
        result = []
        for i in tqdm(range(len(images)), desc=f"  {desc}", leave=False, ncols=80):
            img = Image.fromarray(images[i])
            if vtpreprocess is not None:
                img = vtpreprocess(img)
            result.append(img)
        self.tensors = torch.stack(result)
        self.labels  = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.tensors[index], int(self.labels[index])


# ── DataLoader factory ────────────────────────────────────────────────────────
def load_npy_split(data_dir: str, mag: str, split: str):
    mag_dir = os.path.join(data_dir, mag)
    images  = np.load(os.path.join(mag_dir, f"{split}_images.npy"), mmap_mode="r")
    labels  = np.load(os.path.join(mag_dir, f"{split}_labels.npy"))
    return images, labels


def build_loaders(
    data_dir:         str,
    mag:              str,
    preprocess=None,
    augment=None,
    balance_augment=None,
    normalise=None,
    vtpreprocess=None,
    batch_size:       int  = 32,
    num_workers:      int  = 4,
    prefetch_factor:  int  = 2,
    use_sampler:      bool = False,   # WeightedRandomSampler instead of manual oversampling
    pin_memory:       bool = False,   # False on NFS/HPC systems to avoid socket errors
) -> dict[str, DataLoader]:
    """Returns {'train': ..., 'val': ..., 'test': ...} DataLoaders."""

    tr_imgs, tr_lbls = load_npy_split(data_dir, mag, "train")
    va_imgs, va_lbls = load_npy_split(data_dir, mag, "val")
    te_imgs, te_lbls = load_npy_split(data_dir, mag, "test")

    train_ds = NpyTrainDataset(
        tr_imgs, tr_lbls,
        preprocess=preprocess,
        augment=augment,
        balance_augment=balance_augment,
        normalise=normalise,
    )
    val_ds  = NpyValTestDataset(va_imgs, va_lbls, vtpreprocess=vtpreprocess, desc="val")
    test_ds = NpyValTestDataset(te_imgs, te_lbls, vtpreprocess=vtpreprocess, desc="test")

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    # Optional: use WeightedRandomSampler for more statistically correct balancing
    if use_sampler:
        counts   = Counter(tr_lbls.tolist())
        weights  = [1.0 / counts[int(l)] for l in tr_lbls]
        sampler  = WeightedRandomSampler(weights, num_samples=len(tr_lbls), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kw)

    return {
        "train": train_loader,
        "val":   DataLoader(val_ds,   shuffle=False, **kw),
        "test":  DataLoader(test_ds,  shuffle=False, **kw),
    }