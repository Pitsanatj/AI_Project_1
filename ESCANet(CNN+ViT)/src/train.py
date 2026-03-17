"""
src/train.py
━━━━━━━━━━━━
Single-epoch training step (AMP-aware).
Returns a dict of metrics for the epoch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

try:
    from torch.amp import GradScaler  # PyTorch >= 2.3
except ImportError:
    from torch.cuda.amp import GradScaler  # PyTorch < 2.3


def train_one_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    criterion:      nn.Module,
    optimizer:      torch.optim.Optimizer,
    scaler:         GradScaler,
    device:         torch.device,
    use_amp:        bool,
    num_classes:    int,
    epoch:          int = 0,
    total_epochs:   int = 0,
    supcon_loss_fn  = None,   # pytorch-metric-learning SupConLoss instance or None
    supcon_alpha:   float = 0.0,  # weight of SupCon term; 0 → pure CE
) -> dict[str, float]:
    """
    Run one training epoch with a live progress bar.

    When supcon_alpha > 0 the combined loss is used:
        loss = (1 - supcon_alpha) * CE(logits, labels)
             +        supcon_alpha * SupConLoss(proj_feats, labels)

    The model must have been built with proj_dim > 0 for SupCon to work.

    Returns
    -------
    dict with keys: loss, accuracy, roc_auc, _probs, _labels
    """
    model.train()
    lb = LabelBinarizer()
    lb.fit(range(num_classes))

    running_loss = 0.0
    all_probs:  list[np.ndarray] = []
    all_labels: list[int]        = []

    device_type  = device.type  # 'cuda' or 'cpu'
    use_supcon   = supcon_alpha > 0.0 and supcon_loss_fn is not None

    epoch_str = f"Epoch {epoch+1:03d}/{total_epochs}" if total_epochs else f"Epoch {epoch+1:03d}"

    pbar = tqdm(
        loader,
        desc=f"  {epoch_str} [train]",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            if use_supcon:
                outputs, proj = model(inputs, return_proj=True)
                ce_loss  = criterion(outputs, labels)
                sc_loss  = supcon_loss_fn(proj, labels)
                loss     = (1.0 - supcon_alpha) * ce_loss + supcon_alpha * sc_loss
            else:
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        probs = F.softmax(outputs.detach().float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / ((batch_idx + 1) * loader.batch_size)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    pbar.close()

    all_probs_arr  = np.concatenate(all_probs)
    all_labels_arr = np.array(all_labels)
    epoch_loss = running_loss / len(loader.dataset)
    acc        = accuracy_score(all_labels_arr, np.argmax(all_probs_arr, axis=1))
    auc        = roc_auc_score(lb.transform(all_labels_arr), all_probs_arr, multi_class="ovr")

    return {
        "loss":     epoch_loss,
        "accuracy": acc,
        "roc_auc":  auc,
        "_probs":   all_probs_arr,
        "_labels":  all_labels_arr,
    }