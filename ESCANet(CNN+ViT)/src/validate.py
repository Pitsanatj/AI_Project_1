"""
src/validate.py
━━━━━━━━━━━━━━━
Validation and test evaluation loops (AMP-aware, no-grad).
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


def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    use_amp:     bool,
    num_classes: int,
    split:       str = "val",
) -> dict:
    """
    Run a full evaluation pass (val or test) with a live progress bar.

    Returns
    -------
    dict with keys: loss, accuracy, roc_auc, _probs (N,C), _labels (N,)
    """
    model.eval()
    lb = LabelBinarizer()
    lb.fit(range(num_classes))

    running_loss = 0.0
    all_probs:  list[np.ndarray] = []
    all_labels: list[int]        = []

    device_type = device.type  # 'cuda' or 'cpu'

    pbar = tqdm(
        loader,
        desc=f"  [{split:>5}]",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            avg_loss = running_loss / ((batch_idx + 1) * loader.batch_size)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    pbar.close()

    all_probs_arr  = np.concatenate(all_probs)
    all_labels_arr = np.array(all_labels)
    epoch_loss     = running_loss / len(loader.dataset)
    acc            = accuracy_score(all_labels_arr, np.argmax(all_probs_arr, axis=1))
    auc            = roc_auc_score(
        lb.transform(all_labels_arr), all_probs_arr, multi_class="ovr"
    )

    return {
        "loss":     epoch_loss,
        "accuracy": acc,
        "roc_auc":  auc,
        "_probs":   all_probs_arr,
        "_labels":  all_labels_arr,
    }