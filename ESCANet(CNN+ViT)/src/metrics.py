"""
src/metrics.py
━━━━━━━━━━━━━━
Compute, log (MLflow + TensorBoard), and plot all evaluation metrics.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelBinarizer

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


# ⚠ Label order MUST match the .npy label encoding.
# BreakHis .npy files use benign-first order (NOT alphabetical).
# Previous alphabetical order ["A","DC","F","LC","MC","PC","PT","TA"] was WRONG
# and caused all per-class metrics to be reported for the wrong class.
CLASS_LABELS_SHORT = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
CLASS_LABELS_FULL  = [
    "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
    "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma",
]


def compute_all(
    probs_arr:  np.ndarray,   # (N, C)
    labels_arr: np.ndarray,   # (N,)
    class_labels: list[str] = CLASS_LABELS_SHORT,
) -> dict:
    """Return a flat dict of all scalar metrics."""
    preds   = np.argmax(probs_arr, axis=1)
    cm      = confusion_matrix(labels_arr, preds)
    lb_     = LabelBinarizer()
    lb_.fit(labels_arr)

    # Specificity per class
    spec = {}
    for i, lbl in enumerate(class_labels):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        spec[f"specificity_{lbl}"] = float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0

    return {
        "accuracy":  float(accuracy_score(labels_arr, preds)),
        "precision": float(precision_score(labels_arr, preds, average="weighted", zero_division=0)),
        "recall":    float(recall_score(labels_arr, preds, average="weighted", zero_division=0)),
        "f1":        float(f1_score(labels_arr, preds, average="weighted", zero_division=0)),
        "jaccard":   float(jaccard_score(labels_arr, preds, average="weighted", zero_division=0)),
        "roc_auc":   float(roc_auc_score(lb_.transform(labels_arr), probs_arr, multi_class="ovr")),
        **spec,
        "_cm":       cm,
        "_lb":       lb_,
        "_preds":    preds,
        "_report":   classification_report(labels_arr, preds, target_names=class_labels, zero_division=0),
    }


def log_metrics(
    metrics:       dict,
    prefix:        str,               # e.g. "test/40X"
    step:          int | None = None,
    writer=None,                       # TensorBoard SummaryWriter
    mlflow_run=None,
):
    """Push scalar metrics to TensorBoard and/or MLflow."""
    scalar_keys = ["accuracy", "precision", "recall", "f1", "jaccard", "roc_auc"]
    scalar_keys += [k for k in metrics if k.startswith("specificity")]

    for key in scalar_keys:
        val = metrics[key]
        tag = f"{prefix}/{key}"
        if _HAS_TB and writer is not None:
            if step is not None:
                writer.add_scalar(tag, val, step)
            else:
                writer.add_scalar(tag, val)
        if _HAS_MLFLOW and mlflow_run is not None:
            mlflow.log_metric(tag.replace("/", "_"), val, step=step)


def save_text_metrics(
    metrics:     dict,
    output_path: str,
    mag:         str,
    split:       str,
    model_name:  str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Model: {model_name}  |  Mag: {mag}  |  Split: {split}\n")
        f.write("=" * 60 + "\n")
        for k in ["accuracy", "precision", "recall", "f1", "jaccard", "roc_auc"]:
            f.write(f"{k:>12}: {metrics[k]:.4f}\n")
        f.write("\nSpecificity per class:\n")
        for k, v in metrics.items():
            if k.startswith("specificity"):
                f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['_cm']}\n\n")
        f.write(f"Classification Report:\n{metrics['_report']}\n")
    print(f"  Metrics saved → {output_path}")


def save_figures(
    metrics:      dict,
    figure_dir:   str,
    mag:          str,
    model_name:   str,
    class_labels: list[str] = CLASS_LABELS_SHORT,
):
    os.makedirs(figure_dir, exist_ok=True)

    # ── Confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        metrics["_cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels,
    )
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"{model_name} {mag} – Confusion Matrix")
    cm_path = os.path.join(figure_dir, f"{model_name}_{mag}_cm.pdf")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {cm_path}")

    # ── ROC curves
    lb_     = metrics["_lb"]
    probs   = metrics.get("_probs")
    labels  = metrics.get("_labels")
    if probs is None or labels is None:
        return

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(len(class_labels)):
        fpr_d[i], tpr_d[i], _ = roc_curve(lb_.transform(labels)[:, i], probs[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])

    plt.figure(figsize=(8, 8))
    for i in range(len(class_labels)):
        plt.plot(fpr_d[i], tpr_d[i], lw=2, label=f"{class_labels[i]} (AUC={auc_d[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 0.2]); plt.ylim([0.1, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.title(f"{model_name} {mag} – ROC Curve")
    roc_path = os.path.join(figure_dir, f"{model_name}_{mag}_roc.pdf")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve      → {roc_path}")