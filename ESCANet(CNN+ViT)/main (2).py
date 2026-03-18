"""
main.py  (optimised)
━━━━━━━━━━━━━━━━━━━
ECSAnet – BreakHis training pipeline
Reads pre-saved .npy data (from prepare_data.py) and runs
  preprocess → augment → train → validate → test
for each requested magnification, with MLflow experiment tracking.

OPTIMISATIONS vs original
─────────────────────────
1. torch.compile(model)      → ~15–30% faster forward/backward (PyTorch 2+)
2. cudnn.benchmark = True    → picks fastest conv kernels per input size
3. Checkpoint every N epochs → eliminates per-epoch disk I/O overhead
4. GradScaler device fix     → already present; kept clean
5. AMP autocast context      → unchanged (already optimal)
6. Pre-cached dataset        → biggest win: stain-norm done once at startup
   (see src/dataset.py)

Example
-------
python main.py --data_dir ./npy_data
python main.py \\
    --data_dir      ./npy_data \\
    --magnification 40X 200X \\
    --model         ecsamet \\
    --optimizer     sgd \\
    --augment       standard \\
    --preprocess    reinhard \\
    --target_image  ./npy_data/target_image.png \\
    --epochs        50 \\
    --batch_size    32 \\
    --lr            0.001 \\
    --ckpt_interval 5 \\
    --compile \\
    --output_dir    ./outputs \\
    --experiment    BreakHis-ECSAnet


python main.py --data_dir ./npy_data --magnification 40x --model lictnet --optimizer adam

Supervised Contrastive Learning (experiments E2 / E3 / E4)
───────────────────────────────────────────────────────────
Requires: pip install pytorch-metric-learning

  loss = (1-α)×CrossEntropy + α×SupConLoss(τ=0.07)

The projection head FC(1280→256→128) is trained jointly with the backbone
and discarded at test time (only logits are used for evaluation).

--fine_tune_from <path>  load pre-trained E0 weights before training
--freeze_blocks  N       freeze first N EfficientNetV2-S feature blocks (default 0)
--supcon_alpha   α       SupCon loss weight: 0.3 → E2, 0.5 → E3 (default 0.0)
--supcon_temp    τ       SupCon temperature (default 0.07)
--proj_dim       D       projection head output dim (default 128)

Example — E2 (SupCon α=0.3, fine-tune from E0 checkpoint):
python main.py \\
    --data_dir       ./npy_data \\
    --magnification  40X \\
    --model          ecsamet \\
    --fine_tune_from ./outputs/models/40X/best_model.pth \\
    --freeze_blocks  5 \\
    --supcon_alpha   0.3 \\
    --lr             0.0001 \\
    --experiment     ECSAnet-E2-SupCon

Macenko stain normalisation (experiment E1 / E4)
─────────────────────────────────────────────────
Pure NumPy/OpenCV — no extra dependencies.

Example — E4 (Macenko + best SupCon):
python main.py \\
    --data_dir       ./npy_data \\
    --magnification  40X \\
    --model          ecsamet \\
    --preprocess     macenko \\
    --target_image   ./npy_data/target_image.png \\
    --fine_tune_from ./outputs/models/40X/best_model.pth \\
    --supcon_alpha   0.3 \\
    --lr             0.0001 \\
    --experiment     ECSAnet-E4-Combined
"""

import argparse
import csv
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for HPC/server)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import mlflow
import mlflow.pytorch

from src.augment    import build_augment, build_balance_augment
from src.dataset    import build_loaders
from src.metrics    import (
    CLASS_LABELS_SHORT, compute_all, log_metrics, save_figures, save_text_metrics,
)
from src.model      import build_model
from src.preprocess import (
    StainNormalizationTransform, MacenkoStainNormalizationTransform,
    OtsuSegmentationTransform,
    build_final_normalise, build_preprocess, build_vtpreprocess,
    build_paper_preprocess, build_paper_vtpreprocess,
)
from src.train    import train_one_epoch
from src.validate import evaluate
from src.gradcam  import (
    GradCAM, get_default_target_layer,
    generate_gradcam_batch, save_gradcam_grid,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ECSAnet BreakHis Training Pipeline (optimised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--target_image", default=None)
    p.add_argument("--magnification", nargs="+",
                   choices=["40X", "100X", "200X", "400X", "all"], default=["all"])

    # Model
    p.add_argument("--model", default="ecsamet")
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--weights_dir", default=None,
                   help="Path to local pretrained weights dir (offline mode). "
                        "Populate it first: python src/download_pretrained_weights.py")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile() to the model (PyTorch 2+, ~15-30%% faster)")

    # Preprocessing
    p.add_argument("--preprocess", choices=["reinhard", "macenko", "otsu", "none"], default="reinhard")
    p.add_argument("--img_size", type=int, default=384)

    # Augmentation
    p.add_argument("--augment",
                   choices=["standard", "augmix", "combined", "none"], default="standard")

    # Optimiser
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    # Scheduler
    p.add_argument("--scheduler_factor",   type=float, default=0.1)
    p.add_argument("--scheduler_patience", type=int,   default=3)

    # Training
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--patience",     type=int, default=25)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--no_amp", action="store_true")

    # Hybrid model args (only used when --model hybrid_ecsanet)
    p.add_argument("--cnn_backbone", choices=["s", "m"], default="s",
                   help="EfficientNetV2 variant: 's' (21M) or 'm' (54M) [hybrid only]")
    p.add_argument("--vit_branch", choices=["none", "deit_s", "vit_b16"], default="vit_b16",
                   help="ViT branch: 'none'=pure CNN  'deit_s'=384-dim  'vit_b16'=768-dim [hybrid only]")
    p.add_argument("--use_cbam", action="store_true",
                   help="CBAM(1280) on CNN branch before global avg-pool [hybrid only]")
    p.add_argument("--fusion_dim", type=int, default=512,
                   help="Hidden dim of the fusion head [hybrid only, default 512]")

    # GRU model args (only used when --model efficientnet_gru)
    p.add_argument("--gru_hidden", type=int, default=512,
                   help="GRU hidden state dimension (default 512) [efficientnet_gru only]")
    p.add_argument("--gru_layers", type=int, default=2,
                   help="Number of stacked GRU layers (default 2) [efficientnet_gru only]")
    p.add_argument("--gru_dropout", type=float, default=0.5,
                   help="Dropout rate in GRU + classifier (paper default 0.5) [efficientnet_gru only]")
    p.add_argument("--bidirectional", action="store_true",
                   help="Bidirectional GRU (doubles hidden dim) [efficientnet_gru only]")

    # Supervised Contrastive Learning (E2 / E3 / E4)
    p.add_argument("--supcon_alpha", type=float, default=0.0,
                   help="SupCon loss weight α: 0=CE only, 0.3=E2, 0.5=E3. "
                        "Requires: pip install pytorch-metric-learning")
    p.add_argument("--supcon_temp", type=float, default=0.07,
                   help="SupCon temperature τ (default 0.07)")
    p.add_argument("--proj_dim", type=int, default=128,
                   help="Projection head output dim for SupCon (default 128)")
    p.add_argument("--fine_tune_from", default=None,
                   help="Path to pre-trained weights to load before training "
                        "(e.g. E0 best_model.pth for fine-tuning in E2/E3/E4)")
    p.add_argument("--freeze_blocks", type=int, default=0,
                   help="Freeze the first N EfficientNetV2-S feature blocks "
                        "during fine-tuning (default 0 = train all)")

    # Checkpointing
    p.add_argument("--ckpt_interval", type=int, default=5,
                   help="Save checkpoint every N epochs (0 = only best model)")
    p.add_argument("--resume_from", default=None,
                   help="Path to a checkpoint (.pth) to resume training from. "
                        "Restores epoch, model/optimizer/scheduler states, "
                        "best_val_loss, and patience counter.")
    p.add_argument("--auto_resume", action="store_true",
                   help="Auto-resume from the latest checkpoint found in "
                        "outputs/checkpoints/<run_tag>/<mag>/. "
                        "Ignored if --resume_from is also given.")

    # Output
    p.add_argument("--output_dir",   default="./outputs")
    p.add_argument("--experiment",   default="ECSAnet-BreakHis")
    p.add_argument("--run_name",     default=None)

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(name, model, lr, momentum, weight_decay):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")
    name = name.lower()
    if name == "sgd":
        return optim.SGD(trainable, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")


def resolve_target_image(args) -> str | None:
    if args.target_image and os.path.isfile(args.target_image):
        return args.target_image
    candidate = os.path.join(args.data_dir, "target_image.png")
    return candidate if os.path.isfile(candidate) else None


def resolve_magnifications(mag_arg: list[str]) -> list[str]:
    all_mags = ["40X", "100X", "200X", "400X"]
    return all_mags if "all" in mag_arg else [m for m in all_mags if m in mag_arg]



def find_latest_checkpoint(ckpt_dir: str) -> "str | None":
    """Return the path to the highest-epoch checkpoint in *ckpt_dir*, or None."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None
    ckpts = sorted(ckpt_path.glob("ckpt_epoch_*.pth"))
    return str(ckpts[-1]) if ckpts else None


# ── Per-magnification training pipeline ───────────────────────────────────────
def train_magnification(mag, loaders, model_name, args, device, use_amp, writer, output_dir):
    print(f"\n{'='*62}")
    print(f"  {model_name.upper()} | {mag}")
    print(f"{'='*62}")

    use_supcon   = args.supcon_alpha > 0.0 and model_name.lower() == "ecsamet"
    _proj_dim    = args.proj_dim if use_supcon else 0

    model = build_model(
        model_name, args.num_classes, args.weights_dir,
        proj_dim=_proj_dim,
        cnn_backbone=args.cnn_backbone,
        vit_branch=args.vit_branch,
        use_cbam=args.use_cbam,
        fusion_dim=args.fusion_dim,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    # ── Load pre-trained weights for fine-tuning (E2/E3/E4) ──────────────────
    if args.fine_tune_from:
        if os.path.isfile(args.fine_tune_from):
            state = torch.load(args.fine_tune_from, map_location=device)
            # Accept both raw state-dicts and checkpoint dicts
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  ✓ Loaded weights from {args.fine_tune_from}")
            if missing:
                print(f"    Missing keys (new layers, expected): {missing}")
            if unexpected:
                print(f"    Unexpected keys (ignored): {unexpected}")
        else:
            print(f"  ⚠  --fine_tune_from path not found: {args.fine_tune_from}")

    # ── Freeze early backbone blocks (optional, for fine-tuning) ─────────────
    if args.freeze_blocks > 0 and model_name.lower() == "ecsamet":
        raw = getattr(model, "_orig_mod", model)
        for i in range(min(args.freeze_blocks, len(raw.features))):
            for p in raw.features[i].parameters():
                p.requires_grad = False
        frozen_params = sum(
            p.numel() for p in raw.features[:args.freeze_blocks].parameters()
        )
        print(f"  ✓ Frozen first {args.freeze_blocks} feature blocks "
              f"({frozen_params/1e6:.2f}M params)")

    # ── Supervised Contrastive loss ───────────────────────────────────────────
    supcon_loss_fn = None
    if use_supcon:
        try:
            from pytorch_metric_learning.losses import SupConLoss
            supcon_loss_fn = SupConLoss(temperature=args.supcon_temp)
            print(f"  ✓ SupConLoss active  α={args.supcon_alpha}  τ={args.supcon_temp}  "
                  f"proj_dim={_proj_dim}")
        except ImportError:
            raise ImportError(
                "pytorch-metric-learning is required for SupCon.\n"
                "Install with:  pip install pytorch-metric-learning"
            )

    # ── torch.compile — big throughput win on PyTorch 2+ ─────────────────────
    if args.compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile …", flush=True)
        try:
            model = torch.compile(model)
            print("  ✓ torch.compile applied")
        except Exception as e:
            print(f"  ⚠  torch.compile failed ({e}), continuing without.")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.momentum, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=args.scheduler_factor,
                                  patience=args.scheduler_patience)
    _gs_device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler     = GradScaler(_gs_device, enabled=use_amp)

    best_val_loss    = float("inf")
    best_state       = None
    patience_counter = 0
    start_epoch      = 0
    model_save_dir   = os.path.join(output_dir, "models",      mag)
    ckpt_save_dir    = os.path.join(output_dir, "checkpoints", mag)
    os.makedirs(model_save_dir, exist_ok=True)
    if args.ckpt_interval > 0:
        os.makedirs(ckpt_save_dir, exist_ok=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    # Priority: explicit --resume_from > --auto_resume > fresh start
    effective_resume = args.resume_from
    if not effective_resume and args.auto_resume:
        effective_resume = find_latest_checkpoint(ckpt_save_dir)
        if effective_resume:
            print(f"  Auto-resume: found {Path(effective_resume).name}")
        else:
            print(f"  Auto-resume: no checkpoint in {ckpt_save_dir} — starting fresh")

    if effective_resume:
        if os.path.isfile(effective_resume):
            ckpt = torch.load(effective_resume, map_location=device)
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss",    ckpt.get("val_loss", float("inf")))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"  ✓ Resumed from {effective_resume}  "
                  f"(epoch {ckpt['epoch']+1}, best_val_loss={best_val_loss:.4f}, "
                  f"patience={patience_counter})")
        else:
            print(f"  ⚠  checkpoint not found: {effective_resume}  (starting fresh)")

    t0 = time.time()

    # ── CSV logger ────────────────────────────────────────────────────────────
    csv_dir  = os.path.join(output_dir, "metrics")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{model_name}_{mag}_history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch",
        "train_loss", "train_acc", "train_auc",
        "val_loss",   "val_acc",   "val_auc",
        "lr",
    ])

    # ── History buffers (for plotting) ────────────────────────────────────────
    history = {
        "epoch":      [],
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss":   [], "val_acc":   [], "val_auc":   [],
        "lr":         [],
    }

    for epoch in range(start_epoch, args.epochs):
        tr = train_one_epoch(
            model, loaders["train"], criterion, optimizer,
            scaler, device, use_amp, args.num_classes,
            epoch=epoch, total_epochs=args.epochs,
            supcon_loss_fn=supcon_loss_fn,
            supcon_alpha=args.supcon_alpha,
        )
        va = evaluate(
            model, loaders["val"], criterion, device, use_amp, args.num_classes,
        )

        for tag, val in [
            (f"train/{mag}/loss",     tr["loss"]),
            (f"train/{mag}/accuracy", tr["accuracy"]),
            (f"train/{mag}/roc_auc",  tr["roc_auc"]),
            (f"val/{mag}/loss",       va["loss"]),
            (f"val/{mag}/accuracy",   va["accuracy"]),
            (f"val/{mag}/roc_auc",    va["roc_auc"]),
        ]:
            writer.add_scalar(tag, val, epoch)
            mlflow.log_metric(tag.replace("/", "_"), val, step=epoch)

        for pg in optimizer.param_groups:
            writer.add_scalar(f"lr/{mag}", pg["lr"], epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:03d}  "
            f"train_loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} auc={tr['roc_auc']:.4f}  |  "
            f"val_loss={va['loss']:.4f} acc={va['accuracy']:.4f} auc={va['roc_auc']:.4f}"
        )

        # ── CSV row ──────────────────────────────────────────────────────────
        csv_writer.writerow([
            epoch + 1,
            f"{tr['loss']:.6f}",  f"{tr['accuracy']:.6f}", f"{tr['roc_auc']:.6f}",
            f"{va['loss']:.6f}",  f"{va['accuracy']:.6f}", f"{va['roc_auc']:.6f}",
            f"{current_lr:.8f}",
        ])
        csv_file.flush()

        # ── History ──────────────────────────────────────────────────────────
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["train_auc"].append(tr["roc_auc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["accuracy"])
        history["val_auc"].append(va["roc_auc"])
        history["lr"].append(current_lr)

        # ── Best model ───────────────────────────────────────────────────────
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            # Unwrap compiled model before saving
            raw_model = getattr(model, "_orig_mod", model)
            best_state = {k: v.cpu() for k, v in raw_model.state_dict().items()}
            torch.save(best_state, os.path.join(model_save_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > args.patience:
            print(f"  ⚡ Early stopping at epoch {epoch+1}")
            break

        # ── Checkpoint (every N epochs instead of every epoch) ───────────────
        if args.ckpt_interval > 0 and (epoch + 1) % args.ckpt_interval == 0:
            raw_model = getattr(model, "_orig_mod", model)
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":             va["loss"],
                "best_val_loss":        best_val_loss,
                "patience_counter":     patience_counter,
            }, os.path.join(ckpt_save_dir, f"ckpt_epoch_{epoch:03d}.pth"))

        scheduler.step(va["loss"])

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed/60:.1f} min")

    # ── Close CSV ─────────────────────────────────────────────────────────────
    csv_file.close()
    print(f"  Training history CSV → {csv_path}")

    # ── Plot training curves (last 20 epochs) ─────────────────────────────────
    _plot_training_curves(history, model_name, mag, output_dir)

    # ── Test evaluation ───────────────────────────────────────────────────────
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(
        torch.load(os.path.join(model_save_dir, "best_model.pth"), map_location=device)
    )
    te = evaluate(raw_model, loaders["test"], criterion, device, use_amp, args.num_classes)

    te_metrics = compute_all(te["_probs"], te["_labels"], CLASS_LABELS_SHORT)
    te_metrics["_probs"]  = te["_probs"]
    te_metrics["_labels"] = te["_labels"]

    log_metrics(te_metrics, prefix=f"test/{mag}", writer=writer, mlflow_run=True)

    figure_dir  = os.path.join(output_dir, "figures")
    metrics_dir = os.path.join(output_dir, "metrics")
    save_figures(te_metrics, figure_dir, mag, model_name)
    save_text_metrics(
        te_metrics,
        os.path.join(metrics_dir, f"{model_name}_{mag}_test_metrics.txt"),
        mag, "test", model_name,
    )
    mlflow.log_artifacts(figure_dir,  artifact_path="figures")
    mlflow.log_artifacts(metrics_dir, artifact_path="metrics")
    mlflow.pytorch.log_model(raw_model, artifact_path=f"model_{mag}")

    # ── Full metrics table ────────────────────────────────────────────────────
    _print_metrics_table(te_metrics, mag, model_name)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    try:
        print(f"\n  Generating Grad-CAM visualisations …")
        images_np, true_lbls, pred_lbls, cams = generate_gradcam_batch(
            model        = raw_model,
            model_name   = model_name,
            loader       = loaders["test"],
            device       = device,
            n_images     = 16,
            class_labels = CLASS_LABELS_SHORT,
        )
        gradcam_path = os.path.join(figure_dir, f"{model_name}_{mag}_gradcam.pdf")
        save_gradcam_grid(
            images_np    = images_np,
            true_labels  = true_lbls,
            pred_labels  = pred_lbls,
            cams         = cams,
            class_labels = CLASS_LABELS_SHORT,
            save_path    = gradcam_path,
            n_cols       = 4,
        )
        mlflow.log_artifact(gradcam_path, artifact_path="figures")
    except Exception as e:
        print(f"  ⚠  Grad-CAM skipped: {e}")

    return raw_model


# ── Helper: plot training curves ──────────────────────────────────────────────
def _plot_training_curves(history: dict, model_name: str, mag: str, output_dir: str):
    """
    Plot Accuracy (train vs val) และ Loss (train vs val) per epoch.
    บันทึก 2 ไฟล์:
      - _curves_half.pdf  → ครึ่งแรก  (epoch 1 … n//2)
      - _curves_full.pdf  → ทั้งหมด   (epoch 1 … n)
    บันทึกที่ output_dir/figures/
    """
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    epochs = history["epoch"]
    n      = len(epochs)

    # กำหนด 2 slice: ครึ่งแรก และทั้งหมด
    half   = max(1, n // 2)
    slices = [
        (slice(0, half), "half", f"epoch 1–{epochs[half-1]}"),
        (slice(0, n),    "full", f"epoch 1–{epochs[-1]}"),
    ]

    for sl, suffix, label in slices:
        ep     = epochs[sl]
        t_acc  = history["train_acc"][sl]
        v_acc  = history["val_acc"][sl]
        t_loss = history["train_loss"][sl]
        v_loss = history["val_loss"][sl]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"{model_name.upper()} | {mag}  ({label})",
            fontsize=13, fontweight="bold",
        )

        # ── Accuracy ──────────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(ep, t_acc, "o-",  color="#2196F3", label="Train Acc", linewidth=2, markersize=4)
        ax.plot(ep, v_acc, "s--", color="#FF5722", label="Val Acc",   linewidth=2, markersize=4)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title("Accuracy vs Epoch", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([max(0, min(t_acc + v_acc) - 0.05), 1.02])
        best_v_idx = int(np.argmax(v_acc))
        ax.annotate(
            f"best val\n{v_acc[best_v_idx]:.4f}",
            xy=(ep[best_v_idx], v_acc[best_v_idx]),
            xytext=(ep[best_v_idx], max(0, v_acc[best_v_idx] - 0.06)),
            arrowprops=dict(arrowstyle="->", color="#FF5722"),
            fontsize=8, color="#FF5722", ha="center",
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        ax = axes[1]
        ax.plot(ep, t_loss, "o-",  color="#2196F3", label="Train Loss", linewidth=2, markersize=4)
        ax.plot(ep, v_loss, "s--", color="#FF5722", label="Val Loss",   linewidth=2, markersize=4)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Loss vs Epoch", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        best_vl_idx = int(np.argmin(v_loss))
        loss_range  = max(v_loss) - min(v_loss)
        ax.annotate(
            f"best val\n{v_loss[best_vl_idx]:.4f}",
            xy=(ep[best_vl_idx], v_loss[best_vl_idx]),
            xytext=(ep[best_vl_idx], v_loss[best_vl_idx] + max(loss_range * 0.15, 0.01)),
            arrowprops=dict(arrowstyle="->", color="#FF5722"),
            fontsize=8, color="#FF5722", ha="center",
        )

        fig.tight_layout()
        curve_path = os.path.join(figure_dir, f"{model_name}_{mag}_curves_{suffix}.pdf")
        plt.savefig(curve_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Training curves ({suffix:<4}) → {curve_path}")


# ── Helper: metrics summary table ─────────────────────────────────────────────
def _print_metrics_table(metrics: dict, mag: str, model_name: str):
    """Print metrics table หลัง test evaluation."""
    w = 56
    print(f"\n  {'═'*w}")
    print(f"  {'TEST RESULTS':^{w}}")
    print(f"  {model_name.upper() + ' | ' + mag:^{w}}")
    print(f"  {'─'*w}")
    rows = [
        ("Accuracy  (Acc)",  metrics["accuracy"]),
        ("Precision (Prec)", metrics["precision"]),
        ("Recall    (Rec)",  metrics["recall"]),
        ("F1-Score  (F1)",   metrics["f1"]),
        ("IoU / Jaccard",    metrics["jaccard"]),
        ("ROC-AUC   (AUC)",  metrics["roc_auc"]),
    ]
    for name, val in rows:
        bar_len = int(val * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        print(f"  {name:<20} {val:.4f}  {bar}")
    print(f"  {'─'*w}")
    print(f"  Specificity per class:")
    for k, v in metrics.items():
        if k.startswith("specificity"):
            cls = k.replace("specificity_", "")
            print(f"    {cls:<6} {v:.4f}")
    print(f"  {'═'*w}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seed_everything(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Speed knobs ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        cudnn.benchmark = True          # fastest cuDNN kernel per input shape
        cudnn.deterministic = False     # allow non-deterministic (faster)
        print("  cuDNN benchmark: ON")

    mags = resolve_magnifications(args.magnification)
    print(f"Magnifications: {mags}")

    # ── Transforms ────────────────────────────────────────────────────────────
    target_img_path = resolve_target_image(args)
    if args.preprocess == "reinhard" and target_img_path:
        stain_norm = StainNormalizationTransform(target_img_path)
        print(f"  Stain normalisation (Reinhard) target: {target_img_path}")
    elif args.preprocess == "macenko" and target_img_path:
        stain_norm = MacenkoStainNormalizationTransform(target_img_path)
        print(f"  Stain normalisation (Macenko)  target: {target_img_path}")
    else:
        stain_norm = None
        if args.preprocess in ("reinhard", "macenko"):
            print(f"  WARNING: {args.preprocess} requested but no target image found. Skipping.")

    # ── Build preprocessing pipelines ─────────────────────────────────────────
    if args.preprocess == "otsu":
        # Paper pipeline: Resize → CenterCrop → Otsu segmentation → uint8/float32
        print(f"  Preprocessing: Otsu segmentation (paper pipeline, img_size={args.img_size})")
        preprocess   = build_paper_preprocess(args.img_size, use_otsu=True)
        vtpreprocess = build_paper_vtpreprocess(args.img_size, use_otsu=True)
    else:
        preprocess   = build_preprocess(args.img_size, stain_norm)
        vtpreprocess = build_vtpreprocess(args.img_size, stain_norm)
    augment         = build_augment(args.augment)
    balance_augment = build_balance_augment()
    normalise       = build_final_normalise()

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or (
        f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model":          args.model,
            "magnifications": str(mags),
            "optimizer":      args.optimizer,
            "lr":             args.lr,
            "momentum":       args.momentum,
            "weight_decay":   args.weight_decay,
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "patience":       args.patience,
            "augment":        args.augment,
            "preprocess":     args.preprocess,
            "img_size":       args.img_size,
            "use_amp":        use_amp,
            "num_classes":    args.num_classes,
            "seed":           args.seed,
            "compiled":       args.compile,
            "ckpt_interval":  args.ckpt_interval,
        })

        for mag in mags:
            loaders = build_loaders(
                data_dir=args.data_dir,
                mag=mag,
                preprocess=preprocess,
                augment=augment,
                balance_augment=balance_augment,
                normalise=normalise,
                vtpreprocess=vtpreprocess,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            print(f"\n{mag}  train={len(loaders['train'].dataset):,}  "
                  f"val={len(loaders['val'].dataset):,}  "
                  f"test={len(loaders['test'].dataset):,}")

            now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(args.output_dir, "runs", run_name, mag, now_str)
            os.makedirs(log_dir, exist_ok=True)
            writer  = SummaryWriter(log_dir)

            train_magnification(
                mag=mag, loaders=loaders, model_name=args.model,
                args=args, device=device, use_amp=use_amp,
                writer=writer, output_dir=args.output_dir,
            )
            writer.close()

        print(f"\n✅  All magnifications done!  MLflow run: {run.info.run_id}")
        print(f"   Outputs → {args.output_dir}")

if __name__ == "__main__":
    main()