"""
ECSAnet Training Script for Lenovo LiCO Cluster
================================================
Converted from Google Colab notebook.
- ใช้ argument parser แทน hardcoded paths
- รองรับ multi-GPU ผ่าน SLURM environment variables
- บันทึกไฟล์ไปยัง local cluster paths (ไม่ใช้ Google Drive)
- Seed 42 ทุกจุด (torch, numpy, random, CUDA) เพื่อ reproducibility
- รองรับ offline weights ผ่าน --weights_dir
- รองรับ resume จาก checkpoint ด้วย --resume               ← NEW
"""

import os
import random
import time
import uuid
import argparse
from datetime import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from staintools import ReinhardColorNormalizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, jaccard_score,
    classification_report, roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')  # ใช้ non-interactive backend (ไม่ต้องมี display)
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
# Global Seed — ใช้ค่าเดียวกันกับ split_data.py
# ─────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """ตั้ง seed ทุกจุดเพื่อให้ผลลัพธ์ reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # สำหรับ multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[ECSAnet] Global seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """Worker init function สำหรับ DataLoader ให้แต่ละ worker มี seed ต่างกัน"""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='ECSAnet Training on LiCO')

    # Paths
    parser.add_argument('--split_dir', type=str,
                        default='./MyBreakHis_Split')
    parser.add_argument('--target_image', type=str,
                        default='./target_image.png')
    parser.add_argument('--output_dir', type=str,
                        default='./ECSAnet_outputs')
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='Path ไปยัง pretrained weights directory (offline mode)')

    # Training parameters
    parser.add_argument('--magnification', type=str, default='40X',
                        choices=['40X', '100X', '200X', '400X'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=SEED)

    # ── Resume ──────────────────────────────────────────────────────────────
    parser.add_argument('--resume', type=str, default=None,
                        metavar='CHECKPOINT_PATH',
                        help='Path ไปยัง checkpoint .pth ที่ต้องการ resume\n'
                             'ใช้ "latest" เพื่อโหลด checkpoint ล่าสุดใน\n'
                             'output_dir/Checkpoints/<mag>/ อัตโนมัติ')

    return parser.parse_args()


# ─────────────────────────────────────────────
# Resume Helper
# ─────────────────────────────────────────────
def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """คืน path ของ checkpoint ล่าสุดใน ckpt_dir (ตาม epoch number)"""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        ckpt_dir.glob('checkpoint_epoch_*.pth'),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    return str(checkpoints[-1]) if checkpoints else None


def load_checkpoint(ckpt_path: str, model, optimizer, scheduler, device):
    """
    โหลด checkpoint และคืนค่า state ที่จำเป็นสำหรับการ resume

    Returns
    -------
    start_epoch      : int   — epoch ถัดไปที่ต้องเริ่ม (saved_epoch + 1)
    best_val_loss    : float — best validation loss ที่เคยได้
    patience_counter : int   — early-stopping counter ณ เวลาที่ save
    """
    print(f"[ECSAnet] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start_epoch      = ckpt['epoch'] + 1          # เริ่ม epoch ถัดไป
    best_val_loss    = ckpt.get('best_val_loss', np.inf)
    patience_counter = ckpt.get('patience_counter', 0)

    print(f"  → Resumed from epoch {ckpt['epoch']+1}  "
          f"best_val_loss={best_val_loss:.4f}  "
          f"patience={patience_counter}")
    return start_epoch, best_val_loss, patience_counter


# ─────────────────────────────────────────────
# Dataset Utilities
# ─────────────────────────────────────────────
IMG_EXTENSIONS = ['.png', '.jpg']

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


class StainNormalizationTransform:
    def __init__(self, target_image_path):
        self.normalizer = ReinhardColorNormalizer()
        if not os.path.isfile(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")
        target_image = np.array(Image.open(target_image_path))
        self.normalizer.fit(target_image)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError('Input must be a PIL Image.')
        image_np = np.array(image, dtype=np.uint8)
        normalized = self.normalizer.transform(image_np)
        if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
            raise ValueError('NaN or Inf values after stain normalization!')
        return Image.fromarray(normalized)


class BalancedOversampledDataset(Dataset):
    def __init__(self, main_dir, preprocess=None, transform=None, balance=None):
        self.main_dir = main_dir
        self.preprocess = preprocess
        self.transform = transform
        self.balance = balance

        self.classes = sorted(os.listdir(main_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.imgs_labelled = []
        self.class_counts = Counter()

        for cls in self.classes:
            cls_path = os.path.join(main_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            cls_index = self.class_to_idx[cls]
            for img_name in os.listdir(cls_path):
                if is_image_file(img_name):
                    self.imgs_labelled.append((os.path.join(cls_path, img_name), cls_index))
                    self.class_counts[cls_index] += 1

        self.max_class_count = max(self.class_counts.values())
        self.samples_by_class = [[] for _ in self.classes]
        for img_path, class_idx in self.imgs_labelled:
            self.samples_by_class[class_idx].append(img_path)

    def __len__(self):
        return len(self.classes) * self.max_class_count * 3

    def __getitem__(self, index):
        class_index = index % len(self.classes)
        img_index = (index // len(self.classes)) % self.max_class_count
        img_list = self.samples_by_class[class_index]
        img_path = img_list[img_index % len(img_list)]
        image = Image.open(img_path).convert("RGB")

        if self.preprocess:
            image = self.preprocess(image)
        if len(img_list) < self.max_class_count and self.balance:
            image = self.balance(image)
        if self.transform:
            image = self.transform(image)

        return image, class_index


class ValTestDataset(Dataset):
    def __init__(self, main_dir, vtpreprocess=None):
        self.main_dir = main_dir
        self.vtpreprocess = vtpreprocess
        self.classes = sorted(os.listdir(main_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        all_imgs = []
        for dp, dn, fn in os.walk(os.path.expanduser(main_dir)):
            for f in fn:
                if is_image_file(f):
                    all_imgs.append(os.path.join(dp, f))

        self.imgs_labelled = [
            (img, self.class_to_idx[os.path.basename(os.path.dirname(img))])
            for img in all_imgs
        ]

    def __len__(self):
        return len(self.imgs_labelled)

    def __getitem__(self, index):
        img_path, label = self.imgs_labelled[index]
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.vtpreprocess:
            image = self.vtpreprocess(image)
        return image, label


# ─────────────────────────────────────────────
# Model: ECSAnet
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ECSAnet(nn.Module):
    def __init__(self, num_classes=8, weights_dir: str | None = None):
        super().__init__()

        if weights_dir is not None:
            weights_path = Path(weights_dir) / "efficientnet_v2_s.pth"
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {weights_path}\n"
                    "Run: python download_pretrained_weights.py --weights_dir "
                    f"{weights_dir}"
                )
            print(f"  [ECSAnet] Loading offline weights from: {weights_path}")
            self.base_model = efficientnet_v2_s(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            self.base_model.load_state_dict(state)
        else:
            print("  [ECSAnet] Downloading EfficientNetV2-S pretrained weights ...")
            self.base_model = efficientnet_v2_s(
                weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1, progress=True
            )
        out_channels = self.base_model.features[6][14].block[3][0].out_channels
        self.cbam = CBAM(out_channels)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.base_model.features[:7](x)
        x = self.cbam(x)
        x = self.base_model.features[7:](x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x)
        return x


# ─────────────────────────────────────────────
# Metrics & Plotting Helpers
# ─────────────────────────────────────────────
CLASS_LABELS = ['A', 'DC', 'F', 'LC', 'MC', 'PC', 'PT', 'TA']

def compute_metrics(all_labels, all_preds, all_probs, num_classes):
    lb = LabelBinarizer()
    lb.fit(range(num_classes))
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    confusion = confusion_matrix(all_labels, all_preds)
    jaccard   = jaccard_score(all_labels, all_preds, average='weighted', zero_division=0)
    report    = classification_report(all_labels, all_preds, target_names=CLASS_LABELS, zero_division=0)
    roc_auc   = roc_auc_score(lb.transform(all_labels), all_probs, multi_class='ovr')

    specificity = {}
    for i, label in enumerate(CLASS_LABELS):
        TN = confusion.sum() - (confusion[i, :].sum() + confusion[:, i].sum() - confusion[i, i])
        FP = confusion[:, i].sum() - confusion[i, i]
        specificity[label] = TN / (TN + FP) if (TN + FP) > 0 else 0

    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                confusion=confusion, jaccard=jaccard, report=report,
                roc_auc=roc_auc, specificity=specificity,
                all_labels_onehot=lb.transform(all_labels))


def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"Jaccard:   {metrics['jaccard']:.4f}\n")
        f.write(f"ROC AUC:   {metrics['roc_auc']:.4f}\n\n")
        f.write("Specificity per class:\n")
        for label, spec in metrics['specificity'].items():
            f.write(f"  {label}: {spec:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion']}\n\n")
        f.write(f"Classification Report:\n{metrics['report']}\n")
    print(f"  → Metrics saved: {filepath}")


def save_confusion_matrix(confusion, figure_dir, model_name, mag):
    os.makedirs(figure_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    path = os.path.join(figure_dir, f'{model_name}_{mag}_cm.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  → Confusion matrix saved: {path}")


def save_roc_curve(all_labels_onehot, all_probs, figure_dir, model_name, mag):
    os.makedirs(figure_dir, exist_ok=True)
    fpr, tpr, roc_auc_per = {}, {}, {}
    for i in range(len(CLASS_LABELS)):
        fpr[i], tpr[i], _ = roc_curve(all_labels_onehot[:, i], all_probs[:, i])
        roc_auc_per[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 8))
    for i in range(len(CLASS_LABELS)):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC (AUC={roc_auc_per[i]:.2f}) - {CLASS_LABELS[i]}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 0.2])
    plt.ylim([0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    path = os.path.join(figure_dir, f'{model_name}_{mag}_roc_auc.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  → ROC curve saved: {path}")


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Seed ────────────────────────────────────
    set_seed(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    args.output_dir = os.path.expandvars(args.output_dir)
    args.split_dir  = os.path.expandvars(args.split_dir)

    # ── Output directories ──────────────────────
    model_name   = 'ECSAnet'
    mag          = args.magnification
    figure_dir   = os.path.join(args.output_dir, 'Figures')
    metrics_dir  = os.path.join(args.output_dir, 'Metrics')
    model_dir    = os.path.join(args.output_dir, 'Models', mag)
    ckpt_dir     = os.path.join(args.output_dir, 'Checkpoints', mag)
    log_dir      = os.path.join(args.output_dir, 'Logs', model_name, mag,
                                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}")

    for d in [figure_dir, metrics_dir, model_dir, ckpt_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[ECSAnet] Output directory : {args.output_dir}")
    print(f"[ECSAnet] Split directory  : {args.split_dir}")
    print(f"[ECSAnet] Magnification    : {mag}")
    print(f"[ECSAnet] TensorBoard logs : {log_dir}")
    print(f"[ECSAnet] Weights dir      : {args.weights_dir or '(online download)'}")

    # ── Device ─────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECSAnet] Using device     : {device}")
    if torch.cuda.is_available():
        print(f"[ECSAnet] GPU              : {torch.cuda.get_device_name(0)}")

    # ── Stain Normalization ─────────────────────
    print(f"[ECSAnet] Loading stain normalizer from: {args.target_image}")
    stain_normalization = StainNormalizationTransform(args.target_image)

    # ── Transforms ─────────────────────────────
    preprocess = v2.Compose([
        v2.Resize(size=(384, 384), interpolation=InterpolationMode.BILINEAR),
        v2.CenterCrop(size=(384, 384)),
        stain_normalization,
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
    ])
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(0, 10)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vtpreprocess = v2.Compose([
        v2.Resize(size=(384, 384), interpolation=InterpolationMode.BILINEAR),
        v2.CenterCrop(size=(384, 384)),
        stain_normalization,
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    balance = v2.Compose([
        v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.)
    ])

    # ── Dataset paths ───────────────────────────
    train_dir = os.path.join(args.split_dir, 'train', mag)
    val_dir   = os.path.join(args.split_dir, 'val',   mag)
    test_dir  = os.path.join(args.split_dir, 'test',  mag)

    # ── DataLoaders ─────────────────────────────
    train_dataset = BalancedOversampledDataset(train_dir, preprocess=preprocess,
                                               transform=transform, balance=balance)
    val_dataset   = ValTestDataset(val_dir,  vtpreprocess=vtpreprocess)
    test_dataset  = ValTestDataset(test_dir, vtpreprocess=vtpreprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(f"[ECSAnet] Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Model ───────────────────────────────────
    model = ECSAnet(
        num_classes=args.num_classes,
        weights_dir=args.weights_dir,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    writer    = SummaryWriter(log_dir)

    lb = LabelBinarizer()
    lb.fit(range(args.num_classes))

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = np.inf
    patience_counter = 0

    resume_path = args.resume
    if resume_path == 'latest':
        resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path is None:
            print("[ECSAnet] --resume latest: ไม่พบ checkpoint, เริ่มใหม่ตั้งแต่ต้น")
        else:
            print(f"[ECSAnet] --resume latest: พบ checkpoint → {resume_path}")

    if resume_path is not None:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        # โหลด best_model ที่เคย save ด้วย (ถ้ามี) เพื่อให้ best_model_state ถูกต้อง
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.isfile(best_model_path):
            best_model_state = torch.load(best_model_path, map_location=device)
            print(f"  → Best model state loaded from: {best_model_path}")
        else:
            best_model_state = model.state_dict()
    else:
        best_model_state = model.state_dict()

    # ── Training ────────────────────────────────
    start_time = time.time()
    print(f"\n{'='*55}")
    print(f"  Starting training: {model_name} @ {mag}")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch + 1} / {args.epochs}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        all_train_probs, all_train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            all_train_probs.append(probs.cpu().detach().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        all_train_probs = np.concatenate(all_train_probs)
        running_loss /= len(train_loader.dataset)
        train_accuracy = accuracy_score(all_train_labels, np.argmax(all_train_probs, axis=1))
        train_roc_auc  = roc_auc_score(lb.transform(all_train_labels), all_train_probs, multi_class='ovr')

        writer.add_scalar('Train/Loss',     running_loss,   epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        writer.add_scalar('Train/ROC_AUC',  train_roc_auc,  epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_probs, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                probs = F.softmax(outputs, dim=1)
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        all_val_probs = np.concatenate(all_val_probs)
        val_loss     /= len(val_loader.dataset)
        val_accuracy  = accuracy_score(all_val_labels, np.argmax(all_val_probs, axis=1))
        val_roc_auc   = roc_auc_score(lb.transform(all_val_labels), all_val_probs, multi_class='ovr')

        writer.add_scalar('Val/Loss',     val_loss,     epoch)
        writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
        writer.add_scalar('Val/ROC_AUC',  val_roc_auc,  epoch)

        for param_group in optimizer.param_groups:
            writer.add_scalar('LR', param_group['lr'], epoch)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {running_loss:.4f} Acc: {train_accuracy:.4f} AUC: {train_roc_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} AUC: {val_roc_auc:.4f}")

        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            best_path = os.path.join(model_dir, 'best_model.pth')
            torch.save(best_model_state, best_path)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # ── บันทึก checkpoint พร้อม state ครบสำหรับ resume ──────────────
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':                 running_loss,
            'best_val_loss':        best_val_loss,        # ← NEW
            'patience_counter':     patience_counter,     # ← NEW
        }, os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth'))

        if patience_counter > args.patience:
            print(f"\n⚠ Early stopping at epoch {epoch+1}!")
            break

        scheduler.step(val_loss)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Training Metrics ────────────────────────
    print("\n--- Training Metrics (last epoch) ---")
    train_metrics = compute_metrics(all_train_labels,
                                    np.argmax(all_train_probs, axis=1),
                                    all_train_probs, args.num_classes)
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}  F1: {train_metrics['f1']:.4f}  AUC: {train_metrics['roc_auc']:.4f}")
    save_metrics(train_metrics, os.path.join(metrics_dir, f'{model_name}_{mag}_train_metrics.txt'))

    # ── Testing ─────────────────────────────────
    print("\n--- Testing ---")
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    model.eval()

    all_preds, all_labels, all_probs_test = [], [], []
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() / len(test_loader)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_test.extend(probs.cpu().numpy())

    test_metrics = compute_metrics(all_labels, all_preds,
                                   np.array(all_probs_test), args.num_classes)

    writer.add_scalar('Test/Accuracy',  test_metrics['accuracy'])
    writer.add_scalar('Test/ROC_AUC',   test_metrics['roc_auc'])
    writer.add_scalar('Test/Loss',      test_loss)

    print(f"  Accuracy: {test_metrics['accuracy']:.4f}  F1: {test_metrics['f1']:.4f}  AUC: {test_metrics['roc_auc']:.4f}")
    save_metrics(test_metrics, os.path.join(metrics_dir, f'{model_name}_{mag}_test_metrics.txt'))

    # ── Figures ─────────────────────────────────
    save_confusion_matrix(test_metrics['confusion'], figure_dir, model_name, mag)
    save_roc_curve(test_metrics['all_labels_onehot'], np.array(all_probs_test),
                   figure_dir, model_name, mag)

    writer.close()
    print(f"\n✅ Done! All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()