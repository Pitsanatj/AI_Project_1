"""
src/models/hybrid_ecsanet.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HybridECSAnet — EfficientNetV2-{S|M} + ViT (parallel, late-fusion).

Architecture
────────────
┌─ CNN branch ──────────────────────────────────────────────────┐
│  EfficientNetV2-{S|M}  (ImageNet1K pretrained)                │
│  Optional: CBAM(1280) on the final 1280-ch feature maps       │
│  → GlobalAvgPool → Flatten → (B, 1280)                        │
└───────────────────────────────────────────────────────────────┘
             ║                    ┌─ ViT branch ───────────────┐
             ║                    │  ViT-B/16  → (B, 768)  OR │
             ║                    │  DeiT-S    → (B, 384)  OR │
             ║                    │  "none"  → skip            │
             ║                    └───────────────────────────-┘
             ╚══════════════ cat ════════════════════╝
                                   ↓
           FC(joint_dim → fusion_dim) → BN1d → ReLU → Dropout(p)
                                   ↓
                          FC(fusion_dim → C)

Ablation variants (B-series experiments)
──────────────────────────────────────────
  cnn_backbone | vit_branch | use_cbam → Exp
  ─────────────┼────────────┼──────────────────────────────────
  "s"          | "none"     | True     → (use ecsamet instead)
  "s"          | "deit_s"   | False    → B1
  "s"          | "vit_b16"  | False    → B2
  "s"          | "vit_b16"  | True     → B3
  "m"          | "vit_b16"  | False    → B4  ★ paper's HybridEffNetV2M-ViT
  "m"          | "vit_b16"  | True     → B5  ★ proposed
  (B6 = B5 + Macenko stain norm — controlled via --preprocess, same model)

References
──────────
  "Hybrid Deep Learning EfficientNetV2 and Vision Transformer Model for
   Breast Cancer Histopathological Image Classification" (2024).
   Late-fusion: EfficientNetV2-M (1280-dim) + ViT-B/16 (768-dim)
   → FC(2048→512→C).

  Li et al., "VTCNet: A Feature Fusion DL Model Based on CNN and ViT for the
   Classification of Breast Cancer" (2024).
   Parallel CNN (custom) + ViT, feature-level fusion.

  This work adds: optional CBAM on CNN branch, ablation of backbone size,
  ablation of ViT variant (DeiT-S vs ViT-B/16).

CBAM placement note
────────────────────
  CBAM is placed AFTER all features (1280-ch, spatial ~12×12 for 384px input),
  just before GlobalAvgPool. This ensures compatibility with both EfficientNetV2-S
  (160→1280 in stage 7) and EfficientNetV2-M (without hardcoding intermediate
  channel counts that differ between variants).
  Semantic: CBAM acts as a "pre-fusion attention gate" on the CNN branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    vit_b_16,
    ViT_B_16_Weights,
)


# ── Shared CBAM building blocks ────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


# ── CNN branch ─────────────────────────────────────────────────────────────────
class _CNNBranch(nn.Module):
    """
    EfficientNetV2-{S|M} with optional CBAM at the final 1280-channel feature map.

    Both EfficientNetV2-S and M produce 1280-ch feature maps before GlobalAvgPool,
    so CBAM(1280) works unchanged for both variants.
    """
    CNN_DIM = 1280   # fixed for both EfficientNetV2-S and EfficientNetV2-M

    def __init__(
        self,
        backbone: str = "s",       # "s" | "m"
        use_cbam: bool = False,
        weights_dir: str | None = None,
    ):
        super().__init__()
        from ..weights_manager import load_torchvision_backbone

        if backbone == "s":
            net = load_torchvision_backbone(
                efficientnet_v2_s, "efficientnet_v2_s", "IMAGENET1K_V1", weights_dir
            )
        elif backbone == "m":
            net = load_torchvision_backbone(
                efficientnet_v2_m, "efficientnet_v2_m", "IMAGENET1K_V1", weights_dir
            )
        else:
            raise ValueError(f"Unknown cnn_backbone '{backbone}'. Choose 's' or 'm'.")

        self.features = net.features          # all EfficientNetV2 stages
        self.avgpool  = net.avgpool           # AdaptiveAvgPool2d(1)
        self.cbam     = CBAM(self.CNN_DIM) if use_cbam else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 1280, H', W')
        if self.cbam is not None:
            x = self.cbam(x)          # channel + spatial attention
        x = self.avgpool(x)           # (B, 1280, 1, 1)
        return torch.flatten(x, 1)    # (B, 1280)


# ── ViT branch ─────────────────────────────────────────────────────────────────
class _ViTBranch(nn.Module):
    """
    Pretrained ViT-B/16 or DeiT-S with classification head removed.

    Input is resized to 224×224 internally (both variants pretrained at 224).
    Returns the CLS token embedding: (B, D_vit).
    """
    VIT_DIMS = {
        "vit_b16": 768,
        "deit_s":  384,
    }
    VIT_SIZE = 224   # native pretraining resolution for both

    def __init__(
        self,
        vit_branch: str = "vit_b16",
        weights_dir: str | None = None,
    ):
        super().__init__()
        if vit_branch not in self.VIT_DIMS:
            raise ValueError(
                f"Unknown vit_branch '{vit_branch}'. Choose 'vit_b16' or 'deit_s'."
            )
        self.out_dim  = self.VIT_DIMS[vit_branch]
        self.vit_size = self.VIT_SIZE

        if vit_branch == "vit_b16":
            from ..weights_manager import load_torchvision_backbone
            net = load_torchvision_backbone(
                vit_b_16, "vit_b_16", "IMAGENET1K_V1", weights_dir
            )
            net.heads.head = nn.Identity()   # expose 768-dim CLS token
            self.vit = net

        elif vit_branch == "deit_s":
            from ..weights_manager import load_timm_model
            # num_classes=0 → timm strips classifier and returns 384-dim CLS token
            self.vit = load_timm_model(
                "deit_small_patch16_224", weights_dir, num_classes=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to ViT native resolution when input is not already 224×224
        if x.shape[-1] != self.vit_size or x.shape[-2] != self.vit_size:
            x = F.interpolate(
                x, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
        return self.vit(x)   # (B, D_vit)


# ── Hybrid model ──────────────────────────────────────────────────────────────
class HybridECSAnet(nn.Module):
    """
    Hybrid ECSAnet: EfficientNetV2-{S|M} || ViT (parallel, late fusion).

    Parameters
    ----------
    num_classes  : int   — output classes (8 for BreakHis 8-class)
    weights_dir  : str   — local pretrained weights directory (None = download)
    cnn_backbone : str   — "s" (21M) | "m" (54M)
    vit_branch   : str   — "vit_b16" (768-dim) | "deit_s" (384-dim) | "none"
    use_cbam     : bool  — True → CBAM(1280) on CNN branch before fusion
    fusion_dim   : int   — FC hidden dim in the classification head (paper: 512)
    dropout      : float — Dropout probability in the classification head
    """

    def __init__(
        self,
        num_classes:  int   = 8,
        weights_dir:  str | None = None,
        cnn_backbone: str   = "s",
        vit_branch:   str   = "vit_b16",
        use_cbam:     bool  = False,
        fusion_dim:   int   = 512,
        dropout:      float = 0.2,
    ):
        super().__init__()

        # ── CNN branch ────────────────────────────────────────────────────────
        self.cnn_branch = _CNNBranch(cnn_backbone, use_cbam, weights_dir)
        cnn_dim = _CNNBranch.CNN_DIM                     # 1280

        # ── ViT branch (optional) ─────────────────────────────────────────────
        if vit_branch == "none":
            self.vit_branch = None
            joint_dim = cnn_dim                          # 1280
        else:
            self.vit_branch = _ViTBranch(vit_branch, weights_dir)
            joint_dim = cnn_dim + self.vit_branch.out_dim   # 1280+768 or 1280+384

        # ── Fusion head (matches HybridEffNetV2M-ViT paper) ──────────────────
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feats = self.cnn_branch(x)           # (B, 1280)

        if self.vit_branch is not None:
            vit_feats = self.vit_branch(x)       # (B, D_vit)
            feats = torch.cat([cnn_feats, vit_feats], dim=1)
        else:
            feats = cnn_feats                    # pure CNN path (B, 1280)

        return self.classifier(feats)            # (B, num_classes)
