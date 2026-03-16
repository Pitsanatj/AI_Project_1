"""
src/models/ecsamet.py
━━━━━━━━━━━━━━━━━━━━━
ECSAnet — EfficientNet-V2-S + CBAM attention injected after feature stage 6.
Classifier: Dropout → FC-1024 → SiLU → Dropout → FC-1024 → SiLU → FC-C

SupCon support
──────────────
Pass proj_dim > 0 to enable the L2-normalised projection head used for
Supervised Contrastive pre-training (experiment E2/E3/E4).

  forward(x)                  → logits                 (inference / CE-only)
  forward(x, return_proj=True)→ (logits, proj_feats)   (SupCon training)

The projection head is discarded at test time; only logits are used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s


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


class ECSAnet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        weights_dir: str | None = None,
        proj_dim: int = 0,
    ):
        """
        Parameters
        ----------
        proj_dim : int
            Dimension of the SupCon projection head output (e.g. 128).
            Set to 0 (default) to disable the projection head.
        """
        super().__init__()
        from ..weights_manager import load_torchvision_backbone
        backbone       = load_torchvision_backbone(
            efficientnet_v2_s, "efficientnet_v2_s", "IMAGENET1K_V1", weights_dir
        )
        out_channels   = backbone.features[6][14].block[3][0].out_channels
        self.features  = backbone.features
        self.cbam      = CBAM(out_channels)
        self.avgpool   = backbone.avgpool
        num_features   = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        # ── Projection head for Supervised Contrastive Learning ───────────────
        # FC(1280→256→proj_dim) + L2 normalisation.
        # Built only when proj_dim > 0; discarded at inference.
        if proj_dim > 0:
            self.proj_head: nn.Module | None = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, proj_dim),
            )
        else:
            self.proj_head = None

    def forward(
        self, x: torch.Tensor, return_proj: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        return_proj : bool
            When True (SupCon training), return (logits, proj_feats).
            proj_feats are L2-normalised embeddings from the projection head.
            Raises RuntimeError if proj_dim was 0 at construction time.
        """
        x = self.features[:7](x)
        x = self.cbam(x)
        x = self.features[7:](x)
        x = self.avgpool(x)
        feats  = torch.flatten(x, 1)           # (B, 1280)
        logits = self.classifier(feats)        # (B, num_classes)

        if return_proj:
            if self.proj_head is None:
                raise RuntimeError(
                    "return_proj=True requires proj_dim > 0 at construction."
                )
            proj = F.normalize(self.proj_head(feats), dim=-1)  # (B, proj_dim)
            return logits, proj

        return logits
