"""
src/models/shufflenet_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
ShuffleNet-V2 x1.0 — ultra-lightweight CNN baseline.
Params: ~2.3M
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _fc_head(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes: int = 8, weights_dir: str | None = None):
        super().__init__()
        from ..weights_manager import load_torchvision_backbone
        backbone = load_torchvision_backbone(
            models.shufflenet_v2_x1_0, "shufflenet_v2_x1_0",
            models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1, weights_dir,
        )
        in_feat = backbone.fc.in_features
        backbone.fc = _fc_head(in_feat, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
