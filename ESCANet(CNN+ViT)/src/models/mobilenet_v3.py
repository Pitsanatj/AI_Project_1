"""
src/models/mobilenet_v3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
MobileNet-V3-Large — strong lightweight CNN baseline.
Params: ~5.5M
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


class MobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 8, weights_dir: str | None = None):
        super().__init__()
        from ..weights_manager import load_torchvision_backbone
        backbone = load_torchvision_backbone(
            models.mobilenet_v3_large, "mobilenet_v3_large",
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, weights_dir,
        )
        in_feat = backbone.classifier[0].in_features
        backbone.classifier = _fc_head(in_feat, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
