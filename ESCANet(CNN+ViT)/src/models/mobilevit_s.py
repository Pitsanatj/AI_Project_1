"""
src/models/mobilevit_s.py
━━━━━━━━━━━━━━━━━━━━━━━━━
MobileViT-S — lightweight hybrid CNN-Transformer (Apple, 2021).
Efficient global context at low parameter count.
Native resolution: 256×256 (input resized internally if needed).
Params: ~5.6M

Requires: pip install timm
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileViTS(nn.Module):
    NATIVE_SIZE = 256

    def __init__(self, num_classes: int = 8, weights_dir: str | None = None):
        super().__init__()
        from ..weights_manager import load_timm_model

        self.model = load_timm_model(
            "mobilevit_s", weights_dir, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.NATIVE_SIZE or x.shape[-2] != self.NATIVE_SIZE:
            x = F.interpolate(x, size=(self.NATIVE_SIZE, self.NATIVE_SIZE),
                              mode="bilinear", align_corners=False)
        return self.model(x)
