"""
src/models/swin_t.py
━━━━━━━━━━━━━━━━━━━━
Swin Transformer Tiny (Swin-T) — fully trainable baseline.
Native resolution: 224×224 (input resized internally if needed).
Params: ~28M

Requires: pip install timm
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinTiny(nn.Module):
    NATIVE_SIZE = 224

    def __init__(self, num_classes: int = 8, weights_dir: str | None = None):
        super().__init__()
        from ..weights_manager import load_timm_model

        self.model = load_timm_model(
            "swin_tiny_patch4_window7_224", weights_dir, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.NATIVE_SIZE or x.shape[-2] != self.NATIVE_SIZE:
            x = F.interpolate(x, size=(self.NATIVE_SIZE, self.NATIVE_SIZE),
                              mode="bilinear", align_corners=False)
        return self.model(x)
