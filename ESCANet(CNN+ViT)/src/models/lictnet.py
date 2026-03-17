"""
src/models/lictnet.py
━━━━━━━━━━━━━━━━━━━━━
LiCT-Net — Lightweight Convolutional Transformer Network (TENCON 2024).
Fixed to match paper exactly.

Key fixes vs previous version:
  1. FFN is CONVOLUTIONAL (DWConv3 + Conv1x1 + residual) — NOT Linear
     Paper Eq.3: FFN_out = FFN_in ⊕ Conv_k=1( DWConv_k=3(FFN_in) )
  2. Backbone is NOT frozen — paper explicitly states "end-to-end trainable"
  3. Head is single FC layer — paper says "GAP → dense layer"
  4. img_size should be 256 (paper uses 256×256, not 384)

Architecture (paper Fig.1):
  FastViT-T8 backbone (E-to-E, stage-3 features C=192)
  → LSTL  (SAA + Conv-FFN with residual connections)
  → GAP
  → Linear(192 → num_classes)

Requires: pip install timm
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── GSAB ──────────────────────────────────────────────────────────────────────
class GSAB(nn.Module):
    """
    Global Spatial Attention Block (paper eq. 5-6).
      F' = Conv1x1( DWConv_dilated5( DWConv5(F) ) )
      F_GSAB = F ⊗ F'
    Large kernel + dilated conv → enlarge receptive field for global context.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.ReLU(inplace=True),
        )
        # dilation=3, kernel=5 → padding = (5-1)//2 * 3 = 6
        self.dw_conv5d = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=6, dilation=3,
                      groups=channels, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pw_conv(self.dw_conv5d(self.dw_conv5(x)))
        return x * attn                         # eq. 6: F_GSAB = F ⊗ F'


# ── LSAB ──────────────────────────────────────────────────────────────────────
class LSAB(nn.Module):
    """
    Local Spatial Attention Block (paper eq. 7-8).
      F' = Conv1x1( Concat( AvgPool(F), MaxPool(F) ) )
      F_LSAB = F ⊗ sigmoid(F')
    Channel-wise pooling → captures local feature relationships.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pw_conv  = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        attn   = self.sigmoid(self.pw_conv(pooled))
        return x * attn                         # eq. 8: F_LSAB = F ⊗ σ(F')


# ── SAA ───────────────────────────────────────────────────────────────────────
class SAA(nn.Module):
    """
    Spatially Aware Attention (paper eq. 4).
      F_gl = F_GSAB ⊕ F_LSAB
    GSAB (global) and LSAB (local) run in parallel then summed.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gsab = GSAB(channels)
        self.lsab = LSAB(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gsab(x) + self.lsab(x)     # eq. 4


# ── LSTL ──────────────────────────────────────────────────────────────────────
class LSTL(nn.Module):
    """
    Local-Global Spatially Aware Transformer Layer (paper eq. 1-3).

      X'     = F  ⊕ SAA( LayerNorm(F) )          eq.1 — attention + residual
      LSTL_out = X' ⊕ FFN( LayerNorm(X') )        eq.2 — FFN + residual

    FFN (paper eq. 3) — CONVOLUTIONAL, not linear:
      FFN_out = FFN_in ⊕ Conv1x1( DWConv_k=3(FFN_in) )

    ⚠  Previous implementation used Linear layers here — that is WRONG.
       The paper's FFN keeps spatial structure (B,C,H,W) throughout.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm1   = nn.LayerNorm(channels)
        self.norm2   = nn.LayerNorm(channels)
        self.saa     = SAA(channels)

        # Conv-FFN per paper eq.3: FFN_in → DWConv3 → Conv1x1 → + FFN_in
        self.ffn_dw  = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.ffn_pw  = nn.Conv2d(channels, channels, 1, bias=False)
        self.ffn_act = nn.ReLU(inplace=True)

    @staticmethod
    def _ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm over channel dim of (B,C,H,W)."""
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Conv-FFN with residual: x + Conv1x1(DWConv3(x))"""
        return x + self.ffn_pw(self.ffn_act(self.ffn_dw(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.saa(self._ln(self.norm1, x)) + x   # eq.1
        x = self._ffn(self._ln(self.norm2, x)) + x  # eq.2 + eq.3
        return x


# ── LiCTNet ───────────────────────────────────────────────────────────────────
class LiCTNet(nn.Module):
    """
    Full LiCT-Net model (paper Fig.1).

    backbone → LSTL → GAP → FC(num_classes)

    ⚠  Backbone is NOT frozen — paper explicitly states end-to-end (E-to-E)
       trainable. Freezing backbone was the main cause of low accuracy (30%).
    ⚠  Use --img_size 256 to match paper (paper uses 256×256 input).
    """
    BACKBONE_CHANNELS = 192     # FastViT-T8 stage-3 actual output channels

    def __init__(self, num_classes: int = 8, weights_dir: str | None = None):
        super().__init__()
        from ..weights_manager import load_timm_model

        self.backbone = load_timm_model(
            "fastvit_t8", weights_dir,
            features_only=True, out_indices=(2,),
        )

        # ── E-to-E trainable: backbone is NOT frozen (paper design) ──────────
        # All parameters train from the start with pretrained init
        for p in self.backbone.parameters():
            p.requires_grad = True

        C = self.BACKBONE_CHANNELS
        self.lstl = LSTL(C)
        self.gap  = nn.AdaptiveAvgPool2d(1)

        # Paper says "GAP → dense layer" — single FC, not 192→512→C
        self.head = nn.Linear(C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)[0]     # (B, 192, H/16, W/16)
        x = self.lstl(x)
        x = self.gap(x)
        return self.head(torch.flatten(x, 1))