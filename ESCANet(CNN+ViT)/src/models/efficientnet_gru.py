"""
src/models/efficientnet_gru.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EfficientNetV2-S + GRU + Attention — adapted from:
  "A Hybrid Deep Learning Model for Breast Cancer Detection Using
   EfficientNetV2 and GRU with Attention" (Scientific Reports, 2025)

Architecture
────────────
  EfficientNetV2-S (pretrained, ImageNet)
       ↓  spatial feature map  (B, 1280, H', W')
  Reshape → sequence  (B, T=H'×W', 1280)
       ↓
  GRU  (num_layers, hidden_size, optional bidirectional)
       ↓  H = [h1, h2, ..., hT]  (B, T, hidden_dim)
  Attention  (tanh scoring → softmax → context vector)
       ↓  c  (B, hidden_dim)
  Dropout(p) → FC(hidden_dim → num_classes)

Adaptations from original paper (binary → 8-class)
────────────────────────────────────────────────────
  • num_classes = 8  (BreakHis 8-class, not binary benign/malignant)
  • Loss: CrossEntropyLoss  (not BinaryCrossEntropy)
  • 70:20:10 train/val/test split  (matches existing data split)
  • Stain normalisation is handled externally via preprocess pipeline

Paper hyperparams (Table 2)
────────────────────────────
  epochs=50, batch=32, lr=0.001, optimizer=Adam
  β1=0.9, β2=0.999, ε=1e-8, dropout=0.5, L2=0.0001
  early_stopping_patience=5, lr_scheduler_factor=0.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s


# ── Attention layer (paper eq. 17-19) ────────────────────────────────────────
class AttentionLayer(nn.Module):
    """
    Additive (Bahdanau-style) attention over GRU hidden state sequence.

    Paper eq. 17:  e_t = tanh(W_e · h_t + b_e)
    Paper eq. 18:  α_t = softmax(e_t)
    Paper eq. 19:  c   = Σ α_t · h_t
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Score network: hidden_dim → 1 (via tanh + linear)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, H: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : (B, T, hidden_dim)  — GRU hidden state sequence

        Returns
        -------
        context : (B, hidden_dim)
        alpha   : (B, T)         — attention weights (sum to 1)
        """
        e      = torch.tanh(self.W(H))           # (B, T, hidden_dim)
        e      = self.v(e).squeeze(-1)            # (B, T)  raw scores
        alpha  = F.softmax(e, dim=1)              # (B, T)  attention weights
        context = (alpha.unsqueeze(-1) * H).sum(dim=1)   # (B, hidden_dim)
        return context, alpha


# ── EfficientNetGRU ───────────────────────────────────────────────────────────
class EfficientNetGRU(nn.Module):
    """
    EfficientNetV2-S + GRU + Attention hybrid model.

    Parameters
    ----------
    num_classes  : int   — output classes (8 for BreakHis 8-class)
    weights_dir  : str   — local pretrained weights dir (None = download)
    gru_hidden   : int   — GRU hidden state dimension  (paper default 512)
    gru_layers   : int   — number of stacked GRU layers (default 2)
    gru_dropout  : float — dropout in GRU + before classifier (paper 0.5)
    bidirectional: bool  — bidirectional GRU (doubles hidden dim)
    """

    # EfficientNetV2-S final feature map channels (before GAP)
    FEAT_DIM = 1280

    def __init__(
        self,
        num_classes:   int   = 8,
        weights_dir:   str | None = None,
        gru_hidden:    int   = 512,
        gru_layers:    int   = 2,
        gru_dropout:   float = 0.5,
        bidirectional: bool  = False,
    ):
        super().__init__()
        from ..weights_manager import load_torchvision_backbone

        # ── EfficientNetV2-S backbone (features only, no classifier/avgpool) ──
        backbone      = load_torchvision_backbone(
            efficientnet_v2_s, "efficientnet_v2_s", "IMAGENET1K_V1", weights_dir
        )
        self.features = backbone.features   # (B, 1280, H', W')

        # ── GRU ───────────────────────────────────────────────────────────────
        # input_size = 1280 (feature channels per spatial position)
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)
        self.gru = nn.GRU(
            input_size   = self.FEAT_DIM,
            hidden_size  = gru_hidden,
            num_layers   = gru_layers,
            batch_first  = True,
            dropout      = gru_dropout if gru_layers > 1 else 0.0,
            bidirectional= bidirectional,
        )

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = AttentionLayer(gru_out_dim)

        # ── Classifier ────────────────────────────────────────────────────────
        self.dropout    = nn.Dropout(gru_dropout)
        self.classifier = nn.Linear(gru_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)  — preprocessed & normalised image

        Returns
        -------
        logits : (B, num_classes)
        """
        # ── Feature extraction ───────────────────────────────────────────────
        feat_map = self.features(x)              # (B, 1280, H', W')
        B, C, Hf, Wf = feat_map.shape

        # ── Reshape spatial map → sequence ───────────────────────────────────
        # Paper: 7×7×1280 → 49×1280 (T=H'×W', each timestep = 1280-dim vector)
        seq = feat_map.permute(0, 2, 3, 1)       # (B, H', W', 1280)
        seq = seq.reshape(B, Hf * Wf, C)         # (B, T, 1280)

        # ── GRU ──────────────────────────────────────────────────────────────
        H_out, _ = self.gru(seq)                 # (B, T, gru_out_dim)

        # ── Attention ────────────────────────────────────────────────────────
        context, _alpha = self.attention(H_out)  # (B, gru_out_dim)

        # ── Classification ───────────────────────────────────────────────────
        out    = self.dropout(context)
        logits = self.classifier(out)            # (B, num_classes)
        return logits