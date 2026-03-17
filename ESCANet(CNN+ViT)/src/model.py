"""
src/model.py
━━━━━━━━━━━━
Central model registry — imports each architecture from its own file
under src/models/ and exposes build_model() for use by main.py.

  --model ecsamet        → src/models/ecsamet.py
  --model lictnet        → src/models/lictnet.py
  --model swin_t         → src/models/swin_t.py
  --model deit_s         → src/models/deit_s.py
  --model mobilevit_s    → src/models/mobilevit_s.py
  --model fastvit_sa12   → src/models/fastvit_sa12.py
  --model mobilenet_v3   → src/models/mobilenet_v3.py
  --model shufflenet_v2  → src/models/shufflenet_v2.py
"""

from __future__ import annotations

import time
import torch.nn as nn

from .models.ecsamet        import ECSAnet
from .models.hybrid_ecsanet import HybridECSAnet
from .models.lictnet        import LiCTNet
from .models.swin_t       import SwinTiny
from .models.deit_s       import DeiTSmall
from .models.mobilevit_s  import MobileViTS
from .models.fastvit_sa12 import FastViTSA12
from .models.mobilenet_v3  import MobileNetV3
from .models.shufflenet_v2 import ShuffleNetV2

MODEL_REGISTRY: dict[str, type] = {
    # ── Paper models
    "ecsamet":          ECSAnet,
    "hybrid_ecsanet":   HybridECSAnet,
    "lictnet":          LiCTNet,
    # ── Transformer-based
    "swin_t":           SwinTiny,
    "deit_s":           DeiTSmall,
    # ── Lightweight Transformer
    "mobilevit_s":      MobileViTS,
    "fastvit_sa12":     FastViTSA12,
    # ── Lightweight CNN
    "mobilenet_v3":     MobileNetV3,
    "shufflenet_v2":    ShuffleNetV2,
}


def build_model(
    model_name:   str,
    num_classes:  int   = 8,
    weights_dir:  str | None = None,
    proj_dim:     int   = 0,
    # Hybrid model args (ignored for all other models)
    cnn_backbone: str   = "s",
    vit_branch:   str   = "vit_b16",
    use_cbam:     bool  = False,
    fusion_dim:   int   = 512,
) -> nn.Module:
    """
    Instantiate a model by registry key.

    Paper models       : ecsamet | hybrid_ecsanet | lictnet
    Transformer        : swin_t  | deit_s
    Lightweight Transf.: mobilevit_s | fastvit_sa12
    Lightweight CNN    : mobilenet_v3 | shufflenet_v2

    weights_dir  : local pretrained weights dir (None = download).
    proj_dim     : SupCon projection head dim (ecsamet only, 0 = disabled).

    Hybrid-only args (only used when model_name == "hybrid_ecsanet"):
    cnn_backbone : "s" (EfficientNetV2-S, 21M) | "m" (EfficientNetV2-M, 54M)
    vit_branch   : "vit_b16" (768-dim) | "deit_s" (384-dim) | "none" (pure CNN)
    use_cbam     : True → CBAM(1280) after CNN features, before avg pool
    fusion_dim   : hidden dim of the classification head (default 512)
    """
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    print(f"  [model] Building '{key}' with num_classes={num_classes} ...", flush=True)
    if weights_dir:
        print(f"  [model] Weights source: local dir → {weights_dir}", flush=True)
    else:
        print(f"  [model] ⚠  No --weights_dir set → attempting internet download ...", flush=True)
        print(f"  [model] ⚠  If this hangs, the cluster has no internet. Use --weights_dir instead.", flush=True)

    t0 = time.time()

    if key == "hybrid_ecsanet":
        model = HybridECSAnet(
            num_classes=num_classes,
            weights_dir=weights_dir,
            cnn_backbone=cnn_backbone,
            vit_branch=vit_branch,
            use_cbam=use_cbam,
            fusion_dim=fusion_dim,
        )
    elif proj_dim > 0 and key == "ecsamet":
        model = MODEL_REGISTRY[key](num_classes=num_classes, weights_dir=weights_dir,
                                    proj_dim=proj_dim)
    else:
        model = MODEL_REGISTRY[key](num_classes=num_classes, weights_dir=weights_dir)
    elapsed = time.time() - t0

    # Count parameters
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [model] ✓ '{key}' ready in {elapsed:.1f}s  |  "
        f"params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable",
        flush=True
    )

    return model