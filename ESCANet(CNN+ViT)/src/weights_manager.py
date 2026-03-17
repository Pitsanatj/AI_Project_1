"""
src/weights_manager.py
━━━━━━━━━━━━━━━━━━━━━━
Offline-friendly pretrained weight loading.

Workflow
--------
1. On a machine WITH internet, download all backbone weights once:
       python src/download_pretrained_weights.py --weights_dir ./pretrained_weights

2. Copy ``pretrained_weights/`` to your cloud server.

3. Train without internet:
       python main.py --data_dir ./npy_data --weights_dir ./pretrained_weights

If ``weights_dir`` is None the original online behaviour is preserved.
"""

from __future__ import annotations

from pathlib import Path
import torch

# ── Expected filename for each backbone ───────────────────────────────────────
TIMM_FILES: dict[str, str] = {
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.pth",
    "deit_small_patch16_224":       "deit_small_patch16_224.pth",
    "mobilevit_s":                  "mobilevit_s.pth",
    "fastvit_sa12":                 "fastvit_sa12.pth",
    "fastvit_t8":                   "fastvit_t8.pth",
}

TV_FILES: dict[str, str] = {
    "efficientnet_v2_s":   "efficientnet_v2_s.pth",
    "efficientnet_v2_m":   "efficientnet_v2_m.pth",   # HybridECSAnet M backbone
    "vit_b_16":            "vit_b_16.pth",             # HybridECSAnet ViT-B/16 branch
    "mobilenet_v3_large":  "mobilenet_v3_large.pth",
    "shufflenet_v2_x1_0":  "shufflenet_v2_x1_0.pth",
}


def _resolve(weights_dir: str, filename: str) -> Path:
    path = Path(weights_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Local pretrained weights not found: {path}\n"
            f"Download them first (internet required):\n"
            f"  python src/download_pretrained_weights.py "
            f"--weights_dir {weights_dir}"
        )
    return path


def load_timm_model(model_name: str, weights_dir: str | None, **create_kwargs):
    """
    Create a timm model and load pretrained weights.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required:  pip install timm") from exc

    if weights_dir is None:
        print(f"  [weights] Downloading '{model_name}' from internet (timm) ...", flush=True)
        print(f"  [weights] ⚠  This may hang if the cluster has no internet access!", flush=True)
        model = timm.create_model(model_name, pretrained=True, **create_kwargs)
        print(f"  [weights] ✓ '{model_name}' downloaded OK", flush=True)
        return model

    path = _resolve(weights_dir, TIMM_FILES[model_name])
    print(f"  [weights] Loading '{model_name}' from local file: {path}", flush=True)
    model = timm.create_model(model_name, pretrained=False, **create_kwargs)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Filter out keys whose shape doesn't match the current model (e.g. the
    # 1000-class ImageNet head vs. our num_classes head).  strict=False alone
    # skips missing/unexpected keys but still raises on shape mismatches.
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in state_dict if k not in filtered]
    if skipped:
        print(f"  [weights] Skipped {len(skipped)} mismatched key(s): {skipped}", flush=True)

    model.load_state_dict(filtered, strict=False)
    print(f"  [weights] ✓ '{model_name}' loaded from local file OK", flush=True)
    return model


def load_torchvision_backbone(constructor, model_key: str, online_weights,
                               weights_dir: str | None):
    """
    Create a torchvision model and load pretrained weights.
    """
    if weights_dir is None:
        print(f"  [weights] Downloading '{model_key}' from internet (torchvision) ...", flush=True)
        print(f"  [weights] ⚠  This may hang if the cluster has no internet access!", flush=True)
        model = constructor(weights=online_weights)
        print(f"  [weights] ✓ '{model_key}' downloaded OK", flush=True)
        return model

    path = _resolve(weights_dir, TV_FILES[model_key])
    print(f"  [weights] Loading '{model_key}' from local file: {path}", flush=True)
    model = constructor(weights=None)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  [weights] ✓ '{model_key}' loaded from local file OK", flush=True)
    return model