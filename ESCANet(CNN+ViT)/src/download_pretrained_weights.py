"""
src/download_pretrained_weights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download all pretrained backbone weights used by this project to a local
directory.  Run ONCE on a machine with internet access, then copy the
directory to the cloud server.

Usage
-----
python src/download_pretrained_weights.py
python src/download_pretrained_weights.py --weights_dir ./pretrained_weights

After copying to server, train offline:
    python main.py --data_dir ./npy_data --weights_dir ./pretrained_weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# ── timm backbones ─────────────────────────────────────────────────────────────
TIMM_MODELS = [
    "swin_tiny_patch4_window7_224",
    "deit_small_patch16_224",
    "mobilevit_s",
    "fastvit_sa12",
    "fastvit_t8",
]


def download_timm(out_dir: Path) -> None:
    import timm

    for name in TIMM_MODELS:
        dst = out_dir / f"{name}.pth"
        if dst.exists():
            print(f"  [skip] {name}  (already saved)")
            continue
        print(f"  Downloading timm/{name} …", flush=True)
        model = timm.create_model(name, pretrained=True)
        torch.save(model.state_dict(), dst)
        del model
        print(f"         → {dst}")


# ── torchvision backbones ──────────────────────────────────────────────────────
def download_torchvision(out_dir: Path) -> None:
    from torchvision.models import (
        efficientnet_v2_s,   EfficientNet_V2_S_Weights,
        efficientnet_v2_m,   EfficientNet_V2_M_Weights,
        vit_b_16,            ViT_B_16_Weights,
        mobilenet_v3_large,  MobileNet_V3_Large_Weights,
        shufflenet_v2_x1_0,  ShuffleNet_V2_X1_0_Weights,
    )

    tv_models = [
        ("efficientnet_v2_s",  efficientnet_v2_s,   EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        ("efficientnet_v2_m",  efficientnet_v2_m,   EfficientNet_V2_M_Weights.IMAGENET1K_V1),
        ("vit_b_16",           vit_b_16,            ViT_B_16_Weights.IMAGENET1K_V1),
        ("mobilenet_v3_large", mobilenet_v3_large,  MobileNet_V3_Large_Weights.IMAGENET1K_V2),
        ("shufflenet_v2_x1_0", shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    ]

    for name, constructor, weights in tv_models:
        dst = out_dir / f"{name}.pth"
        if dst.exists():
            print(f"  [skip] {name}  (already saved)")
            continue
        print(f"  Downloading torchvision/{name} …", flush=True)
        model = constructor(weights=weights)
        torch.save(model.state_dict(), dst)
        del model
        print(f"         → {dst}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pretrained backbone weights for offline deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights_dir", default="./pretrained_weights",
        help="Directory where .pth files will be saved",
    )
    args = parser.parse_args()

    out_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving weights to: {out_dir.resolve()}\n")

    print("── timm models ──────────────────────────────────────────────────────")
    download_timm(out_dir)

    print("\n── torchvision models ───────────────────────────────────────────────")
    download_torchvision(out_dir)

    print(f"\nAll done.  Copy '{out_dir}' to your server and run:")
    print(f"  python main.py --data_dir ./npy_data --weights_dir {args.weights_dir}")


if __name__ == "__main__":
    main()
