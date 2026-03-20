"""
download_pretrained_weights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download pretrained weights สำหรับ ECSAnet (EfficientNetV2-S backbone)
รัน ONCE บนเครื่องที่มี internet แล้ว copy ไปยัง LiCO server

Usage
-----
  python download_pretrained_weights.py
  python download_pretrained_weights.py --weights_dir ./pretrained_weights

หลัง copy ไปยัง LiCO แล้ว train แบบ offline:
  python train.py --weights_dir ./pretrained_weights --magnification 40X
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# ── torchvision backbones สำหรับ ECSAnet ──────────────────────────────────────
# เพิ่ม model อื่นได้ถ้าต้องการ compare กัน

def download_torchvision(out_dir: Path) -> None:
    from torchvision.models import (
        efficientnet_v2_s, EfficientNet_V2_S_Weights,
        efficientnet_v2_m, EfficientNet_V2_M_Weights,
    )

    tv_models = [
        (
            "efficientnet_v2_s",
            efficientnet_v2_s,
            EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        ),
        (
            "efficientnet_v2_m",
            efficientnet_v2_m,
            EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        ),
    ]

    for name, constructor, weights in tv_models:
        dst = out_dir / f"{name}.pth"
        if dst.exists():
            print(f"  [skip] {name}  (already saved at {dst})")
            continue
        print(f"  Downloading torchvision/{name} ...", flush=True)
        model = constructor(weights=weights)
        torch.save(model.state_dict(), dst)
        del model
        print(f"         → {dst}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained backbone weights for ECSAnet (offline deployment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights_dir",
        default="./pretrained_weights",
        help="Directory ที่จะบันทึกไฟล์ .pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  ECSAnet — Pretrained Weights Downloader")
    print("=" * 55)
    print(f"  Saving to: {out_dir.resolve()}\n")

    print("── torchvision models ───────────────────────────────")
    download_torchvision(out_dir)

    print(f"\n✅ Done.  ไฟล์ทั้งหมดบันทึกที่: {out_dir.resolve()}")
    print("\nขั้นตอนถัดไป:")
    print(f"  1. Copy folder ไปยัง LiCO:")
    print(f"     scp -r {out_dir} user@lico-hostname:~/ECSAnet/")
    print(f"  2. Train แบบ offline:")
    print(f"     python train.py --weights_dir {args.weights_dir} --magnification 40X")


if __name__ == "__main__":
    main()