"""
split_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
แบ่งข้อมูล BreakHis dataset เป็น train / val / test
แบบ random split ระดับ image (seed=42)

โครงสร้าง input:
  breakhis_raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
  ├── benign/SOB/adenosis/          → class: A
  ├── benign/SOB/fibroadenoma/      → class: F
  ├── benign/SOB/phyllodes_tumor/   → class: PT
  ├── benign/SOB/tubular_adenoma/   → class: TA
  ├── malignant/SOB/ductal_carcinoma/    → class: DC
  ├── malignant/SOB/lobular_carcinoma/   → class: LC
  ├── malignant/SOB/mucinous_carcinoma/  → class: MC
  └── malignant/SOB/papillary_carcinoma/ → class: PC

โครงสร้าง output:
  MyBreakHis_Split/
  ├── train/ 40X/ {A, DC, F, LC, MC, PC, PT, TA}/
  ├── val/   40X/ {A, DC, F, LC, MC, PC, PT, TA}/
  ├── test/  40X/ {A, DC, F, LC, MC, PC, PT, TA}/
  └── (เหมือนกันสำหรับ 100X, 200X, 400X)

Usage:
  python split_data.py
  python split_data.py --raw_dir ../breakhis_raw/.../breast --out_dir ./MyBreakHis_Split
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42

CLASS_MAP = {
    "adenosis":           "A",
    "fibroadenoma":       "F",
    "phyllodes_tumor":    "PT",
    "tubular_adenoma":    "TA",
    "ductal_carcinoma":   "DC",
    "lobular_carcinoma":  "LC",
    "mucinous_carcinoma": "MC",
    "papillary_carcinoma":"PC",
}

MAGNIFICATIONS  = ["40X", "100X", "200X", "400X"]
IMG_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def collect_images(raw_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """
    Return: { class_label: { mag: [image_path, ...] } }
    สแกนทุก image ใต้ breast/ โดยตรง ไม่ยุ่งกับ patient level
    """
    breast_dir = raw_dir
    if not (breast_dir / "benign").exists():
        candidates = list(breast_dir.rglob("breast"))
        if candidates:
            breast_dir = candidates[0]
            print(f"  Found breast/ at: {breast_dir}")
        else:
            raise FileNotFoundError(
                f"Cannot find benign/malignant under: {raw_dir}\n"
                "Set --raw_dir to the 'breast' folder."
            )

    data: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for category in ["benign", "malignant"]:
        cat_dir = breast_dir / category / "SOB"
        if not cat_dir.exists():
            print(f"  [warn] Missing: {cat_dir}")
            continue

        for subtype_dir in sorted(cat_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            label = CLASS_MAP.get(subtype_dir.name.lower())
            if label is None:
                print(f"  [warn] Unknown subtype '{subtype_dir.name}', skipping.")
                continue

            # วนทุก patient folder → ทุก mag folder → เก็บ image paths
            for patient_dir in sorted(subtype_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                for mag in MAGNIFICATIONS:
                    mag_dir = patient_dir / mag
                    if not mag_dir.exists():
                        continue
                    for p in sorted(mag_dir.iterdir()):
                        if is_image(p):
                            data[label][mag].append(p)

    return data


def split_and_copy(
    data: dict[str, dict[str, list[Path]]],
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    dry_run: bool,
) -> None:
    summary: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    )

    for label in sorted(data.keys()):
        for mag in MAGNIFICATIONS:
            images = data[label].get(mag, [])
            if not images:
                continue

            # ── Split: train / (val+test) ──────────────────────────────────
            train_imgs, valtest_imgs = train_test_split(
                images,
                test_size=(val_ratio + test_ratio),
                random_state=SEED,
            )
            # ── Split: val / test ──────────────────────────────────────────
            val_imgs, test_imgs = train_test_split(
                valtest_imgs,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=SEED,
            )

            for split_name, split_imgs in [
                ("train", train_imgs),
                ("val",   val_imgs),
                ("test",  test_imgs),
            ]:
                dest = out_dir / split_name / mag / label
                if not dry_run:
                    dest.mkdir(parents=True, exist_ok=True)
                for src in split_imgs:
                    if not dry_run:
                        shutil.copy2(src, dest / src.name)
                summary[label][mag][split_name] += len(split_imgs)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary — image counts per split")
    print("=" * 60)

    for mag in MAGNIFICATIONS:
        has_data = any(mag in summary[lbl] for lbl in summary)
        if not has_data:
            continue
        print(f"\n  Magnification: {mag}")
        print(f"  {'Class':<6} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print(f"  {'-'*40}")
        tot_tr = tot_vl = tot_ts = 0
        for label in sorted(summary.keys()):
            c = summary[label].get(mag, {})
            tr, vl, ts = c.get("train", 0), c.get("val", 0), c.get("test", 0)
            tot_tr += tr; tot_vl += vl; tot_ts += ts
            print(f"  {label:<6} {tr:>8} {vl:>8} {ts:>8} {tr+vl+ts:>8}")
        print(f"  {'TOTAL':<6} {tot_tr:>8} {tot_vl:>8} {tot_ts:>8} "
              f"{tot_tr+tot_vl+tot_ts:>8}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split BreakHis into train/val/test (random image-level, seed=42)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        default="../breakhis_raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
        help="Path ไปยัง breast/ folder",
    )
    parser.add_argument(
        "--out_dir",
        default="./MyBreakHis_Split",
        help="Output directory",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.20)
    parser.add_argument("--test_ratio",  type=float, default=0.10)
    parser.add_argument("--dry_run", action="store_true",
                        help="แสดงผลโดยไม่ copy ไฟล์จริง")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train + val + test must sum to 1.0"

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    print("=" * 60)
    print("  BreakHis Splitter  (random image-level, seed=42)")
    print("=" * 60)
    print(f"  Raw data  : {raw_dir.resolve()}")
    print(f"  Output    : {out_dir.resolve()}")
    print(f"  Splits    : train={args.train_ratio:.0%}  "
          f"val={args.val_ratio:.0%}  test={args.test_ratio:.0%}")
    print(f"  Dry run   : {args.dry_run}\n")

    print("Scanning dataset...")
    data = collect_images(raw_dir)

    if not data:
        print("ERROR: No images found. Check --raw_dir.")
        return

    total_imgs = sum(len(imgs) for lbl in data.values() for imgs in lbl.values())
    print(f"Found {len(data)} classes, {total_imgs} images total\n")

    split_and_copy(data, out_dir, args.train_ratio, args.val_ratio,
                   args.test_ratio, args.dry_run)

    if args.dry_run:
        print("\n⚠ Dry run — no files copied.")
    else:
        print(f"\n✅ Done! Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()