"""
install_requirements.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ตรวจสอบและติดตั้ง dependencies สำหรับ ECSAnet บน LiCO
รองรับ Singularity container (pytorch_25_04_py3.sif / Python 3.12)

ปัญหาที่พบและการแก้ไข:
  1. spams==2.6.5.4 build ล้มเหลวบน Python 3.12
     เพราะ numpy.distutils ถูกลบออกตั้งแต่ numpy>=1.24
     → ใช้ spams-bin (pre-compiled wheel) แทน
  2. Container มี PyTorch + CUDA ครบแล้ว → ข้ามการติดตั้ง
  3. staintools ต้องการ opencv → container มีให้แล้ว

Usage:
  python install_requirements.py
  python install_requirements.py --skip_spams   (ถ้าไม่ใช้ stain norm)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
import sys

# ── [HOTFIX] ล้าง OpenCV GUI ออกจาก .local ก่อนเริ่มงาน ──────────────────────
# def fix_opencv_issue():
#     print("=" * 60)
#     print("  HOTFIX: Removing problematic OpenCV (GUI) from .local...")
#     print("=" * 60)
#     # ลบตัวเก่าที่พัง (ต้องใช้ --user เพื่อให้แน่ใจว่าลบใน home ของเรา)
#     pkgs_to_remove = ["opencv-python", "opencv-contrib-python", "opencv-python-headless"]
#     for pkg in pkgs_to_remove:
#         subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], check=False)
    
#     # ติดตั้งตัว Headless ลงไปใหม่ที่ .local
#     print("\n  Installing opencv-python-headless...")
#     subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--user", "--quiet"], check=False)
#     print("  ✓ OpenCV Headless fix applied.\n")

# เรียกใช้ทันทีที่รันสคริปต์
fix_opencv_issue()
# ── Package list ──────────────────────────────────────────────────────────────
#
# CONTAINER_PACKAGES  — มีใน container แล้ว ตรวจสอบเท่านั้น ไม่ติดตั้งซ้ำ
# INSTALL_PACKAGES    — ต้องติดตั้งเพิ่ม (ไม่มีใน container)
# SPAMS_VARIANTS      — ลองตามลำดับจนกว่าจะสำเร็จ (Python 3.12 compat)

CONTAINER_PACKAGES: list[tuple[str, str]] = [
    ("torch",       "PyTorch (container)"),
    ("torchvision", "torchvision (container)"),
    ("PIL",         "Pillow (container)"),
    ("cv2",         "opencv-python-headless (container)"),
    ("numpy",       "numpy (container)"),
    ("sklearn",     "scikit-learn (container)"),
    ("matplotlib",  "matplotlib (container)"),
    ("seaborn",     "seaborn (container)"),
    ("tensorboard", "tensorboard (container)"),
    ("tqdm",        "tqdm (container)"),
]

INSTALL_PACKAGES: list[tuple[str, str]] = [
    ("staintools", "staintools"),
]

# ลำดับความพยายามติดตั้ง spams:
#   1. spams-bin  — pre-compiled wheel รองรับ Python 3.12, ไม่ต้องการ numpy.distutils
#   2. spams      — source build (ใช้ได้บน Python <=3.11 เท่านั้น)
SPAMS_INSTALL_ORDER: list[tuple[str, str]] = [
    ("spams", "spams-bin"),
    ("spams", "spams"),
]


# ── Core helpers ──────────────────────────────────────────────────────────────

def is_installed(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def install_package(import_name: str, pip_name: str) -> bool:
    """
    ติดตั้ง package และ hot-load เข้า environment ปัจจุบัน
    Returns True ถ้าสำเร็จ
    """
    if is_installed(import_name):
        print(f"  ✓ {import_name:<20} already installed")
        return True

    print(f"  ↓ {import_name:<20} not found — installing '{pip_name}' ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--user", "--quiet"],
            stdout=sys.stdout, stderr=sys.stderr,
        )
        print(f"    ✓ Installed '{pip_name}'")
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Failed to install '{pip_name}': {e}")
        return False

    # hot-load เข้า current interpreter
    spec = importlib.util.find_spec(import_name)
    if spec:
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[import_name] = module
            print(f"    ✓ Loaded '{import_name}' into current session")
        except Exception as exc:
            # บาง package เช่น staintools ต้อง restart — ไม่ใช่ error จริง
            print(f"    ~ '{import_name}' installed; will be available on next run "
                  f"({exc})")
        return True

    print(f"    ✗ Installed but cannot locate '{import_name}'")
    return False


def install_spams() -> bool:
    """
    ลอง install spams ตามลำดับ SPAMS_INSTALL_ORDER
    จนกว่าจะสำเร็จ 1 ตัว

    Python 3.12 issue:
      numpy >=1.24 ลบ numpy.distutils ออก
      spams (source) ต้องการ numpy.distutils → ModuleNotFoundError
      spams-bin คือ pre-compiled wheel ที่ไม่ต้องการ numpy.distutils
    """
    if is_installed("spams"):
        print(f"  ✓ {'spams':<20} already installed")
        return True

    python_ver = sys.version_info
    print(f"  ↓ {'spams':<20} not found")
    print(f"    Python {python_ver.major}.{python_ver.minor} detected", end="")

    order = list(SPAMS_INSTALL_ORDER)
    if python_ver < (3, 12):
        print(" → trying spams (source) first")
        order = list(reversed(order))
    else:
        print(" → numpy.distutils removed → trying spams-bin first")

    for import_name, pip_name in order:
        print(f"    Trying: pip install {pip_name} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name,
                 "--user", "--quiet"],
                stdout=sys.stdout, stderr=sys.stderr,
            )
            if is_installed(import_name):
                print(f"    ✓ spams installed via '{pip_name}'")
                return True
        except subprocess.CalledProcessError:
            print(f"    ✗ '{pip_name}' failed, trying next ...")
            continue

    print("\n  ✗ spams installation failed with all methods.")
    print("    Manual options:")
    print("      pip install spams-bin --user")
    print("      conda install -c conda-forge spams")
    print("    หรือรัน: python install_requirements.py --skip_spams")
    return False


# ── Verification ──────────────────────────────────────────────────────────────

def verify_container_packages() -> bool:
    all_ok = True
    for import_name, label in CONTAINER_PACKAGES:
        if is_installed(import_name):
            ver = ""
            try:
                mod = importlib.import_module(import_name)
                ver = getattr(mod, "__version__", "")
            except Exception:
                pass
            ver_str = f"  v{ver}" if ver else ""
            print(f"  ✓ {import_name:<20} ok{ver_str}  ({label})")
        else:
            print(f"  ✗ {import_name:<20} MISSING from container!")
            all_ok = False
    return all_ok


def verify_torch_details() -> None:
    try:
        import torch
        print(f"\n  PyTorch version : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"  CUDA version    : {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"  GPU [{i}]         : {name}  ({mem} GB)")
        else:
            print("  CUDA            : NOT available")
            print("  ⚠ Training will use CPU — extremely slow for ECSAnet")
    except ImportError:
        print("  ✗ torch import failed — is this a PyTorch container?")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install extra dependencies for ECSAnet inside Singularity container",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip_spams", action="store_true",
        help="ข้ามการติดตั้ง spams (ถ้าไม่ใช้ stain normalization)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  ECSAnet — Dependency Installer")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("=" * 60)

    # ── 1: ตรวจสอบ packages ที่มีใน container ────────────────
    print("\n[1/3] Container packages (verify only — no install)")
    print("-" * 45)
    ok_container = verify_container_packages()

    # ── 2: ติดตั้ง staintools + spams ────────────────────────
    print("\n[2/3] Extra packages (staintools + spams)")
    print("-" * 45)
    ok_staintools = True
    for import_name, pip_name in INSTALL_PACKAGES:
        ok_staintools = install_package(import_name, pip_name) and ok_staintools

    ok_spams = True
    if not args.skip_spams:
        ok_spams = install_spams()
    else:
        print(f"  ~ {'spams':<20} skipped (--skip_spams)")
        print("    ⚠ stain normalization จะไม่ทำงาน — ตรวจสอบ train.py ด้วย")

    # ── 3: PyTorch + GPU summary ─────────────────────────────
    print("\n[3/3] PyTorch + CUDA (from container — no install needed)")
    print("-" * 45)
    verify_torch_details()

    # ── Result ────────────────────────────────────────────────
    all_ok = ok_container and ok_staintools and ok_spams
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All dependencies ready — สามารถรัน split_data.py และ train.py ได้เลย")
    else:
        if not ok_container:
            print("⚠ Container packages บางตัวหายไป — ตรวจสอบ .sif file")
        if not ok_staintools:
            print("⚠ staintools ติดตั้งไม่สำเร็จ")
        if not ok_spams:
            print("⚠ spams ติดตั้งไม่สำเร็จ")
            print("  → ลอง: pip install spams-bin --user")
            print("  → หรือ: python install_requirements.py --skip_spams")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)