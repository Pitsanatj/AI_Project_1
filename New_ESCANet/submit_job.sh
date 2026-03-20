#!/bin/bash
# ============================================================
#  SLURM Job Script สำหรับ Lenovo LiCO
#  ECSAnet Training — BreakHis Dataset
# ============================================================
#
# วิธี submit:
#
#   ครั้งแรก (fresh):
#     sbatch submit_job.sh
#
#   Resume — set RESUME_CHECKPOINT ก่อน sbatch:
#     RESUME_CHECKPOINT="latest" sbatch submit_job.sh
#     RESUME_CHECKPOINT="./ECSAnet_outputs/Checkpoints/40X/checkpoint_epoch_27.pth" sbatch submit_job.sh
#
#   หรือแก้ค่า RESUME_CHECKPOINT ในไฟล์นี้โดยตรงแล้ว sbatch
#
# ============================================================

#SBATCH --job-name=ECSAnet_40X
#SBATCH --output=logs/ecsa_40x_%j.out
#SBATCH --error=logs/ecsa_40x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================================
# ★ แก้ตรงนี้เพื่อ resume ★
#   ""        → เริ่มใหม่ตั้งแต่ต้น
#   "latest"  → หา checkpoint ล่าสุดอัตโนมัติ
#   "path"    → ระบุ .pth โดยตรง
# ============================================================
RESUME_CHECKPOINT=""

# ============================================================
# Environment Setup
# ============================================================

mkdir -p logs

echo "=============================================="
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURMD_NODENAME"
echo "  Start     : $(date)"
echo "  Work dir  : $(pwd)"
echo "  Resume    : ${RESUME_CHECKPOINT:-none (fresh training)}"
echo "=============================================="

# module อาจ error ใน singularity container — ข้ามได้ปลอดภัย
module purge 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# ── Activate environment ─────────────────────────────────────
# source ./venv/bin/activate
# conda activate ecsa_env

nvidia-smi
echo ""

# ============================================================
# Path Configuration
# ============================================================

SPLIT_DIR="./MyBreakHis_Split"
TARGET_IMAGE="./target_image.png"
WEIGHTS_DIR="./pretrained_weights"
OUTPUT_DIR="./ECSAnet_outputs"
MAGNIFICATION="100X"

# ============================================================
# Build --resume flag
# ============================================================
RESUME_FLAG=""
if [ -n "${RESUME_CHECKPOINT}" ]; then
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
    echo "[submit] Resume mode  : ${RESUME_CHECKPOINT}"
else
    echo "[submit] Fresh training (no resume)"
fi

# ============================================================
# Run Training
# ============================================================
echo "Starting ECSAnet training for ${MAGNIFICATION}..."
echo ""

python train.py \
    --split_dir      "$SPLIT_DIR" \
    --target_image   "$TARGET_IMAGE" \
    --weights_dir    "$WEIGHTS_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --magnification  "$MAGNIFICATION" \
    --epochs         50 \
    --batch_size     16 \
    --lr             0.001 \
    --num_workers    2 \
    --patience       25 \
    --num_classes    8 \
    --seed           42 \
    $RESUME_FLAG

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  End time  : $(date)"
echo "  Exit code : $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE