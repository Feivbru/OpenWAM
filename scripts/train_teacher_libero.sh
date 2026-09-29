#!/usr/bin/env bash
# Train OpenWAM FPD teacher (clean video + action only) on LIBERO.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_teacher_libero.sh
#   bash scripts/train_teacher_libero.sh training.debug=true training.batch_size=4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export OPENWAM_WAN22_TI2V_5B="${OPENWAM_WAN22_TI2V_5B:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
WAN_PATH="$OPENWAM_WAN22_TI2V_5B"

CFG=train_teacher_libero

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29584}"
LOG="logs/teacher_libero_$(date +%Y%m%d_%H%M%S).log"

echo "Teacher train: config=${CFG} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "  log=${LOG}"
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name="${CFG}" \
  model.video_backbone.model_path="${WAN_PATH}" \
  project.wandb.mode="${WANDB_MODE}" \
  "$@" 2>&1 | tee "${LOG}"
