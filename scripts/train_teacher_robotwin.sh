#!/usr/bin/env bash
# Train OpenWAM FPD teacher (clean video + action only) from Full / Clean2Random.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6,7 bash scripts/train_teacher_robotwin.sh
#   VARIANT=clean2random bash scripts/train_teacher_robotwin.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="4,5,6,7"
# Helps when DeepSpeed still allocates large temporary buffers (matches the
# hint in torch OOM messages). FastWAM mainly relies on contiguous_gradients=false.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="online"
export WANDB_API_KEY="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"
export OPENWAM_WAN22_TI2V_5B="${OPENWAM_WAN22_TI2V_5B:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
WAN_PATH="$OPENWAM_WAN22_TI2V_5B"

VARIANT="${VARIANT:-full}"
case "$VARIANT" in
  full) CFG=train_teacher_robotwin_full ;;
  clean2random|c2r) CFG=train_teacher_robotwin_clean2random ;;
  *) echo "Unknown VARIANT=${VARIANT} (full|clean2random)" >&2; exit 1 ;;
esac

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29574}"

echo "Teacher train: config=${CFG} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name="${CFG}" \
  model.video_backbone.model_path="${WAN_PATH}" \
  "$@" 2>&1 | tee logs/teacher_full_$(date +%Y%m%d_%H%M%S).log
