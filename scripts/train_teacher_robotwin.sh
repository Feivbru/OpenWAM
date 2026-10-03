#!/usr/bin/env bash
# Train OpenWAM FPD teacher (clean video + action only) from Full / Clean2Random.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC=6 bash scripts/train_teacher_robotwin.sh
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC=6 VARIANT=c2r bash scripts/train_teacher_robotwin.sh
#   CONDA_ENV=openwam_re VARIANT=c2r NPROC=6 bash scripts/train_teacher_robotwin.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
# expandable_segments:True correlates with unfrozen-DiT NaN on this H20 stack.
unset PYTORCH_CUDA_ALLOC_CONF
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export OPENWAM_WAN22_TI2V_5B="${OPENWAM_WAN22_TI2V_5B:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
WAN_PATH="$OPENWAM_WAN22_TI2V_5B"

VARIANT="${VARIANT:-full}"
case "$VARIANT" in
  full) CFG=train_teacher_robotwin_full; LOG_TAG=teacher_full ;;
  clean2random|c2r) CFG=train_teacher_robotwin_clean2random; LOG_TAG=teacher_c2r ;;
  *) echo "Unknown VARIANT=${VARIANT} (full|clean2random)" >&2; exit 1 ;;
esac

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29574}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/${LOG_TAG}_${STAMP}.log"

echo "Teacher train: config=${CFG} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "  wandb=${WANDB_MODE} log=${LOG}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-<UNSET>}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name="${CFG}" \
  model.video_backbone.model_path="${WAN_PATH}" \
  project.wandb.mode="${WANDB_MODE}" \
  "$@" 2>&1 | tee "${LOG}"

echo "Teacher train done. log=${LOG}"
