#!/usr/bin/env bash
# Train OpenWAM FPD (student + frozen teacher) on LIBERO.
#
# Usage:
#   TEACHER_CKPT=/path/to/teacher_run bash scripts/train_fpd_libero.sh
#   Extra Hydra overrides are forwarded, e.g.:
#     bash scripts/train_fpd_libero.sh training.batch_size=1 training.debug=true
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

CFG=train_fpd_libero

TEACHER_CKPT="${TEACHER_CKPT:-}"
if [[ -z "${TEACHER_CKPT}" ]]; then
  echo "TEACHER_CKPT is required (teacher run dir with checkpoint_step_*.safetensors)" >&2
  exit 1
fi
if [[ ! -d "${TEACHER_CKPT}" ]]; then
  echo "Teacher ckpt dir not found: ${TEACHER_CKPT}" >&2
  exit 1
fi
if ! compgen -G "${TEACHER_CKPT}/checkpoint_step_*.safetensors" >/dev/null; then
  echo "No checkpoint_step_*.safetensors under ${TEACHER_CKPT}" >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29585}"
LOG="logs/fpd_libero_$(date +%Y%m%d_%H%M%S).log"

echo "FPD train: config=${CFG} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "  teacher=${TEACHER_CKPT}"
echo "  wandb=${WANDB_MODE}"
echo "  log=${LOG}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name="${CFG}" \
  model.video_backbone.model_path="${WAN_PATH}" \
  fpd.teacher_ckpt_path="${TEACHER_CKPT}" \
  project.wandb.mode="${WANDB_MODE}" \
  "$@" 2>&1 | tee "${LOG}"
