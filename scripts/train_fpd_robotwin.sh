#!/usr/bin/env bash
# Train OpenWAM FPD (student + frozen teacher) on RoboTwin Full / Clean2Random.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6,7 NPROC=2 bash scripts/train_fpd_robotwin.sh
#   CUDA_VISIBLE_DEVICES=6,7 NPROC=2 VARIANT=c2r CONDA_ENV=openwam_re \
#     TEACHER_CKPT=outputs/openwam_teacher_robotwin_clean2random/2026-10-02_21-29-18 \
#     bash scripts/train_fpd_robotwin.sh
#   Extra Hydra overrides are forwarded, e.g.:
#     bash scripts/train_fpd_robotwin.sh training.batch_size=2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
# expandable_segments:True correlates with unfrozen-DiT NaN on this H20 stack.
unset PYTORCH_CUDA_ALLOC_CONF
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export OPENWAM_WAN22_TI2V_5B="${OPENWAM_WAN22_TI2V_5B:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
WAN_PATH="$OPENWAM_WAN22_TI2V_5B"

VARIANT="${VARIANT:-full}"
case "$VARIANT" in
  full)
    CFG=train_fpd_robotwin_full
    DEFAULT_TEACHER="${ROOT}/outputs/openwam_teacher_robotwin_full/2026-09-24_17-41-21"
    ;;
  clean2random|c2r)
    CFG=train_fpd_robotwin_clean2random
    DEFAULT_TEACHER="${ROOT}/outputs/openwam_teacher_robotwin_clean2random/2026-10-02_21-29-18"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT} (full|clean2random)" >&2
    exit 1
    ;;
esac

TEACHER_CKPT="${TEACHER_CKPT:-$DEFAULT_TEACHER}"
# Allow relative paths from ROOT.
if [[ "${TEACHER_CKPT}" != /* ]]; then
  TEACHER_CKPT="${ROOT}/${TEACHER_CKPT}"
fi
if [[ -z "${TEACHER_CKPT}" ]]; then
  echo "TEACHER_CKPT is required for VARIANT=${VARIANT}" >&2
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

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29575}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/fpd_${VARIANT}_${STAMP}.log"

echo "FPD train: config=${CFG} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "  teacher=${TEACHER_CKPT}"
echo "  wandb=${WANDB_MODE} log=${LOG}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-<UNSET>}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name="${CFG}" \
  model.video_backbone.model_path="${WAN_PATH}" \
  fpd.teacher_ckpt_path="${TEACHER_CKPT}" \
  project.wandb.mode="${WANDB_MODE}" \
  "$@" 2>&1 | tee "${LOG}"

echo "FPD train done. log=${LOG}"
