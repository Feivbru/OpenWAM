#!/usr/bin/env bash
# Train OpenWAM FPD teacher (clean video + action only) on EBench.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC=6 bash scripts/train_teacher_ebench.sh
#   Extra Hydra overrides are forwarded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
unset PYTORCH_CUDA_ALLOC_CONF
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export OPENWAM_WAN22_TI2V_5B="${OPENWAM_WAN22_TI2V_5B:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
WAN_PATH="$OPENWAM_WAN22_TI2V_5B"

CFG=train_teacher_ebench
DATASET_DIR="${DATASET_DIR:-${ROOT}/assets/benchmark_data/ebench}"
OPENWAM_CKPT="${OPENWAM_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench}"

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29592}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/teacher_ebench_${STAMP}.log"

if [[ ! -d "${OPENWAM_CKPT}" ]]; then
  echo "Missing student/teacher warm-start dir: ${OPENWAM_CKPT}" >&2
  echo "Download OpenWAM_Alpha → OpenWAM-Alpha-Sim-EBench, or set OPENWAM_CKPT." >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing EBench dataset dir: ${DATASET_DIR}" >&2
  exit 1
fi

echo "Teacher train: config=${CFG} gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "  ckpt=${OPENWAM_CKPT}"
echo "  data=${DATASET_DIR}"
echo "  wandb=${WANDB_MODE} log=${LOG}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-<UNSET>}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name="${CFG}" \
  model.video_backbone.model_path="${WAN_PATH}" \
  training.finetune_ckpt_path="${OPENWAM_CKPT}" \
  dataloader.dataset_dir="${DATASET_DIR}" \
  project.wandb.mode="${WANDB_MODE}" \
  "$@" 2>&1 | tee "${LOG}"

echo "Teacher train done. log=${LOG}"
