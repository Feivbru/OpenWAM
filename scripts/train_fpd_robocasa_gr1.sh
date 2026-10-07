#!/usr/bin/env bash
# Train OpenWAM FPD (student + frozen teacher) on RoboCasa-GR1.
#
# Usage:
#   TEACHER_CKPT=outputs/openwam_teacher_robocasa_gr1/<run> \
#     CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC=6 CONDA_ENV=openwam_re \
#     bash scripts/train_fpd_robocasa_gr1.sh
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

CFG=train_fpd_robocasa_gr1
DATASET_DIR="${DATASET_DIR:-${ROOT}/assets/benchmark_data/robocasa-gr1}"
T5_CACHE_DIRNAME="${T5_CACHE_DIRNAME:-umt5_openwam}"
OPENWAM_CKPT="${OPENWAM_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1}"
TEACHER_ROOT="${ROOT}/outputs/openwam_teacher_robocasa_gr1"

TEACHER_CKPT="${TEACHER_CKPT:-}"
if [[ -z "${TEACHER_CKPT}" ]]; then
  # Newest run dir under outputs that already has a safetensors checkpoint.
  TEACHER_CKPT="$(
    find "${TEACHER_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | while read -r _ts dir; do
          if compgen -G "${dir}/checkpoint_step_*.safetensors" >/dev/null; then
            echo "${dir}"
            break
          fi
        done
  )"
fi
if [[ "${TEACHER_CKPT}" != /* && -n "${TEACHER_CKPT}" ]]; then
  TEACHER_CKPT="${ROOT}/${TEACHER_CKPT}"
fi
if [[ -z "${TEACHER_CKPT}" ]]; then
  echo "TEACHER_CKPT is required (or produce a run under ${TEACHER_ROOT})" >&2
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

CONDA_ENV="${CONDA_ENV:-openwam_re}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29591}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/fpd_robocasa_gr1_${STAMP}.log"
FIRST_GPU="${GPU_ARR[0]}"

need_t5=0
for bucket in "${DATASET_DIR}"/gr1_*; do
  [[ -d "${bucket}" ]] || continue
  if [[ ! -d "${bucket}/meta/${T5_CACHE_DIRNAME}" ]] \
    || ! compgen -G "${bucket}/meta/${T5_CACHE_DIRNAME}/task_*.pt" >/dev/null; then
    need_t5=1
    break
  fi
done
if [[ "${need_t5}" -eq 1 || "${FORCE_T5_PRECOMPUTE:-0}" == "1" ]]; then
  echo "Precomputing RoboCasa-GR1 T5 cache on GPU ${FIRST_GPU} ..."
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python scripts/precompute_robocasa_gr1_t5_embeds.py \
    --dataset_dir "${DATASET_DIR}" \
    --wan_path "${WAN_PATH}" \
    --openwam_ckpt "${OPENWAM_CKPT}" \
    --cache_dirname "${T5_CACHE_DIRNAME}"
fi

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
