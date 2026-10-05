#!/usr/bin/env bash
# One-shot pick_blocks Piper Foundation finetune with VRTC enabled:
#   1) ensure EEF sidecar exists (offline FK; same script as banana)
#   2) precompute single-instruction UMT5 cache (skip loading T5 at train time)
#   3) launch train_pick_blocks_piper_vrtc (vrtc.enabled=true, fu_frames=4)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_pick_blocks_piper_vrtc.sh
#   SKIP_PRECOMPUTE=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_pick_blocks_piper_vrtc.sh
#   CONDA_ENV=openwam SKIP_PRECOMPUTE=1 bash scripts/train_pick_blocks_piper_vrtc.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
# expandable_segments:True correlates with unfrozen-DiT NaN on this H20 stack;
# unset so we never inherit a bad default from the caller environment either.
unset PYTORCH_CUDA_ALLOC_CONF
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

DATASET_DIR="${DATASET_DIR:-/data/zixian_guo/projects/haoming/project/PI/riri/data/pick_blocks}"
WAN_PATH="${OPENWAM_WAN22_TI2V_5B:-${ROOT}/assets/video_backbone_ckpt/Wan2.2-TI2V-5B}"
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
T5_CACHE_DIR="${DATASET_DIR}/meta/umt5_openwam"
EEF_META="${DATASET_DIR}/derived/eef/meta.json"
MANIFEST="${DATASET_DIR}/splits/manifest.json"

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29585}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/pick_blocks_piper_vrtc
LOG="logs/pick_blocks_piper_vrtc/train_${STAMP}.log"

echo "[train_pick_blocks_piper_vrtc] gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} dataset=${DATASET_DIR}"
echo "[train_pick_blocks_piper_vrtc] wan=${WAN_PATH}"
echo "[train_pick_blocks_piper_vrtc] foundation=${FOUNDATION_CKPT}"
echo "[train_pick_blocks_piper_vrtc] log=${LOG}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[train_pick_blocks_piper_vrtc] missing splits/manifest.json at ${MANIFEST}" >&2
  exit 1
fi

# ── 1) EEF sidecar ───────────────────────────────────────────────────────────
if [[ ! -f "${EEF_META}" ]]; then
  echo "[train_pick_blocks_piper_vrtc] missing EEF sidecar; running preprocess_banana_piper_eef.py"
  python scripts/preprocess_banana_piper_eef.py --dataset-dir "${DATASET_DIR}"
else
  echo "[train_pick_blocks_piper_vrtc] EEF sidecar OK: ${EEF_META}"
fi

# ── 2) T5 cache (single instruction) ─────────────────────────────────────────
NEED_T5=0
if [[ "${SKIP_PRECOMPUTE:-0}" != "1" ]]; then
  if [[ ! -d "${T5_CACHE_DIR}" ]] || ! compgen -G "${T5_CACHE_DIR}/task_*.pt" > /dev/null; then
    NEED_T5=1
  fi
fi
if [[ "${FORCE_T5_PRECOMPUTE:-0}" == "1" ]]; then
  NEED_T5=1
fi
if [[ "${NEED_T5}" == "1" ]]; then
  echo "[train_pick_blocks_piper_vrtc] precomputing UMT5 cache -> ${T5_CACHE_DIR}"
  FIRST_GPU="${GPU_ARR[0]}"
  T5_EXTRA=()
  if [[ "${FORCE_T5_PRECOMPUTE:-0}" == "1" ]]; then
    T5_EXTRA+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python scripts/precompute_banana_piper_t5_embeds.py \
    --dataset_dir "${DATASET_DIR}" \
    --wan_path "${WAN_PATH}" \
    --openwam_ckpt "${FOUNDATION_CKPT}" \
    "${T5_EXTRA[@]}"
else
  echo "[train_pick_blocks_piper_vrtc] T5 cache OK: ${T5_CACHE_DIR}"
fi

# ── 3) Train ─────────────────────────────────────────────────────────────────
echo "[train_pick_blocks_piper_vrtc] launching configs/train_pick_blocks_piper_vrtc.yaml"
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name=train_pick_blocks_piper_vrtc \
  model.video_backbone.model_path="${WAN_PATH}" \
  training.finetune_ckpt_path="${FOUNDATION_CKPT}" \
  dataloader.dataset_dir="${DATASET_DIR}" \
  "$@" 2>&1 | tee "${LOG}"

echo "[train_pick_blocks_piper_vrtc] done. log=${LOG}"
