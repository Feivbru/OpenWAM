#!/usr/bin/env bash
# Precompute LIBERO UMT5 caches (task_N.pt under meta/umt5_openwam/).
#
# Smoke (first 4 tasks):
#   CUDA_VISIBLE_DEVICES=2 bash scripts/precompute_libero_t5_embeds.sh smoke
# Full (~40 tasks):
#   CUDA_VISIBLE_DEVICES=2 bash scripts/precompute_libero_t5_embeds.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29582}"

MODE="${1:-full}"
EXTRA=()
if [[ "$MODE" == "smoke" ]]; then
  EXTRA+=(--max_tasks 4)
  shift || true
fi
EXTRA+=("$@")

echo "[precompute_libero_t5] gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} mode=${MODE}"
if [[ "${NPROC}" -le 1 ]]; then
  python scripts/precompute_libero_t5_embeds.py --skip_existing "${EXTRA[@]}"
else
  torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    scripts/precompute_libero_t5_embeds.py \
    --skip_existing \
    "${EXTRA[@]}"
fi
