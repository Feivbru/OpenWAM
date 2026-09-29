#!/usr/bin/env bash
# Precompute OpenWAM-wrap RoboTwin T5 caches (seen[:10] -> umt5_openwam/episodeN.pt).
#
# Smoke (GPUs 6,7):
#   CUDA_VISIBLE_DEVICES=6,7 bash scripts/precompute_robotwin_t5_embeds.sh smoke
# Full:
#   CUDA_VISIBLE_DEVICES=6,7 bash scripts/precompute_robotwin_t5_embeds.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29581}"

MODE="${1:-full}"
EXTRA=()
if [[ "$MODE" == "smoke" ]]; then
  EXTRA+=(--tasks adjust_bottle --variants clean_50 --max_episodes 8)
  shift || true
fi
EXTRA+=("$@")

echo "[precompute_robotwin_t5] gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} mode=${MODE}"
torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  scripts/precompute_robotwin_t5_embeds.py \
  --skip_existing \
  "${EXTRA[@]}"
