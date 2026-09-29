#!/usr/bin/env bash
# Run inside an interactive session that can see /dev/nvidia* (e.g. your openwam shell):
#   bash scripts/run_fpd_bs_probe_tmux.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WRITE_BACK=1
bash scripts/probe_fpd_batch_size.sh
BEST=$(rg -o 'RECOMMENDED_BATCH_SIZE=[0-9]+' logs/fpd_bs_probe_*.log | tail -1 | cut -d= -f2)
echo "Best BS=${BEST}; starting smoke..."
BATCH_SIZE="${BEST}" bash scripts/smoke_fpd_robotwin.sh
