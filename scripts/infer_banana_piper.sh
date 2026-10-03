#!/usr/bin/env bash
# Banana Piper offline inference smoke against an OpenWAM deploy server.
#
# Two-step workflow (server must already be up, or start it in another tmux pane):
#
#   # Terminal A — policy server (needs a real banana FT ckpt dir):
#   CUDA_VISIBLE_DEVICES=6 bash scripts/deploy.sh /path/to/banana_ckpt --port 8848
#
#   # Terminal B — one-frame smoke from the banana dataset:
#   bash scripts/infer_banana_piper.sh
#   bash scripts/infer_banana_piper.sh --episode 1 --frame-index 10
#   HOST=127.0.0.1 PORT=8848 DATASET_DIR=/path/to/banana bash scripts/infer_banana_piper.sh
#
# Extra args are forwarded to scripts/infer_banana_piper_smoke.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8848}"
DATASET_DIR="${DATASET_DIR:-/data/zixian_guo/projects/haoming/project/PI/riri/data/banana}"
CONDA_ENV="${CONDA_ENV:-openwam}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[infer_banana_piper] host=${HOST} port=${PORT}"
echo "[infer_banana_piper] dataset=${DATASET_DIR}"
echo "[infer_banana_piper] conda=${CONDA_ENV}"

python scripts/infer_banana_piper_smoke.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --dataset-dir "${DATASET_DIR}" \
  "$@"
