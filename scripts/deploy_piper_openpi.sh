#!/usr/bin/env bash
# Start OpenWAM as an openpi-compatible Piper EEF10 policy server.
#
# Wire protocol: README_piper.md (msgpack + physical EEF10).
# Real-robot client: openpi_client.WebsocketClientPolicy → EndPoseCtrl.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt
#   CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt --port 8000
#   CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt --training-rtc --max-delay 8
#
# Health check:
#   curl -s http://127.0.0.1:8848/healthz
#   curl -s http://192.168.3.37:8848/healthz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/ckpt_dir [extra deploy.py args...]" >&2
  exit 1
fi

CKPT_DIR="$1"
shift

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[deploy_piper_openpi] ckpt=${CKPT_DIR}"
echo "[deploy_piper_openpi] gpus=${CUDA_VISIBLE_DEVICES} conda=${CONDA_ENV}"

exec python scripts/deploy.py \
  --ckpt-dir "${CKPT_DIR}" \
  --protocol openpi \
  "$@"
