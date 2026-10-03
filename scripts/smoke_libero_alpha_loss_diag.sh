#!/usr/bin/env bash
# Diagnose OpenWAM LIBERO teacher loss scale vs Alpha release + norm ablation.
#
# Runs two forward-only smokes in parallel (no optimizer):
#   GPU2: normalize_mode=min-max (canonical, same as teacher train)
#   GPU3: normalize_mode=null     (raw disk actions, no affine)
# Optional GPU4: z-score (should look different if diag is sensitive)
#
# Usage:
#   bash scripts/smoke_libero_alpha_loss_diag.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-openwam}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

STEPS="${STEPS:-20}"
BATCH="${BATCH:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/smoke_libero_alpha_loss_${STAMP}"
mkdir -p "${LOGDIR}"

echo "[diag] steps=${STEPS} batch=${BATCH} logs=${LOGDIR}"

run_one() {
  local gpu="$1"
  local mode="$2"
  local log="${LOGDIR}/norm_${mode}_gpu${gpu}.log"
  echo "[diag] launch gpu=${gpu} normalize_mode=${mode} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/smoke_libero_alpha_loss_diag.py \
    --steps "${STEPS}" \
    --batch_size "${BATCH}" \
    --normalize_mode "${mode}" \
    --use_t5_cache \
    --video_mode clean \
    >"${log}" 2>&1 &
  echo $! >"${LOGDIR}/norm_${mode}_gpu${gpu}.pid"
}

# Prefer free-ish cards among 2-7 (riri holds ~22GB each; OpenWAM+skip-TE should fit).
run_one 2 min-max
run_one 3 null
run_one 4 z-score

echo "[diag] waiting..."
fail=0
for pidf in "${LOGDIR}"/*.pid; do
  pid="$(cat "${pidf}")"
  if ! wait "${pid}"; then
    echo "[diag] FAILED pid=${pid} (${pidf})" >&2
    fail=1
  fi
done

echo
echo "========== RESULTS =========="
rg -n "SUMMARY|loss_action:|normalize_mode:|sample0 active" "${LOGDIR}"/*.log || true
echo "============================="
echo "full logs: ${LOGDIR}"
exit "${fail}"
