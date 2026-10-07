#!/usr/bin/env bash
# Reusable RoboCasa GR1 eval (batched infer + multi-GPU sim).
#
# Defaults:
#   - release OpenWAM-Alpha-Sim-RoboCasa-GR1 (or latest FPD under outputs/)
#   - INFER_GPUS=2, SIM_GPUS=0,1,3,4,5,6,7, N_SIMS=1
#   - MODE=smoke|full (default full), NUM_EPISODES=10 (smoke forces 1 unless set)
#
# Usage:
#   bash scripts/eval_robocasa_gr1.sh
#   MODE=smoke bash scripts/eval_robocasa_gr1.sh
#   CKPT_DIR=outputs/openwam_fpd_robocasa_gr1/<run> MODE=full bash scripts/eval_robocasa_gr1.sh
#   INFER_GPUS=2 SIM_GPUS=3,4,5,6,7 N_SIMS=1 bash scripts/eval_robocasa_gr1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/robocasa_gr1

MODE="${MODE:-full}" # smoke | full
NUM_EPISODES="${NUM_EPISODES:-}"
INFER_GPUS="${INFER_GPUS:-2}"
SIM_GPUS="${SIM_GPUS:-0,1,3,4,5,6,7}"
N_SIMS="${N_SIMS:-1}"
MAX_INFER_BATCH="${MAX_INFER_BATCH:-2}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
WAN_PATH="${WAN_PATH:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
BASE_PORT="${BASE_PORT:-9300}"
SERVER_PYTHON="${SERVER_PYTHON:-/data/anaconda3/envs/openwam_re/bin/python}"

RELEASE_CKPT="${RELEASE_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1}"
FPD_ROOT="${FPD_ROOT:-${ROOT}/outputs/openwam_fpd_robocasa_gr1}"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_NAME="${CKPT_NAME:-}"
PREFER_FPD="${PREFER_FPD:-0}"

resolve_latest_fpd() {
  local best_dir="" best_ckpt="" best_mtime=0
  local d ckpt mtime
  shopt -s nullglob
  for d in "${FPD_ROOT}"/*/; do
    [[ -d "$d" ]] || continue
    [[ "$(basename "$d")" == *_debug ]] && continue
    ckpt="$(ls -1 "$d"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1 || true)"
    [[ -n "$ckpt" ]] || continue
    mtime="$(stat -c %Y "$ckpt" 2>/dev/null || echo 0)"
    if (( mtime >= best_mtime )); then
      best_mtime=$mtime
      best_dir="${d%/}"
      best_ckpt="$(basename "$ckpt")"
    fi
  done
  shopt -u nullglob
  if [[ -z "$best_dir" ]]; then
    return 1
  fi
  CKPT_DIR="$best_dir"
  CKPT_NAME="$best_ckpt"
  return 0
}

if [[ -z "${CKPT_DIR}" ]]; then
  if [[ "${PREFER_FPD}" == "1" ]] && resolve_latest_fpd; then
    :
  elif [[ -d "${RELEASE_CKPT}" ]]; then
    CKPT_DIR="${RELEASE_CKPT}"
  elif resolve_latest_fpd; then
    :
  else
    echo "[eval_robocasa_gr1] no checkpoint found (release or FPD)" >&2
    exit 2
  fi
fi
if [[ -z "${CKPT_NAME}" ]]; then
  CKPT_NAME="$(ls -1 "${CKPT_DIR}"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1 | xargs -n1 basename || true)"
fi
[[ -n "${CKPT_NAME}" && -f "${CKPT_DIR}/${CKPT_NAME}" ]] || {
  echo "[eval_robocasa_gr1] missing checkpoint under ${CKPT_DIR}" >&2
  exit 2
}

if [[ "${MODE}" == "smoke" ]]; then
  NUM_EPISODES="${NUM_EPISODES:-1}"
else
  NUM_EPISODES="${NUM_EPISODES:-10}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="$(basename "${CKPT_DIR}")"
RUN_TAG="${RUN_TAG:-${MODE}_${CKPT_TAG}_${STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/robocasa_gr1/${RUN_TAG}}"
LOG="${LOG:-${ROOT}/logs/robocasa_gr1_${RUN_TAG}.log}"

export MODE NUM_EPISODES INFER_GPUS SIM_GPUS N_SIMS MAX_INFER_BATCH
export ENCODER_DEVICE WAN_PATH BASE_PORT SERVER_PYTHON RUN_TAG OUTPUT_DIR

echo "[eval_robocasa_gr1] mode=${MODE} ckpt=${CKPT_DIR}/${CKPT_NAME}"
echo "[eval_robocasa_gr1] infer=${INFER_GPUS} sim=${SIM_GPUS} n_sims=${N_SIMS} episodes=${NUM_EPISODES}"
echo "[eval_robocasa_gr1] output=${OUTPUT_DIR}"
echo "[eval_robocasa_gr1] log=${LOG}"

bash "${ROOT}/benchmarks/robocasa_gr1/run_eval.sh" \
  "${CKPT_DIR}" \
  "${CKPT_NAME}" \
  "$@" \
  2>&1 | tee "${LOG}"
