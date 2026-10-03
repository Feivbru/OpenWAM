#!/usr/bin/env bash
# Reusable LIBERO-plus eval for OpenWAM checkpoints (managed servers + clients).
#
# Defaults target a quick post-train sanity check of the latest LIBERO FPD run:
#   - auto-pick newest outputs/openwam_fpd_libero/*/checkpoint_step_*.safetensors
#   - sample 10% tasks per suite (--task-sample-ratio 0.1)
#   - batched: 1 infer GPU (2) + sim GPUs 0,1,3-7, N_SIMS=2 → 14 concurrent
#   - T5 encoder_server on the same infer GPU (socket-isolated process)
#
# Usage (from OpenWAM repo root):
#   bash scripts/eval_libero_plus.sh
#   CKPT_DIR=outputs/openwam_fpd_libero/2026-10-01_12-42-09 bash scripts/eval_libero_plus.sh
#   MODE=full bash scripts/eval_libero_plus.sh
#   MODE=smoke bash scripts/eval_libero_plus.sh
#   MODE=spot TASK_IDS=0,50,100,150 bash scripts/eval_libero_plus.sh
#   INFER_GPUS=2 SIM_GPUS=3,4,5,6,7 N_SIMS=2 bash scripts/eval_libero_plus.sh
#   SERVER_BACKEND=legacy GPUS=2,4,6 RENDER_GPUS=3,5,7 REPLICAS_PER_GPU=1 bash scripts/eval_libero_plus.sh
#
# Extra args after "--" are forwarded to run_all_suites.py / run_eval.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/libero-plus

# shellcheck disable=SC1091
source "${ROOT}/benchmarks/libero-plus/openwam_libero_plus_env.sh"

export SERVER_PYTHON="${SERVER_PYTHON:-/data/anaconda3/envs/openwam/bin/python}"
export LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/data/anaconda3/envs/libero-plus/bin/python}"
export LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-/data/zixian_guo/projects/haoming/project/PI/ImageWAM/third_party/LIBERO-plus}"

MODE="${MODE:-sample}" # sample | full | smoke | spot
TASK_SAMPLE_RATIO="${TASK_SAMPLE_RATIO:-0.1}"
TASK_SAMPLE_SEED="${TASK_SAMPLE_SEED:-42}"
TASK_IDS="${TASK_IDS:-0,50,100,150}"
NUM_TRIALS="${NUM_TRIALS:-1}"

SERVER_BACKEND="${SERVER_BACKEND:-batched}" # batched | legacy
INFER_GPUS="${INFER_GPUS:-2}"
# GPU0/1 EGL speed matches free cards; they currently have ~8GB free under other jobs.
SIM_GPUS="${SIM_GPUS:-0,1,3,4,5,6,7}"
GPUS="${GPUS:-${INFER_GPUS}}"
RENDER_GPUS="${RENDER_GPUS:-${SIM_GPUS}}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
N_SIMS="${N_SIMS:-2}"
MAX_INFER_BATCH="${MAX_INFER_BATCH:-2}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
WAN_PATH="${WAN_PATH:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
BASE_PORT="${BASE_PORT:-8920}"
COMPILE_ENABLED="${COMPILE_ENABLED:-false}"

FPD_ROOT="${FPD_ROOT:-${ROOT}/outputs/openwam_fpd_libero}"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_NAME="${CKPT_NAME:-}"

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
    echo "[eval_libero_plus] no FPD checkpoint under ${FPD_ROOT}" >&2
    exit 2
  fi
  CKPT_DIR="$best_dir"
  CKPT_NAME="$best_ckpt"
}

if [[ -z "${CKPT_DIR}" ]]; then
  resolve_latest_fpd
elif [[ -z "${CKPT_NAME}" ]]; then
  CKPT_NAME="$(ls -1 "${CKPT_DIR}"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1 | xargs -n1 basename || true)"
  [[ -n "${CKPT_NAME}" ]] || {
    echo "[eval_libero_plus] no checkpoint_step_*.safetensors under ${CKPT_DIR}" >&2
    exit 2
  }
fi

[[ -f "${CKPT_DIR}/${CKPT_NAME}" ]] || {
  echo "[eval_libero_plus] missing ${CKPT_DIR}/${CKPT_NAME}" >&2
  exit 2
}
[[ -f "${CKPT_DIR}/config.yaml" ]] || {
  echo "[eval_libero_plus] missing ${CKPT_DIR}/config.yaml" >&2
  exit 2
}

# Train with use_t5_cache skips saving UMT5; legacy deploy needs it in-process.
# Batched mode uses encoder_server + WAN_PATH, so materialize is optional there.
AUTO_MATERIALIZE_UMT5="${AUTO_MATERIALIZE_UMT5:-}"
if [[ -z "${AUTO_MATERIALIZE_UMT5}" ]]; then
  if [[ "${SERVER_BACKEND}" == "batched" ]]; then
    AUTO_MATERIALIZE_UMT5=0
  else
    AUTO_MATERIALIZE_UMT5=1
  fi
fi
if [[ "${AUTO_MATERIALIZE_UMT5}" == "1" ]]; then
  TE_KEYS="$("${SERVER_PYTHON}" - <<PY
from safetensors import safe_open
p = r"${CKPT_DIR}/${CKPT_NAME}"
with safe_open(p, framework="pt", device="cpu") as f:
    print(sum(1 for k in f.keys() if k.startswith("video_backbone.text_encoder.")))
PY
)"
  if [[ "${TE_KEYS}" == "0" ]]; then
    echo "[eval_libero_plus] ckpt has 0 text_encoder keys (use_t5_cache train); materializing UMT5..."
    MAT_LOG="$(mktemp)"
    "${SERVER_PYTHON}" "${ROOT}/scripts/materialize_libero_ckpt_with_umt5.py" --src "${CKPT_DIR}" | tee "${MAT_LOG}"
    CKPT_DIR="$(sed -n 's/^CKPT_DIR=//p' "${MAT_LOG}" | tail -1)"
    CKPT_NAME="$(sed -n 's/^CKPT_NAME=//p' "${MAT_LOG}" | tail -1)"
    rm -f "${MAT_LOG}"
    [[ -n "${CKPT_DIR}" && -n "${CKPT_NAME}" && -f "${CKPT_DIR}/${CKPT_NAME}" ]] || {
      echo "[eval_libero_plus] materialize failed" >&2
      exit 2
    }
    echo "[eval_libero_plus] using materialized ckpt=${CKPT_DIR}/${CKPT_NAME}"
  else
    echo "[eval_libero_plus] text_encoder keys=${TE_KEYS}"
  fi
fi
[[ -x "${SERVER_PYTHON}" ]] || {
  echo "[eval_libero_plus] SERVER_PYTHON not executable: ${SERVER_PYTHON}" >&2
  exit 2
}
[[ -x "${LIBERO_PLUS_PYTHON}" ]] || {
  echo "[eval_libero_plus] LIBERO_PLUS_PYTHON not executable: ${LIBERO_PLUS_PYTHON}" >&2
  exit 2
}
[[ -d "${LIBERO_PLUS_PATH}" ]] || {
  echo "[eval_libero_plus] LIBERO_PLUS_PATH missing: ${LIBERO_PLUS_PATH}" >&2
  exit 2
}

STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="$(basename "${CKPT_DIR}")"
case "${MODE}" in
  sample) RUN_TAG_DEFAULT="fpd_${CKPT_TAG}_sample${TASK_SAMPLE_RATIO}_${STAMP}" ;;
  full)   RUN_TAG_DEFAULT="fpd_${CKPT_TAG}_full_${STAMP}" ;;
  smoke)  RUN_TAG_DEFAULT="fpd_${CKPT_TAG}_smoke_${STAMP}" ;;
  spot)   RUN_TAG_DEFAULT="fpd_${CKPT_TAG}_spot_${STAMP}" ;;
  *)
    echo "[eval_libero_plus] unknown MODE=${MODE} (use sample|full|smoke|spot)" >&2
    exit 2
    ;;
esac
export RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/libero-plus/${RUN_TAG}}"
export SERVER_BACKEND INFER_GPUS SIM_GPUS GPUS RENDER_GPUS REPLICAS_PER_GPU N_SIMS MAX_INFER_BATCH ENCODER_DEVICE WAN_PATH BASE_PORT
LOG="${LOG:-${ROOT}/logs/libero_plus_${RUN_TAG}.log}"

EXTRA_ARGS=()
case "${MODE}" in
  sample)
    EXTRA_ARGS+=(--task-sample-ratio "${TASK_SAMPLE_RATIO}" --task-sample-seed "${TASK_SAMPLE_SEED}")
    ;;
  spot)
    EXTRA_ARGS+=(--task-ids "${TASK_IDS}")
    ;;
  smoke)
    EXTRA_ARGS+=(--smoke)
    ;;
  full) ;;
esac
EXTRA_ARGS+=(--num-trials "${NUM_TRIALS}" --compile-enabled "${COMPILE_ENABLED}")

# Forward leftover CLI args after optional "--"
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS+=("$@")
fi

echo "[eval_libero_plus] mode=${MODE} backend=${SERVER_BACKEND}"
echo "[eval_libero_plus] ckpt=${CKPT_DIR}/${CKPT_NAME}"
echo "[eval_libero_plus] infer=${INFER_GPUS} sim=${SIM_GPUS} n_sims=${N_SIMS} max_infer_batch=${MAX_INFER_BATCH}"
echo "[eval_libero_plus] encoder_device=${ENCODER_DEVICE}"
echo "[eval_libero_plus] output=${OUTPUT_DIR}"
echo "[eval_libero_plus] log=${LOG}"
echo "[eval_libero_plus] extra=${EXTRA_ARGS[*]}"

bash "${ROOT}/benchmarks/libero-plus/run_eval.sh" \
  "${CKPT_DIR}" \
  "${CKPT_NAME}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG}"
