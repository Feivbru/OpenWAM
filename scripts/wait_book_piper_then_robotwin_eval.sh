#!/usr/bin/env bash
# Wait for book_piper train to finish, then relaunch full RoboTwin batched_eval (FPD C2R).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TRAIN_BASH_PID="${1:-644360}"
TRAIN_MARKER="${TRAIN_MARKER:-train_book_piper}"
POLL_SEC="${POLL_SEC:-60}"
GPU_FREE_MIB="${GPU_FREE_MIB:-8000}"
GPU_WAIT_ROUNDS="${GPU_WAIT_ROUNDS:-40}"

CKPT_DIR="${CKPT_DIR:-${ROOT}/outputs/openwam_fpd_robotwin_clean2random/2026-10-03_00-09-11}"
EVAL_NAME="${EVAL_NAME:-fpd_c2r_full_rerun}"
INFER_GPUS="${INFER_GPUS:-2}"
SIM_GPUS="${SIM_GPUS:-3,4,5,6,7}"
N_SIMS="${N_SIMS:-4}"
MAX_INFER_BATCH="${MAX_INFER_BATCH:-2}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
CONDA_ENV="${CONDA_ENV:-openwam}"

log() { echo "[$(date '+%F %T')] $*"; }

train_still_running() {
  if kill -0 "${TRAIN_BASH_PID}" 2>/dev/null; then
    return 0
  fi
  if pgrep -f "${TRAIN_MARKER}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

gpus_busy() {
  local idx used
  while IFS=',' read -r idx used _; do
    idx="$(echo "$idx" | tr -d ' ')"
    used="$(echo "$used" | tr -d ' MiB')"
    case "$idx" in
      2|3|4|5|6|7)
        if [[ "${used:-0}" =~ ^[0-9]+$ ]] && (( used > GPU_FREE_MIB )); then
          return 0
        fi
        ;;
    esac
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  return 1
}

log "monitor start: wait train pid=${TRAIN_BASH_PID} marker=${TRAIN_MARKER}"
log "then: RoboTwin FULL batched_eval ckpt=${CKPT_DIR}"
log "  infer=${INFER_GPUS} sim=${SIM_GPUS} n_sims=${N_SIMS} encoder=${ENCODER_DEVICE} name=${EVAL_NAME}"

while train_still_running; do
  etime="$(ps -p "${TRAIN_BASH_PID}" -o etime= 2>/dev/null | tr -d ' ' || echo gone)"
  log "book_piper still running (bash_etime=${etime}); sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

log "train gone; waiting for GPUs 2-7 < ${GPU_FREE_MIB} MiB used"
for ((i=1; i<=GPU_WAIT_ROUNDS; i++)); do
  if gpus_busy; then
    mem="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '3,8p' | tr '\n' ' ')"
    log "GPU mem still high round ${i}/${GPU_WAIT_ROUNDS}: ${mem}"
    sleep "${POLL_SEC}"
  else
    log "GPUs 2-7 free enough"
    break
  fi
done

# Clear leftover eval/servers if any.
if pgrep -f 'batched_eval\.sh|batched_server\.py|encoder_server\.py' >/dev/null 2>&1; then
  log "killing leftover batched_eval/server processes"
  pkill -f 'batched_eval\.sh' 2>/dev/null || true
  pkill -f 'batched_server\.py' 2>/dev/null || true
  pkill -f 'encoder_server\.py' 2>/dev/null || true
  sleep 3
fi

if [[ ! -f "${CKPT_DIR}/checkpoint_step_10000.safetensors" && ! -f "${CKPT_DIR}/config.yaml" ]]; then
  log "ERROR: ckpt dir missing or incomplete: ${CKPT_DIR}"
  exit 2
fi

log "launching batched_eval..."
export ENCODER_DEVICE MAX_INFER_BATCH CONDA_ENV
exec bash benchmarks/robotwin/batched_eval.sh \
  -d "${CKPT_DIR}" \
  -m both \
  -n "${EVAL_NAME}" \
  --infer-gpus "${INFER_GPUS}" \
  --sim-gpus "${SIM_GPUS}" \
  --n-sims "${N_SIMS}" \
  all
