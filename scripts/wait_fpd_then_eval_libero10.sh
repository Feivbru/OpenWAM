#!/usr/bin/env bash
# Wait for the running FPD RoboTwin train to finish, then launch full libero_10 eval.
# Usage: bash scripts/wait_fpd_then_eval_libero10.sh [train_bash_pid]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TRAIN_BASH_PID="${1:-3070808}"
TRAIN_MARKER="${TRAIN_MARKER:-train_fpd_robotwin_clean2random}"
POLL_SEC="${POLL_SEC:-60}"
GPU_FREE_MIB="${GPU_FREE_MIB:-8000}"   # per-GPU used mem threshold on 2-7 before starting eval
GPU_WAIT_ROUNDS="${GPU_WAIT_ROUNDS:-30}"

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
  # Return 0 if any of GPUs 2-7 still has high used memory (likely train not fully released).
  local line idx used
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
log "then: libero_10 FULL eval infer=2 sim=0,1,3,4,5,6,7 n_sims=1"
log "ckpt=outputs/openwam_fpd_libero/2026-10-01_12-42-09_with_umt5"

while train_still_running; do
  etime="$(ps -p "${TRAIN_BASH_PID}" -o etime= 2>/dev/null | tr -d ' ' || echo gone)"
  log "FPD train still running (bash_etime=${etime}); sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

log "train process gone; waiting for GPUs 2-7 to drop below ${GPU_FREE_MIB} MiB used"
for ((i=1; i<=GPU_WAIT_ROUNDS; i++)); do
  if gpus_busy; then
    mem="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '3,8p' | tr '\n' ' ')"
    log "GPU mem still high round ${i}/${GPU_WAIT_ROUNDS}: ${mem}"
    sleep "${POLL_SEC}"
  else
    log "GPUs 2-7 look free enough"
    break
  fi
done

# Avoid colliding with a leftover eval.
if pgrep -f 'run_all_suites.py|batched_server.py' >/dev/null 2>&1; then
  log "ERROR: leftover eval/server processes still present; abort launch"
  pgrep -af 'run_all_suites.py|batched_server.py|single_eval.py' || true
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
export CKPT_DIR="outputs/openwam_fpd_libero/2026-10-01_12-42-09_with_umt5"
export MODE=full
export SERVER_BACKEND=batched
export INFER_GPUS=2
export SIM_GPUS=0,1,3,4,5,6,7
export N_SIMS=1
export MAX_INFER_BATCH=2
export RUN_TAG="fpd_with_umt5_libero10_full_${STAMP}"

log "launching eval RUN_TAG=${RUN_TAG}"
log "command: MODE=full --suites long infer=${INFER_GPUS} sim=${SIM_GPUS} n_sims=${N_SIMS}"

exec bash scripts/eval_libero_plus.sh -- --suites long
