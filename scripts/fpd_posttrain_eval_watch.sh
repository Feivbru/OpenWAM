#!/usr/bin/env bash
# tmux-safe: wait for the current FPD train to finish, then launch split-topology eval.
# Owns eval launch only — health fixes are handled by Cursor CLI in another tmux session.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TORCH_PID="${TORCH_PID:-3714787}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/openwam_fpd_robotwin_full/2026-09-24_22-18-53}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT/logs/fpd_full_20260924_221624.log}"
LOCK="${LOCK:-/tmp/fpd_posttrain_eval.lock}"
EVAL_NAME="${EVAL_NAME:-fpd_full_split_auto}"
LOG="$ROOT/logs/fpd_posttrain_eval_watch_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="$ROOT/logs/fpd_posttrain_eval.status"

exec > >(tee -a "$LOG") 2>&1

echo "[fpd_posttrain_eval] start $(date -Is)"
echo "  torch_pid=$TORCH_PID"
echo "  run_dir=$RUN_DIR"
echo "  train_log=$TRAIN_LOG"
echo "  status_file=$STATUS_FILE"
echo "waiting" >"$STATUS_FILE"

wait_train() {
  while true; do
    if ! kill -0 "$TORCH_PID" 2>/dev/null; then
      return 0
    fi
    if [[ -r /proc/"$TORCH_PID"/cmdline ]]; then
      if ! tr '\0' ' ' </proc/"$TORCH_PID"/cmdline | grep -q 'train_fpd_robotwin_full\|train.py --config-name=train_fpd_robotwin_full'; then
        echo "[fpd_posttrain_eval] pid $TORCH_PID no longer FPD train"
        return 0
      fi
    else
      return 0
    fi
    sleep 30
  done
}

echo "[fpd_posttrain_eval] waiting for torchrun to exit..."
wait_train
echo "[fpd_posttrain_eval] torchrun gone at $(date -Is); waiting for checkpoint"

CKPT=""
for _ in $(seq 1 90); do
  latest="$(ls -1 "$RUN_DIR"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${latest}" ]]; then
    CKPT="$latest"
    break
  fi
  # If train children are gone, still give final save a short grace window
  if ! pgrep -f 'scripts/train.py --config-name=train_fpd_robotwin_full' >/dev/null 2>&1; then
    sleep 20
    latest="$(ls -1 "$RUN_DIR"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1 || true)"
    [[ -n "${latest}" ]] && CKPT="$latest"
    break
  fi
  sleep 10
done

if [[ -z "$CKPT" ]]; then
  echo "[fpd_posttrain_eval] FAIL: no checkpoint under $RUN_DIR"
  echo "failed_no_ckpt $(date -Is)" >"$STATUS_FILE"
  exit 2
fi

echo "[fpd_posttrain_eval] ckpt=$CKPT"

# Wait until train GPUs are free (avoid fighting leftover procs)
echo "[fpd_posttrain_eval] waiting for train python procs to clear..."
for _ in $(seq 1 60); do
  if ! pgrep -f 'scripts/train.py --config-name=train_fpd_robotwin_full' >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
sleep 10

if [[ -e "$LOCK" ]]; then
  echo "[fpd_posttrain_eval] lock exists ($LOCK); another eval launcher may be active — abort"
  echo "skipped_lock $(date -Is)" >"$STATUS_FILE"
  exit 3
fi
echo "$$ $(date -Is) $CKPT" >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "launching $(date -Is)" >"$STATUS_FILE"
echo "[fpd_posttrain_eval] launching batched_eval (infer=2 sim=3-7 n-sims=4 max_batch=${MAX_INFER_BATCH:-2})"

# T5 defaults to same infer GPU (cuda:0 under CVD); override ENCODER_DEVICE=cpu to free VRAM.
export ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
export MAX_INFER_BATCH="${MAX_INFER_BATCH:-2}"

# Run eval in foreground of this tmux pane so logs stay attached
bash benchmarks/robotwin/batched_eval.sh \
  -d "$RUN_DIR" \
  -m both \
  -n "$EVAL_NAME" \
  --infer-gpus 2 \
  --sim-gpus 3,4,5,6,7 \
  --n-sims 4 \
  all

ec=$?
echo "[fpd_posttrain_eval] batched_eval exited ec=$ec at $(date -Is)"
if (( ec == 0 )); then
  echo "done_ok $(date -Is)" >"$STATUS_FILE"
else
  echo "done_fail_$ec $(date -Is)" >"$STATUS_FILE"
fi
exit "$ec"
