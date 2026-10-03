#!/usr/bin/env bash
# Wait for banana FT to finish, then launch RoboTwin C2R teacher on GPUs 2-7.
# Run inside tmux so SSH disconnect does not kill the chain:
#   tmux new-window -t openwam_piper -n c2r_chain \
#     "bash scripts/wait_banana_then_teacher_c2r.sh"
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

WAITLOG="logs/wait_banana_then_teacher_c2r_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$WAITLOG") 2>&1

echo "===== $(date) wait banana then C2R teacher ====="
echo "waitlog=$WAITLOG pid=$$"

while true; do
  if ! pgrep -f 'config-name=train_banana_piper' >/dev/null; then
    echo "$(date) banana train processes gone"
    break
  fi
  BLOG=$(ls -t logs/banana_piper/train_*.log 2>/dev/null | head -1 || true)
  prog=""
  if [[ -n "${BLOG}" ]]; then
    prog=$(tr '\r' '\n' < "$BLOG" | grep 'Training:' | tail -1 | sed 's/.*Training:/Training:/' | cut -c1-120 || true)
  fi
  echo "$(date '+%H:%M:%S') banana still running; ${prog}"
  sleep 120
done

echo "$(date) waiting GPU 2-7 drain"
for _ in $(seq 1 40); do
  busy=0
  while read -r used; do
    mb=${used%% *}
    if [[ "${mb}" -gt 2048 ]]; then busy=1; fi
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3,4,5,6,7)
  if [[ $busy -eq 0 ]]; then
    echo "$(date) GPUs 2-7 free"
    break
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv -i 2,3,4,5,6,7
  sleep 30
done

echo "$(date) launching C2R teacher on 2-7 (openwam_re, NPROC=6)"
export CONDA_ENV=openwam_re
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export NPROC=6
export VARIANT=c2r
export WANDB_MODE=online
bash scripts/train_teacher_robotwin.sh
EC=$?
echo "$(date) C2R teacher finished exit=$EC"
exit "$EC"
