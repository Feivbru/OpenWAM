#!/usr/bin/env bash
# Wait until the occupying GPU job finishes, then start LIBERO teacher training.
#
# Current occupant (2026-10-01):
#   tmux serve
#     └─ run_queue.sh  (WATCH_PID)
#          └─ scheduler.py
#               └─ task.py × N  (openpi on GPUs 2-7)
#
# Usage:
#   WATCH_PID=101082 bash scripts/watch_then_train_teacher_libero.sh
#   # or auto-detect the live run_queue.sh:
#   bash scripts/watch_then_train_teacher_libero.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

POLL_SEC="${POLL_SEC:-30}"
FREE_MEM_MIB="${FREE_MEM_MIB:-70000}"   # each of GPUs 2-7 must have >= this free
GPU_CHECK_LIST="${GPU_CHECK_LIST:-2,3,4,5,6,7}"
LOG="logs/watch_then_teacher_libero_$(date +%Y%m%d_%H%M%S).log"

detect_watch_pid() {
  # Prefer the rtc_queue run_queue.sh; fall back to its scheduler.py.
  local pid
  pid="$(pgrep -nf 'beiyu_trainingRTC/openpi-stu/outputs/rtc_queue_.*/run_queue\.sh' || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}"
    return
  fi
  pid="$(pgrep -nf 'beiyu_trainingRTC/openpi-stu/outputs/rtc_queue_.*/scheduler\.py' || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}"
    return
  fi
  echo ""
}

WATCH_PID="${WATCH_PID:-$(detect_watch_pid)}"

exec > >(tee -a "${LOG}") 2>&1

echo "[watch] log=${LOG}"
echo "[watch] root=${ROOT}"
echo "[watch] poll=${POLL_SEC}s free_mem>=${FREE_MEM_MIB}MiB gpus=${GPU_CHECK_LIST}"

if [[ -z "${WATCH_PID}" ]]; then
  echo "[watch] no run_queue/scheduler found — will only wait for GPU free memory"
else
  if ! kill -0 "${WATCH_PID}" 2>/dev/null; then
    echo "[watch] WATCH_PID=${WATCH_PID} already dead"
    WATCH_PID=""
  else
    echo "[watch] watching PID=${WATCH_PID}: $(ps -o pid=,etime=,cmd= -p "${WATCH_PID}" | sed 's/^ *//')"
  fi
fi

alive() { kill -0 "$1" 2>/dev/null; }

gpus_free_enough() {
  local csv used free idx
  # memory.free in MiB
  while IFS=',' read -r idx free; do
    idx="$(echo "${idx}" | tr -d ' ')"
    free="$(echo "${free}" | tr -d ' MiB')"
    case ",${GPU_CHECK_LIST}," in
      *",${idx},"*)
        if [[ "${free}" -lt "${FREE_MEM_MIB}" ]]; then
          return 1
        fi
        ;;
    esac
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 0
}

# 1) wait for watched process to exit
if [[ -n "${WATCH_PID}" ]]; then
  while alive "${WATCH_PID}"; do
    echo "[watch] $(date '+%F %T') still running pid=${WATCH_PID} etime=$(ps -o etime= -p "${WATCH_PID}" | tr -d ' ')"
    sleep "${POLL_SEC}"
  done
  echo "[watch] $(date '+%F %T') watched process exited"
fi

# 2) wait until target GPUs have enough free memory (children / next queue jobs)
echo "[watch] waiting for GPUs ${GPU_CHECK_LIST} free>=${FREE_MEM_MIB}MiB ..."
while ! gpus_free_enough; do
  echo "[watch] $(date '+%F %T') GPUs not free yet:"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader \
    | awk -F',' -v list=",${GPU_CHECK_LIST}," '
        {
          gsub(/ /,"",$1);
          if (index(list, "," $1 ",")>0) print "  gpu"$0
        }'
  # if a new queue appeared, keep waiting on it too
  new_pid="$(detect_watch_pid || true)"
  if [[ -n "${new_pid}" ]] && alive "${new_pid}"; then
    echo "[watch] detected new queue pid=${new_pid}, waiting on it"
    while alive "${new_pid}"; do
      sleep "${POLL_SEC}"
    done
    echo "[watch] new queue exited"
  fi
  sleep "${POLL_SEC}"
done

echo "[watch] $(date '+%F %T') GPUs free — launching teacher"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# train_teacher_libero.sh already sets CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
bash scripts/train_teacher_libero.sh
echo "[watch] teacher finished with exit=$?"
