#!/usr/bin/env bash
# FastWAM-style in-process OpenWAM RoboTwin evaluation (does NOT touch batched_eval).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/robotwin_local.sh
#   CKPT_DIR=... EVAL_PHASES=clean EVALUATION.task_name=shake_bottle_horizontally \
#     CUDA_VISIBLE_DEVICES=4 bash scripts/robotwin_local.sh
#
# Avoid GPUs used by an active batched_eval (often 2,3) unless intentional.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OPENWAM_ROOT}"

# Optional NVIDIA headless Vulkan (same helper as FastWAM if present).
if [[ -f "${OPENWAM_ROOT}/../FastWAM/scripts/env_nvidia_vulkan.sh" ]]; then
  # shellcheck source=/dev/null
  source "${OPENWAM_ROOT}/../FastWAM/scripts/env_nvidia_vulkan.sh"
fi

PYTHON="${PYTHON:-/data/anaconda3/envs/openwam/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] PYTHON not executable: ${PYTHON}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export ROBOTWIN_DISABLE_RT="${ROBOTWIN_DISABLE_RT:-0}"

CKPT_DIR="${CKPT_DIR:-${OPENWAM_ROOT}/outputs/openwam_fpd_robotwin_full/2026-09-24_22-18-53}"
EVAL_PHASES="${EVAL_PHASES:-both}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"

# Derive num_gpus from visible devices.
IFS=',' read -r -a _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#_gpu_arr[@]}"
if (( NUM_GPUS < 1 )); then
  echo "[ERROR] CUDA_VISIBLE_DEVICES empty" >&2
  exit 1
fi

EXTRA_ARGS=()
case "${EVAL_PHASES}" in
  both|all|clean|random|randomized)
    EXTRA_ARGS+=("MULTIRUN.phases=${EVAL_PHASES}")
    ;;
  *)
    echo "Unknown EVAL_PHASES=${EVAL_PHASES} (expected both|clean|random|randomized)" >&2
    exit 1
    ;;
esac

# Forward any extra Hydra overrides from the caller.
EXTRA_ARGS+=("$@")

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OPENWAM_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/robotwin_local_${STAMP}.log"

echo "Starting OpenWAM RoboTwin local (in-process) evaluation"
echo "  python: ${PYTHON}"
echo "  ckpt:   ${CKPT_DIR}"
echo "  gpus:   ${CUDA_VISIBLE_DEVICES} (num_gpus=${NUM_GPUS})"
echo "  max_tasks_per_gpu: ${MAX_TASKS_PER_GPU}"
echo "  phases: ${EVAL_PHASES}"
echo "  log:    ${LOG_FILE}"

"${PYTHON}" benchmarks/robotwin/local_eval/run_robotwin_manager.py \
  ckpt="${CKPT_DIR}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1
