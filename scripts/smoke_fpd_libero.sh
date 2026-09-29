#!/usr/bin/env bash
# Smoke: LIBERO teacher (debug) then FPD (debug) on GPUs 2,3.
#
# Usage:
#   bash scripts/smoke_fpd_libero.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NPROC="${NPROC:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[smoke_fpd_libero] teacher debug on gpus=${CUDA_VISIBLE_DEVICES}"
bash scripts/train_teacher_libero.sh \
  training.debug=true \
  training.batch_size=1 \
  project.wandb.mode="${WANDB_MODE}"

# Newest teacher run under outputs/openwam_teacher_libero
TEACHER_ROOT="${ROOT}/outputs/openwam_teacher_libero"
if [[ ! -d "${TEACHER_ROOT}" ]]; then
  echo "Teacher output root missing: ${TEACHER_ROOT}" >&2
  exit 1
fi
TEACHER_CKPT="$(ls -1dt "${TEACHER_ROOT}"/*/ 2>/dev/null | head -1 | sed 's:/*$::')"
if [[ -z "${TEACHER_CKPT}" ]] || ! compgen -G "${TEACHER_CKPT}/checkpoint_step_*.safetensors" >/dev/null; then
  echo "No teacher checkpoint found under ${TEACHER_ROOT}" >&2
  exit 1
fi
echo "[smoke_fpd_libero] teacher=${TEACHER_CKPT}"

echo "[smoke_fpd_libero] FPD debug"
TEACHER_CKPT="${TEACHER_CKPT}" bash scripts/train_fpd_libero.sh \
  training.debug=true \
  training.batch_size=1 \
  project.wandb.mode="${WANDB_MODE}"

echo "[smoke_fpd_libero] done"
