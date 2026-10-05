#!/usr/bin/env bash
# End-to-end RoboCasa-GR1 FPD pipeline:
#   1) train teacher (5k) from release OpenWAM-Alpha-Sim-RoboCasa-GR1
#   2) pick the newest teacher run with a checkpoint
#   3) train FPD student (10k) from the same release + that teacher
#
# Usage (tmux recommended):
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC=6 CONDA_ENV=openwam_re \
#     bash scripts/train_teacher_then_fpd_robocasa_gr1.sh
#
# Env:
#   SKIP_TEACHER=1   skip teacher stage; use TEACHER_CKPT or latest under outputs/
#   TEACHER_CKPT=... force a teacher dir for the FPD stage
#   Extra args after -- are forwarded to BOTH train stages as Hydra overrides.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export NPROC="${NPROC:-}"
export CONDA_ENV="${CONDA_ENV:-openwam_re}"
export WANDB_MODE="${WANDB_MODE:-online}"

STAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="logs/teacher_then_fpd_robocasa_gr1_${STAMP}.log"
exec > >(tee -a "${CHAIN_LOG}") 2>&1

echo "===== $(date) RoboCasa-GR1 teacher → FPD chain ====="
echo "chain_log=${CHAIN_LOG} pid=$$ gpus=${CUDA_VISIBLE_DEVICES} conda=${CONDA_ENV}"

EXTRA_ARGS=("$@")

latest_teacher() {
  local root="${ROOT}/outputs/openwam_teacher_robocasa_gr1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | while read -r _ts dir; do
        if compgen -G "${dir}/checkpoint_step_*.safetensors" >/dev/null; then
          echo "${dir}"
          return 0
        fi
      done
  return 1
}

if [[ "${SKIP_TEACHER:-0}" != "1" ]]; then
  echo "$(date) === stage 1/2: teacher ==="
  bash "${ROOT}/scripts/train_teacher_robocasa_gr1.sh" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  echo "$(date) teacher stage exit=0"
else
  echo "$(date) SKIP_TEACHER=1 — skipping teacher stage"
fi

if [[ -n "${TEACHER_CKPT:-}" ]]; then
  if [[ "${TEACHER_CKPT}" != /* ]]; then
    TEACHER_CKPT="${ROOT}/${TEACHER_CKPT}"
  fi
else
  TEACHER_CKPT="$(latest_teacher || true)"
fi
if [[ -z "${TEACHER_CKPT}" || ! -d "${TEACHER_CKPT}" ]]; then
  echo "[ERROR] no teacher checkpoint dir found after teacher stage" >&2
  exit 1
fi
if ! compgen -G "${TEACHER_CKPT}/checkpoint_step_*.safetensors" >/dev/null; then
  echo "[ERROR] no checkpoint_step_*.safetensors under ${TEACHER_CKPT}" >&2
  exit 1
fi
echo "$(date) using teacher=${TEACHER_CKPT}"
ls -lh "${TEACHER_CKPT}"/checkpoint_step_*.safetensors

echo "$(date) === stage 2/2: FPD (release student) ==="
export TEACHER_CKPT
bash "${ROOT}/scripts/train_fpd_robocasa_gr1.sh" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
echo "$(date) FPD stage exit=0"
echo "$(date) ===== chain done ====="
