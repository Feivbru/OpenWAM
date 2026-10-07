#!/usr/bin/env bash
# End-to-end RoboCasa GR1 launcher: batched OpenWAM servers + sim clients.
#
# Usage:
#   bash benchmarks/robocasa_gr1/run_eval.sh CKPT_DIR CKPT_NAME [scheduler options]
#
# Env overrides:
#   SERVER_PYTHON, ROBOCASA_GR1_PYTHON, ROBOCASA_GR1_PATH
#   INFER_GPUS=2 SIM_GPUS=0,1,3,4,5,6,7 N_SIMS=1
#   NUM_EPISODES=10 BASE_PORT=9300 OUTPUT_DIR=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -ge 2 ]]; then
  CKPT_DIR="$1"
  CKPT_NAME="$2"
  shift 2
else
  CKPT_DIR="${CKPT_DIR:-}"
  CKPT_NAME="${CKPT_NAME:-}"
fi
[[ -n "${CKPT_DIR}" && -n "${CKPT_NAME}" ]] || {
  echo "Usage: bash benchmarks/robocasa_gr1/run_eval.sh CKPT_DIR CKPT_NAME [options]" >&2
  exit 2
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/openwam_robocasa_gr1_env.sh"

SERVER_PYTHON="${SERVER_PYTHON:-/data/anaconda3/envs/openwam_re/bin/python}"
CLIENT_PYTHON="${ROBOCASA_GR1_PYTHON:-/data/anaconda3/envs/robocasa-gr1/bin/python}"
ROBOCASA_PATH="${ROBOCASA_GR1_PATH:-${REPO_ROOT}/third_party/robocasa-gr1-tabletop-tasks}"
POLICY_CONFIG="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"

INFER_GPUS="${INFER_GPUS:-2}"
SIM_GPUS="${SIM_GPUS:-0,1,3,4,5,6,7}"
N_SIMS="${N_SIMS:-1}"
MAX_INFER_BATCH="${MAX_INFER_BATCH:-2}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
WAN_PATH="${WAN_PATH:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
BASE_PORT="${BASE_PORT:-9300}"
NUM_EPISODES="${NUM_EPISODES:-10}"
SEED="${SEED:-0}"
INFERENCE_HORIZON="${INFERENCE_HORIZON:-}"
ENV_IDS="${ENV_IDS:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/robocasa_gr1/${RUN_TAG}}"
mkdir -p "${OUTPUT_DIR}"

EXTRA=()
if [[ "${MODE:-}" == "smoke" ]]; then
  EXTRA+=(--smoke)
  NUM_EPISODES="${NUM_EPISODES:-1}"
fi
if [[ -n "${INFERENCE_HORIZON}" ]]; then
  EXTRA+=(--inference-horizon "${INFERENCE_HORIZON}")
fi
if [[ -n "${ENV_IDS}" ]]; then
  EXTRA+=(--env-ids "${ENV_IDS}")
fi

echo "[robocasa-gr1] checkpoint=${CKPT_DIR}/${CKPT_NAME}"
echo "[robocasa-gr1] infer=${INFER_GPUS} sim=${SIM_GPUS} n_sims=${N_SIMS} episodes=${NUM_EPISODES}"
echo "[robocasa-gr1] inference_horizon=${INFERENCE_HORIZON:-null} policy_config=${POLICY_CONFIG}"
echo "[robocasa-gr1] output=${OUTPUT_DIR}"

"${SERVER_PYTHON}" "${SCRIPT_DIR}/scheduler.py" \
  --ckpt-dir "${CKPT_DIR}" \
  --ckpt-name "${CKPT_NAME}" \
  --server-python "${SERVER_PYTHON}" \
  --client-python "${CLIENT_PYTHON}" \
  --robocasa-path "${ROBOCASA_PATH}" \
  --policy-config "${POLICY_CONFIG}" \
  --infer-gpus "${INFER_GPUS}" \
  --sim-gpus "${SIM_GPUS}" \
  --n-sims "${N_SIMS}" \
  --max-infer-batch "${MAX_INFER_BATCH}" \
  --encoder-device "${ENCODER_DEVICE}" \
  --wan-path "${WAN_PATH}" \
  --base-port "${BASE_PORT}" \
  --num-episodes "${NUM_EPISODES}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA[@]}" \
  "$@" \
  2>&1 | tee "${OUTPUT_DIR}/launcher.log"
