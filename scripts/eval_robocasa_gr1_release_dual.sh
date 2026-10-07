#!/usr/bin/env bash
# Dual-group RoboCasa GR1 release eval (no smoke).
#
# Topology (default):
#   Group A: infer+encode GPU 6, sim GPUs 0,1,2  (n_sims=1, max_infer_batch=3)
#   Group B: infer+encode GPU 7, sim GPUs 3,4,5  (n_sims=1, max_infer_batch=3)
#   24 official tasks split 12+12; 10 episodes each; n_action_steps=8; discrete hands OFF.
#
# Usage (from OpenWAM repo root):
#   bash scripts/eval_robocasa_gr1_release_dual.sh
#   OUTPUT_DIR=outputs/robocasa_gr1/my_run bash scripts/eval_robocasa_gr1_release_dual.sh
#
# See README_ROBOCASA.md for knobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-/data/anaconda3/envs/openwam_re/bin/python}"
CLIENT_PYTHON="${ROBOCASA_GR1_PYTHON:-/data/anaconda3/envs/robocasa-gr1/bin/python}"
RELEASE_CKPT="${RELEASE_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1}"
CKPT_DIR="${CKPT_DIR:-${RELEASE_CKPT}}"
CKPT_NAME="${CKPT_NAME:-}"
POLICY_CONFIG="${POLICY_CONFIG_PATH:-${ROOT}/benchmarks/robocasa_gr1/policy_config.yml}"
WAN_PATH="${WAN_PATH:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"

INFER_A="${INFER_A:-6}"
SIM_A="${SIM_A:-0,1,2}"
PORT_A="${PORT_A:-9300}"
INFER_B="${INFER_B:-7}"
SIM_B="${SIM_B:-3,4,5}"
PORT_B="${PORT_B:-9400}"

N_SIMS="${N_SIMS:-1}"
MAX_INFER_BATCH="${MAX_INFER_BATCH:-3}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-10}"
SEED="${SEED:-0}"
INFERENCE_HORIZON="${INFERENCE_HORIZON:-8}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-release_dual_h${INFERENCE_HORIZON}_${STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/robocasa_gr1/${RUN_TAG}}"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/group_a" "${OUTPUT_DIR}/group_b"

if [[ -z "${CKPT_NAME}" ]]; then
  CKPT_NAME="$(ls -1 "${CKPT_DIR}"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1 | xargs -n1 basename || true)"
fi
[[ -n "${CKPT_NAME}" && -f "${CKPT_DIR}/${CKPT_NAME}" ]] || {
  echo "[release-dual] missing checkpoint under ${CKPT_DIR}" >&2
  exit 2
}
[[ -f "${POLICY_CONFIG}" ]] || {
  echo "[release-dual] missing policy config: ${POLICY_CONFIG}" >&2
  exit 2
}

# Contiguous 12+12 split of the official 24 env ids.
mapfile -t GROUP_ENVS < <(
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SERVER_PYTHON}" - <<'PY'
from benchmarks.robocasa_gr1.task_list import split_official_env_ids
for group in split_official_env_ids(2):
    print(",".join(group))
PY
)
ENV_IDS_A="${GROUP_ENVS[0]}"
ENV_IDS_B="${GROUP_ENVS[1]}"
[[ -n "${ENV_IDS_A}" && -n "${ENV_IDS_B}" ]] || {
  echo "[release-dual] failed to split official env ids" >&2
  exit 2
}

cat > "${OUTPUT_DIR}/run_config.json" <<EOF
{
  "created": "$(date -Iseconds)",
  "ckpt_dir": "${CKPT_DIR}",
  "ckpt_name": "${CKPT_NAME}",
  "policy_config": "${POLICY_CONFIG}",
  "inference_horizon": ${INFERENCE_HORIZON},
  "num_episodes": ${NUM_EPISODES},
  "seed": ${SEED},
  "n_sims": ${N_SIMS},
  "max_infer_batch": ${MAX_INFER_BATCH},
  "group_a": {"infer": "${INFER_A}", "sim": "${SIM_A}", "base_port": ${PORT_A}, "env_ids": "${ENV_IDS_A}"},
  "group_b": {"infer": "${INFER_B}", "sim": "${SIM_B}", "base_port": ${PORT_B}, "env_ids": "${ENV_IDS_B}"}
}
EOF

echo "[release-dual] ckpt=${CKPT_DIR}/${CKPT_NAME}"
echo "[release-dual] A: infer=${INFER_A} sim=${SIM_A} port=${PORT_A}"
echo "[release-dual] B: infer=${INFER_B} sim=${SIM_B} port=${PORT_B}"
echo "[release-dual] horizon=${INFERENCE_HORIZON} episodes=${NUM_EPISODES} batch=${MAX_INFER_BATCH}"
echo "[release-dual] output=${OUTPUT_DIR}"

launch_group() {
  local name="$1" infer="$2" sim="$3" port="$4" env_ids="$5" group_dir="$6"
  mkdir -p "${group_dir}"
  # Per-group output dirs avoid launcher/manifest races; summarize.py globs **/result.json.
  (
    export MODE=full
    export INFER_GPUS="${infer}"
    export SIM_GPUS="${sim}"
    export N_SIMS="${N_SIMS}"
    export MAX_INFER_BATCH="${MAX_INFER_BATCH}"
    export ENCODER_DEVICE="${ENCODER_DEVICE}"
    export WAN_PATH="${WAN_PATH}"
    export BASE_PORT="${port}"
    export NUM_EPISODES="${NUM_EPISODES}"
    export SEED="${SEED}"
    export INFERENCE_HORIZON="${INFERENCE_HORIZON}"
    export ENV_IDS="${env_ids}"
    export POLICY_CONFIG_PATH="${POLICY_CONFIG}"
    export SERVER_PYTHON="${SERVER_PYTHON}"
    export ROBOCASA_GR1_PYTHON="${CLIENT_PYTHON}"
    export OUTPUT_DIR="${group_dir}"
    export RUN_TAG="${RUN_TAG}_${name}"
    bash "${ROOT}/benchmarks/robocasa_gr1/run_eval.sh" "${CKPT_DIR}" "${CKPT_NAME}"
  ) > "${group_dir}/orchestrator.log" 2>&1
}

launch_group a "${INFER_A}" "${SIM_A}" "${PORT_A}" "${ENV_IDS_A}" "${OUTPUT_DIR}/group_a" &
PID_A=$!
launch_group b "${INFER_B}" "${SIM_B}" "${PORT_B}" "${ENV_IDS_B}" "${OUTPUT_DIR}/group_b" &
PID_B=$!

echo "[release-dual] started group_a pid=${PID_A} group_b pid=${PID_B}"
echo "[release-dual] logs: ${OUTPUT_DIR}/group_{a,b}/launcher.log"

EC_A=0
EC_B=0
wait "${PID_A}" || EC_A=$?
wait "${PID_B}" || EC_B=$?

echo "[release-dual] group_a exit=${EC_A} group_b exit=${EC_B}"
"${SERVER_PYTHON}" "${ROOT}/benchmarks/robocasa_gr1/summarize.py" "${OUTPUT_DIR}" || true

if (( EC_A != 0 || EC_B != 0 )); then
  echo "[release-dual] FAILED (a=${EC_A} b=${EC_B})" >&2
  exit 1
fi
echo "[release-dual] OK — see ${OUTPUT_DIR}/summary.json"
exit 0
