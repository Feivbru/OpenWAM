#!/usr/bin/env bash
# Diagnostic: websocket single vs batched single/dual sim, with ego-view mp4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${ROOT}/outputs/robocasa_gr1/video_diag_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}/videos" "${OUT}/logs"
echo "${OUT}" | tee /tmp/gr1_video_diag_out.txt

CKPT_DIR="${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1"
CKPT_NAME="checkpoint_step_100000.safetensors"
ENV_CUP="gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
ENV_POTATO="gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"

write_cfg() {
  local dest="$1" port="$2" video_subdir="$3" env_id="$4"
  /data/anaconda3/envs/robocasa-gr1/bin/python - "$dest" "$port" "${OUT}/videos/${video_subdir}" "$env_id" <<'PY'
import sys, yaml
from pathlib import Path
dest, port, video_dir, env_id = sys.argv[1:5]
cfg = yaml.safe_load(Path("benchmarks/robocasa_gr1/policy_config.yml").read_text())
cfg.update({
    "host": "127.0.0.1",
    "port": int(port),
    "num_episodes": 1,
    "seed": 0,
    "save_video": True,
    "video_dir": video_dir,
    "env_id": env_id,
})
Path(dest).write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", dest)
PY
}

# ---------- A: websocket deploy (reuse if already up on 8848) ----------
if ! ss -ltn | rg -q ':8848'; then
  echo "[A] starting deploy on GPU2:8848"
  CUDA_VISIBLE_DEVICES=2 /data/anaconda3/envs/openwam_re/bin/python scripts/deploy.py \
    --ckpt-dir "${CKPT_DIR}" --device cuda:0 --port 8848 \
    > "${OUT}/logs/deploy.log" 2>&1 &
  echo $! > "${OUT}/deploy.pid"
  for _ in $(seq 1 180); do
    ss -ltn | rg -q ':8848' && break
    sleep 2
  done
else
  echo "[A] reusing existing deploy on :8848"
fi
ss -ltn | rg -q ':8848' || { echo "deploy not ready"; tail -40 "${OUT}/logs/deploy.log" || true; exit 1; }

CFG_WS="${OUT}/ws_cup.yml"
write_cfg "${CFG_WS}" 8848 "ws_single" "${ENV_CUP}"
echo "[A] websocket single CupToDrawer + video"
POLICY_CONFIG_PATH="${CFG_WS}" \
ROBOCASA_GR1_GPU=3 MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 \
bash benchmarks/robocasa_gr1/single_eval.sh "${ENV_CUP}" 8848 127.0.0.1 \
  2>&1 | tee "${OUT}/logs/ws_single_cup.log"
echo "[A] done"

# stop websocket deploy before batched reuses GPU2
if [[ -f "${OUT}/deploy.pid" ]]; then
  kill "$(cat "${OUT}/deploy.pid")" 2>/dev/null || true
  sleep 2
fi
# also kill any leftover deploy on 8848 we started / orphaned
pkill -f "scripts/deploy.py --ckpt-dir ${CKPT_DIR}" 2>/dev/null || true
sleep 2

# ---------- B: batched single-sim ----------
echo "[B] batched single-sim CupToDrawer + video"
MODE=full NUM_EPISODES=1 INFER_GPUS=2 SIM_GPUS=3 N_SIMS=1 \
RUN_TAG="video_batch1_$(basename "${OUT}")" \
OUTPUT_DIR="${OUT}/batch_single" \
bash scripts/eval_robocasa_gr1.sh \
  --env-ids "${ENV_CUP}" \
  --save-video \
  2>&1 | tee "${OUT}/logs/batch_single.log"
# copy videos to unified tree if present
mkdir -p "${OUT}/videos/batch_single"
cp -a "${OUT}/batch_single/videos/." "${OUT}/videos/batch_single/" 2>/dev/null || true
echo "[B] done"

# ---------- C: batched dual-sim ----------
echo "[C] batched dual-sim Cup+Potato + video"
MODE=full NUM_EPISODES=1 INFER_GPUS=2 SIM_GPUS=3,4 N_SIMS=1 \
RUN_TAG="video_batch2_$(basename "${OUT}")" \
OUTPUT_DIR="${OUT}/batch_dual" \
bash scripts/eval_robocasa_gr1.sh \
  --env-ids "${ENV_CUP},${ENV_POTATO}" \
  --save-video \
  2>&1 | tee "${OUT}/logs/batch_dual.log"
mkdir -p "${OUT}/videos/batch_dual"
cp -a "${OUT}/batch_dual/videos/." "${OUT}/videos/batch_dual/" 2>/dev/null || true
echo "[C] done"

echo "==== videos ===="
find "${OUT}/videos" -name '*.mp4' -ls || true
echo "OUT=${OUT}"
