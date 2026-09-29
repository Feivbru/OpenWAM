#!/usr/bin/env bash
# Launch / babysit Cursor CLI against the FPD post-train eval pipeline.
# Survives SSH disconnect when run inside tmux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/logs/fpd_cli_agent_prompt.txt}"
AGENT_LOG="$ROOT/logs/fpd_cli_agent_$(date +%Y%m%d_%H%M%S).log"
STATUS="$ROOT/logs/fpd_cli_agent.status"

echo "starting $(date -Is)" | tee "$STATUS"
echo "[fpd_cli_agent] log=$AGENT_LOG"

# Restart loop: if the print-mode agent exits early while pipeline still running, relaunch.
while true; do
  echo "[fpd_cli_agent] agent start $(date -Is)" | tee -a "$STATUS"
  set +e
  # line-buffer so tmux/tee show progress while agent runs for hours
  stdbuf -oL -eL agent -p \
    --force \
    --trust \
    --sandbox disabled \
    --workspace "$ROOT" \
    --output-format text \
    "$(cat "$PROMPT_FILE")" \
    2>&1 | stdbuf -oL -eL tee -a "$AGENT_LOG"
  ec=${PIPESTATUS[0]}
  set -e
  echo "[fpd_cli_agent] agent exit ec=$ec at $(date -Is)" | tee -a "$STATUS"

  # Stop looping if train+eval both finished cleanly
  train_alive=0
  kill -0 "${TORCH_PID:-3714787}" 2>/dev/null && train_alive=1 || true
  pgrep -f 'scripts/train.py --config-name=train_fpd_robotwin_full' >/dev/null 2>&1 && train_alive=1 || true

  eval_alive=0
  pgrep -f 'batched_eval.sh|batched_server|robotwin.*eval' >/dev/null 2>&1 && eval_alive=1 || true

  st="$(cat "$ROOT/logs/fpd_posttrain_eval.status" 2>/dev/null || echo missing)"
  echo "[fpd_cli_agent] train_alive=$train_alive eval_alive=$eval_alive eval_status=$st" | tee -a "$STATUS"

  if [[ "$train_alive" -eq 0 && "$eval_alive" -eq 0 && "$st" == done_ok* ]]; then
    echo "pipeline_complete $(date -Is)" | tee -a "$STATUS"
    exit 0
  fi
  if [[ "$train_alive" -eq 0 && "$st" == failed* ]]; then
    echo "pipeline_failed $(date -Is) — agent will retry after backoff" | tee -a "$STATUS"
  fi

  echo "[fpd_cli_agent] sleeping 180s before relaunch..." | tee -a "$STATUS"
  sleep 180
done
