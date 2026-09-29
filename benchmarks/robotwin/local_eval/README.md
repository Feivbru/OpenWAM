# OpenWAM FastWAM-style local RoboTwin eval

Two evaluation stacks:

| Path | When to use |
|------|-------------|
| `benchmarks/robotwin/batched_eval.sh` | Shared WS inference server + many sim slots (saves VRAM; good for colocate). |
| `scripts/robotwin_local.sh` | FastWAM-like: each worker loads OpenWAM **in-process** with Sapien (faster wall-clock per episode when VRAM allows; 1 model/GPU). |

## Requirements

- Conda env `openwam` with both `openwam` and `sapien` (and curobo) importable in **one** Python.
- Do **not** stop an active `batched_eval` unless intended; pick free GPUs via `CUDA_VISIBLE_DEVICES` (often avoid 2,3 while `fpd_colocate6_norender` runs).

## Launch

```bash
cd OpenWAM
# Prefer GPUs not used by colocate batched_eval (often 2,3):
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/robotwin_local.sh

# Smoke one task / clean only / 1 episode:
ROBOTWIN_TEST_NUM=1 EVAL_PHASES=clean \
  CUDA_VISIBLE_DEVICES=4 \
  bash scripts/robotwin_local.sh \
  EVALUATION.task_name=shake_bottle_horizontally \
  EVALUATION.eval_num_episodes=1
```

Defaults:

- `CKPT_DIR=outputs/openwam_fpd_robotwin_full/2026-09-24_22-18-53`
- `MULTIRUN.max_tasks_per_gpu=1` (OpenWAM FPD is large; unlike FastWAM's up-to-3 replicas/GPU)
- `action_type=ee` (matches training `action_mode: eef`; **not** FastWAM `qpos`)
- `skip_text_encoder=true` (FPD ckpts omit UMT5; prompts encoded on CPU in-process)

## Outputs

```
evaluate_results/robotwin_local/<ckpt_tag>/<run_ts>/
  manager.log
  summary.csv          # rates as percent with 2 decimals
  summary.json
  failed_tasks.txt
  <task>/_result_clean.txt
  <task>/_result_random.txt
  eval_<task>_*.log
```

## Notes

- `eval_skip_front_camera: true` is enabled in OpenWAM RoboTwin task configs (same idea as FastWAM).
- `ROBOTWIN_DISABLE_RT` defaults to `0` (RT on). Only set to `1` for debugging.
- Smoke timing (1 ep `shake_bottle_horizontally` / `demo_clean`): expert setup ~53s, eval episode ~214s including first-time compile; warm workers are expected much closer to the ~40s/ep non-colocate ballpark.
