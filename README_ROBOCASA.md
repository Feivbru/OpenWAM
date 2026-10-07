# RoboCasa GR1 Evaluation (OpenWAM)

Quick start for evaluating the **OpenWAM-Alpha-Sim-RoboCasa-GR1** release checkpoint on the official 24 `gr1_unified/*` tabletop tasks.

## Prerequisites

| Piece | Default |
|---|---|
| Policy / server env | `openwam_re` (`/data/anaconda3/envs/openwam_re/bin/python`) |
| Sim / client env | `robocasa-gr1` (`/data/anaconda3/envs/robocasa-gr1/bin/python`) |
| Sim repo | `third_party/robocasa-gr1-tabletop-tasks` (or `ROBOCASA_GR1_PATH`) |
| Wan TI2V weights | `WAN_PATH` → Wan2.2-TI2V-5B |
| Release ckpt | `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1` |

Client env bootstrap: see [`benchmarks/robocasa_gr1/README.md`](benchmarks/robocasa_gr1/README.md).

## Recommended: dual-group release eval

Default topology (8× H20 style):

| Group | Infer + T5 encode | Sim GPUs | Clients |
|---|---|---|---|
| A | GPU **6** | **0,1,2** | 3 (`n_sims=1` each), `max_infer_batch=3` |
| B | GPU **7** | **3,4,5** | 3, `max_infer_batch=3` |

- 24 official tasks split **12 + 12**
- **10** episodes / task
- **`n_action_steps` / `inference_horizon` = 8** (align with GR00T RoboCasa GR1)
- Fourier hand **discrete projection OFF** (`project_discrete_hands: false`)
- Release checkpoint under `assets/openwam_ckpt/.../OpenWAM-Alpha-Sim-RoboCasa-GR1`

```bash
cd /path/to/OpenWAM
# Prefer a tmux session so the job survives disconnects
tmux new -s robocasa_gr1_release
bash scripts/eval_robocasa_gr1_release_dual.sh
```

Outputs land in `outputs/robocasa_gr1/release_dual_h8_<timestamp>/`:

- `run_config.json` — frozen topology / knobs
- `group_{a,b}/` — per-group scheduler tree (`launcher.log`, `results/<task>/result.json`, …)
- root `summary.json` / CSV — merged over both groups via `summarize.py`

### Common overrides

```bash
# Custom output dir
OUTPUT_DIR=outputs/robocasa_gr1/my_run \
  bash scripts/eval_robocasa_gr1_release_dual.sh

# Different GPU map (example: infer 4/5, sim 0-3 + 6-7)
INFER_A=4 SIM_A=0,1,2 PORT_A=9300 \
INFER_B=5 SIM_B=3,6,7 PORT_B=9400 \
  bash scripts/eval_robocasa_gr1_release_dual.sh

# Full action chunk instead of 8
INFERENCE_HORIZON=  # empty → omit server flag; also set n_action_steps: null in policy YAML
# Or keep server truncation only:
INFERENCE_HORIZON=8 NUM_EPISODES=10 \
  bash scripts/eval_robocasa_gr1_release_dual.sh

# Another checkpoint
CKPT_DIR=/path/to/ckpt CKPT_NAME=checkpoint_step_XXXXX.safetensors \
  bash scripts/eval_robocasa_gr1_release_dual.sh
```

## Client policy config

Template: [`benchmarks/robocasa_gr1/policy_config.yml`](benchmarks/robocasa_gr1/policy_config.yml).

| Key | Release default | Meaning |
|---|---|---|
| `n_action_steps` | `8` | Client executes only the first N actions of each chunk, then re-queries |
| `project_discrete_hands` | `false` | Nearest-neighbour snap of Fourier hand commands; off for this release run |
| `num_episodes` | patched by launcher | Episodes per env id |
| `max_steps` | `720` | Episode horizon |
| `send_state` / `state_dim` | `true` / `33` | Live FK → EEF33 proprio |

Copy and point the launcher at your copy:

```bash
POLICY_CONFIG_PATH=/path/to/my_policy.yml \
  bash scripts/eval_robocasa_gr1_release_dual.sh
```

Server-side truncation uses the same number via `INFERENCE_HORIZON` → `batched_server --inference-horizon` (keeps encoder/GPU work aligned with what the client will execute).

## Single-group / custom topology

Lower-level entry (one scheduler process):

```bash
INFER_GPUS=6 SIM_GPUS=0,1,2 \
N_SIMS=1 MAX_INFER_BATCH=3 \
INFERENCE_HORIZON=8 NUM_EPISODES=10 \
POLICY_CONFIG_PATH=benchmarks/robocasa_gr1/policy_config.yml \
OUTPUT_DIR=outputs/robocasa_gr1/group_a_only \
bash benchmarks/robocasa_gr1/run_eval.sh \
  assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1 \
  checkpoint_step_100000.safetensors \
  --env-ids 'gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env,...'
```

Wrapper with release/FPD auto-resolve: `scripts/eval_robocasa_gr1.sh`.

### Env knobs (`run_eval.sh`)

| Env | Default | Role |
|---|---|---|
| `INFER_GPUS` | `2` | Comma-separated infer GPUs (one encoder+batched server each) |
| `SIM_GPUS` | `0,1,3,...` | Render / MuJoCo GPUs (must be disjoint from infer) |
| `N_SIMS` | `1` | Client processes per sim GPU |
| `MAX_INFER_BATCH` | `2` | `batched_server --max-batch` |
| `INFERENCE_HORIZON` | _(empty)_ | Truncate returned chunk on the server |
| `ENV_IDS` | _(all 24)_ | Comma-separated gym ids |
| `NUM_EPISODES` | `10` | Episodes per task |
| `BASE_PORT` | `9300` | Infer WS port base (`+10` per infer GPU) |
| `POLICY_CONFIG_PATH` | `benchmarks/robocasa_gr1/policy_config.yml` | Client YAML |
| `WAN_PATH` | Motus Wan2.2-TI2V-5B path | Shared T5 / VAE assets for encoder |
| `OUTPUT_DIR` | `outputs/robocasa_gr1/<tag>` | Results root |

`MODE=smoke` runs 1–2 tasks only — **not** used by the release dual script.

## Official WS (single server) smoke

For drip-feed debugging against `scripts/deploy.py` (not batched):

```bash
# Terminal 1 — policy server (pick a free GPU)
CUDA_VISIBLE_DEVICES=2 python scripts/deploy.py \
  --ckpt-dir assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1 \
  --device cuda:0 --port 8848 --inference-mode sync --inference-horizon 8

# Terminal 2 — client
ROBOCASA_GR1_GPU=3 \
POLICY_CONFIG_PATH=benchmarks/robocasa_gr1/policy_config.yml \
bash benchmarks/robocasa_gr1/single_eval.sh \
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env 8848 127.0.0.1
```

Patch `num_episodes` in a copied YAML or pass overrides through `single_eval.py --num-episodes`.

## Summarize

```bash
python benchmarks/robocasa_gr1/summarize.py outputs/robocasa_gr1/<run_dir>
```

## Notes

- Images are **256×256** from GrootRoboCasaEnv; the policy canvas is L-shape **384×320**. Blurry rollout videos are expected at 256.
- EEF33 → joint IK (`hold_on_failure`) is OpenWAM-specific; official GR00T joint policies do not use this path.
- Infer and sim GPU lists **must not overlap**.
