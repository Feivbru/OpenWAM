# OpenWAM examples

Real-robot / smoke clients that speak the **native OpenWAM JSON WebSocket**
protocol (not openpi-client / msgpack).

## Piper EEF10 client

[`piper_eef_client.py`](piper_eef_client.py) is a port of the previous
`infer/example/eef_main.py` client:

| | old (`eef_main.py`) | this example |
|---|---|---|
| Protocol | openpi msgpack (`WebsocketClientPolicy`) | OpenWAM JSON (`WSPolicyClient`) |
| Default port | 8000 | **8848** |
| Deps | `openpi-client`, `tyro`, … | `websockets`, `Pillow`, `numpy` (+ optional `piper_sdk` / RealSense) |
| Action I/O | full chunk `actions[H,10]` open-loop | **step** (default) or **chunk** via `--return-action-chunk` |

### Action return modes (server + client must match)

| Mode | Server | Client | Wire |
|------|--------|--------|------|
| **step** (default) | drip-feed from buffer | one WS call per control tick | `{action: [...]}` |
| **chunk** | return full executable horizon, clear buffer | one WS call per chunk, local open-loop | `{action, actions: [[...], ...]}` |

```bash
# Server — chunk mode
CUDA_VISIBLE_DEVICES=7 bash scripts/deploy.sh /path/to/ckpt --port 8848 --return-action-chunk

# Client — chunk mode (optional --open-loop-horizon N to use only the first N steps)
python examples/piper_eef_client.py \
  --host 127.0.0.1 --port 8848 \
  --return-action-chunk \
  --open-loop-horizon 16 \
  --prompt '...' --can-name can0 --camera-backend opencv
```

Ping/pong advertises `return_action_chunk` so a mismatched pair fails fast.

### 1. Start the OpenWAM JSON server

From the OpenWAM repo root, with the `openwam` conda env:

```bash
conda activate openwam
cd /path/to/OpenWAM

# Native JSON PolicyServer (NOT --protocol openpi)
CUDA_VISIBLE_DEVICES=7 bash scripts/deploy.sh /path/to/your_ckpt --port 8848
```

Useful flags (see `scripts/deploy.py` / `configs/deploy.yaml`):

```bash
bash scripts/deploy.sh /path/to/ckpt --port 8848 \
  --inference-horizon 16   # optional: replan every N actions instead of full chunk
```

### 2. Start the client

```bash
conda activate openwam
cd /path/to/OpenWAM

# Live robot (OpenCV cameras; adjust ids / CAN)
python examples/piper_eef_client.py \
  --host 127.0.0.1 --port 8848 \
  --prompt 'Pick up the red block and place it in the box.' \
  --can-name can0 \
  --camera-backend opencv \
  --head-camera 0 --wrist-camera 2 \
  --show-cameras

# RealSense cameras
python examples/piper_eef_client.py \
  --host 127.0.0.1 --port 8848 \
  --camera-backend realsense \
  --head-camera-serial 339322074804 \
  --wrist-camera-serial 346522074547 \
  --can-name can0

# Smoke against a running server without robot / cameras
python examples/piper_eef_client.py \
  --host 127.0.0.1 --port 8848 \
  --camera-backend fake \
  --dry-run \
  --max-timesteps 5 \
  --log-actions
```

### Notes

- Run from the **OpenWAM repo root** so `benchmarks.utils` imports work.
- Banana / pick_blocks multiview ckpts: keep `--resize-lshape` (default on).
- Prompt is forwarded **verbatim**; use the exact instruction string used in training / T5 cache.
- Between episodes the client calls `reset()` once at start; call again if you restart a task.
- `piper_sdk` is only required when not using `--dry-run`. RealSense needs `pyrealsense2` (optional).

## VRTC cube-prefetch topology toy

[`vrtc_topology_sim.py`](vrtc_topology_sim.py) is a **discrete-event** sandbox (no GPU /
robot) for the VRTC wire + replan rules:

- wire always returns **one cube**; client open-loop horizon = cube length
- `replan_cubes=2` starts background infer; merge **overwrites** wait after the
  last delivered cube (skip head so output sits right after the snapshotted real)

```bash
python examples/vrtc_topology_sim.py                  # n=0..100 full log
python examples/vrtc_topology_sim.py --quiet          # summary only
python examples/vrtc_topology_sim.py --n-max 30 --csv /tmp/vrtc_topo.csv
```

For a **real** three-thread run (Client / Server / Env with actual
``time.sleep`` + ``perf_counter`` wall clock), use:

```bash
python examples/vrtc_topology_realtime.py --n-max 20
python examples/vrtc_topology_realtime.py --n-max 12 --quiet
```

### VRTC deploy smoke (fake robot)

With a VRTC checkpoint and GPU, exercise cube return + prefetch:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/deploy.py \
  --ckpt-dir outputs/pick_blocks_piper_vrtc/<run> \
  --port 8857 --return-action-chunk --device cuda:0

python examples/piper_eef_client.py \
  --host 127.0.0.1 --port 8857 \
  --camera-backend fake --dry-run \
  --return-action-chunk --open-loop-horizon 4 \
  --max-timesteps 40 \
  --prompt 'Pick up the red block and place it in the box.'
```

Pong may include ``vrtc.cube_action_len`` (= ``video_stride``); keep client
open-loop horizon equal to that.
