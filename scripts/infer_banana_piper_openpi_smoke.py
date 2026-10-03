#!/usr/bin/env python3
"""Smoke against the openpi Piper EEF10 server (msgpack WebsocketClientPolicy wire).

Prereqs — start server first::

    CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt --port 8000

Then::

    python scripts/infer_banana_piper_openpi_smoke.py --host 127.0.0.1 --port 8000 \\
      --dataset-dir /data/.../riri/data/banana --episode 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openwam.deploy import msgpack_numpy  # noqa: E402
from scripts.infer_banana_piper_smoke import _load_frame_obs  # noqa: E402

HEAD_KEY = "observation.images.top_head"
WRIST_KEY = "observation.images.hand_right"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/zixian_guo/projects/haoming/project/PI/riri/data/banana"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise SystemExit("need websockets>=12 with sync client") from exc

    obs_frame, prompt, _gmax = _load_frame_obs(args.dataset_dir.resolve(), args.episode, args.frame_index)
    payload = {
        "observation/top_image": np.asarray(obs_frame[HEAD_KEY], dtype=np.uint8),
        "observation/right_wrist_image": np.asarray(obs_frame[WRIST_KEY], dtype=np.uint8),
        "observation/state": np.asarray(obs_frame["state_eef10"], dtype=np.float32),
        "prompt": prompt,
    }

    url = f"ws://{args.host}:{args.port}"
    print(f"[openpi-smoke] connect {url}")
    kwargs = {"max_size": None, "compression": None, "open_timeout": 10.0, "ping_interval": None}
    import inspect

    supported = set(inspect.signature(connect).parameters)
    with connect(url, **{k: v for k, v in kwargs.items() if k in supported}) as ws:
        meta = msgpack_numpy.unpackb(ws.recv())
        print(f"[openpi-smoke] metadata={meta}")
        for key in ("action_horizon", "raw_action_dim", "wire_action_space", "action_representation"):
            if key not in meta:
                raise SystemExit(f"metadata missing {key}")
        if int(meta["raw_action_dim"]) != 10:
            raise SystemExit(f"raw_action_dim={meta['raw_action_dim']} != 10")

        ws.send(msgpack_numpy.packb(payload))
        try:
            raw = ws.recv(timeout=args.timeout)
        except TypeError:
            raw = ws.recv()
        if isinstance(raw, str):
            print(raw)
            raise SystemExit("server returned text error frame")
        result = msgpack_numpy.unpackb(raw)

    actions = np.asarray(result["actions"], dtype=np.float32)
    h = int(meta["action_horizon"])
    print(f"[openpi-smoke] actions.shape={actions.shape} (expect ({h}, 10))")
    if actions.shape != (h, 10):
        raise SystemExit(f"bad actions shape {actions.shape}")
    if not np.isfinite(actions).all():
        raise SystemExit("non-finite actions")
    print(f"[openpi-smoke] actions[0]={actions[0].tolist()}")
    print(f"[openpi-smoke] policy_timing={result.get('policy_timing')}")
    print("[openpi-smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
