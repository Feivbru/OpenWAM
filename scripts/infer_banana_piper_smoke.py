#!/usr/bin/env python3
"""Offline smoke: one banana episode frame → OpenWAM deploy server → physical EEF10.

Prereqs:
  1. A banana FT checkpoint dir with ``config.yaml`` + ``checkpoint_step_*.safetensors``.
  2. Policy server already running, e.g.::

       CUDA_VISIBLE_DEVICES=6 bash scripts/deploy.sh /path/to/banana_ckpt --port 8848

  3. Dataset with videos + ``derived/eef/episode_XXXXXX.npz`` (from
     ``scripts/preprocess_banana_piper_eef.py``).

Usage::

    cd OpenWAM
    python scripts/infer_banana_piper_smoke.py \\
      --host 127.0.0.1 --port 8848 \\
      --dataset-dir /data/.../riri/data/banana \\
      --episode 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.banana_piper.openwam2banana_interface import OpenWAMBananaPolicy  # noqa: E402
from openwam.dataloader.utils.video_io import decode_video_frames  # noqa: E402

DEFAULT_DATASET = Path("/data/zixian_guo/projects/haoming/project/PI/riri/data/banana")
HEAD_KEY = "observation.images.top_head"
WRIST_KEY = "observation.images.hand_right"


def _find_video(dataset_dir: Path, video_key: str, episode: int) -> Path:
    pattern = f"videos/**/{video_key}/episode_{episode:06d}.mp4"
    hits = sorted(dataset_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no video for episode {episode} key={video_key} under {dataset_dir}")
    return hits[0]


def _load_prompt(dataset_dir: Path, episode: int) -> str:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if int(row.get("episode_index", -1)) == int(episode):
                    tasks = row.get("tasks") or []
                    if tasks:
                        return str(tasks[0])
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        with tasks_path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if first:
                return str(json.loads(first).get("task", "")).strip()
    raise FileNotFoundError(f"could not resolve prompt for episode {episode}")


def _load_frame_obs(dataset_dir: Path, episode: int, frame_index: int) -> tuple[dict, str, float]:
    eef_path = dataset_dir / "derived" / "eef" / f"episode_{episode:06d}.npz"
    meta_path = dataset_dir / "derived" / "eef" / "meta.json"
    if not eef_path.is_file():
        raise FileNotFoundError(
            f"missing {eef_path}; run scripts/preprocess_banana_piper_eef.py first"
        )
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gripper_max = float(meta["gripper_max_m"])
    with np.load(eef_path) as data:
        state_eef10 = np.asarray(data["state_eef10"], dtype=np.float32)
    if frame_index < 0 or frame_index >= state_eef10.shape[0]:
        raise IndexError(
            f"frame_index={frame_index} out of range for episode {episode} "
            f"(T={state_eef10.shape[0]})"
        )

    head_path = _find_video(dataset_dir, HEAD_KEY, episode)
    wrist_path = _find_video(dataset_dir, WRIST_KEY, episode)
    # decode_video_frames resizes to (height, width); smoke wants native RGB for
    # client-side L-shape slot resize — pass a large canvas then the client
    # resizes again. Use metadata sizes from the first decode without forcing
    # a tiny canvas: height/width here are the PIL resize target inside video_io.
    # Read once at native-ish 480x640 then let the client LANCZOS to slot sizes.
    head_imgs = decode_video_frames(str(head_path), [frame_index], height=480, width=640)
    wrist_imgs = decode_video_frames(str(wrist_path), [frame_index], height=480, width=640)
    head = np.asarray(head_imgs[0].convert("RGB"), dtype=np.uint8)
    wrist = np.asarray(wrist_imgs[0].convert("RGB"), dtype=np.uint8)

    obs = {
        HEAD_KEY: head,
        WRIST_KEY: wrist,
        "state_eef10": state_eef10[frame_index],
    }
    prompt = _load_prompt(dataset_dir, episode)
    return obs, prompt, gripper_max


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    obs, prompt, gripper_max = _load_frame_obs(dataset_dir, args.episode, args.frame_index)
    print(f"[smoke] dataset={dataset_dir}")
    print(f"[smoke] episode={args.episode} frame={args.frame_index}")
    print(f"[smoke] prompt={prompt!r}")
    print(f"[smoke] head={obs[HEAD_KEY].shape} wrist={obs[WRIST_KEY].shape}")
    print(f"[smoke] state_eef10={np.asarray(obs['state_eef10']).tolist()}")

    policy = OpenWAMBananaPolicy(
        host=args.host,
        port=args.port,
        request_timeout=args.timeout,
        gripper_max_m=gripper_max,
        debug=bool(args.debug),
    )
    try:
        policy.reset()
        action = policy.act(obs, prompt)
    finally:
        policy.close()

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    print(f"[smoke] action_eef10 shape={action.shape}")
    print(f"[smoke] action_eef10={action.tolist()}")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
