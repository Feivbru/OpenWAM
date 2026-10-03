#!/usr/bin/env python3
"""Offline FK: banana LeRobot joint7 -> EEF10 sidecar (does not modify original data).

Writes::

    <dataset_dir>/derived/eef/
      meta.json
      episode_XXXXXX.npz   # state_eef10 (T,10), action_eef10 (T,10), optionally joint copies

EEF10 layout matches OpenWAM single-arm contract:
  [xyz_m(3), rot6d(6), gripper_open_scale(1)] with -1=closed, +1=open.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.utils.piper_fk import (
    JOINT7_DIM,
    joints7_to_eef10,
    make_piper_fk,
)


def _read_joint_columns(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(parquet_path)
    # Avoid scalar columns that can trip pyarrow histogram bugs on this corpus.
    state = pf.read(columns=["observation.state"]).column(0).combine_chunks()
    action = pf.read(columns=["action"]).column(0).combine_chunks()
    state_np = state.values.to_numpy().reshape(len(state), JOINT7_DIM).astype(np.float32)
    action_np = action.values.to_numpy().reshape(len(action), JOINT7_DIM).astype(np.float32)
    return state_np, action_np


def _scan_gripper_max(parquet_paths: list[Path]) -> float:
    gmax = 0.0
    for path in parquet_paths:
        state, action = _read_joint_columns(path)
        gmax = max(gmax, float(state[:, 6].max()), float(action[:, 6].max()))
    if gmax <= 0:
        raise ValueError("gripper max is non-positive across the dataset")
    return gmax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/zixian_guo/projects/haoming/project/PI/riri/data/banana"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/derived/eef",
    )
    parser.add_argument(
        "--dh-is-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="piper_sdk DH offset flag (1 = current URDF / S-V1.6-3+)",
    )
    parser.add_argument(
        "--gripper-max-m",
        type=float,
        default=None,
        help="If unset, scanned from all episodes (state/action joint_6)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    out_dir = (args.out_dir or (dataset_dir / "derived" / "eef")).resolve()
    data_root = dataset_dir / "data"
    parquet_paths = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        print(f"no parquet files under {data_root}", file=sys.stderr)
        return 1

    # Only write under <dataset>/derived/... so original parquet/videos stay untouched.
    try:
        rel = out_dir.relative_to(dataset_dir)
    except ValueError:
        print(f"refusing out_dir outside dataset_dir: {out_dir}", file=sys.stderr)
        return 1
    if not rel.parts or rel.parts[0] != "derived":
        print(f"refusing to write outside derived/: {out_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    gripper_max = float(args.gripper_max_m) if args.gripper_max_m is not None else _scan_gripper_max(parquet_paths)
    fk = make_piper_fk(dh_is_offset=args.dh_is_offset)

    meta = {
        "format": "banana_piper_eef10_v1",
        "source_dataset": str(dataset_dir),
        "dh_is_offset": int(args.dh_is_offset),
        "fk_backend": "piper_sdk.C_PiperForwardKinematics",
        "xyz_unit": "meter",
        "rotation": "rot6d_first_two_columns_of_R",
        "gripper_unit_source": "meter_sdk_total_opening",
        "gripper_max_m": gripper_max,
        "gripper_mapping": "open_scale = 2*(g/gripper_max_m)-1 ; -1=closed +1=open",
        "eef10_layout": ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper_open_scale"],
        "original_data_untouched": True,
        "n_episodes": len(parquet_paths),
        "episodes": [],
    }

    for path in parquet_paths:
        ep_idx = int(path.stem.split("_")[-1])
        state_j, action_j = _read_joint_columns(path)
        state_eef = joints7_to_eef10(fk, state_j, gripper_max)
        action_eef = joints7_to_eef10(fk, action_j, gripper_max)
        out_path = out_dir / f"episode_{ep_idx:06d}.npz"
        np.savez_compressed(
            out_path,
            state_eef10=state_eef,
            action_eef10=action_eef,
            state_joint7=state_j,
            action_joint7=action_j,
        )
        meta["episodes"].append(
            {
                "episode_index": ep_idx,
                "length": int(state_eef.shape[0]),
                "path": out_path.name,
                "state_xyz_min": state_eef[:, :3].min(axis=0).tolist(),
                "state_xyz_max": state_eef[:, :3].max(axis=0).tolist(),
            }
        )
        print(f"wrote {out_path.name} T={state_eef.shape[0]}")

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta_path} gripper_max_m={gripper_max:.6f} dh_is_offset={args.dh_is_offset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
