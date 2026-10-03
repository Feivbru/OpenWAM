"""Compute Banana Piper EEF10 normalization statistics from the train split.

Pools action_eef10 and state_eef10 from ``derived/eef/*.npz`` for episodes listed
in ``splits/manifest.json`` train. rot6d dims 3..8 are pinned to identity.
"""

from __future__ import annotations

import argparse
import os
import socket
import uuid
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.banana_piper import (
    GRIPPER_CONVENTION,
    GRIPPER_TRANSFORM,
    RAW_GRIPPER_CONVENTION,
    ROT6D_DIMS_EEF10,
    ROTATION_CONVENTION,
    STATS_POPULATION,
    BananaPiperDataset,
)
from openwam.dataloader.utils.normalization import pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _iter_episode_arrays(bucket: BananaPiperDataset):
    for _, row in bucket._eps_df.iterrows():  # noqa: SLF001
        ep = int(row["episode_index"])
        payload = bucket._load_eef_episode(ep)  # noqa: SLF001
        yield payload["action_eef10"], payload["state_eef10"]


def _compute_global_stats(dataset: BananaPiperDataset, reservoir_cap: int):
    if not isinstance(dataset, BananaPiperDataset):
        raise TypeError(f"expected BananaPiperDataset, got {type(dataset).__name__}")
    raw_dim = dataset._raw_action_dim  # noqa: SLF001
    accumulator = Accumulator(dim=raw_dim, reservoir_cap=reservoir_cap)
    action_rows = 0
    state_rows = 0
    for action, state in _iter_episode_arrays(dataset):
        action = np.asarray(action, np.float32).reshape(-1, raw_dim)
        state = np.asarray(state, np.float32).reshape(-1, raw_dim)
        accumulator.update_batch(action)
        accumulator.update_batch(state)
        action_rows += action.shape[0]
        state_rows += state.shape[0]
    if action_rows == 0 or state_rows == 0:
        raise ValueError("cannot compute Banana Piper stats from an empty dataset")

    stats = accumulator.finalize()
    stats.update(
        {
            "num_timesteps": accumulator.count,
            "pool": "action_state_eef10",
            "action_rows": action_rows,
            "state_rows": state_rows,
            "gripper_convention": GRIPPER_CONVENTION,
            "raw_gripper_convention": RAW_GRIPPER_CONVENTION,
            "gripper_transform": GRIPPER_TRANSFORM,
            "rotation_convention": ROTATION_CONVENTION,
            "rotation_transform": "piper_fk_matrix_to_rot6d",
            "action_alignment": "absolute_leader_command_eef_via_fk",
            "stats_population": STATS_POPULATION,
            "split": dataset._split,  # noqa: SLF001
            "num_episodes": len(dataset._eps_df),  # noqa: SLF001
        }
    )
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF10)
    return dataset.action_mode, raw_dim, stats, action_rows, state_rows


def build_and_save_banana_piper_stats(
    dataset: BananaPiperDataset,
    output: str | Path,
    reservoir_cap: int = 1_000_000,
):
    output = Path(output)
    action_mode, raw_dim, global_stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap)
    payload = {}
    if output.exists():
        try:
            previous = np.load(output, allow_pickle=True).item()
            if isinstance(previous, dict):
                payload.update(previous)
        except (ValueError, EOFError):
            pass
    payload[action_mode] = global_stats
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, payload)
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return action_mode, raw_dim, action_rows, state_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/banana_piper.yaml")
    parser.add_argument("--output", required=True, help="Deploy-compatible .npy output")
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    dataset = BananaPiperDataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))
    action_mode, raw_dim, action_rows, state_rows = build_and_save_banana_piper_stats(
        dataset, output, args.reservoir_cap
    )
    print(
        f"wrote {output} mode={action_mode} dim={raw_dim} "
        f"action_rows={action_rows} state_rows={state_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
