"""Banana AgileX Piper single-arm LeRobot v2.1 reader (EEF10 from derived FK sidecar).

Original joint parquet / videos are left untouched. Training supervision uses
``derived/eef/episode_XXXXXX.npz`` produced by
``scripts/preprocess_banana_piper_eef.py``.

EEF10: ``[xyz_m3, rot6d6, gripper_open_scale1]`` with ``-1=closed / +1=open``,
scattered into α left-arm slots ``0:10`` when ``unify_action_map: ["0-9"]``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.bases.lerobot_v3_reader import _read_data_table_cached
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_ARM10,
    apply_normalization,
    load_stats_file,
    load_stats_metadata,
)

logger = logging.getLogger(__name__)

_ACTION_MODE = "eef"
EEF10_DIM = 10
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
RAW_GRIPPER_CONVENTION = "meter_sdk_total_opening"
GRIPPER_TRANSFORM = "2_times_g_over_gmax_minus_1"
ROTATION_CONVENTION = "piper_fk_rot6d_R_cols01"
NORMALIZATION_STATS_FILENAME = "normalization_stats.npy"
STATS_POPULATION = "train_split_manifest"
EEF_SIDECAR_DIRNAME = "derived/eef"


def _as_priority(value: Optional[Sequence[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    return next((key for key in priorities if key in features), None)


class BananaPiperDataset(LeRobotV3Reader):
    """LeRobot v2.1 banana Piper reader with offline EEF10 sidecar."""

    DATASET_NAME = "BananaPiper"
    ACTION_DIM = EEF10_DIM
    NEEDED_COLS = ("task_index", "episode_index", "frame_index")
    PROMPT_FILE_REQUIRED = False
    DEPLOY_ACTION_MODE = _ACTION_MODE

    HEAD_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = ("observation.images.top_head",)
    WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = ("observation.images.hand_right",)
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "gripper_convention",
        "head_camera_priority",
        "wrist_camera_priority",
        "normalization_stats_path",
        "eef_dir",
        "split_manifest_path",
        "use_t5_cache",
        "t5_cache_dirname",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = _ACTION_MODE,
        gripper_convention: str = GRIPPER_CONVENTION,
        head_camera_priority: Optional[Sequence[str]] = None,
        wrist_camera_priority: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
        eef_dir: Optional[str] = None,
        split_manifest_path: Optional[str] = None,
        use_t5_cache: bool = False,
        t5_cache_dirname: str = "umt5_openwam",
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != _ACTION_MODE:
            raise ValueError(f"Banana Piper supports only action_mode='eef', got {action_mode!r}")
        convention = str(gripper_convention).strip()
        if convention != GRIPPER_CONVENTION:
            raise ValueError(
                f"Banana Piper requires gripper_convention={GRIPPER_CONVENTION!r}, got {gripper_convention!r}"
            )
        unify_on = bool(unify_action)
        if unify_on and unify_action_map is None:
            raise ValueError(
                "Banana Piper unify_action=true requires an explicit unify_action_map; "
                'set ["0-9"] for the canonical single-arm left-slot mapping'
            )
        self.action_mode = mode
        self._head_priority = _as_priority(head_camera_priority, self.HEAD_CAMERA_PRIORITY)
        self._wrist_priority = _as_priority(wrist_camera_priority, self.WRIST_CAMERA_PRIORITY)
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._resolved_stats_path: Optional[str] = None
        self._eef_dir_override = Path(eef_dir) if eef_dir else None
        self._split_manifest_path = Path(split_manifest_path) if split_manifest_path else None
        self._eef_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._eef_meta: Optional[dict] = None
        self.use_t5_cache = bool(use_t5_cache)
        self.t5_cache_dirname = str(t5_cache_dirname or "umt5_openwam")
        self._t5_by_task: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._prompt_to_task_idx: Dict[str, int] = {}
        # Must be ready before super().__init__ because _load_stats runs before _post_init.
        self._eef_dir = self._eef_dir_override or (Path(dataset_dir) / EEF_SIDECAR_DIRNAME)
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=unify_on,
            unify_action_map=unify_action_map,
            **kwargs,
        )

    # ── cameras / layout ───────────────────────────────────────────────────

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        if self._target_camera is not None:
            return self._target_camera, None, None
        head = _pick_feature(features, self._head_priority)
        wrist = _pick_feature(features, self._wrist_priority)
        # Single wrist camera sits in the right L-shape slot; left stays black.
        return head, None, wrist

    def _expected_canvas_hw(self) -> Tuple[int, int]:
        return (384, 320) if self._multiview else (256, 320)

    def _post_init(self, info: dict) -> None:
        expected_size = self._expected_canvas_hw()
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"Banana Piper {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        # Remap LeRobot v2.1 templates onto the chunk_index/file_index contract
        # used by LeRobotV3Reader IO helpers (file_index == episode_index).
        self._data_path_template = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
        self._video_path_template = (
            "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
        )
        meta_path = self._eef_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Banana Piper EEF sidecar missing: {meta_path}. "
                "Run scripts/preprocess_banana_piper_eef.py first."
            )
        with meta_path.open(encoding="utf-8") as handle:
            self._eef_meta = json.load(handle)
        if self._eef_meta.get("format") != "banana_piper_eef10_v1":
            raise ValueError(f"unexpected EEF sidecar format in {meta_path}: {self._eef_meta.get('format')!r}")

        # Offline UMT5 cache: one file per task_index under meta/<dirname>/.
        if self.use_t5_cache:
            self._load_t5_cache()

    def _t5_cache_dir(self) -> Path:
        return self._dataset_dir / "meta" / self.t5_cache_dirname

    def _load_t5_cache(self) -> None:
        """Preload task UMT5 embeds (banana has a single instruction)."""
        if not getattr(self, "_task_idx_to_text", None):
            raise RuntimeError("use_t5_cache=true but task prompt map is empty")
        cache_dir = self._t5_cache_dir()
        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"use_t5_cache=true but missing T5 cache dir: {cache_dir}. "
                "Run scripts/precompute_banana_piper_t5_embeds.py first."
            )
        self._prompt_to_task_idx = {
            str(text).strip(): int(task_idx) for task_idx, text in self._task_idx_to_text.items()
        }
        missing = []
        for task_idx, text in self._task_idx_to_text.items():
            path = cache_dir / f"task_{int(task_idx)}.pt"
            if not path.is_file():
                missing.append(int(task_idx))
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cached_prompt = str(payload.get("prompt", "")).strip()
            expected = str(text).strip()
            if cached_prompt and cached_prompt != expected:
                raise ValueError(
                    f"T5 cache prompt mismatch for task_{task_idx}: "
                    f"cache={cached_prompt!r} dataset={expected!r}"
                )
            ctx = payload["context"]
            if not isinstance(ctx, torch.Tensor):
                ctx = torch.as_tensor(ctx)
            ctx = ctx.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
            L = int(payload.get("seq_len", ctx.shape[0]))
            if ctx.ndim != 2:
                raise ValueError(f"task_{task_idx} t5 context must be [L, D], got {tuple(ctx.shape)}")
            if ctx.shape[0] < L:
                raise ValueError(f"task_{task_idx}: context length {ctx.shape[0]} < seq_len {L}")
            if ctx.shape[0] > L:
                ctx = ctx[:L].contiguous()
            self._t5_by_task[int(task_idx)] = (ctx, torch.tensor(L, dtype=torch.long))
        if missing:
            raise FileNotFoundError(
                f"use_t5_cache=true but missing {len(missing)}/{len(self._task_idx_to_text)} "
                f"task cache files under {cache_dir} (e.g. task_{missing[0]}.pt). "
                "Run scripts/precompute_banana_piper_t5_embeds.py."
            )
        logger.info(
            "BananaPiper T5 cache ON: dirname=%s n_tasks=%d dir=%s",
            self.t5_cache_dirname,
            len(self._t5_by_task),
            cache_dir,
        )

    def _getitem_impl(self, idx: int) -> dict:
        sample = super()._getitem_impl(idx)
        if not self.use_t5_cache:
            return sample
        prompt = str(sample["prompt"]).strip()
        task_idx = self._prompt_to_task_idx.get(prompt)
        if task_idx is None:
            raise KeyError(f"use_t5_cache=true but prompt not in task map: {prompt!r}")
        ctx, seq_lens = self._t5_by_task[task_idx]
        sample["t5_context"] = ctx
        sample["t5_seq_lens"] = seq_lens
        sample["t5_task_index"] = int(task_idx)
        return sample

    # ── episode index (LeRobot v2.1) ───────────────────────────────────────

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        episodes_path = self._dataset_dir / "meta" / "episodes.jsonl"
        rows = []
        with episodes_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep_idx = int(ep["episode_index"])
                length = int(ep["length"])
                chunk = ep_idx // int(info.get("chunks_size", 1000))
                row = {
                    "episode_index": ep_idx,
                    "length": length,
                    "data/chunk_index": chunk,
                    "data/file_index": ep_idx,
                    "_data_row_offset": 0,
                    "tasks": ep.get("tasks"),
                }
                for cam in self._video_cameras():
                    row[f"videos/{cam}/chunk_index"] = chunk
                    row[f"videos/{cam}/file_index"] = ep_idx
                    row[f"_video_frame_offset/{cam}"] = 0
                rows.append(row)
        eps = pd.DataFrame(rows)
        if eps.empty:
            raise ValueError(f"BananaPiper({self._dataset_id}): no episodes in {episodes_path}")

        split_ids = self._load_split_episode_ids(self._split)
        before = len(eps)
        eps = eps[eps["episode_index"].isin(split_ids)].reset_index(drop=True)
        if eps.empty:
            raise ValueError(
                f"BananaPiper({self._dataset_id}): split={self._split!r} selected 0/{before} episodes"
            )
        logger.info(
            "BananaPiper(%s): split=%s kept %d/%d episodes via manifest",
            self._dataset_id,
            self._split,
            len(eps),
            before,
        )
        return eps

    def _load_split_episode_ids(self, split: str) -> set[int]:
        manifest_path = self._split_manifest_path or (self._dataset_dir / "splits" / "manifest.json")
        with Path(manifest_path).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        key = "train" if split == "train" else "validation" if split in ("val", "validation") else split
        if key not in manifest.get("splits", {}):
            raise KeyError(f"split key {key!r} missing in {manifest_path}")
        return {int(x) for x in manifest["splits"][key]}

    def _load_data_table(self, chunk_idx: int, file_idx: int):
        path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        return _read_data_table_cached(str(path), tuple(self.NEEDED_COLS))

    # ── prompts (tasks.jsonl) ──────────────────────────────────────────────

    def _load_prompts(self) -> None:
        tasks_parquet = self._dataset_dir / "meta" / "tasks.parquet"
        if tasks_parquet.is_file():
            tasks_df = pd.read_parquet(tasks_parquet)
            task_idx = tasks_df["task_index"].to_numpy()
            task_str = tasks_df.index.to_numpy()
            self._task_idx_to_text = dict(zip(task_idx.tolist(), task_str.tolist()))
            return

        tasks_jsonl = self._dataset_dir / "meta" / "tasks.jsonl"
        mapping: Dict[int, str] = {}
        if tasks_jsonl.is_file():
            with tasks_jsonl.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    mapping[int(row["task_index"])] = str(row.get("task") or row.get("task_index"))
        # Fallback: episode-level tasks from episodes.jsonl
        if not mapping:
            for _, row in self._eps_df.iterrows():
                tasks = row.get("tasks") or []
                if tasks:
                    mapping[0] = str(tasks[0])
                    break
        if not mapping:
            raise FileNotFoundError(
                f"BananaPiper({self._dataset_id}): no meta/tasks.jsonl or tasks.parquet and no episode tasks"
            )
        self._task_idx_to_text = mapping

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        tasks = row.get("tasks")
        if isinstance(tasks, (list, tuple)) and tasks:
            text = str(tasks[0]).strip()
            if text:
                return text
        return super()._resolve_prompt(row, win)

    # ── EEF sidecar IO ─────────────────────────────────────────────────────

    def _load_eef_episode(self, episode_index: int) -> Dict[str, np.ndarray]:
        cached = self._eef_cache.get(int(episode_index))
        if cached is not None:
            return cached
        path = self._eef_dir / f"episode_{int(episode_index):06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing EEF sidecar for episode {episode_index}: {path}")
        with np.load(path) as data:
            payload = {
                "state_eef10": np.asarray(data["state_eef10"], dtype=np.float32),
                "action_eef10": np.asarray(data["action_eef10"], dtype=np.float32),
            }
        if payload["state_eef10"].shape[1] != EEF10_DIM or payload["action_eef10"].shape[1] != EEF10_DIM:
            raise ValueError(f"bad EEF10 shapes in {path}")
        self._eef_cache[int(episode_index)] = payload
        return payload

    def _window_eef(self, win: pd.DataFrame, key: str) -> np.ndarray:
        ep = int(win["episode_index"].iloc[0])
        frames = win["frame_index"].to_numpy(dtype=np.int64)
        arr = self._load_eef_episode(ep)[key]
        if frames.min() < 0 or frames.max() >= arr.shape[0]:
            raise IndexError(
                f"frame_index out of range for episode {ep}: "
                f"frames=[{frames.min()}, {frames.max()}] length={arr.shape[0]}"
            )
        return arr[frames]

    def _raw_action_eef10(self, win) -> np.ndarray:
        return self._window_eef(win, "action_eef10")

    def _raw_state_eef10(self, win) -> np.ndarray:
        return self._window_eef(win, "state_eef10")

    def _action_20d(self, win) -> np.ndarray:
        return apply_normalization(self._raw_action_eef10(win), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._raw_state_eef10(win)[0:1]
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)

    # ── normalization stats ────────────────────────────────────────────────

    @staticmethod
    def _stats_contract_matches(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            raw = np.load(path, allow_pickle=True).item()
            block = raw.get(_ACTION_MODE, raw) if isinstance(raw, dict) else {}
            return (
                block.get("gripper_convention") == GRIPPER_CONVENTION
                and block.get("raw_gripper_convention") == RAW_GRIPPER_CONVENTION
                and block.get("gripper_transform") == GRIPPER_TRANSFORM
                and block.get("rotation_convention") == ROTATION_CONVENTION
                and block.get("stats_population") == STATS_POPULATION
            )
        except (OSError, ValueError, EOFError, AttributeError):
            return False

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if self._source_stats_path:
            stats_path = Path(self._source_stats_path)
            if not self._stats_contract_matches(stats_path):
                raise ValueError(f"Banana Piper stats {stats_path} do not match the required contract")
        else:
            stats_path = self._eef_dir / NORMALIZATION_STATS_FILENAME
            if not self._stats_contract_matches(stats_path):
                self._build_default_stats(stats_path)
        self._resolved_stats_path = str(stats_path)
        global_stats = load_stats_file(
            stats_path,
            action_mode=self.action_mode,
            normalize_mode=str(self._normalize_mode),
            dim=self._raw_action_dim,
        )
        self._check_stats_contract(stats_path)
        self.normalization_stats_path = str(stats_path)
        return global_stats

    def _check_stats_contract(self, stats_path: Path) -> None:
        metadata = load_stats_metadata(stats_path, action_mode=self.action_mode)
        expected = {
            "gripper_convention": GRIPPER_CONVENTION,
            "raw_gripper_convention": RAW_GRIPPER_CONVENTION,
            "gripper_transform": GRIPPER_TRANSFORM,
            "rotation_convention": ROTATION_CONVENTION,
            "stats_population": STATS_POPULATION,
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"Banana Piper stats contract mismatch in {stats_path}: {mismatches}")

    def _build_default_stats(self, path: Path) -> None:
        from openwam.dataloader.utils.stats_computation.banana_piper_stats_computation import (
            build_and_save_banana_piper_stats,
        )

        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        rank = dist.get_rank() if dist_ready else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            logger.info("BananaPiper(%s): building normalization stats at %s", self._dataset_id, path)
            build_and_save_banana_piper_stats(self, path)
            return

        deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
        poll_interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
        while not self._stats_contract_matches(path):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for rank 0 to build Banana Piper stats: {path}")
            time.sleep(poll_interval)


ROT6D_DIMS_EEF10 = ROT6D_DIMS_ARM10

__all__ = [
    "BananaPiperDataset",
    "EEF10_DIM",
    "EEF_SIDECAR_DIRNAME",
    "GRIPPER_CONVENTION",
    "GRIPPER_TRANSFORM",
    "NORMALIZATION_STATS_FILENAME",
    "RAW_GRIPPER_CONVENTION",
    "ROT6D_DIMS_EEF10",
    "ROTATION_CONVENTION",
    "STATS_POPULATION",
]
