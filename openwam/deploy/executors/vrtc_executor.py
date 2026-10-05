"""VRTC cube-pool executor with optional cube-level prefetch.

Wire contract (matches the topology experiment):
  * Each response is **one cube** of ``video_stride`` actions.
  * Client open-loop horizon equals that length.
  * ``replan_cubes``: when ``0 < len(wait) <= replan_cubes``, start a background
    generate whose clear condition ends at the latest real frame. On completion,
    overwrite wait after the last delivered cube (skip the head so the kept
    output sits immediately after the snapshotted real).

``replan_cubes=0`` is the sync special case (only infer when wait is empty).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from openwam.deploy.engine import BaseInferenceEngine
from openwam.vrtc import VrtcConfig, resolve_vrtc_config

logger = logging.getLogger(__name__)


@dataclass
class Cube:
    """One video frame plus the ``video_stride`` actions that follow it."""

    frame: object
    actions: np.ndarray  # (video_stride, action_dim), physical units
    state: Optional[np.ndarray] = None  # proprio at this frame, physical units
    seq: int = 0  # monotonic cube index for prefetch skip alignment

    def update(self, frame, state=None) -> None:
        self.frame = frame
        if state is not None:
            self.state = np.asarray(state, dtype=np.float32)


@dataclass
class _InferJob:
    job_id: int
    clear_snapshot: Tuple[Cube, ...]
    origin_last_clear_seq: int
    future: Future = field(repr=False)


class VrtcSyncExecutor:
    """VRTC cube-pool scheduler (cube return + optional prefetch)."""

    def __init__(self, engine: BaseInferenceEngine, vrtc: VrtcConfig, cfg=None):
        if not vrtc.enabled:
            raise ValueError("VrtcSyncExecutor requires vrtc.enabled=True")
        self.engine = engine
        self.vrtc = vrtc
        self.cfg = cfg

        self.clear_pool: List[Cube] = []
        self.wait_pool: List[Cube] = []
        # Back-compat alias used by unit tests / debugging.
        self.pool = self.clear_pool
        self._action_buffer: deque = deque()
        self._last_prompt: str = ""
        self._last_delivered: Optional[Cube] = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vrtc-infer")
        self._pending: Optional[_InferJob] = None
        self._job_seq: int = 0
        self._closed: bool = False

        logger.info(
            "VrtcSyncExecutor: warmup=%d predict=%d stride=%d replan_cubes=%d",
            vrtc.pool_warmup_cubes,
            vrtc.predict_cubes,
            vrtc.video_stride,
            vrtc.replan_cubes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_action(self, conditions: dict) -> np.ndarray:
        """Return the next physical action; refill one cube when the drip buffer is empty."""
        if len(self._action_buffer) == 0:
            frame, state = self._extract_obs(conditions)
            cube_actions = self._step_cube(frame, state)
            self._action_buffer.extend(np.asarray(a, dtype=np.float32) for a in cube_actions)

        if len(self._action_buffer) == 0:
            raise RuntimeError("VRTC executor produced an empty action buffer")
        return np.asarray(self._action_buffer.popleft(), dtype=np.float32)

    def predict_action_chunk(self, conditions: dict) -> np.ndarray:
        """Return one cube of actions (``video_stride`` steps) for client open-loop."""
        if self._action_buffer:
            actions = np.stack([np.asarray(a, dtype=np.float32) for a in self._action_buffer], axis=0)
            self._action_buffer.clear()
            return actions
        frame, state = self._extract_obs(conditions)
        return np.asarray(self._step_cube(frame, state), dtype=np.float32)

    def reset(self) -> None:
        with self._cv:
            self._cancel_pending_locked()
            self.clear_pool.clear()
            self.wait_pool.clear()
            self._action_buffer.clear()
            self._last_prompt = ""
            self._last_delivered = None

    def shutdown(self) -> None:
        with self._cv:
            self._closed = True
            self._cancel_pending_locked()
            self.clear_pool.clear()
            self.wait_pool.clear()
            self._action_buffer.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Cube scheduler
    # ------------------------------------------------------------------

    def _step_cube(self, frame, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        warmup_n = self.vrtc.pool_warmup_cubes

        with self._cv:
            if self._closed:
                raise RuntimeError("VrtcSyncExecutor is shut down")
            self._drain_done_locked()
            # Arm prefetch on the request thread (never from the infer callback).
            self._maybe_start_replan_locked()

            # ---- Warmup: collect real frames, return proprio-held cube ----
            if len(self.clear_pool) < warmup_n:
                actions = np.repeat(state[None, :], self.vrtc.video_stride, axis=0)
                seq = 0 if not self.clear_pool else self.clear_pool[-1].seq + 1
                cube = Cube(
                    frame=frame,
                    actions=actions.copy(),
                    state=state.copy(),
                    seq=seq,
                )
                self.clear_pool.append(cube)
                self._last_delivered = cube
                logger.debug(
                    "VRTC warmup %d/%d seq=%d",
                    len(self.clear_pool),
                    warmup_n,
                    cube.seq,
                )
                if len(self.clear_pool) == warmup_n and self._pending is None:
                    self._start_infer_locked(reason="post-warmup")
                return actions.copy()

            # ---- Steady: append latest real obs (topology: clear grows by 1) ----
            self._append_real_locked(frame, state)

            # Block until a wait cube is available (prefetch may still be running).
            while not self.wait_pool:
                if self._pending is None:
                    self._start_infer_locked(reason="wait-empty")
                if self._pending is None:
                    raise RuntimeError("VRTC wait empty and failed to start inference")
                job = self._pending
                logger.debug("VRTC client block — waiting for job#%d", job.job_id)
                self._cv.release()
                try:
                    job.future.result()
                finally:
                    self._cv.acquire()
                self._drain_done_locked()

            cube = self.wait_pool.pop(0)
            self._last_delivered = cube
            # Prefer the delivered cube's actions on the matching clear entry.
            if self.clear_pool and self.clear_pool[-1].seq == cube.seq:
                self.clear_pool[-1].actions = np.asarray(cube.actions, dtype=np.float32).copy()
            logger.debug(
                "VRTC deliver seq=%d remaining=%d",
                cube.seq,
                len(self.wait_pool),
            )
            self._maybe_start_replan_locked()
            return np.asarray(cube.actions, dtype=np.float32).copy()

    def _append_real_locked(self, frame, state: np.ndarray) -> None:
        """Append the newest real observation as the next clear cube."""
        if not self.clear_pool:
            raise RuntimeError("clear_pool empty in steady state")
        if self._last_delivered is None:
            raise RuntimeError("missing last_delivered in steady state")
        next_seq = self.clear_pool[-1].seq + 1
        real = Cube(
            frame=frame,
            actions=np.asarray(self._last_delivered.actions, dtype=np.float32).copy(),
            state=np.asarray(state, dtype=np.float32).copy(),
            seq=next_seq,
        )
        self.clear_pool.append(real)
        max_keep = self.vrtc.pool_warmup_cubes + self.vrtc.predict_cubes
        if len(self.clear_pool) > max_keep:
            self.clear_pool = self.clear_pool[-max_keep:]
            # Keep alias in sync if list was replaced.
            self.pool = self.clear_pool

    # ------------------------------------------------------------------
    # Prefetch / merge
    # ------------------------------------------------------------------

    def _maybe_start_replan_locked(self) -> None:
        if self._pending is not None or self._closed:
            return
        if len(self.clear_pool) < self.vrtc.pool_warmup_cubes:
            return
        # replan_cubes=0 → sync-only (only wait-empty path starts infer).
        if self.vrtc.replan_cubes <= 0:
            return
        if 0 < len(self.wait_pool) <= self.vrtc.replan_cubes:
            self._start_infer_locked(reason=f"replan_cubes<={self.vrtc.replan_cubes}")

    def _start_infer_locked(self, reason: str) -> None:
        if self._pending is not None:
            raise RuntimeError("infer already pending")
        if len(self.clear_pool) < self.vrtc.pool_warmup_cubes:
            raise RuntimeError("cannot infer before warmup")

        snap = tuple(self.clear_pool[-self.vrtc.pool_warmup_cubes :])
        snap_cubes = tuple(
            Cube(
                frame=c.frame,
                actions=np.asarray(c.actions, dtype=np.float32).copy(),
                state=None if c.state is None else np.asarray(c.state, dtype=np.float32).copy(),
                seq=c.seq,
            )
            for c in snap
        )
        self._job_seq += 1
        job_id = self._job_seq
        origin_seq = snap_cubes[-1].seq
        prompt = self._last_prompt

        def _worker() -> List[Cube]:
            with torch.no_grad():
                return self._model_predict(list(snap_cubes), prompt=prompt)

        future = self._pool.submit(_worker)
        self._pending = _InferJob(
            job_id=job_id,
            clear_snapshot=snap_cubes,
            origin_last_clear_seq=origin_seq,
            future=future,
        )
        logger.info(
            "VRTC INFER_START job#%d reason=%s origin_seq=%d clear_seqs=%s",
            job_id,
            reason,
            origin_seq,
            [c.seq for c in snap_cubes],
        )
        # No done-callback: draining from the request thread avoids a race
        # between the worker thread and ``future.result()`` under the lock.

    def _drain_done_locked(self) -> None:
        if self._pending is None:
            return
        fut = self._pending.future
        if not fut.done():
            return
        job = self._pending
        self._pending = None
        predicted = fut.result()

        if self._last_delivered is None:
            skip = 0
        else:
            skip = max(0, self._last_delivered.seq - job.origin_last_clear_seq)
        kept = predicted[skip:]
        old = [c.seq for c in self.wait_pool]
        self.wait_pool = list(kept)
        logger.info(
            "VRTC INFER_MERGE job#%d pred_seqs=%s skip=%d old_wait=%s -> wait=%s",
            job.job_id,
            [c.seq for c in predicted],
            skip,
            old,
            [c.seq for c in self.wait_pool],
        )
        # Replan is armed by the request thread after drain/deliver — never here.

    def _cancel_pending_locked(self) -> None:
        if self._pending is None:
            return
        fut = self._pending.future
        self._pending = None
        if not fut.cancel():
            try:
                fut.result()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _model_predict(self, cubes: List[Cube], prompt: str = "") -> List[Cube]:
        """Run one generate call; return newly predicted cubes with assigned seqs."""
        v = self.vrtc
        if len(cubes) < v.pool_warmup_cubes:
            raise RuntimeError(
                f"VRTC predict needs >= {v.pool_warmup_cubes} cubes, got {len(cubes)}"
            )

        clear_cubes = cubes[: v.pool_warmup_cubes]
        frames = [c.frame for c in clear_cubes]
        proprio = clear_cubes[0].state
        if proprio is None:
            raise RuntimeError("VRTC condition cube is missing state/proprio")

        action_prefix = np.concatenate([c.actions for c in clear_cubes[1:]], axis=0)
        if action_prefix.shape[0] != v.clear_action_steps:
            raise RuntimeError(
                f"action_prefix length {action_prefix.shape[0]} != "
                f"clear_action_steps {v.clear_action_steps}"
            )

        conditions = {
            "prompt": prompt,
            "first_frame_image": frames,
            "proprio": proprio,
            "action_prefix": action_prefix,
            "num_frames": v.num_frames,
            "video_num_frames": v.video_num_frames,
        }
        result = self.engine.generate(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < v.num_action_steps:
            raise RuntimeError(
                f"VRTC generate returned actions shape {actions.shape}, "
                f"expected (>= {v.num_action_steps}, D)"
            )

        future_actions = actions[v.clear_action_steps : v.num_action_steps]
        if future_actions.shape[0] != v.noisy_action_steps:
            raise RuntimeError(
                f"noisy future actions {future_actions.shape[0]} != {v.noisy_action_steps}"
            )

        origin = clear_cubes[-1].seq
        predicted: List[Cube] = []
        for i in range(v.predict_cubes):
            sl = future_actions[i * v.video_stride : (i + 1) * v.video_stride]
            predicted.append(
                Cube(
                    frame=clear_cubes[-1].frame,
                    actions=sl.copy(),
                    state=None,
                    seq=origin + 1 + i,
                )
            )
        return predicted

    def _extract_obs(self, conditions: dict):
        self._last_prompt = str(conditions.get("prompt") or "")

        images = conditions.get("first_frame_image")
        if not images:
            obs = conditions.get("observation") or {}
            img = obs.get("image")
            if img is None:
                raise ValueError("VRTC executor requires an observation image")
            frame = img
        else:
            frame = images[0] if isinstance(images, (list, tuple)) else images

        proprio = conditions.get("proprio")
        if proprio is None:
            obs = conditions.get("observation") or {}
            proprio = obs.get("state")
        if proprio is None:
            raise ValueError("VRTC executor requires proprio/state")
        return frame, np.asarray(proprio, dtype=np.float32)


def build_vrtc_executor(engine, cfg) -> Optional[VrtcSyncExecutor]:
    """Return a VRTC executor when ``vrtc.enabled``, else ``None``."""
    vrtc = resolve_vrtc_config(cfg)
    if not vrtc.enabled:
        return None
    return VrtcSyncExecutor(engine=engine, vrtc=vrtc, cfg=cfg)
