"""Unit tests for VRTC config + cube-pool scheduler with prefetch (no GPU)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from openwam.deploy.executors.vrtc_executor import Cube, VrtcSyncExecutor
from openwam.vrtc import VrtcConfig, resolve_vrtc_config


def test_vrtc_config_defaults_disabled():
    cfg = VrtcConfig(enabled=False)
    assert cfg.enabled is False


def test_vrtc_config_derived_counts():
    cfg = VrtcConfig(enabled=True, fu_frames=4, num_frames=33, video_stride=4, replan_cubes=2)
    assert cfg.clear_video_frames == 5
    assert cfg.clear_action_steps == 16
    assert cfg.noisy_action_steps == 16
    assert cfg.clear_latent_frames == 2
    assert cfg.predict_cubes == 4
    assert cfg.pool_warmup_cubes == 5
    assert cfg.video_num_frames == 9
    assert cfg.replan_cubes == 2


def test_vrtc_config_rejects_bad_fu_frames():
    with pytest.raises(ValueError, match="multiple"):
        VrtcConfig(enabled=True, fu_frames=3, num_frames=33, video_stride=4)


def test_vrtc_config_rejects_bad_replan():
    with pytest.raises(ValueError, match="replan_cubes"):
        VrtcConfig(enabled=True, fu_frames=4, num_frames=33, video_stride=4, replan_cubes=5)


def test_resolve_vrtc_from_dict():
    cfg = {
        "vrtc": {"enabled": True, "fu_frames": 4, "replan_cubes": 1},
        "dataloader": {"num_frames": 33, "video_stride": 4},
    }
    v = resolve_vrtc_config(cfg)
    assert v.enabled
    assert v.clear_latent_frames == 2
    assert v.replan_cubes == 1


class _FakeEngine:
    def __init__(self, action_dim=10, delay_s: float = 0.0):
        self.action_dim = action_dim
        self.delay_s = delay_s
        self.calls = []
        self._lock = threading.Lock()

    def generate(self, conditions):
        with self._lock:
            self.calls.append(conditions)
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        n = int(conditions["num_frames"]) - 1
        actions = np.arange(n * self.action_dim, dtype=np.float32).reshape(n, self.action_dim)
        actions[16:] = 100 + np.arange(16 * self.action_dim, dtype=np.float32).reshape(16, self.action_dim)
        return {"actions": actions, "video": None}


def _obs(i: int, dim: int = 10) -> dict:
    state = np.full(dim, float(i), dtype=np.float32)
    return {
        "prompt": "pick block",
        "first_frame_image": [f"frame-{i}"],
        "proprio": state,
    }


def test_vrtc_sync_warmup_and_predict():
    vrtc = VrtcConfig(enabled=True, fu_frames=4, num_frames=33, video_stride=4, replan_cubes=0)
    engine = _FakeEngine(action_dim=10)
    ex = VrtcSyncExecutor(engine=engine, vrtc=vrtc)

    # Warmup: 5 cubes × 4 actions of state copies.
    for i in range(5):
        state = np.full(10, float(i), dtype=np.float32)
        for _ in range(4):
            a = ex.predict_action(_obs(i))
            np.testing.assert_allclose(a, state)

    assert len(ex.clear_pool) == 5
    # First infer kicked off at end of warmup — wait for the worker.
    deadline = time.time() + 2.0
    while len(engine.calls) < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert len(engine.calls) == 1

    # Next cube tick waits for / adopts first predict, returns noisy future head.
    first = ex.predict_action(_obs(5))
    assert first.shape == (10,)
    assert float(first[0]) == 100.0
    # One cube delivered → 3 remain (replan_cubes=0 so no prefetch yet).
    assert len(ex.wait_pool) == 3
    assert len(ex.clear_pool) == 6
    cond = engine.calls[0]
    assert len(cond["first_frame_image"]) == 5
    assert cond["action_prefix"].shape == (16, 10)
    ex.shutdown()


def test_vrtc_predict_action_chunk_one_cube():
    vrtc = VrtcConfig(enabled=True, fu_frames=4, num_frames=33, video_stride=4, replan_cubes=0)
    engine = _FakeEngine(action_dim=10)
    ex = VrtcSyncExecutor(engine=engine, vrtc=vrtc)
    for i in range(5):
        chunk = ex.predict_action_chunk(_obs(i))
        assert chunk.shape == (4, 10)

    chunk = ex.predict_action_chunk(_obs(5))
    assert chunk.shape == (4, 10)
    assert float(chunk[0, 0]) == 100.0
    ex.shutdown()


def test_vrtc_prefetch_replan_cubes():
    """With replan_cubes=2, a second infer starts once remaining wait hits 2."""
    vrtc = VrtcConfig(enabled=True, fu_frames=4, num_frames=33, video_stride=4, replan_cubes=2)
    engine = _FakeEngine(action_dim=10, delay_s=0.0)
    ex = VrtcSyncExecutor(engine=engine, vrtc=vrtc)

    for i in range(5):
        ex.predict_action_chunk(_obs(i))

    # Ensure first infer finished.
    deadline = time.time() + 2.0
    while len(engine.calls) < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert len(engine.calls) == 1

    # First steady delivery (wait 4→3); no replan yet.
    ex.predict_action_chunk(_obs(5))
    assert len(engine.calls) == 1
    assert len(ex.wait_pool) == 3

    # Second delivery (wait 3→2) triggers prefetch submit.
    ex.predict_action_chunk(_obs(6))
    assert len(ex.wait_pool) == 2
    assert ex._pending is not None or len(engine.calls) >= 2

    deadline = time.time() + 2.0
    while len(engine.calls) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(engine.calls) >= 2

    # Continue until a few more cubes; should not raise / deadlock.
    for i in range(7, 12):
        chunk = ex.predict_action_chunk(_obs(i))
        assert chunk.shape == (4, 10)
    ex.shutdown()


def test_cube_update_shares_pool_ref():
    c = Cube(frame="a", actions=np.zeros((4, 3), dtype=np.float32), state=np.zeros(3))
    pool = [c]
    wait = [c]
    wait[0].update("b", np.ones(3, dtype=np.float32))
    assert pool[0].frame == "b"
    np.testing.assert_allclose(pool[0].state, np.ones(3))
