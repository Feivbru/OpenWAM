from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.robocasa_gr1 import openwam2robocasa_gr1_interface as adapter
from benchmarks.utils import transport


class _FakeKinematics:
    @classmethod
    def from_env(cls, env):
        return cls()

    def observation_to_eef33(self, obs):
        return np.zeros(adapter.EEF33_DIM, dtype=np.float32)

    def eef33_to_action_dict(self, raw, **kwargs):
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        ik = SimpleNamespace(converged=True, position_error=0.0, rotation_error=0.0)
        return {"eef33": raw.copy()}, ik


class _FakeClient:
    def __init__(self, *, chunks=None, drip=None):
        self.predict_calls = 0
        self.reset_calls = 0
        self._chunks = list(chunks or [])
        self._drip = drip

    def ping(self):
        return {"type": transport.PONG}

    def reset(self, slot_id=None):
        self.reset_calls += 1
        return {"type": transport.RESET_ACK}

    def predict(self, payload):
        self.predict_calls += 1
        if self._drip is not None:
            return {"action": np.asarray(self._drip, dtype=np.float32).tolist()}
        if not self._chunks:
            raise AssertionError("unexpected extra predict() after chunk was consumed")
        chunk = self._chunks.pop(0)
        return {"action": chunk[0], "actions": chunk}

    def close(self):
        pass


def _obs() -> dict:
    return {
        "video.ego_view_pad_res256_freq20": np.zeros((16, 16, 3), dtype=np.uint8),
        "annotation.human.coarse_action": "unlocked_waist: pick up the cup",
    }


def _make_policy(
    monkeypatch,
    client: _FakeClient,
    *,
    n_action_steps: int | None = None,
    project_discrete_hands: bool = False,
) -> adapter.OpenWAMRoboCasaGR1Policy:
    monkeypatch.setattr(adapter, "GR1Kinematics", _FakeKinematics)
    monkeypatch.setattr(adapter, "WSPolicyClient", lambda *args, **kwargs: client)
    action_space = SimpleNamespace(spaces={})
    return adapter.OpenWAMRoboCasaGR1Policy(
        action_space=action_space,
        env=object(),
        send_state=True,
        n_action_steps=n_action_steps,
        project_discrete_hands=project_discrete_hands,
    )


def _eef33(fill: float) -> list[float]:
    return np.full(adapter.EEF33_DIM, fill, dtype=np.float32).tolist()


def test_act_executes_full_server_chunk_without_requery(monkeypatch):
    horizon = 4
    chunk = [_eef33(float(i + 1)) for i in range(horizon)]
    client = _FakeClient(chunks=[chunk])
    policy = _make_policy(monkeypatch, client)
    policy.reset()

    seen = []
    for _ in range(horizon):
        seen.append(policy.act(_obs())["eef33"][0])

    assert client.predict_calls == 1
    np.testing.assert_allclose(seen, [1.0, 2.0, 3.0, 4.0])
    assert not policy._pending


def test_act_drip_feed_without_actions_field_queries_every_step(monkeypatch):
    client = _FakeClient(drip=_eef33(7.0))
    policy = _make_policy(monkeypatch, client)
    policy.reset()

    for _ in range(3):
        np.testing.assert_allclose(policy.act(_obs())["eef33"][0], 7.0)

    assert client.predict_calls == 3


def test_reset_discards_unused_chunk_tail(monkeypatch):
    client = _FakeClient(chunks=[[_eef33(1.0), _eef33(2.0), _eef33(3.0)], [_eef33(9.0)]])
    policy = _make_policy(monkeypatch, client)
    policy.reset()
    policy.act(_obs())
    assert len(policy._pending) == 2
    policy.reset()
    assert not policy._pending
    np.testing.assert_allclose(policy.act(_obs())["eef33"][0], 9.0)
    assert client.predict_calls == 2
    assert client.reset_calls == 2


def test_n_action_steps_truncates_chunk_and_requeries(monkeypatch):
    chunk_a = [_eef33(float(i + 1)) for i in range(4)]
    chunk_b = [_eef33(float(i + 10)) for i in range(4)]
    client = _FakeClient(chunks=[chunk_a, chunk_b])
    policy = _make_policy(monkeypatch, client, n_action_steps=2)
    policy.reset()

    seen = [policy.act(_obs())["eef33"][0] for _ in range(4)]
    assert client.predict_calls == 2
    np.testing.assert_allclose(seen, [1.0, 2.0, 10.0, 11.0])
    assert not policy._pending


def test_n_action_steps_rejects_non_positive(monkeypatch):
    monkeypatch.setattr(adapter, "GR1Kinematics", _FakeKinematics)
    monkeypatch.setattr(adapter, "WSPolicyClient", lambda *args, **kwargs: _FakeClient())
    with pytest.raises(ValueError, match="n_action_steps"):
        adapter.OpenWAMRoboCasaGR1Policy(
            action_space=SimpleNamespace(spaces={}),
            env=object(),
            n_action_steps=0,
        )
