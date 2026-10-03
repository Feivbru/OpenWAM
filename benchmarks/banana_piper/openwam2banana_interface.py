"""OpenWAM WebSocket client for banana AgileX Piper (physical EEF10).

Training contract (see ``BananaPiperDataset``):
  - cameras: ``top_head`` → head slot; ``hand_right`` → right wrist; left black
  - proprio / action: physical EEF10
    ``[xyz_m3, rot6d6, gripper_open_scale1]`` with ``-1=closed / +1=open``
  - server normalizes + unify-scatters to α left-arm slots ``0:10`` and
    unnormalizes + gathers on the way out

This client does **not** run IK or talk to Piper SDK. Callers that need joint
commands convert the returned EEF10 themselves.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from benchmarks.utils import WSPolicyClient, build_payload, encode_numpy_b64, resize_for_lshape_slot
from openwam.dataloader.utils.piper_fk import EEF10_DIM, JOINT7_DIM, joints7_to_eef10, make_piper_fk

BANANA_ACTION_MODE = "eef"
BANANA_EEF10_DIM = EEF10_DIM
ACTION_CHUNK_MODES = ("first", "all")

DEFAULT_HEAD_KEY = "observation.images.top_head"
DEFAULT_RIGHT_WRIST_KEY = "observation.images.hand_right"
# Short aliases accepted in obs dicts (real-robot / smoke).
_HEAD_ALIASES = (DEFAULT_HEAD_KEY, "top_head", "head_camera", "head")
_RIGHT_WRIST_ALIASES = (DEFAULT_RIGHT_WRIST_KEY, "hand_right", "right_wrist_camera", "right_wrist")


def _as_eef10(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (BANANA_EEF10_DIM,):
        raise ValueError(f"{name} must be length-{BANANA_EEF10_DIM} EEF10, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def _pick_image(obs: dict, aliases: tuple[str, ...]) -> Optional[np.ndarray]:
    for key in aliases:
        if key in obs and obs[key] is not None:
            image = np.asarray(obs[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Camera '{key}' must be HxWx3 RGB, got {image.shape}")
            return np.ascontiguousarray(image.astype(np.uint8, copy=False))
    return None


class OpenWAMBananaPolicy:
    """WebSocket policy client for banana Piper EEF10 checkpoints."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        action_mode: str = BANANA_ACTION_MODE,
        send_state: bool = True,
        action_chunk_mode: str = "all",
        gripper_max_m: Optional[float] = None,
        dh_is_offset: int = 1,
        debug: bool = False,
        debug_dir: str = "./debug_banana_piper",
        _client=None,
    ) -> None:
        if str(action_mode).strip().lower() != BANANA_ACTION_MODE:
            raise ValueError(
                f"banana client requires action_mode={BANANA_ACTION_MODE!r}, got {action_mode!r}"
            )
        chunk_mode = str(action_chunk_mode).strip().lower()
        if chunk_mode not in ACTION_CHUNK_MODES:
            raise ValueError(f"action_chunk_mode must be one of {ACTION_CHUNK_MODES}, got {action_chunk_mode!r}")

        self._client = _client or WSPolicyClient(f"ws://{host}:{port}", timeout=request_timeout)
        self._send_state = bool(send_state)
        self._action_chunk_mode = chunk_mode
        self._gripper_max_m = float(gripper_max_m) if gripper_max_m is not None else None
        self._dh_is_offset = int(dh_is_offset)
        self._fk = None
        self._pending: deque[np.ndarray] = deque()
        self._debug = bool(debug)
        self._debug_dir = Path(debug_dir)
        self._episode = -1
        self._step = 0
        raw_slot = os.environ.get("BANANA_SLOT", "").strip() or os.environ.get("ROBOTWIN_SLOT", "").strip()
        self._slot_id = int(raw_slot) if raw_slot else None
        if self._debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        pong = self._client.ping()
        if pong.get("type") != "pong":
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")
        if pong.get("representation") is not None and pong.get("representation") != BANANA_ACTION_MODE:
            raise RuntimeError(
                f"representation mismatch: banana requires {BANANA_ACTION_MODE!r}, "
                f"server advertises {pong.get('representation')!r}"
            )
        print(
            f"[OpenWAMBananaPolicy] server=ws://{host}:{port} "
            f"action_mode={BANANA_ACTION_MODE} action_dim={BANANA_EEF10_DIM} "
            f"send_state={self._send_state} action_chunk_mode={self._action_chunk_mode} "
            f"slot_id={self._slot_id}"
        )

    def close(self) -> None:
        self._client.close()

    def reset(self) -> None:
        self._episode += 1
        self._step = 0
        self._pending.clear()
        ack = self._client.reset(slot_id=self._slot_id)
        if ack.get("type") != "reset_ack":
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def act(self, obs: dict, prompt: str) -> np.ndarray:
        """Return one physical EEF10 action for ``obs``.

        With ``action_chunk_mode='all'``, if the server returns a multi-step
        ``actions`` list the remainder is buffered locally. Current
        ``PolicyServer`` returns one action per request and buffers server-side;
        in that case each ``act`` call is one server round-trip / one EEF10.
        """
        if self._action_chunk_mode == "all" and self._pending:
            action = self._pending.popleft()
            self._step += 1
            return action

        head = _pick_image(obs, _HEAD_ALIASES)
        if head is None:
            raise KeyError(
                f"banana obs missing head camera; tried keys {_HEAD_ALIASES}"
            )
        right = _pick_image(obs, _RIGHT_WRIST_ALIASES)

        payload = build_payload(
            head=encode_numpy_b64(resize_for_lshape_slot(head, "head_camera")),
            left_wrist=None,
            right_wrist=(
                encode_numpy_b64(resize_for_lshape_slot(right, "right_wrist_camera"))
                if right is not None
                else None
            ),
            prompt=str(prompt),
            state=self._state(obs),
        )
        if self._slot_id is not None:
            payload["slot_id"] = self._slot_id

        response = self._client.predict(payload)
        raw_chunk = response.get("actions") if self._action_chunk_mode == "all" else None
        if not raw_chunk:
            if "action" not in response:
                raise RuntimeError(f"OpenWAM response missing action field: {response}")
            raw_chunk = [response["action"]]

        converted: list[np.ndarray] = []
        for raw in raw_chunk:
            action = _as_eef10(raw, name="server action")
            converted.append(action)

        self._maybe_debug(obs, payload, converted[0])
        if self._action_chunk_mode == "all" and len(converted) > 1:
            self._pending.extend(converted[1:])
        self._step += 1
        return converted[0]

    def _ensure_fk(self):
        if self._fk is None:
            self._fk = make_piper_fk(dh_is_offset=self._dh_is_offset)
        return self._fk

    def _state(self, obs: dict) -> list[float] | None:
        if not self._send_state:
            return None

        if "state_eef10" in obs and obs["state_eef10"] is not None:
            return _as_eef10(obs["state_eef10"], name="state_eef10").tolist()

        if "state" in obs and obs["state"] is not None:
            state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            if state.shape == (BANANA_EEF10_DIM,):
                return _as_eef10(state, name="state").tolist()
            if state.shape == (JOINT7_DIM,):
                return self._joint7_to_eef10(state).tolist()
            raise ValueError(
                f"obs['state'] must be EEF10 ({BANANA_EEF10_DIM}) or joint7 ({JOINT7_DIM}), got {state.shape}"
            )

        if "joint7" in obs and obs["joint7"] is not None:
            return self._joint7_to_eef10(obs["joint7"]).tolist()

        raise KeyError(
            "banana proprio required: provide state_eef10, state (EEF10 or joint7), or joint7"
        )

    def _joint7_to_eef10(self, joint7) -> np.ndarray:
        if self._gripper_max_m is None:
            raise ValueError(
                "joint7 → EEF10 needs gripper_max_m "
                "(pass OpenWAMBananaPolicy(gripper_max_m=...) from derived/eef/meta.json)"
            )
        return _as_eef10(
            joints7_to_eef10(self._ensure_fk(), joint7, self._gripper_max_m),
            name="joint7→eef10",
        )

    def _maybe_debug(self, obs: dict, payload: dict, action: np.ndarray) -> None:
        if not self._debug:
            return
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "action_mode": BANANA_ACTION_MODE,
            "state": payload.get("state"),
            "physical_eef10_action": action.tolist(),
            "obs_keys": sorted(str(k) for k in obs.keys()),
            "prompt": payload.get("prompt"),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


__all__ = [
    "ACTION_CHUNK_MODES",
    "BANANA_ACTION_MODE",
    "BANANA_EEF10_DIM",
    "OpenWAMBananaPolicy",
]
