"""OpenPI-compatible WebSocket policy server for Piper / banana EEF10.

Wire protocol matches ``openpi_client.WebsocketClientPolicy`` (see
``README_piper.md``):

- On connect: server sends metadata (msgpack)
- Loop: client sends obs dict → server returns result dict (msgpack bytes)
- Errors: UTF-8 text frame with traceback
- Optional ``GET /healthz``

Bus representation is **physical raw EEF10** only (never 80-D / normalized).
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any, Optional

import numpy as np
from PIL import Image

from openwam.deploy import msgpack_numpy
from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError

logger = logging.getLogger(__name__)

EEF10_DIM = 10
DEFAULT_MAX_DELAY = 8

# Client image key → OpenWAM ObsPreprocessor slot
_IMAGE_KEY_ALIASES = {
    "head": (
        "observation/top_image",
        "observation/top_head",
        "top_image",
        "top_head",
        "observation.images.top_head",
    ),
    "right_wrist": (
        "observation/right_wrist_image",
        "right_wrist_image",
        "hand_right",
        "observation.images.hand_right",
    ),
}


def _as_rgb_uint8(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ObsValidationError(f"{name} must be HxWx3 RGB, got {getattr(arr, 'shape', None)}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _pick_image(obs: dict, aliases: tuple[str, ...]) -> Optional[np.ndarray]:
    for key in aliases:
        if key in obs and obs[key] is not None:
            return _as_rgb_uint8(obs[key], name=key)
    return None


def _as_eef10(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (EEF10_DIM,):
        raise ObsValidationError(f"{name} must be length-{EEF10_DIM} EEF10, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ObsValidationError(f"{name} contains NaN/Inf")
    return arr


def resolve_action_horizon(cfg) -> int:
    """Action chunk length = raw window ``num_frames - 1`` (OpenWAM contract)."""
    from omegaconf import OmegaConf

    num_frames = int(OmegaConf.select(cfg, "inference.num_frames", default=None) or 33)
    if num_frames < 2:
        raise ValueError(f"inference.num_frames must be >= 2, got {num_frames}")
    return num_frames - 1


def build_piper_metadata(cfg, *, max_delay: Optional[int] = None, training_rtc: Optional[bool] = None) -> dict:
    """Metadata dict sent as the first msgpack frame after connect."""
    from omegaconf import OmegaConf

    action_horizon = resolve_action_horizon(cfg)
    md_delay = OmegaConf.select(cfg, "piper_openpi.max_delay", default=None)
    if max_delay is None:
        max_delay = int(md_delay) if md_delay is not None else DEFAULT_MAX_DELAY
    else:
        max_delay = int(max_delay)

    md_rtc = OmegaConf.select(cfg, "piper_openpi.training_rtc", default=None)
    if training_rtc is None:
        training_rtc = bool(md_rtc) if md_rtc is not None else False

    meta = {
        "action_horizon": int(action_horizon),
        "raw_action_dim": EEF10_DIM,
        "wire_action_space": "absolute",
        "action_representation": "eef10_rot6d",
        "gripper_convention": "minus1_closed_plus1_open",
        "rot6d_convention": "piper_fk_rot6d_R_cols01",
    }
    if training_rtc:
        if action_horizon <= max_delay:
            raise ValueError(
                f"action_horizon ({action_horizon}) must be > max_delay ({max_delay}) when training_rtc=true"
            )
        meta["training_rtc"] = True
        meta["max_delay"] = int(max_delay)
    return meta


class PiperOpenPIPolicy:
    """Stateless OpenWAM → openpi ``infer`` adapter (physical EEF10 in/out)."""

    def __init__(self, engine, cfg, metadata: Optional[dict] = None):
        self.engine = engine
        self.cfg = cfg
        self.metadata = dict(metadata or build_piper_metadata(cfg))
        self._obs_preprocessor = ObsPreprocessor.from_cfg(cfg, engine)
        self._action_horizon = int(self.metadata["action_horizon"])
        self._max_delay = int(self.metadata.get("max_delay", DEFAULT_MAX_DELAY))

    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()
        if not isinstance(obs, dict):
            raise ObsValidationError(f"obs must be a dict, got {type(obs).__name__}")

        openwam_obs = self._to_openwam_obs(obs)
        openwam_obs = self._obs_preprocessor.preprocess(openwam_obs)
        conditions = self._build_conditions(openwam_obs)

        result = self.engine.generate(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != EEF10_DIM:
            raise RuntimeError(
                f"model returned actions shape {actions.shape}, expected (H, {EEF10_DIM}) physical EEF10"
            )

        # Truncate / pad to the advertised horizon (pad by repeating last).
        h = self._action_horizon
        if actions.shape[0] < h:
            pad = np.repeat(actions[-1:], h - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        elif actions.shape[0] > h:
            actions = actions[:h].copy()
        else:
            actions = actions.copy()

        actions = self._snap_binary_dims(actions)

        rtc_out = None
        rtc_in = obs.get("rtc")
        if rtc_in is not None:
            rtc_out, actions = self._apply_rtc_prefix(actions, rtc_in)

        if not np.isfinite(actions).all():
            raise RuntimeError("non-finite values in physical EEF10 actions")

        out = {
            "actions": actions.astype(np.float32, copy=False),
            "policy_timing": {"infer_ms": (time.monotonic() - t0) * 1000.0},
        }
        if rtc_out is not None:
            out["rtc"] = rtc_out
        return out

    def _to_openwam_obs(self, obs: dict) -> dict:
        head = _pick_image(obs, _IMAGE_KEY_ALIASES["head"])
        if head is None:
            raise ObsValidationError(
                "missing head image; expected one of "
                + ", ".join(_IMAGE_KEY_ALIASES["head"])
            )
        right = _pick_image(obs, _IMAGE_KEY_ALIASES["right_wrist"])

        state = obs.get("observation/state", obs.get("state"))
        if state is None:
            raise ObsValidationError("missing observation/state (physical EEF10)")
        state = _as_eef10(state, name="observation/state")

        prompt = obs.get("prompt", "") or ""

        return {
            "images": {
                "head_camera": Image.fromarray(head),
                "left_wrist_camera": None,
                "right_wrist_camera": Image.fromarray(right) if right is not None else None,
            },
            "prompt": str(prompt),
            "state": state,
        }

    def _build_conditions(self, obs: dict) -> dict:
        conditions = {
            "observation": obs,
            "prompt": obs.get("prompt", "") or "",
        }
        img = obs.get("image")
        if img is not None:
            conditions["first_frame_image"] = [img]
        if obs.get("state") is not None:
            conditions["proprio"] = obs["state"]
        return conditions

    def _snap_binary_dims(self, actions: np.ndarray) -> np.ndarray:
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if not dims:
            return actions
        out = np.array(actions, dtype=np.float32, copy=True)
        for d in dims:
            if d >= out.shape[-1]:
                raise ValueError(
                    f"binary_command_dims includes {d} but action width is {out.shape[-1]}"
                )
            out[..., d] = np.where(out[..., d] > 0.5, 1.0, -1.0)
        return out

    def _apply_rtc_prefix(self, actions: np.ndarray, rtc_in: Any) -> tuple[dict, np.ndarray]:
        if not isinstance(rtc_in, dict):
            raise ObsValidationError("rtc must be a dict when provided")
        try:
            delay = int(rtc_in["delay"])
            start_index = int(rtc_in["start_index"])
            request_id = int(rtc_in["request_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObsValidationError(f"rtc missing/invalid delay|start_index|request_id ({exc})") from exc
        if delay < 0 or delay > self._max_delay:
            raise ObsValidationError(f"rtc.delay={delay} out of range [0, {self._max_delay}]")
        if delay > actions.shape[0]:
            raise ObsValidationError(
                f"rtc.delay={delay} exceeds action_horizon={actions.shape[0]}"
            )

        prefix_raw = rtc_in.get("prefix", None)
        if delay == 0:
            if prefix_raw is not None:
                prefix = np.asarray(prefix_raw, dtype=np.float32)
                if prefix.size != 0 and not (prefix.ndim == 2 and prefix.shape[0] == 0):
                    raise ObsValidationError("rtc.delay=0 expects empty/omitted prefix")
        else:
            if prefix_raw is None:
                raise ObsValidationError("rtc.prefix required when delay > 0")
            prefix = np.asarray(prefix_raw, dtype=np.float32)
            if prefix.shape != (delay, EEF10_DIM):
                raise ObsValidationError(
                    f"rtc.prefix shape must be ({delay}, {EEF10_DIM}), got {prefix.shape}"
                )
            if not np.isfinite(prefix).all():
                raise ObsValidationError("rtc.prefix contains NaN/Inf")
            # Hard constraint: overwrite with the client's committed absolute EEF10.
            actions = actions.copy()
            actions[:delay] = prefix

        rtc_out = {
            "request_id": request_id,
            "start_index": start_index,
            "delay": delay,
            "action_space": "absolute",
        }
        return rtc_out, actions


class OpenPIPiperServer:
    """websockets asyncio server compatible with openpi WebsocketClientPolicy."""

    def __init__(
        self,
        policy: PiperOpenPIPolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[dict] = None,
    ):
        self._policy = policy
        self._host = host
        self._port = int(port)
        self._metadata = dict(metadata if metadata is not None else policy.metadata)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def run(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        if host is not None:
            self._host = host
        if port is not None:
            self._port = int(port)
        asyncio.run(self._run())

    async def _run(self) -> None:
        import websockets.asyncio.server as ws_server

        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=None,
        ):
            logger.info(
                "OpenPI Piper server on ws://%s:%d metadata=%s",
                self._host,
                self._port,
                {k: self._metadata[k] for k in self._metadata},
            )
            await asyncio.Future()

    async def _handler(self, websocket) -> None:
        import websockets
        import websockets.frames

        logger.info("Client connected: %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw = await websocket.recv()
                if isinstance(raw, str):
                    raise ObsValidationError("expected binary msgpack obs frame, got text")
                obs = msgpack_numpy.unpackb(raw)

                infer_t0 = time.monotonic()
                result = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_t0) * 1000.0

                timing = dict(result.get("policy_timing") or {})
                timing["infer_ms"] = float(timing.get("infer_ms", infer_ms))
                result["policy_timing"] = timing
                result["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Client disconnected: %s", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                logger.exception("OpenPI Piper infer failed")
                try:
                    await websocket.send(tb)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except Exception:
                    pass
                break


def _health_check(connection, request):
    """Return 200 OK for GET /healthz; otherwise continue WebSocket upgrade."""
    path = getattr(request, "path", None) or ""
    if path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def build_openpi_piper_server(engine, cfg, *, host: str, port: int, max_delay: Optional[int] = None, training_rtc: Optional[bool] = None):
    """Construct :class:`OpenPIPiperServer` around an already-built engine."""
    metadata = build_piper_metadata(cfg, max_delay=max_delay, training_rtc=training_rtc)
    policy = PiperOpenPIPolicy(engine, cfg, metadata=metadata)
    return OpenPIPiperServer(policy=policy, host=host, port=port, metadata=metadata)


__all__ = [
    "EEF10_DIM",
    "OpenPIPiperServer",
    "PiperOpenPIPolicy",
    "build_openpi_piper_server",
    "build_piper_metadata",
    "resolve_action_horizon",
]
