#!/usr/bin/env python3
"""OpenWAM-native Piper EEF10 client (JSON WebSocket, no openpi-client).

Talks to the default OpenWAM ``PolicyServer`` (``--protocol openwam``, port
8848 by default). Wire contract:

  Client → ``{"type":"obs", "images":{...}, "prompt":..., "state":[10 floats]}``
  Server → ``{"type":"action", "action":[10 floats], ...}``
           With ``--return-action-chunk`` on both sides, also ``"actions":[[...],...]``.

EEF10 physical layout (matches banana / pick_blocks Piper training)::

  [xyz_m(3) | rot6d R[:,0]|R[:,1] (6) | gripper open_scale (1)]
  gripper: -1=closed, +1=open

Arm execution uses Piper ``EndPoseCtrl`` (firmware IK).

Run from the OpenWAM repo root (so ``benchmarks.utils`` imports resolve)::

  conda activate openwam
  # Terminal A — server (JSON protocol)
  CUDA_VISIBLE_DEVICES=7 bash scripts/deploy.sh /path/to/ckpt --port 8848

  # Terminal B — client
  python examples/piper_eef_client.py \\
      --host 127.0.0.1 --port 8848 \\
      --prompt 'Pick up the red block and place it in the box.' \\
      --can-name can0 \\
      --camera-backend opencv \\
      --head-camera 0 --wrist-camera 2
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Allow `python examples/piper_eef_client.py` from the OpenWAM repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.utils import (  # noqa: E402
    WSPolicyClient,
    build_payload,
    encode_numpy_b64,
    resize_for_lshape_slot,
)

logger = logging.getLogger(__name__)

EEF10_DIM = 10
PIPER_CONTROL_FREQUENCY = 30.0


# ---------------------------------------------------------------------------
# Image / camera helpers
# ---------------------------------------------------------------------------


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {image.shape}.")
    return np.ascontiguousarray(image)


class OpenCVCamera:
    def __init__(self, camera_id: int | str, *, name: str) -> None:
        import cv2

        self._cv2 = cv2
        self._name = name
        self._camera = cv2.VideoCapture(camera_id)
        if not self._camera.isOpened():
            raise RuntimeError(f"Failed to open {name} camera: {camera_id}")

    def read_rgb(self) -> np.ndarray:
        ok, frame = self._camera.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame from {self._name} camera.")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._camera.release()


class RealSenseCamera:
    def __init__(
        self,
        serial: str,
        *,
        name: str,
        width: int,
        height: int,
        fps: int,
        timeout_ms: int,
        warmup_frames: int,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "camera-backend=realsense requires pyrealsense2. "
                "Install it in the openwam env, or use --camera-backend opencv."
            ) from exc

        self._rs = rs
        self._name = name
        self._timeout_ms = int(timeout_ms)
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        self._pipeline.start(config)
        for _ in range(max(0, int(warmup_frames))):
            self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)

    def read_rgb(self) -> np.ndarray:
        try:
            frame_set = self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
        except RuntimeError as exc:
            raise RuntimeError(f"Timed out waiting for RealSense frame from {self._name}.") from exc
        while True:
            newer = self._pipeline.poll_for_frames()
            if not newer:
                break
            frame_set = newer
        color = frame_set.get_color_frame()
        if not color:
            raise RuntimeError(f"Failed to read color frame from {self._name} RealSense camera.")
        bgr = np.asanyarray(color.get_data())
        return bgr[..., ::-1].copy()

    def close(self) -> None:
        self._pipeline.stop()


class FakeCamera:
    """Synthetic RGB for --dry-run / --fake-cameras smoke tests."""

    def __init__(self, *, name: str, width: int = 640, height: int = 480, seed: int = 0) -> None:
        self._name = name
        self._rng = np.random.default_rng(seed)
        self._h = int(height)
        self._w = int(width)

    def read_rgb(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=(self._h, self._w, 3), dtype=np.uint8)

    def close(self) -> None:
        return


def make_camera(
    *,
    backend: str,
    serial: str,
    opencv_id: int | str,
    name: str,
    width: int,
    height: int,
    fps: int,
    timeout_ms: int,
    warmup_frames: int,
    fake_seed: int = 0,
):
    backend = str(backend).strip().lower()
    if backend == "realsense":
        return RealSenseCamera(
            serial,
            name=name,
            width=width,
            height=height,
            fps=fps,
            timeout_ms=timeout_ms,
            warmup_frames=warmup_frames,
        )
    if backend == "opencv":
        return OpenCVCamera(opencv_id, name=name)
    if backend == "fake":
        return FakeCamera(name=name, width=width, height=height, seed=fake_seed)
    raise ValueError(f"Unsupported camera backend {backend!r}; use realsense|opencv|fake.")


class CameraVisualizer:
    def __init__(self, *, enabled: bool, window_name: str, scale: float) -> None:
        self._enabled = bool(enabled)
        self._window_name = window_name
        self._scale = float(scale)
        self._cv2 = None
        self._window_created = False
        if self._enabled:
            if self._scale <= 0:
                raise ValueError("camera_preview_scale must be > 0.")
            import cv2

            self._cv2 = cv2
            self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
            self._window_created = True

    def show(self, head_rgb: np.ndarray, wrist_rgb: np.ndarray) -> bool:
        if not self._enabled:
            return True
        assert self._cv2 is not None
        head = _as_uint8_rgb(head_rgb)
        wrist = _as_uint8_rgb(wrist_rgb)
        if head.shape[:2] != wrist.shape[:2]:
            wrist = self._cv2.resize(wrist, (head.shape[1], head.shape[0]))
        preview = np.concatenate([head, wrist], axis=1)
        if self._scale != 1.0:
            w = max(1, int(round(preview.shape[1] * self._scale)))
            h = max(1, int(round(preview.shape[0] * self._scale)))
            preview = self._cv2.resize(preview, (w, h))
        self._cv2.imshow(self._window_name, preview[..., ::-1])
        return (self._cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        if self._window_created:
            assert self._cv2 is not None
            self._cv2.destroyWindow(self._window_name)
            self._window_created = False


# ---------------------------------------------------------------------------
# EEF10 ↔ Piper EndPoseCtrl
# ---------------------------------------------------------------------------


def gripper_sdk_units_to_meters(gripper_sdk: int | float) -> float:
    return float(gripper_sdk) * 1e-6


def meters_to_open_scale(gripper_m: float, *, gripper_max_m: float) -> float:
    if not math.isfinite(gripper_m) or gripper_max_m <= 0:
        raise ValueError("gripper_m must be finite and gripper_max_m must be > 0.")
    ratio = float(np.clip(gripper_m / gripper_max_m, 0.0, 1.0))
    return float(2.0 * ratio - 1.0)


def open_scale_to_meters(open_scale: float, *, gripper_max_m: float) -> float:
    if not math.isfinite(open_scale) or gripper_max_m <= 0:
        raise ValueError("open_scale must be finite and gripper_max_m must be > 0.")
    scale = float(np.clip(open_scale, -1.0, 1.0))
    return float((scale + 1.0) * 0.5 * gripper_max_m)


def open_scale_to_gripper_sdk(
    open_scale: float,
    *,
    gripper_max_m: float,
    open_mm: float,
    closed_mm: float,
    threshold_mm: float,
    binarize_gripper: bool,
) -> int:
    gripper_m = open_scale_to_meters(open_scale, gripper_max_m=gripper_max_m)
    action_mm = gripper_m * 1000.0
    if binarize_gripper:
        target_mm = closed_mm if action_mm <= threshold_mm else open_mm
    else:
        target_mm = float(np.clip(action_mm, min(closed_mm, open_mm), max(closed_mm, open_mm)))
    return round(target_mm * 1000.0)


def rpy_deg_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3, 3), got {rotation.shape}.")
    r20 = float(rotation[2, 0])
    if r20 < -1.0 + 1e-4:
        pitch = 90.0
        yaw = 0.0
        roll = math.degrees(math.atan2(float(rotation[0, 1]), float(rotation[1, 1])))
    elif r20 > 1.0 - 1e-4:
        pitch = -90.0
        yaw = 0.0
        roll = -math.degrees(math.atan2(float(rotation[0, 1]), float(rotation[1, 1])))
    else:
        pitch = math.degrees(
            math.atan2(-r20, math.sqrt(float(rotation[0, 0]) ** 2 + float(rotation[1, 0]) ** 2))
        )
        cos_pitch = math.cos(math.radians(pitch))
        yaw = math.degrees(math.atan2(float(rotation[1, 0]) / cos_pitch, float(rotation[0, 0]) / cos_pitch))
        roll = math.degrees(math.atan2(float(rotation[2, 1]) / cos_pitch, float(rotation[2, 2]) / cos_pitch))
    return float(roll), float(pitch), float(yaw)


def rotation_matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3, 3), got {rotation.shape}.")
    return np.concatenate([rotation[:, 0], rotation[:, 1]], axis=0).astype(np.float32)


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(rot6d)):
        raise ValueError("rot6d contains non-finite values.")
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        raise ValueError("rot6d first column has near-zero norm.")
    b1 = a1 / n1
    a2_proj = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2_proj)
    if n2 < 1e-8:
        raise ValueError("rot6d second column is parallel to the first.")
    b2 = a2_proj / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def eef10_to_end_pose_sdk_units(action: np.ndarray) -> tuple[int, int, int, int, int, int]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (EEF10_DIM,):
        raise ValueError(f"Expected EEF10 action shape ({EEF10_DIM},), got {action.shape}.")
    if not np.all(np.isfinite(action)):
        raise ValueError("EEF10 action contains non-finite values.")
    xyz_m = action[:3]
    rotation = rot6d_to_rotation_matrix(action[3:9])
    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy_deg(rotation)
    return (
        round(float(xyz_m[0]) * 1e6),
        round(float(xyz_m[1]) * 1e6),
        round(float(xyz_m[2]) * 1e6),
        round(roll_deg * 1000.0),
        round(pitch_deg * 1000.0),
        round(yaw_deg * 1000.0),
    )


def _as_eef10(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (EEF10_DIM,):
        raise ValueError(f"{name} must be length-{EEF10_DIM} EEF10, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


@contextlib.contextmanager
def prevent_keyboard_interrupt() -> Iterator[None]:
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class GripperHoldClose:
    def __init__(
        self,
        *,
        enabled: bool,
        gripper_max_m: float,
        close_trigger_mm: float,
        release_trigger_mm: float,
        hold_mm: float,
    ) -> None:
        self.enabled = enabled
        self.gripper_max_m = gripper_max_m
        self.close_trigger_mm = close_trigger_mm
        self.release_trigger_mm = release_trigger_mm
        self.hold_mm = hold_mm
        self._holding = False

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if not self.enabled:
            return action
        gripper_mm = open_scale_to_meters(float(action[-1]), gripper_max_m=self.gripper_max_m) * 1000.0
        if self._holding:
            if gripper_mm >= self.release_trigger_mm:
                self._holding = False
                logger.info("Released gripper hold-close at policy target %.1f mm.", gripper_mm)
            else:
                action[-1] = meters_to_open_scale(self.hold_mm * 1e-3, gripper_max_m=self.gripper_max_m)
                return action
        elif gripper_mm <= self.close_trigger_mm:
            self._holding = True
            logger.info("Activated gripper hold-close at policy target %.1f mm.", gripper_mm)
            action[-1] = meters_to_open_scale(self.hold_mm * 1e-3, gripper_max_m=self.gripper_max_m)
        return action


class PiperArmEEF:
    """Piper arm using physical EEF10 proprio/actions and EndPoseCtrl."""

    def __init__(
        self,
        can_name: str,
        *,
        dry_run: bool,
        move_speed_percent: int,
        enable_timeout_s: float,
        gripper_max_m: float,
        gripper_open_mm: float,
        gripper_closed_mm: float,
        gripper_threshold_mm: float,
        gripper_effort: int,
        binarize_gripper: bool,
    ) -> None:
        self._dry_run = bool(dry_run)
        self._move_speed_percent = move_speed_percent
        self._gripper_max_m = gripper_max_m
        self._gripper_open_mm = gripper_open_mm
        self._gripper_closed_mm = gripper_closed_mm
        self._gripper_threshold_mm = gripper_threshold_mm
        self._binarize_gripper = binarize_gripper
        self._gripper_effort = gripper_effort
        self._piper = None
        self._fake_state = np.zeros(EEF10_DIM, dtype=np.float32)
        self._fake_state[3] = 1.0  # rot6d identity-ish
        self._fake_state[7] = 1.0
        self._fake_state[-1] = 1.0  # open

        if self._dry_run:
            logger.warning("dry-run: Piper CAN commands will not be sent.")
            return

        try:
            from piper_sdk import C_PiperInterface_V2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Piper arm requires piper_sdk. Install it in the openwam env, or pass --dry-run."
            ) from exc

        self._piper = C_PiperInterface_V2(can_name)
        try:
            self._piper.ConnectPort()
            logger.info("Enabling Piper arm on %s (MOVE P / EndPoseCtrl).", can_name)
            deadline = time.monotonic() + enable_timeout_s
            while not self._piper.EnablePiper():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out enabling Piper arm on {can_name} after {enable_timeout_s:.1f}s."
                    )
                time.sleep(0.01)
            self._piper.MotionCtrl_2(0x01, 0x00, move_speed_percent, 0x00)
        except Exception:
            self.close()
            raise

    def read_state(self) -> np.ndarray:
        if self._dry_run or self._piper is None:
            return self._fake_state.copy()

        end_pose = self._piper.GetArmEndPoseMsgs().end_pose
        xyz_m = np.array([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis], dtype=np.float64) * 1e-6
        rpy_deg = np.array([end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis], dtype=np.float64) * 1e-3
        rotation = rpy_deg_to_rotation_matrix(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
        rot6d = rotation_matrix_to_rot6d(rotation)
        gripper_m = gripper_sdk_units_to_meters(self._piper.GetArmGripperMsgs().gripper_state.grippers_angle)
        open_scale = meters_to_open_scale(gripper_m, gripper_max_m=self._gripper_max_m)
        state = np.concatenate(
            [xyz_m.astype(np.float32), rot6d, np.asarray([open_scale], dtype=np.float32)]
        ).astype(np.float32)
        if state.shape != (EEF10_DIM,) or not np.all(np.isfinite(state)):
            raise RuntimeError("Piper returned a non-finite or malformed EEF10 state.")
        return state

    def send_action(self, action: np.ndarray) -> None:
        action = _as_eef10(action, name="action")
        if self._dry_run or self._piper is None:
            self._fake_state = action.copy()
            return
        x, y, z, rx, ry, rz = eef10_to_end_pose_sdk_units(action)
        gripper_sdk = open_scale_to_gripper_sdk(
            float(action[-1]),
            gripper_max_m=self._gripper_max_m,
            open_mm=self._gripper_open_mm,
            closed_mm=self._gripper_closed_mm,
            threshold_mm=self._gripper_threshold_mm,
            binarize_gripper=self._binarize_gripper,
        )
        self._piper.MotionCtrl_2(0x01, 0x00, self._move_speed_percent, 0x00)
        self._piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        self._piper.GripperCtrl(gripper_sdk, self._gripper_effort, 0x01, 0)

    def close(self) -> None:
        if self._piper is not None and hasattr(self._piper, "DisconnectPort"):
            self._piper.DisconnectPort()
        self._piper = None


# ---------------------------------------------------------------------------
# OpenWAM policy client
# ---------------------------------------------------------------------------


class OpenWAMPiperPolicy:
    """Thin wrapper around :class:`WSPolicyClient` for Piper EEF10."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        request_timeout: float = 300.0,
        resize_lshape: bool = True,
        return_action_chunk: bool = False,
        open_loop_horizon: Optional[int] = None,
    ) -> None:
        self._client = WSPolicyClient(f"ws://{host}:{port}", timeout=request_timeout)
        self._resize_lshape = bool(resize_lshape)
        self._return_action_chunk = bool(return_action_chunk)
        self._open_loop_horizon = int(open_loop_horizon) if open_loop_horizon is not None else None
        self._pending: list[np.ndarray] = []
        self._pending_idx = 0
        self._server_return_action_chunk: Optional[bool] = None

        pong = self._client.ping()
        if pong.get("type") != "pong":
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")
        if "return_action_chunk" in pong:
            self._server_return_action_chunk = bool(pong["return_action_chunk"])
            if self._return_action_chunk and not self._server_return_action_chunk:
                raise RuntimeError(
                    "Client --return-action-chunk is set, but the server advertises "
                    "return_action_chunk=false. Restart the server with "
                    "`--return-action-chunk` (or server.return_action_chunk=true)."
                )
            if (not self._return_action_chunk) and self._server_return_action_chunk:
                logger.warning(
                    "Server return_action_chunk=true but client is in step mode; "
                    "extra 'actions' fields will be ignored. Pass --return-action-chunk "
                    "on the client to open-loop locally."
                )
        logger.info(
            "Connected to OpenWAM PolicyServer ws://%s:%d "
            "(representation=%r, client_chunk=%s, server_chunk=%s)",
            host,
            port,
            pong.get("representation"),
            self._return_action_chunk,
            self._server_return_action_chunk,
        )

    def reset(self) -> None:
        self._pending.clear()
        self._pending_idx = 0
        ack = self._client.reset()
        if ack.get("type") != "reset_ack":
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def _build_payload(
        self,
        head_rgb: np.ndarray,
        wrist_rgb: Optional[np.ndarray],
        state: np.ndarray,
        prompt: str,
    ) -> dict:
        head = _as_uint8_rgb(head_rgb)
        if self._resize_lshape:
            head_enc = encode_numpy_b64(resize_for_lshape_slot(head, "head_camera"))
        else:
            head_enc = encode_numpy_b64(head)

        right_enc = None
        if wrist_rgb is not None:
            wrist = _as_uint8_rgb(wrist_rgb)
            if self._resize_lshape:
                right_enc = encode_numpy_b64(resize_for_lshape_slot(wrist, "right_wrist_camera"))
            else:
                right_enc = encode_numpy_b64(wrist)

        return build_payload(
            head=head_enc,
            left_wrist=None,
            right_wrist=right_enc,
            prompt=str(prompt),
            state=_as_eef10(state, name="state").tolist(),
        )

    def predict(self, head_rgb: np.ndarray, wrist_rgb: Optional[np.ndarray], state: np.ndarray, prompt: str) -> np.ndarray:
        """Return one EEF10 action (step mode, or next open-loop step in chunk mode)."""
        if self._return_action_chunk and self._pending_idx < len(self._pending):
            action = self._pending[self._pending_idx]
            self._pending_idx += 1
            return action

        payload = self._build_payload(head_rgb, wrist_rgb, state, prompt)
        with prevent_keyboard_interrupt():
            response = self._client.predict(payload)

        if self._return_action_chunk:
            raw_chunk = response.get("actions")
            if not raw_chunk:
                raise RuntimeError(
                    "Client is in chunk mode but server response has no 'actions'. "
                    "Start the JSON server with --return-action-chunk."
                )
            chunk = [_as_eef10(a, name="server action") for a in raw_chunk]
            if self._open_loop_horizon is not None:
                if self._open_loop_horizon > len(chunk):
                    raise ValueError(
                        f"open_loop_horizon={self._open_loop_horizon} exceeds returned chunk length {len(chunk)}"
                    )
                chunk = chunk[: self._open_loop_horizon]
            self._pending = chunk
            self._pending_idx = 1
            logger.info("Received EEF10 action chunk with shape (%d, %d).", len(chunk), EEF10_DIM)
            return chunk[0]

        if "action" not in response:
            raise RuntimeError(f"OpenWAM response missing action field: {response}")
        return _as_eef10(response["action"], name="server action")

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# CLI / main loop
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenWAM JSON WebSocket Piper EEF10 client (no openpi-client).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8848, help="OpenWAM PolicyServer port (JSON protocol).")
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument(
        "--prompt",
        default='Pick up the book labeled "Wang Guowei Selected Works" and place it in the black grid.',
    )
    p.add_argument("--can-name", default="can0")
    p.add_argument(
        "--camera-backend",
        choices=("opencv", "realsense", "fake"),
        default="opencv",
        help="fake = synthetic images (for server smoke without cameras).",
    )
    p.add_argument("--head-camera-serial", default="339322074804")
    p.add_argument("--wrist-camera-serial", default="346522074547")
    p.add_argument("--head-camera", default="0", help="OpenCV camera id / path for head.")
    p.add_argument("--wrist-camera", default="2", help="OpenCV camera id / path for right wrist.")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--camera-timeout-ms", type=int, default=1000)
    p.add_argument("--camera-warmup-frames", type=int, default=5)
    p.add_argument("--show-cameras", action="store_true")
    p.add_argument("--camera-preview-window", default="Piper cameras (OpenWAM EEF10)")
    p.add_argument("--camera-preview-scale", type=float, default=1.5)
    p.add_argument("--max-timesteps", type=int, default=2000)
    p.add_argument("--control-hz", type=float, default=PIPER_CONTROL_FREQUENCY)
    p.add_argument("--move-speed-percent", type=int, default=30)
    p.add_argument("--enable-timeout-s", type=float, default=30.0)
    p.add_argument("--gripper-max-m", type=float, default=0.07)
    p.add_argument("--gripper-open-mm", type=float, default=70.0)
    p.add_argument("--gripper-closed-mm", type=float, default=0.0)
    p.add_argument("--gripper-threshold-mm", type=float, default=35.0)
    p.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gripper-effort", type=int, default=2000)
    p.add_argument("--gripper-hold-close", action="store_true")
    p.add_argument("--gripper-close-trigger-mm", type=float, default=25.0)
    p.add_argument("--gripper-release-trigger-mm", type=float, default=45.0)
    p.add_argument("--gripper-hold-mm", type=float, default=0.0)
    p.add_argument(
        "--resize-lshape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-resize tiles to banana/pick_blocks L-shape slots before send.",
    )
    p.add_argument(
        "--return-action-chunk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request/consume full action chunks: one WS round-trip per chunk, "
        "then open-loop locally. Server must be started with --return-action-chunk.",
    )
    p.add_argument(
        "--open-loop-horizon",
        type=int,
        default=None,
        help="With --return-action-chunk: execute only the first N actions from "
        "each chunk (default: use the full returned chunk).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not enable Piper / send CAN; still talks to the policy server.",
    )
    p.add_argument("--log-actions", action="store_true")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.control_hz <= 0:
        raise ValueError("control_hz must be > 0.")
    if args.max_timesteps < 1:
        raise ValueError("max_timesteps must be >= 1.")
    if not 1 <= args.move_speed_percent <= 100:
        raise ValueError("move_speed_percent must be between 1 and 100.")
    if args.gripper_max_m <= 0:
        raise ValueError("gripper_max_m must be > 0.")
    if args.gripper_hold_close and args.gripper_release_trigger_mm <= args.gripper_close_trigger_mm:
        raise ValueError("gripper_release_trigger_mm must be greater than gripper_close_trigger_mm.")
    if args.open_loop_horizon is not None:
        if args.open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be >= 1.")
        if not args.return_action_chunk:
            raise ValueError("--open-loop-horizon requires --return-action-chunk.")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)

    policy = OpenWAMPiperPolicy(
        args.host,
        args.port,
        request_timeout=args.request_timeout,
        resize_lshape=args.resize_lshape,
        return_action_chunk=args.return_action_chunk,
        open_loop_horizon=args.open_loop_horizon,
    )

    with contextlib.ExitStack() as stack:
        stack.callback(policy.close)

        head_camera = make_camera(
            backend=args.camera_backend,
            serial=args.head_camera_serial,
            opencv_id=args.head_camera,
            name="head",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            timeout_ms=args.camera_timeout_ms,
            warmup_frames=args.camera_warmup_frames,
            fake_seed=0,
        )
        stack.callback(head_camera.close)
        wrist_camera = make_camera(
            backend=args.camera_backend,
            serial=args.wrist_camera_serial,
            opencv_id=args.wrist_camera,
            name="right wrist",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            timeout_ms=args.camera_timeout_ms,
            warmup_frames=args.camera_warmup_frames,
            fake_seed=1,
        )
        stack.callback(wrist_camera.close)

        visualizer = CameraVisualizer(
            enabled=args.show_cameras,
            window_name=args.camera_preview_window,
            scale=args.camera_preview_scale,
        )
        stack.callback(visualizer.close)

        arm = PiperArmEEF(
            args.can_name,
            dry_run=args.dry_run,
            move_speed_percent=args.move_speed_percent,
            enable_timeout_s=args.enable_timeout_s,
            gripper_max_m=args.gripper_max_m,
            gripper_open_mm=args.gripper_open_mm,
            gripper_closed_mm=args.gripper_closed_mm,
            gripper_threshold_mm=args.gripper_threshold_mm,
            gripper_effort=args.gripper_effort,
            binarize_gripper=args.binarize_gripper,
        )
        stack.callback(arm.close)

        policy.reset()
        if args.dry_run:
            logger.warning("dry-run control loop starting (fake/cameras + policy only).")
        else:
            logger.warning("Piper is enabled (EEF10 / EndPoseCtrl). Live control is starting.")

        gripper_hold = GripperHoldClose(
            enabled=args.gripper_hold_close,
            gripper_max_m=args.gripper_max_m,
            close_trigger_mm=args.gripper_close_trigger_mm,
            release_trigger_mm=args.gripper_release_trigger_mm,
            hold_mm=args.gripper_hold_mm,
        )
        control_period = 1.0 / args.control_hz

        for step in range(args.max_timesteps):
            start = time.monotonic()
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            if not visualizer.show(head_rgb, wrist_rgb):
                logger.info("Camera preview requested shutdown.")
                break
            state = arm.read_state()

            action = policy.predict(head_rgb, wrist_rgb, state, args.prompt)
            action = gripper_hold.apply(action)
            if args.log_actions and step % 10 == 0:
                logger.info(
                    "step=%d latency_budget state_xyz=%s action_xyz=%s grip=%.3f",
                    step,
                    np.array2string(state[:3], precision=3),
                    np.array2string(action[:3], precision=3),
                    float(action[-1]),
                )
            arm.send_action(action)

            elapsed = time.monotonic() - start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Piper OpenWAM EEF10 control stopped by user.")


if __name__ == "__main__":
    main()
