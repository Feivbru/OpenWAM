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


class ChunkControlPacer:
    """Pace control steps; optional dynamic Hz that absorbs server RTT.

    Fixed mode: each step sleeps to ``1 / control_hz``.

    Dynamic mode (chunk fetch): after a server round-trip of ``interact_s`` that
    returns ``N`` actions, set

        step_period = max(0, N / control_hz - interact_s) / N

    so ``interact_s + N * step_period ≈ N / control_hz``. Open-loop execution
    Hz is then slightly above the nominal ``control_hz`` whenever RTT > 0.
    """

    def __init__(self, control_hz: float, *, dynamic: bool = False) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be > 0.")
        self.control_hz = float(control_hz)
        self.dynamic = bool(dynamic)
        self._nominal_period = 1.0 / self.control_hz
        self._step_period = self._nominal_period
        self._deadline: Optional[float] = None
        self._steps_left_in_chunk = 0

    @property
    def policy_step_period(self) -> float:
        """Period allocated to one policy action (before interpolation substeps)."""
        if self.dynamic and self._steps_left_in_chunk > 0:
            return float(self._step_period)
        return float(self._nominal_period)

    def note_chunk_fetch(self, n: int, interact_s: float) -> None:
        """Record a fresh chunk of ``n`` actions that cost ``interact_s`` on the wire."""
        if not self.dynamic:
            return
        n = int(n)
        if n < 1:
            raise ValueError("chunk length must be >= 1.")
        budget = n / self.control_hz
        interact_s = max(0.0, float(interact_s))
        exec_budget = budget - interact_s
        if exec_budget <= 0:
            logger.warning(
                "dynamic-control-hz: server interact %.3fs >= chunk budget %.3fs "
                "(N=%d, control_hz=%.3f); open-loop period clamped to 0.",
                interact_s,
                budget,
                n,
                self.control_hz,
            )
            self._step_period = 0.0
        else:
            self._step_period = exec_budget / n
        self._steps_left_in_chunk = n
        self._deadline = None
        logger.info(
            "dynamic-control-hz: N=%d interact=%.3fs budget=%.3fs "
            "step_period=%.4fs (exec_hz≈%.2f, nominal=%.2f)",
            n,
            interact_s,
            budget,
            self._step_period,
            (1.0 / self._step_period) if self._step_period > 0 else float("inf"),
            self.control_hz,
        )

    def consume_policy_step(self) -> None:
        """Advance dynamic-chunk accounting after one policy action is fully sent."""
        if not self.dynamic or self._steps_left_in_chunk <= 0:
            return
        self._steps_left_in_chunk -= 1
        if self._steps_left_in_chunk <= 0:
            self._deadline = None
            self._step_period = self._nominal_period

    def wait_after_step(self, step_start: float) -> None:
        """Sleep so this control tick respects the active period / chunk deadline."""
        if not self.dynamic or self._steps_left_in_chunk <= 0:
            elapsed = time.monotonic() - step_start
            if elapsed < self._nominal_period:
                time.sleep(self._nominal_period - elapsed)
            return

        now = time.monotonic()
        if self._deadline is None:
            # Start open-loop deadlines after the just-finished step work so
            # server RTT is outside the N * step_period window.
            self._deadline = now + self._step_period
        if now < self._deadline:
            time.sleep(self._deadline - now)
        self._deadline += self._step_period
        self.consume_policy_step()


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


def _stack_cameras_vertical(head_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    """Stack head (top) + wrist (bottom); resize wrist to head width if needed."""
    import cv2

    head = _as_uint8_rgb(head_rgb)
    wrist = _as_uint8_rgb(wrist_rgb)
    if head.shape[1] != wrist.shape[1]:
        new_h = max(1, int(round(wrist.shape[0] * (head.shape[1] / wrist.shape[1]))))
        wrist = cv2.resize(wrist, (head.shape[1], new_h))
    elif head.shape[0] != wrist.shape[0]:
        # Same width already; keep native heights for vertical concat.
        pass
    return np.concatenate([head, wrist], axis=0)


class EpisodeVideoRecorder:
    """Write a vertically stacked dual-camera episode video at a fixed Hz.

    Overlay (top-right): open-loop interaction index, 0-based — increments each
    time the client fetches a new action chunk from the server (or each server
    step when not in ``--return-action-chunk`` mode).
    """

    def __init__(self, path: str | Path, *, hz: float = 5.0) -> None:
        if hz <= 0:
            raise ValueError("save_video_hz must be > 0.")
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        self.hz = float(hz)
        self._period = 1.0 / self.hz
        self._writer = None
        self._size: Optional[tuple[int, int]] = None  # (w, h)
        self._last_write_t: Optional[float] = None
        self._frames_written = 0
        self._interact_idx = 0  # currently displayed open-loop index
        self._next_interact_idx = 0  # assigned on the next server fetch

    @property
    def interact_idx(self) -> int:
        return int(self._interact_idx)

    def note_interact(self) -> int:
        """Mark a new open-loop / server interaction; returns the 0-based index."""
        self._interact_idx = self._next_interact_idx
        self._next_interact_idx += 1
        return self._interact_idx

    def maybe_write(self, head_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
        now = time.monotonic()
        if self._last_write_t is not None and (now - self._last_write_t) < self._period:
            return
        frame_rgb = _stack_cameras_vertical(head_rgb, wrist_rgb)
        frame_bgr = self._overlay_interact(frame_rgb[..., ::-1].copy())
        h, w = frame_bgr.shape[:2]
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = self._cv2.VideoWriter(str(self.path), fourcc, self.hz, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter at {self.path}")
            self._size = (w, h)
            logger.info(
                "Saving episode video to %s (%.3f Hz, stacked head/wrist vertical)",
                self.path,
                self.hz,
            )
        assert self._size is not None
        if (w, h) != self._size:
            frame_bgr = self._cv2.resize(frame_bgr, self._size)
        self._writer.write(frame_bgr)
        self._last_write_t = now
        self._frames_written += 1

    def _overlay_interact(self, frame_bgr: np.ndarray) -> np.ndarray:
        label = f"interact {self._interact_idx}"
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, frame_bgr.shape[1] / 640.0 * 0.7)
        thickness = max(1, int(round(scale * 2)))
        (tw, th), baseline = self._cv2.getTextSize(label, font, scale, thickness)
        x = max(8, frame_bgr.shape[1] - tw - 12)
        y = th + 12
        # Dark box behind text for readability.
        pad = 6
        self._cv2.rectangle(
            frame_bgr,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            (0, 0, 0),
            thickness=-1,
        )
        self._cv2.putText(
            frame_bgr,
            label,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            self._cv2.LINE_AA,
        )
        return frame_bgr

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "Episode video closed: %s (%d frames @ %.3f Hz, last interact=%d)",
                self.path,
                self._frames_written,
                self.hz,
                self._interact_idx,
            )


def default_save_video_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return _ROOT / "runs" / f"piper_eef_{stamp}.mp4"


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


def lerp_eef10(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate EEF10: xyz/gripper linear; rot6d via matrix lerp + orthonormalize.

    Mirrors ``infer/example/main.py`` setpoint blending, with a safer rotation path
    than raw rot6d lerp.
    """
    a = np.asarray(a, dtype=np.float32).reshape(EEF10_DIM)
    b = np.asarray(b, dtype=np.float32).reshape(EEF10_DIM)
    t = float(np.clip(alpha, 0.0, 1.0))
    out = np.empty(EEF10_DIM, dtype=np.float32)
    out[:3] = (1.0 - t) * a[:3] + t * b[:3]
    out[-1] = (1.0 - t) * a[-1] + t * b[-1]
    try:
        ra = rot6d_to_rotation_matrix(a[3:9])
        rb = rot6d_to_rotation_matrix(b[3:9])
        r = (1.0 - t) * ra + t * rb
        # Re-orthonormalize blended columns (same convention as rot6d_to_rotation_matrix).
        c0 = r[:, 0]
        n0 = np.linalg.norm(c0)
        if n0 < 1e-8:
            out[3:9] = b[3:9]
        else:
            c0 = c0 / n0
            c1 = r[:, 1] - np.dot(c0, r[:, 1]) * c0
            n1 = np.linalg.norm(c1)
            if n1 < 1e-8:
                out[3:9] = b[3:9]
            else:
                c1 = c1 / n1
                out[3:9] = rotation_matrix_to_rot6d(np.stack([c0, c1, np.cross(c0, c1)], axis=1))
    except ValueError:
        out[3:9] = b[3:9]
    return out


def send_action_interpolated(
    arm: "PiperArmEEF",
    previous: np.ndarray,
    target: np.ndarray,
    *,
    substeps: int,
    step_period: float,
) -> None:
    """Send ``substeps`` setpoints between ``previous`` and ``target`` (main.py style)."""
    substeps = max(1, int(substeps))
    previous = np.asarray(previous, dtype=np.float32).reshape(EEF10_DIM)
    target = np.asarray(target, dtype=np.float32).reshape(EEF10_DIM)
    if substeps == 1:
        arm.send_action(target)
        return
    sub_period = max(0.0, float(step_period) / substeps)
    next_send = time.monotonic()
    for i in range(1, substeps + 1):
        now = time.monotonic()
        if now < next_send:
            time.sleep(next_send - now)
        arm.send_action(lerp_eef10(previous, target, i / substeps))
        next_send += sub_period


def boundary_blend_chunk(
    chunk: list[np.ndarray],
    anchor: np.ndarray,
    *,
    blend_steps: int,
) -> list[np.ndarray]:
    """Blend the head of a freshly received chunk toward ``anchor`` (last command).

    Step ``i`` (0-based) uses ``alpha=(i+1)/n`` so the front stays closer to the
    previous command and the blend window ends on the raw new action. Helps
    absorb cube-boundary jumps / "retract then re-descend" from stale replans.
    """
    if blend_steps <= 0 or not chunk:
        return chunk
    anchor = np.asarray(anchor, dtype=np.float32).reshape(EEF10_DIM)
    n = min(int(blend_steps), len(chunk))
    out = list(chunk)
    for i in range(n):
        alpha = (i + 1) / n
        out[i] = lerp_eef10(anchor, np.asarray(out[i], dtype=np.float32), alpha)
    return out


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
        replan_cubes: Optional[int] = None,
        merge_mode: Optional[str] = None,
    ) -> None:
        self._client = WSPolicyClient(f"ws://{host}:{port}", timeout=request_timeout)
        self._resize_lshape = bool(resize_lshape)
        self._return_action_chunk = bool(return_action_chunk)
        self._open_loop_horizon = int(open_loop_horizon) if open_loop_horizon is not None else None
        self._replan_cubes = int(replan_cubes) if replan_cubes is not None else None
        self._merge_mode = str(merge_mode).strip().lower() if merge_mode is not None else None
        self._pending: list[np.ndarray] = []
        self._pending_idx = 0
        self._server_return_action_chunk: Optional[bool] = None
        # Set when predict() fetched a new chunk: (chunk_len, interact_s).
        self._last_chunk_fetch: Optional[tuple[int, float]] = None

        pong = self._client.ping(replan_cubes=self._replan_cubes, merge_mode=self._merge_mode)
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
        server_vrtc = pong.get("vrtc") if isinstance(pong.get("vrtc"), dict) else None
        if self._replan_cubes is not None or self._merge_mode is not None:
            if not server_vrtc or not server_vrtc.get("enabled"):
                raise RuntimeError(
                    "Client sent VRTC ping overrides, but the server did not advertise "
                    "VRTC on pong. Use a VRTC checkpoint / enable vrtc."
                )
        if self._replan_cubes is not None:
            applied = int(server_vrtc.get("replan_cubes", -1))
            if applied != self._replan_cubes:
                raise RuntimeError(
                    f"Requested replan_cubes={self._replan_cubes} but server pong reports "
                    f"vrtc.replan_cubes={applied}."
                )
        if self._merge_mode is not None:
            applied_mode = str(server_vrtc.get("merge_mode", "")).strip().lower()
            if applied_mode != self._merge_mode:
                raise RuntimeError(
                    f"Requested merge_mode={self._merge_mode!r} but server pong reports "
                    f"vrtc.merge_mode={applied_mode!r}."
                )
        logger.info(
            "Connected to OpenWAM PolicyServer ws://%s:%d "
            "(representation=%r, client_chunk=%s, server_chunk=%s, vrtc=%s)",
            host,
            port,
            pong.get("representation"),
            self._return_action_chunk,
            self._server_return_action_chunk,
            server_vrtc,
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
        self._last_chunk_fetch = None
        if self._return_action_chunk and self._pending_idx < len(self._pending):
            action = self._pending[self._pending_idx]
            self._pending_idx += 1
            return action

        payload = self._build_payload(head_rgb, wrist_rgb, state, prompt)
        t0 = time.monotonic()
        with prevent_keyboard_interrupt():
            response = self._client.predict(payload)
        interact_s = time.monotonic() - t0

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
            self._last_chunk_fetch = (len(chunk), interact_s)
            logger.info(
                "Received EEF10 action chunk with shape (%d, %d) (server_interact=%.3fs).",
                len(chunk),
                EEF10_DIM,
                interact_s,
            )
            return chunk[0]

        if "action" not in response:
            raise RuntimeError(f"OpenWAM response missing action field: {response}")
        return _as_eef10(response["action"], name="server action")

    def pop_chunk_fetch(self) -> Optional[tuple[int, float]]:
        """Return ``(N, interact_s)`` once after a chunk fetch, else ``None``."""
        meta = self._last_chunk_fetch
        self._last_chunk_fetch = None
        return meta

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
    p.add_argument(
        "--save-video",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help="Save episode video: head (top) + wrist (bottom), overlay open-loop "
        "interact index (0-based) at top-right. Omit PATH to write "
        "runs/piper_eef_<timestamp>.mp4 under the OpenWAM repo root.",
    )
    p.add_argument(
        "--save-video-hz",
        type=float,
        default=5.0,
        help="Frame rate for --save-video (wall-clock throttle; default 5 Hz).",
    )
    p.add_argument("--max-timesteps", type=int, default=2000)
    p.add_argument("--control-hz", type=float, default=PIPER_CONTROL_FREQUENCY)
    p.add_argument(
        "--dynamic-control-hz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --return-action-chunk: pace so server_interact + N*step_period "
        "≈ N/control_hz (open-loop exec Hz slightly above nominal). "
        "Default off = fixed 1/control_hz per step.",
    )
    p.add_argument(
        "--interpolation-substeps",
        type=int,
        default=1,
        help="Like infer/example/main.py: linearly blend previous→target EEF10 "
        "across N robot setpoints per policy action (1=off). Use 4 for smoother motion.",
    )
    p.add_argument(
        "--chunk-xyz-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After a chunk fetch, re-read state and add xyz delta to the whole "
        "pending chunk (main.py state_shift idea; EEF absolute poses, xyz only).",
    )
    p.add_argument(
        "--boundary-blend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After a chunk fetch, blend the first --boundary-blend-steps actions "
        "from the last commanded EEF10 toward the new chunk (reduces retract jumps).",
    )
    p.add_argument(
        "--boundary-blend-steps",
        type=int,
        default=4,
        help="With --boundary-blend: number of leading actions to ramp "
        "(default 4 = one VRTC cube).",
    )
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
        "--replan-cubes",
        type=int,
        default=None,
        help="VRTC only: send replan_cubes on the initial ping so the server "
        "records the prefetch threshold for this session (0=sync when wait empty).",
    )
    p.add_argument(
        "--merge-mode",
        choices=("replace", "average", "blend"),
        default=None,
        help="VRTC only: wait-pool merge on INFER_MERGE — replace (default on "
        "server), average overlapping cubes, or blend (within-cube old→new ramp). "
        "Sent on the initial ping.",
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
    if args.replan_cubes is not None and args.replan_cubes < 0:
        raise ValueError("replan_cubes must be >= 0.")
    if args.dynamic_control_hz and not args.return_action_chunk:
        raise ValueError("--dynamic-control-hz requires --return-action-chunk.")
    if args.interpolation_substeps < 1:
        raise ValueError("interpolation_substeps must be >= 1.")
    if args.chunk_xyz_shift and not args.return_action_chunk:
        raise ValueError("--chunk-xyz-shift requires --return-action-chunk.")
    if args.boundary_blend and not args.return_action_chunk:
        raise ValueError("--boundary-blend requires --return-action-chunk.")
    if args.boundary_blend_steps < 1:
        raise ValueError("boundary_blend_steps must be >= 1.")
    if args.save_video_hz <= 0:
        raise ValueError("save_video_hz must be > 0.")


def _resolve_save_video_path(save_video: Optional[str]) -> Optional[Path]:
    if save_video is None:
        return None
    if save_video == "__AUTO__" or str(save_video).strip() == "":
        return default_save_video_path()
    return Path(save_video).expanduser().resolve()


def run(args: argparse.Namespace) -> None:
    _validate_args(args)

    policy = OpenWAMPiperPolicy(
        args.host,
        args.port,
        request_timeout=args.request_timeout,
        resize_lshape=args.resize_lshape,
        return_action_chunk=args.return_action_chunk,
        open_loop_horizon=args.open_loop_horizon,
        replan_cubes=args.replan_cubes,
        merge_mode=args.merge_mode,
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

        video_path = _resolve_save_video_path(args.save_video)
        recorder: Optional[EpisodeVideoRecorder] = None
        if video_path is not None:
            recorder = EpisodeVideoRecorder(video_path, hz=args.save_video_hz)
            stack.callback(recorder.close)

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
        pacer = ChunkControlPacer(args.control_hz, dynamic=args.dynamic_control_hz)
        if args.dynamic_control_hz:
            logger.info(
                "dynamic-control-hz enabled: target cycle N/control_hz with "
                "open-loop period = (N/hz - server_interact) / N"
            )
        if args.interpolation_substeps > 1:
            logger.info(
                "execution interpolation enabled: %d substeps per policy action "
                "(send rate ≈ %.1f Hz)",
                args.interpolation_substeps,
                args.control_hz * args.interpolation_substeps,
            )
        if args.boundary_blend:
            logger.info(
                "boundary-blend enabled: ramp first %d actions of each new chunk "
                "from last command toward the model chunk",
                args.boundary_blend_steps,
            )
        previous_action: Optional[np.ndarray] = None

        for step in range(args.max_timesteps):
            start = time.monotonic()
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            if not visualizer.show(head_rgb, wrist_rgb):
                logger.info("Camera preview requested shutdown.")
                break
            state = arm.read_state()

            action = policy.predict(head_rgb, wrist_rgb, state, args.prompt)
            fetch = policy.pop_chunk_fetch()
            if fetch is not None:
                if recorder is not None:
                    recorder.note_interact()
                pacer.note_chunk_fetch(fetch[0], fetch[1])
                if args.chunk_xyz_shift:
                    # main.py translates the absolute chunk after blocking infer.
                    state_after = arm.read_state()
                    xyz_shift = (state_after[:3] - state[:3]).astype(np.float32)
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[:3] += xyz_shift
                    for pending in policy._pending:
                        pending[:3] += xyz_shift
                    state = state_after
                    if args.log_actions:
                        logger.info(
                            "chunk xyz_shift=%s after server fetch",
                            np.array2string(xyz_shift, precision=4),
                        )
                if args.boundary_blend and previous_action is not None:
                    # Full chunk lives in policy._pending; action is pending[0].
                    blended = boundary_blend_chunk(
                        [np.asarray(a, dtype=np.float32).copy() for a in policy._pending],
                        previous_action,
                        blend_steps=args.boundary_blend_steps,
                    )
                    policy._pending = blended
                    action = blended[0].copy()
                    if args.log_actions:
                        logger.info(
                            "boundary-blend applied to first %d/%d chunk actions",
                            min(args.boundary_blend_steps, len(blended)),
                            len(blended),
                        )
            elif recorder is not None and not args.return_action_chunk:
                # Step mode: every server predict is one interaction.
                recorder.note_interact()
            if recorder is not None:
                recorder.maybe_write(head_rgb, wrist_rgb)
            action = gripper_hold.apply(action)
            if args.log_actions and step % 10 == 0:
                logger.info(
                    "step=%d latency_budget state_xyz=%s action_xyz=%s grip=%.3f",
                    step,
                    np.array2string(state[:3], precision=3),
                    np.array2string(action[:3], precision=3),
                    float(action[-1]),
                )
            if previous_action is None:
                previous_action = np.asarray(state, dtype=np.float32).copy()
            step_period = pacer.policy_step_period
            if args.interpolation_substeps > 1:
                send_action_interpolated(
                    arm,
                    previous_action,
                    action,
                    substeps=args.interpolation_substeps,
                    step_period=step_period,
                )
                pacer.consume_policy_step()
            else:
                arm.send_action(action)
                pacer.wait_after_step(start)
            previous_action = np.asarray(action, dtype=np.float32).copy()


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Piper OpenWAM EEF10 control stopped by user.")


if __name__ == "__main__":
    main()
