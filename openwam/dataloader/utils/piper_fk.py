"""AgileX Piper joint7 → EEF10 forward kinematics (shared by preprocess + deploy).

EEF10 layout matches the OpenWAM single-arm contract::

    [xyz_m(3), rot6d(6), gripper_open_scale(1)] with -1=closed, +1=open.

Requires ``piper_sdk`` at runtime.
"""

from __future__ import annotations

import numpy as np

EEF10_DIM = 10
JOINT7_DIM = 7


def _link_T(fk, alpha: float, a: float, theta: float, d: float) -> list[float]:
    return fk._C_PiperForwardKinematics__LinkTransformtion(alpha, a, theta, d)


def _mat_mul(fk, a: list[float], b: list[float]) -> list[float]:
    return fk._C_PiperForwardKinematics__MatMultiply(a, b, 4, 4, 4)


def make_piper_fk(*, dh_is_offset: int = 1):
    """Construct ``C_PiperForwardKinematics`` (lazy import of ``piper_sdk``)."""
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    return C_PiperForwardKinematics(dh_is_offset=int(dh_is_offset))


def fk_pose_matrix(fk, joints6: np.ndarray) -> np.ndarray:
    """Return 4x4 homogeneous pose (xyz in mm) for a single 6-joint configuration."""
    q = np.asarray(joints6, dtype=np.float64).reshape(6)
    transforms = []
    for i in range(6):
        transforms.append(_link_T(fk, fk._alpha[i], fk._a[i], float(q[i] + fk._theta[i]), fk._d[i]))
    t = transforms[0]
    for i in range(1, 6):
        t = _mat_mul(fk, t, transforms[i])
    return np.asarray(t, dtype=np.float64).reshape(4, 4)


def joints7_to_eef10(fk, joint7: np.ndarray, gripper_max_m: float) -> np.ndarray:
    """Convert ``(T, 7)`` or ``(7,)`` joint+gripper to ``(T, 10)`` / ``(10,)`` EEF10."""
    raw = np.asarray(joint7, dtype=np.float64)
    squeeze = False
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
        squeeze = True
    if raw.ndim != 2 or raw.shape[1] != JOINT7_DIM:
        raise ValueError(f"expected (..., {JOINT7_DIM}), got {np.asarray(joint7).shape}")
    if gripper_max_m <= 0:
        raise ValueError(f"gripper_max_m must be > 0, got {gripper_max_m}")

    out = np.zeros((raw.shape[0], EEF10_DIM), dtype=np.float32)
    for t in range(raw.shape[0]):
        T = fk_pose_matrix(fk, raw[t, :6])
        out[t, 0:3] = (T[:3, 3] / 1000.0).astype(np.float32)  # mm -> m
        out[t, 3:9] = np.concatenate([T[:3, 0], T[:3, 1]]).astype(np.float32)
        g = float(np.clip(raw[t, 6], 0.0, gripper_max_m))
        out[t, 9] = np.float32(2.0 * (g / gripper_max_m) - 1.0)
    if not np.isfinite(out).all():
        raise ValueError("non-finite values in EEF10 conversion")
    return out[0] if squeeze else out


__all__ = [
    "EEF10_DIM",
    "JOINT7_DIM",
    "fk_pose_matrix",
    "joints7_to_eef10",
    "make_piper_fk",
]
