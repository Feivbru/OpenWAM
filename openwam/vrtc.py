"""VRTC (video real-time conditioning) helpers.

Keeps a configurable clear future prefix on video/action during train and
inference, so the model can condition on recently observed moving objects.

Default: disabled. When enabled, ``fu_frames`` must be a positive multiple of
the Wan temporal compression factor (4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

WAN_TEMPORAL_FACTOR = 4

# How newly predicted wait cubes combine with cubes already in the wait pool.
VRTC_MERGE_MODES = ("replace", "average", "blend")


@dataclass(frozen=True)
class VrtcConfig:
    """Resolved VRTC settings derived from Hydra / ckpt config."""

    enabled: bool = False
    fu_frames: int = 4
    condition_frames: int = 1
    num_frames: int = 33
    video_stride: int = 4
    temporal_factor: int = WAN_TEMPORAL_FACTOR
    # Prefetch when ``len(wait_pool) <= replan_cubes`` (0 = sync: only when empty).
    replan_cubes: int = 2
    # Wait-pool merge on INFER_MERGE: replace | average | blend (within-cube ramp).
    merge_mode: str = "replace"

    def __post_init__(self) -> None:
        mode = str(self.merge_mode).strip().lower()
        object.__setattr__(self, "merge_mode", mode)
        if mode not in VRTC_MERGE_MODES:
            raise ValueError(
                f"vrtc.merge_mode must be one of {VRTC_MERGE_MODES}, got {self.merge_mode!r}"
            )
        if not self.enabled:
            return
        if self.fu_frames <= 0:
            raise ValueError(f"vrtc.fu_frames must be > 0, got {self.fu_frames}")
        if self.fu_frames % self.temporal_factor != 0:
            raise ValueError(
                f"vrtc.fu_frames ({self.fu_frames}) must be a multiple of "
                f"temporal_factor ({self.temporal_factor})"
            )
        if self.condition_frames != 1:
            raise ValueError(
                f"vrtc currently supports condition_frames=1 only, got {self.condition_frames}"
            )
        if self.video_stride <= 0:
            raise ValueError(f"video_stride must be > 0, got {self.video_stride}")
        if self.num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {self.num_frames}")
        if self.clear_action_steps >= self.num_action_steps:
            raise ValueError(
                f"vrtc clear_action_steps ({self.clear_action_steps}) must be < "
                f"num_action_steps ({self.num_action_steps})"
            )
        if self.predict_cubes <= 0:
            raise ValueError(
                f"vrtc predict_cubes must be > 0; got num_frames={self.num_frames}, "
                f"video_stride={self.video_stride}, fu_frames={self.fu_frames}"
            )
        if self.replan_cubes < 0:
            raise ValueError(f"vrtc.replan_cubes must be >= 0, got {self.replan_cubes}")
        if self.replan_cubes > self.predict_cubes:
            raise ValueError(
                f"vrtc.replan_cubes ({self.replan_cubes}) must be <= predict_cubes "
                f"({self.predict_cubes})"
            )

    @property
    def num_action_steps(self) -> int:
        return self.num_frames - 1

    @property
    def video_num_frames(self) -> int:
        return (self.num_frames - 1) // self.video_stride + 1

    @property
    def clear_video_frames(self) -> int:
        """Pixel-level clear frames: condition + clear future."""
        return self.condition_frames + self.fu_frames

    @property
    def clear_action_steps(self) -> int:
        return self.fu_frames * self.video_stride

    @property
    def noisy_action_steps(self) -> int:
        return self.num_action_steps - self.clear_action_steps

    @property
    def clear_latent_frames(self) -> int:
        """Wan causal VAE: frame0 → 1 latent, then every ``temporal_factor`` pixels → 1 latent."""
        return self.condition_frames + self.fu_frames // self.temporal_factor

    @property
    def predict_cubes(self) -> int:
        """Cubes newly predicted each inference (also = fu_frames under default 33/4/4)."""
        return self.num_action_steps // self.video_stride - self.fu_frames

    @property
    def pool_warmup_cubes(self) -> int:
        return self.condition_frames + self.fu_frames


def _select(cfg: Any, path: str, default=None):
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.select(cfg, path, default=default)
    except ImportError:
        pass
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def resolve_vrtc_config(cfg: Any = None, *, overrides: Optional[dict] = None) -> VrtcConfig:
    """Build :class:`VrtcConfig` from a root Hydra/ckpt config.

    Looks at ``vrtc.*`` plus ``dataloader.num_frames`` / ``dataloader.video_stride``
    (falling back to ``inference.*`` for deploy-only configs).
    """
    overrides = overrides or {}
    enabled = overrides.get("enabled", _select(cfg, "vrtc.enabled", False))
    fu_frames = overrides.get("fu_frames", _select(cfg, "vrtc.fu_frames", 4))
    condition_frames = overrides.get(
        "condition_frames", _select(cfg, "vrtc.condition_frames", 1)
    )

    num_frames = overrides.get("num_frames", None)
    if num_frames is None:
        num_frames = _select(cfg, "dataloader.num_frames", None)
    if num_frames is None:
        num_frames = _select(cfg, "inference.num_frames", 33)

    video_stride = overrides.get("video_stride", None)
    if video_stride is None:
        video_stride = _select(cfg, "dataloader.video_stride", None)
    if video_stride is None:
        video_stride = _select(cfg, "inference.video_stride", 4)

    replan_cubes = overrides.get("replan_cubes", _select(cfg, "vrtc.replan_cubes", 2))
    merge_mode = overrides.get("merge_mode", _select(cfg, "vrtc.merge_mode", "replace"))

    return VrtcConfig(
        enabled=bool(enabled),
        fu_frames=int(fu_frames),
        condition_frames=int(condition_frames),
        num_frames=int(num_frames),
        video_stride=int(video_stride or 1),
        replan_cubes=int(replan_cubes),
        merge_mode=str(merge_mode),
    )
