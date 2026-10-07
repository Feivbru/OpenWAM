"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — blocking
  buffer-and-replan with a bounded execution horizon.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.
"""

import logging

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)
from openwam.deploy.executors.vrtc_executor import build_vrtc_executor
from openwam.vrtc import resolve_vrtc_config

logger = logging.getLogger(__name__)


class WAMPolicy:
    """Unified policy facade over the sync / async / VRTC execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Root config, retained for policy-level consumers.
        execution_config: ExecutionConfig-like. ``inference_horizon`` applies
            to both modes; ``inference_delay_steps`` applies only to async.
            Ignored when ``vrtc.enabled`` (VRTC uses its own cube-pool sync executor).
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None):
        self.cfg = cfg
        self.engine = engine

        vrtc_executor = build_vrtc_executor(engine, cfg)
        if vrtc_executor is not None:
            vrtc = resolve_vrtc_config(cfg)
            if execution_config is not None:
                exec_cfg = normalize_execution_config(execution_config)
                if exec_cfg.enabled:
                    raise ValueError(
                        "vrtc.enabled=true is incompatible with inference_mode=async; "
                        "VRTC currently ships a sync cube-pool executor only"
                    )
            self._execution_config = normalize_execution_config({"mode": "sync"})
            self._async = False
            self._vrtc = vrtc
            self._executor = vrtc_executor
            logger.info(
                "WAMPolicy: VRTC cube-pool executor "
                "(fu_frames=%d, warmup_cubes=%d, predict_cubes=%d, video_stride=%d, "
                "replan_cubes=%d, merge_mode=%s)",
                vrtc.fu_frames,
                vrtc.pool_warmup_cubes,
                vrtc.predict_cubes,
                vrtc.video_stride,
                vrtc.replan_cubes,
                vrtc.merge_mode,
            )
        else:
            self._vrtc = resolve_vrtc_config(cfg)
            self._execution_config = normalize_execution_config(execution_config)
            self._async = self._execution_config.enabled
            if self._async:
                self._executor = AsyncInferenceExecutor(
                    engine=engine,
                    inference_horizon=self._execution_config.inference_horizon,
                    inference_delay_steps=self._execution_config.inference_delay_steps,
                )
            else:
                self._executor = SyncInferenceExecutor(
                    engine=engine,
                    inference_horizon=self._execution_config.inference_horizon,
                )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given (already preprocessed) observation.

        The final legality projection for two-point command dims
        (``architecture.binary_command_dims``, from the CKPT's dataloader.binary_action_dims) runs
        HERE — after all executor arithmetic. The normalizer already emits exact ±1 for those dims,
        and this final boundary also protects engines or checkpoints that emit
        values between the two legal commands. Threshold 0.5 preserves the
        downstream command contract for the WS server and direct consumers.
        """
        action = self._executor.predict_action(self._build_conditions(obs))
        return self._project_binary_dims(action)

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        """Return an executable action chunk ``(T, D)`` and clear executor buffers.

        Used by JSON PolicyServer when ``server.return_action_chunk=true`` so the
        client can open-loop locally. Applies the same binary-dim projection as
        :meth:`predict_action`.
        """
        if not hasattr(self._executor, "predict_action_chunk"):
            raise RuntimeError(
                f"Executor {type(self._executor).__name__} does not support predict_action_chunk"
            )
        actions = self._executor.predict_action_chunk(self._build_conditions(obs))
        return self._project_binary_dims(actions)

    def _project_binary_dims(self, action: np.ndarray) -> np.ndarray:
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if not dims:
            return np.asarray(action)
        action = np.array(action)
        for d in dims:
            if d >= action.shape[-1]:
                raise ValueError(
                    f"binary_command_dims includes {d} but the action is {action.shape[-1]}-D; "
                    "the ckpt config and the served action width disagree."
                )
            action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def reset(self):
        """Clear executor state between episodes."""
        self._executor.reset()

    def set_replan_cubes(self, replan_cubes: int):
        """Override VRTC ``replan_cubes`` at runtime (client ping handshake)."""
        if self._vrtc is None or not self._vrtc.enabled:
            raise ValueError("replan_cubes override requires vrtc.enabled=true")
        if not hasattr(self._executor, "set_replan_cubes"):
            raise ValueError(
                f"Executor {type(self._executor).__name__} does not support set_replan_cubes"
            )
        self._vrtc = self._executor.set_replan_cubes(int(replan_cubes))
        return self._vrtc

    def set_merge_mode(self, merge_mode: str):
        """Override VRTC wait-pool ``merge_mode`` at runtime (client ping handshake)."""
        if self._vrtc is None or not self._vrtc.enabled:
            raise ValueError("merge_mode override requires vrtc.enabled=true")
        if not hasattr(self._executor, "set_merge_mode"):
            raise ValueError(
                f"Executor {type(self._executor).__name__} does not support set_merge_mode"
            )
        self._vrtc = self._executor.set_merge_mode(str(merge_mode))
        return self._vrtc

    def vrtc_wire_info(self) -> dict | None:
        """VRTC fields advertised on PONG, or ``None`` when VRTC is disabled."""
        vrtc = self._vrtc
        if vrtc is None or not vrtc.enabled:
            return None
        return {
            "enabled": True,
            "video_stride": int(vrtc.video_stride),
            "cube_action_len": int(vrtc.video_stride),
            "replan_cubes": int(vrtc.replan_cubes),
            "merge_mode": str(vrtc.merge_mode),
            "predict_cubes": int(vrtc.predict_cubes),
            "pool_warmup_cubes": int(vrtc.pool_warmup_cubes),
        }

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        self._executor.shutdown()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``first_frame_image``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        img = obs.get("image")
        if img is not None:
            # Single first frame — pipeline expects list[PIL.Image]
            conditions["first_frame_image"] = [img]
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
