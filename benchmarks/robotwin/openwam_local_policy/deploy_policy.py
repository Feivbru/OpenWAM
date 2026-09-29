"""In-process OpenWAM policy for FastWAM-style RoboTwin eval (no WebSocket)."""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BENCHMARKS_ROBOTWIN = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROBOTWIN) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROBOTWIN.parent))
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmarks.robotwin import policy_runtime  # noqa: E402

logger = logging.getLogger("openwam_local_policy")

_DEFAULT_WAN_PATH = (
    "/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B"
)


class LocalOpenWAMPolicy:
    """RoboTwin policy adapter that loads OpenWAM weights in-process."""

    def __init__(
        self,
        ckpt_dir: str,
        *,
        deploy_config: Optional[str] = None,
        device: str = "cuda",
        action_type: str = "ee",
        send_state: bool = True,
        state_dim: Optional[int] = 20,
        skip_text_encoder: bool = True,
        wan_path: Optional[str] = None,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        from openwam.deploy import JointInferenceEngine
        from openwam.deploy.denoise_schedule import make_schedule
        from openwam.deploy.executors import resolve_execution_config
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.obs_preprocess import ObsPreprocessor
        from openwam.deploy.server import (
            PolicyServer,
            _load_deploy_yaml,
            _normalize_compile_enabled_in_cfg,
            _validate_inference_config,
            merge_deploy_cfg,
        )
        from openwam.model.video_backbone.wan.encode import encode_text
        from precompute_robotwin_t5_embeds import _load_text_stack

        if action_type not in ("qpos", "ee"):
            raise ValueError(f"Unsupported action_type={action_type!r}")

        self._action_type = action_type
        self._send_state = send_state
        self._state_dim = state_dim
        self._pending: deque = deque()
        self._task_description = ""
        self._prompt_cache: dict[str, tuple[Any, Any]] = {}
        self._torch = torch
        self._encode_text = encode_text

        ckpt_dir = str(Path(ckpt_dir).expanduser().resolve())
        if not Path(ckpt_dir).is_dir():
            raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

        deploy_path = deploy_config
        if deploy_path is None or str(deploy_path).strip() in ("", "null", "None"):
            deploy_path = str(_PROJECT_ROOT / "configs" / "deploy.yaml")
        cfg = _load_deploy_yaml(deploy_path)
        # Local eval: skip video decode; keep compile/dit_cache from deploy.yaml.
        OmegaConf.update(cfg, "optimization.decode_video", False, merge=False)
        _normalize_compile_enabled_in_cfg(cfg)
        _validate_inference_config(cfg)

        skip_te = bool(skip_text_encoder)
        logger.info(
            "Loading OpenWAM checkpoint from %s (device=%s skip_text_encoder=%s)",
            ckpt_dir,
            device,
            skip_te,
        )
        training_cfg, architecture = load_from_checkpoint_dir(
            ckpt_dir,
            device=device,
            skip_text_encoder=skip_te,
        )
        merged = merge_deploy_cfg(training_cfg, cfg)
        engine = JointInferenceEngine(cfg=merged, architecture=architecture)
        server = PolicyServer(engine=engine, cfg=merged)
        server._init_policy()

        self._server = server
        self._policy = server._policy
        self._obs_preprocessor: ObsPreprocessor = server._obs_preprocessor
        self._engine = engine
        self._arch = architecture
        self._cfg = merged
        self._execution_config = resolve_execution_config(merged)
        horizon = self._execution_config.inference_horizon
        self._inference_horizon = None if horizon is None else int(horizon)

        # Schedule / denoise knobs (same as batched_server generate_batch path).
        inf = merged.inference
        self._denoise_steps = int(inf.denoise_steps)
        self._action_num_frames = int(inf.num_frames)
        self._video_num_frames = int(getattr(inf, "video_num_frames", self._action_num_frames))
        self._height = int(inf.height)
        self._width = int(inf.width)
        ab = getattr(architecture, "action_backbone", None)
        self._shift = float(getattr(ab, "shift_action", None) or 5.0)
        vb = getattr(architecture, "video_backbone", None)
        shift_video = getattr(vb, "shift_video", None) if vb is not None else None
        self._schedule = make_schedule(
            getattr(inf, "denoise_mode", "sync"),
            video_scheduler=architecture.video_scheduler,
            action_scheduler=architecture.action_scheduler,
            num_steps=self._denoise_steps,
            shift=self._shift,
            shift_video=shift_video,
            lead=getattr(inf, "lead_modality", "video"),
            alpha=float(getattr(inf, "variance_shift_alpha", 1.0)),
            offset=float(getattr(inf, "linear_offset", 0.0)),
        )
        opt = getattr(merged, "optimization", None)
        dc = getattr(opt, "dit_cache", None) if opt is not None else None
        self._dit_cache_cfg = None
        if dc is not None and bool(getattr(dc, "enabled", False)):
            self._dit_cache_cfg = {
                "enabled": True,
                "cosine_threshold": float(getattr(dc, "cosine_threshold", 0.99)),
                "max_skips": int(getattr(dc, "max_skips", 3)),
            }

        self._use_external_context = skip_te
        self._tokenizer = None
        self._text_encoder = None
        if skip_te:
            wan = wan_path or _DEFAULT_WAN_PATH
            logger.info("Loading UMT5 text stack on CPU (wan_path=%s)", wan)
            self._tokenizer, self._text_encoder = _load_text_stack(
                wan, ckpt_dir, torch.device("cpu")
            )

        print(
            f"[LocalOpenWAMPolicy] ckpt={ckpt_dir} device={device} "
            f"action_type={action_type} state_dim={state_dim} "
            f"send_state={send_state} inference_horizon={self._inference_horizon} "
            f"skip_text_encoder={skip_te}"
        )

    def reset(self, task_description: str = "") -> None:
        self._task_description = task_description
        self._pending.clear()
        self._server.reset()

    def should_request_observation(self) -> bool:
        return not self._pending

    def _encode_prompt(self, prompt: str):
        hit = self._prompt_cache.get(prompt)
        if hit is not None:
            return hit
        with self._torch.no_grad():
            context, seq_lens = self._encode_text(
                [prompt],
                tokenizer=self._tokenizer,
                text_encoder=self._text_encoder,
                device=self._torch.device("cpu"),
            )
        # [1, L, D] -> [L, D] CPU (same as encoder_server).
        context_cpu = context[0].detach().to("cpu").contiguous()
        seq_cpu = seq_lens[0].detach().to("cpu")
        self._prompt_cache[prompt] = (context_cpu, seq_cpu)
        return context_cpu, seq_cpu

    @staticmethod
    def _as_pil(frame):
        if frame is None:
            return None
        from PIL import Image

        if isinstance(frame, Image.Image):
            return frame if frame.mode == "RGB" else frame.convert("RGB")
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def _generate_chunk(self, cams: dict, prompt: str, state_list: Optional[list]) -> None:
        obs_payload = {
            "images": {
                "head_camera": self._as_pil(cams["head"]),
                "left_wrist_camera": self._as_pil(cams.get("left")),
                "right_wrist_camera": self._as_pil(cams.get("right")),
            },
            "prompt": prompt,
            "state": state_list,
        }
        processed = self._obs_preprocessor.preprocess(obs_payload)

        if self._use_external_context:
            context, seq_lens = self._encode_prompt(prompt)
            proprio = None
            if state_list is not None:
                proprio = self._arch.normalize_deploy_proprio(state_list)
            sample = {
                "first_frame_image": [processed["image"]],
                "context": context,
                "seq_lens": seq_lens,
                "proprio": proprio,
                "prompt": prompt,
                "seed": 42,
            }
            result = self._arch.generate_batch(
                [sample],
                schedule=self._schedule,
                action_num_frames=self._action_num_frames,
                video_num_frames=self._video_num_frames,
                height=self._height,
                width=self._width,
                denoise_steps=self._denoise_steps,
                shift=self._shift,
                dit_cache_cfg=self._dit_cache_cfg,
            )
            actions = result["actions"]
            if hasattr(actions, "shape") and getattr(actions, "ndim", 0) == 3:
                actions = actions[0]
        else:
            conditions = self._policy._build_conditions(processed)
            result = self._engine.generate(conditions)
            actions = result["actions"]
            if hasattr(actions, "cpu"):
                actions = actions.detach().cpu().numpy()

        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[None, :]
        horizon = self._inference_horizon if self._inference_horizon is not None else len(actions)
        if horizon > len(actions):
            raise ValueError(
                f"inference_horizon ({horizon}) > action chunk length ({len(actions)})"
            )
        self._pending.clear()
        for row in actions[:horizon]:
            self._pending.append(np.asarray(row, dtype=np.float32))

    def step(self, example: Optional[dict]) -> np.ndarray:
        if self._pending:
            return self._pending.popleft()
        if not example:
            raise ValueError(
                "[LocalOpenWAMPolicy] Observation is required when the action chunk is empty."
            )

        cams = example["cams"]
        instruction = str(example.get("lang", self._task_description))
        if instruction and instruction != self._task_description:
            self.reset(instruction)

        state_list: Optional[list] = None
        if self._send_state:
            state_arr = example.get("state", None)
            if state_arr is None:
                raise ValueError(
                    "[LocalOpenWAMPolicy] send_state=True requires example['state']."
                )
            state_np = np.asarray(state_arr, dtype=np.float32).reshape(-1)
            if self._state_dim is not None and state_np.size != self._state_dim:
                raise ValueError(
                    f"[LocalOpenWAMPolicy] Extracted state_dim={state_np.size}, "
                    f"expected {self._state_dim}."
                )
            state_list = [float(v) for v in state_np]

        prompt = policy_runtime.build_inference_prompt(instruction)
        self._generate_chunk(cams, prompt, state_list)
        if not self._pending:
            raise RuntimeError("[LocalOpenWAMPolicy] Empty action chunk from generate.")
        return self._pending.popleft()


def get_model(usr_args: dict) -> LocalOpenWAMPolicy:
    ckpt_dir = usr_args.get("ckpt_dir") or usr_args.get("ckpt_setting")
    if ckpt_dir is None or str(ckpt_dir).strip() == "":
        raise ValueError("ckpt_dir (or ckpt_setting as path) is required for openwam_local_policy")
    ckpt_path = Path(str(ckpt_dir))
    if not ckpt_path.is_dir():
        raise FileNotFoundError(
            f"openwam_local_policy expects ckpt_dir to be a checkpoint directory, got: {ckpt_dir}"
        )
    return LocalOpenWAMPolicy(
        ckpt_dir=str(ckpt_path),
        deploy_config=usr_args.get("deploy_config"),
        device=str(usr_args.get("device", "cuda")),
        action_type=str(usr_args.get("action_type", "ee")),
        send_state=policy_runtime.parse_bool(usr_args.get("send_state", True), default=True),
        state_dim=policy_runtime.parse_optional_int(usr_args.get("state_dim", 20), "state_dim"),
        skip_text_encoder=policy_runtime.parse_bool(
            usr_args.get("skip_text_encoder", True), default=True
        ),
        wan_path=usr_args.get("wan_path"),
    )


def reset_model(model: LocalOpenWAMPolicy) -> None:
    model.reset(task_description="")


def eval(TASK_ENV, model: LocalOpenWAMPolicy, observation: Optional[dict]) -> None:
    policy_runtime.apply_step_lim_override(TASK_ENV)

    if observation is None:
        action = model.step(None)
    else:
        instruction = TASK_ENV.get_instruction()
        example = {
            "cams": policy_runtime.build_cams_from_observation(observation),
            "lang": str(instruction),
            "state": (
                policy_runtime.extract_proprio(model._action_type, observation)
                if model._send_state
                else None
            ),
        }
        action = model.step(example)

    action = policy_runtime.maybe_convert_ee_action(action, model._action_type)
    TASK_ENV.take_action(action, action_type=model._action_type)
