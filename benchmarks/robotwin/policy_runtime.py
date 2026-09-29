"""Shared RoboTwin <-> OpenWAM policy helpers (WS client + in-process local)."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import yaml

from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from benchmarks.utils import action_conversion

_STEP_LIMITS_PATH = os.path.join(os.path.dirname(__file__), "step_limits.yml")
_MISSING_TASK_NAME_WARNED = False
_LOGGED_OVERRIDES: set = set()


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", "none", "null", ""):
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def parse_optional_int(value, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer or null, got {value!r}")
    return parsed


def load_step_lim_overrides(path: str = _STEP_LIMITS_PATH) -> Dict[str, int]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        print(f"[policy_runtime] Failed to load step_lim overrides from {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[policy_runtime] {path} must be a task_name->int mapping; ignoring.")
        return {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        if isinstance(v, bool) or not isinstance(v, int):
            print(
                f"[policy_runtime] Skipping step_lim override {k!r}={v!r}: "
                f"value must be a plain int (got {type(v).__name__})"
            )
            continue
        out[str(k)] = v
    return out


_STEP_LIM_OVERRIDES: Dict[str, int] = load_step_lim_overrides()


def apply_step_lim_override(task_env) -> None:
    if not _STEP_LIM_OVERRIDES or getattr(task_env, "take_action_cnt", -1) != 0:
        return
    task_name = getattr(task_env, "task_name", None)
    if not task_name:
        global _MISSING_TASK_NAME_WARNED
        if not _MISSING_TASK_NAME_WARNED:
            _MISSING_TASK_NAME_WARNED = True
            print(
                "[policy_runtime] step_lim overrides loaded but TASK_ENV.task_name "
                f"is missing/empty ({task_name!r}); overrides will not be applied."
            )
        return
    override = _STEP_LIM_OVERRIDES.get(task_name)
    if override is None or getattr(task_env, "step_lim", None) == override:
        return
    prev = getattr(task_env, "step_lim", None)
    task_env.step_lim = override
    key = (task_name, override)
    if key not in _LOGGED_OVERRIDES:
        _LOGGED_OVERRIDES.add(key)
        print(f"[policy_runtime] step_lim override: {task_name} {prev} -> {override}")


def extract_eef_proprio(observation: dict) -> np.ndarray:
    endpose = observation.get("endpose")
    if not isinstance(endpose, dict):
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(
            "action_type='ee' requires RoboTwin endpose proprio matching training action_mode='eef'. "
            "Expected observation['endpose'] with left_endpose, right_endpose, left_gripper, right_gripper. "
            f"Available top-level observation keys: {available}"
        )
    return action_conversion.robotwin_endpose_to_eef20d(
        endpose["left_endpose"],
        endpose["right_endpose"],
        endpose["left_gripper"],
        endpose["right_gripper"],
    )


def extract_proprio(action_type: str, observation: dict) -> np.ndarray:
    if action_type == "ee":
        return extract_eef_proprio(observation)
    try:
        return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    except KeyError as exc:
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(
            "action_type='qpos' requires RoboTwin joint proprio matching training action_mode='joint'. "
            "Expected observation['joint_action']['vector']. "
            f"Available top-level observation keys: {available}"
        ) from exc


def build_cams_from_observation(observation: dict) -> dict:
    obs = observation["observation"]
    return {
        "head": obs["head_camera"]["rgb"],
        "left": obs.get("left_camera", {}).get("rgb"),
        "right": obs.get("right_camera", {}).get("rgb"),
    }


def maybe_convert_ee_action(action: np.ndarray, action_type: str) -> np.ndarray:
    if action_type == "ee" and len(action) == 20:
        return action_conversion.eef20d_to_ee16d(action)
    return action


def build_inference_prompt(instruction: str) -> str:
    return format_prompt_for_inference(instruction)
