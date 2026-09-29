"""OpenWAM single-task RoboTwin evaluation (in-process policy, FastWAM-style)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_NAME = "openwam_local_policy"
POLICY_SOURCE_DIR = PROJECT_ROOT / "benchmarks" / "robotwin" / "openwam_local_policy"
WRAPPER = PROJECT_ROOT / "benchmarks" / "robotwin" / "eval_policy_wrapper.py"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}_{parts[idx + 2]}"
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.name if ckpt_path.is_dir() else ckpt_path.stem


def _append_override(overrides: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    overrides.append(f"{key}={text}")


def _ensure_policy_symlink(*, robotwin_root: Path, policy_source_dir: Path) -> None:
    policy_dir = robotwin_root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    link = policy_dir / POLICY_NAME
    target = policy_source_dir.resolve()
    if link.is_symlink() or link.exists():
        if link.resolve() == target:
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise FileExistsError(f"Refusing to replace non-symlink policy path: {link}")
    link.symlink_to(target, target_is_directory=True)
    print(f"[eval_robotwin_single] symlink {link} -> {target}", flush=True)


def _resolve_cuda_visible_devices(gpu_id: int, *, parent_visible: str | None) -> str:
    if parent_visible is None or parent_visible.strip() == "":
        return str(gpu_id)
    devices = [d.strip() for d in parent_visible.split(",") if d.strip() != ""]
    if not devices:
        return str(gpu_id)
    if gpu_id < 0 or gpu_id >= len(devices):
        raise ValueError(
            f"gpu_id={gpu_id} out of range for CUDA_VISIBLE_DEVICES={parent_visible!r} "
            f"(len={len(devices)})"
        )
    return devices[gpu_id]


def _describe_subprocess_return_code(return_code: int) -> str:
    if return_code >= 0:
        return f"exit code {return_code}"
    try:
        sig = signal.Signals(-return_code)
        return f"killed by {sig.name} ({-return_code})"
    except Exception:
        return f"negative return code {return_code}"


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="sim_robotwin_local")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    if not WRAPPER.is_file():
        raise FileNotFoundError(f"eval_policy_wrapper not found: {WRAPPER}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    if not POLICY_SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {POLICY_SOURCE_DIR}")
    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=POLICY_SOURCE_DIR)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin_local" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = run_output_dir / str(cfg.EVALUATION.task_name)

    deploy_config = cfg.EVALUATION.get("deploy_config", None)
    if deploy_config is not None and str(deploy_config).strip() not in ("", "null", "None"):
        deploy_config = str(_resolve_path(str(deploy_config), base=PROJECT_ROOT))
    else:
        deploy_config = str(PROJECT_ROOT / "configs" / "deploy.yaml")

    overrides: list[str] = []

    def _add(key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        overrides.extend([f"--{key}", text])

    _add("task_name", cfg.EVALUATION.task_name)
    _add("task_config", cfg.EVALUATION.task_config)
    # Short label for logs; real weights path is ckpt_dir.
    _add("ckpt_setting", ckpt_tag)
    _add("ckpt_dir", str(ckpt_path))
    _add("seed", cfg.get("seed", 0))
    _add("policy_name", POLICY_NAME)
    _add("instruction_type", cfg.EVALUATION.instruction_type)
    _add("eval_num_episodes", cfg.EVALUATION.eval_num_episodes)
    _add("eval_output_dir", str(robotwin_eval_base))
    _add("device", cfg.EVALUATION.device)
    _add("deploy_config", deploy_config)
    _add("action_type", cfg.EVALUATION.get("action_type", "ee"))
    _add("send_state", cfg.EVALUATION.get("send_state", True))
    _add("state_dim", cfg.EVALUATION.get("state_dim", 20))
    _add("skip_text_encoder", cfg.EVALUATION.get("skip_text_encoder", True))
    _add(
        "skip_get_obs_within_replan",
        cfg.EVALUATION.get("skip_get_obs_within_replan", True),
    )
    _add("eval_video_log", cfg.EVALUATION.get("eval_video_log", False))

    policy_yml = POLICY_SOURCE_DIR / "deploy_policy.yml"
    cmd = [
        sys.executable,
        "-u",
        str(WRAPPER),
        "--config",
        str(policy_yml),
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    gpu_id = int(cfg.gpu_id)
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = _resolve_cuda_visible_devices(
        gpu_id, parent_visible=parent_visible
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["ROBOTWIN_PATH"] = str(robotwin_root)
    # In-process: same openwam python (must have sapien + openwam).
    env["ROBOTWIN_PYTHON"] = sys.executable
    # Make openwam_local_policy + benchmarks importable.
    py_paths = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "benchmarks" / "robotwin"),
        str(robotwin_root),
        str(robotwin_root / "policy"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for debug_key in ("ROBOTWIN_DISABLE_RT", "ROBOTWIN_DEBUG", "ROBOTWIN_TEST_NUM", "ROBOTWIN_EVAL_SKIP_FRONT_CAMERA"):
        if debug_key in os.environ:
            env[debug_key] = os.environ[debug_key]

    print(
        f"[eval_robotwin_single] gpu_id={gpu_id} "
        f"parent_CUDA_VISIBLE_DEVICES={parent_visible!r} -> "
        f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}",
        flush=True,
    )
    print(f"[eval_robotwin_single] cmd={' '.join(cmd)}", flush=True)

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"RoboTwin evaluation failed: {_describe_subprocess_return_code(return_code)}. "
            f"Log: {log_file}"
        )

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
