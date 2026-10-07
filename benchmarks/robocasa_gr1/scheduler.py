#!/usr/bin/env python3
"""Batched multi-GPU RoboCasa GR1 evaluation scheduler.

Topology mirrors LIBERO-plus / RoboTwin:
  * 1 encoder_server + 1 batched_server per infer GPU
  * (#sim GPUs) × n_sims concurrent single_eval clients (robocasa-gr1 conda)
  * Shared task queue over the official 24 env ids
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from summarize import summarize
from task_list import OFFICIAL_ENV_IDS, env_short_name, smoke_env_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_DIR = REPO_ROOT / "benchmarks" / "robotwin"
DEFAULT_POLICY_CONFIG = SCRIPT_DIR / "policy_config.yml"
DEFAULT_WAN_PATH = Path(
    "/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B"
)
DEFAULT_SERVER_PYTHON = Path("/data/anaconda3/envs/openwam_re/bin/python")
DEFAULT_CLIENT_PYTHON = Path("/data/anaconda3/envs/robocasa-gr1/bin/python")
DEFAULT_ROBOCASA_PATH = REPO_ROOT / "third_party" / "robocasa-gr1-tabletop-tasks"


@dataclass(frozen=True)
class TaskJob:
    env_id: str


@dataclass(frozen=True)
class ReplicaSlot:
    gpu: int
    gpu_slot: int
    replica: int
    port: int
    render_gpu: int | None = None


@dataclass(frozen=True)
class QueuedTask:
    job: TaskJob
    attempt: int = 1


@dataclass
class ServerProcess:
    gpu: int
    replica: int
    port: int
    process: subprocess.Popen
    log_handle: object


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: list[subprocess.Popen] = []

    def add(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.append(process)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(p.poll() is None for p in processes):
            time.sleep(0.2)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def _csv_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return items


def _parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item) for item in _csv_items(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid GPU list: {value!r}") from exc
    if any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("GPU ids must be non-negative")
    if len(gpus) != len(set(gpus)):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return gpus


def _batched_infer_port(base_port: int, gpu_slot: int) -> int:
    return base_port + 10 * gpu_slot


def _partition_sim_gpus(infer_gpus: list[int], sim_gpus: list[int]) -> list[list[int]]:
    n_infer = len(infer_gpus)
    n_sim = len(sim_gpus)
    if n_infer < 1 or n_sim < 1:
        raise ValueError("batched mode needs at least one infer GPU and one sim GPU")
    base = n_sim // n_infer
    rem = n_sim % n_infer
    owned: list[list[int]] = []
    cursor = 0
    for i in range(n_infer):
        cnt = base + (1 if i < rem else 0)
        if cnt < 1:
            raise ValueError(
                f"too many infer GPUs ({n_infer}) for {n_sim} sim GPU(s); each infer needs >=1 sim"
            )
        owned.append(list(sim_gpus[cursor : cursor + cnt]))
        cursor += cnt
    return owned


def _build_batched_slots(
    infer_gpus: list[int],
    sim_gpus: list[int],
    base_port: int,
    n_sims: int,
) -> list[ReplicaSlot]:
    if n_sims < 1:
        raise ValueError("--n-sims must be >= 1")
    owned = _partition_sim_gpus(infer_gpus, sim_gpus)
    slots: list[ReplicaSlot] = []
    for gpu_slot, infer_gpu in enumerate(infer_gpus):
        port = _batched_infer_port(base_port, gpu_slot)
        replica = 0
        for sim_gpu in owned[gpu_slot]:
            for _ in range(n_sims):
                slots.append(
                    ReplicaSlot(
                        gpu=infer_gpu,
                        gpu_slot=gpu_slot,
                        replica=replica,
                        port=port,
                        render_gpu=sim_gpu,
                    )
                )
                replica += 1
    return slots


def _normalize_batched_topology(args: argparse.Namespace) -> None:
    infer = args.infer_gpus if args.infer_gpus is not None else args.gpus
    sim = args.sim_gpus if args.sim_gpus is not None else args.render_gpus
    if sim is None:
        raise ValueError("batched mode requires --sim-gpus or --render-gpus")
    overlap = sorted(set(infer) & set(sim))
    if overlap:
        raise ValueError(f"infer and sim GPUs must be disjoint; overlap={overlap}")
    owned = _partition_sim_gpus(list(infer), list(sim))
    args.gpus = list(infer)
    args.render_gpus = list(sim)
    args.infer_gpus = list(infer)
    args.sim_gpus = list(sim)
    args._batched_owned_sims = owned  # noqa: SLF001
    args.replicas_per_gpu = max(len(chunk) * int(args.n_sims) for chunk in owned)


def _render_device_for_slot(args: argparse.Namespace, slot: ReplicaSlot) -> int:
    if slot.render_gpu is not None:
        return int(slot.render_gpu)
    return args.render_gpus[slot.gpu_slot % len(args.render_gpus)]


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _wait_for_servers(servers: list[ServerProcess], host: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    pending = {s.port: s for s in servers}
    while pending and time.monotonic() < deadline:
        for port, server in list(pending.items()):
            if server.process.poll() is not None:
                raise RuntimeError(
                    f"server on port {port} exited early with code {server.process.returncode}"
                )
            if _port_open(host, port):
                print(f"[ready] port={port} gpu={server.gpu}", flush=True)
                pending.pop(port)
        if pending:
            time.sleep(1.0)
    if pending:
        raise TimeoutError(f"servers not ready within {timeout}s: ports={sorted(pending)}")


def _encoder_command(args: argparse.Namespace, encoder_port: int) -> list[str]:
    return [
        str(args.server_python),
        str(ROBOTWIN_DIR / "encoder_server.py"),
        "--host",
        args.host,
        "--port",
        str(encoder_port),
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--wan-path",
        str(args.wan_path),
        "--device",
        args.encoder_device,
    ]


def _batched_server_command(
    args: argparse.Namespace, *, port: int, encoder_port: int, n_slots: int
) -> list[str]:
    cmd = [
        str(args.server_python),
        str(ROBOTWIN_DIR / "batched_server.py"),
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--ckpt-name",
        args.ckpt_name,
        "--device",
        "cuda:0",
        "--host",
        args.host,
        "--port",
        str(port),
        "--encoder-host",
        args.host,
        "--encoder-port",
        str(encoder_port),
        "--n-slots",
        str(n_slots),
        "--max-batch",
        str(args.max_infer_batch),
        "--denoise-steps",
        str(args.denoise_steps),
    ]
    horizon = getattr(args, "inference_horizon", None)
    if horizon is not None:
        cmd.extend(["--inference-horizon", str(int(horizon))])
    return cmd


def _start_batched_servers(
    args: argparse.Namespace,
    *,
    slots: list[ReplicaSlot],
    output_dir: Path,
    registry: ProcessRegistry,
) -> list[ServerProcess]:
    server_dir = output_dir / "logs" / "servers"
    server_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[ReplicaSlot]] = {}
    for slot in slots:
        groups.setdefault(slot.gpu_slot, []).append(slot)

    servers: list[ServerProcess] = []
    for gpu_slot, group_slots in sorted(groups.items()):
        infer_gpu = group_slots[0].gpu
        port = group_slots[0].port
        encoder_port = port + 1
        n_slots = len(group_slots)

        enc_cmd = _encoder_command(args, encoder_port)
        enc_log = server_dir / f"encoder_gpu{infer_gpu}_port{encoder_port}.log"
        enc_handle = enc_log.open("w", encoding="utf-8")
        enc_handle.write(f"command: {shlex.join(enc_cmd)}\n")
        enc_handle.flush()
        enc_env = dict(os.environ)
        enc_env["PYTHONUNBUFFERED"] = "1"
        if str(args.encoder_device).startswith("cpu"):
            enc_env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            enc_env["CUDA_VISIBLE_DEVICES"] = str(infer_gpu)
        enc_proc = subprocess.Popen(
            enc_cmd,
            cwd=str(REPO_ROOT),
            env=enc_env,
            stdout=enc_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        registry.add(enc_proc)
        print(
            f"[encoder] gpu={infer_gpu} port={encoder_port} device={args.encoder_device} pid={enc_proc.pid}",
            flush=True,
        )

        bat_cmd = _batched_server_command(args, port=port, encoder_port=encoder_port, n_slots=n_slots)
        bat_log = server_dir / f"batched_gpu{infer_gpu}_port{port}.log"
        bat_handle = bat_log.open("w", encoding="utf-8")
        bat_handle.write(f"command: {shlex.join(bat_cmd)}\n")
        bat_handle.flush()
        bat_env = dict(os.environ)
        bat_env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(infer_gpu),
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                ),
            }
        )
        bat_proc = subprocess.Popen(
            bat_cmd,
            cwd=str(REPO_ROOT),
            env=bat_env,
            stdout=bat_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        registry.add(bat_proc)
        servers.append(ServerProcess(infer_gpu, 0, port, bat_proc, bat_handle))
        print(
            f"[batched] gpu={infer_gpu} port={port} encoder_port={encoder_port} "
            f"n_slots={n_slots} max_batch={args.max_infer_batch} pid={bat_proc.pid}",
            flush=True,
        )

    _wait_for_servers(servers, args.host, args.server_start_timeout)
    return servers


def _result_path(output_dir: Path, job: TaskJob) -> Path:
    return output_dir / "results" / env_short_name(job.env_id) / "result.json"


def _result_done(output_dir: Path, job: TaskJob, num_episodes: int) -> bool:
    path = _result_path(output_dir, job)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(payload.get("num_episodes", -1)) == int(num_episodes) and "successes" in payload


def _write_runtime_config(
    template: Path,
    *,
    host: str,
    port: int,
    env_id: str,
    num_episodes: int,
    seed: int,
    dest: Path,
    save_video: bool = False,
    video_dir: Path | None = None,
) -> Path:
    with template.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.update(
        {
            "host": host,
            "port": port,
            "env_id": env_id,
            "num_episodes": num_episodes,
            "seed": seed,
            "save_video": bool(save_video),
        }
    )
    if video_dir is not None:
        cfg["video_dir"] = str(video_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def _client_env(
    base_env: dict,
    *,
    robocasa_path: Path,
    render_gpu: int,
    slot_id: int,
) -> dict:
    env = dict(base_env)
    # Mirror openwam_robocasa_gr1_env.sh essentials without requiring bash source.
    env["ROBOCASA_GR1_PATH"] = str(robocasa_path)
    env["ROBOCASA_GR1_PYTHON"] = env.get("ROBOCASA_GR1_PYTHON", str(DEFAULT_CLIENT_PYTHON))
    env["PYTHONPATH"] = f"{robocasa_path}:{SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(render_gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
    env["ROBOCASA_GR1_SLOT"] = str(int(slot_id))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_client(
    args: argparse.Namespace,
    *,
    slot: ReplicaSlot,
    job: TaskJob,
    output_dir: Path,
    registry: ProcessRegistry,
    stop_event: threading.Event,
    attempt: int = 1,
) -> bool:
    if _result_done(output_dir, job, args.num_episodes):
        print(f"[skip] {job.env_id} already done", flush=True)
        return True

    short = env_short_name(job.env_id)
    log_path = output_dir / "logs" / "clients" / f"{short}_attempt{attempt:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_json = _result_path(output_dir, job)
    result_json.parent.mkdir(parents=True, exist_ok=True)

    runtime_cfg = output_dir / "runtime_configs" / f"{short}_slot{slot.replica}.yml"
    _write_runtime_config(
        args.policy_config,
        host=args.host,
        port=slot.port,
        env_id=job.env_id,
        num_episodes=args.num_episodes,
        seed=args.seed,
        dest=runtime_cfg,
        save_video=bool(getattr(args, "save_video", False)),
        video_dir=(output_dir / "videos" / short) if getattr(args, "save_video", False) else None,
    )

    command = [
        "bash",
        "-lc",
        (
            f"source '{SCRIPT_DIR / 'openwam_robocasa_gr1_env.sh'}' && "
            f"exec '{args.client_python}' '{SCRIPT_DIR / 'single_eval.py'}' "
            f"--config '{runtime_cfg}' "
            f"--host '{args.host}' --port '{slot.port}' "
            f"--env-id '{job.env_id}' "
            f"--num-episodes '{args.num_episodes}' "
            f"--seed '{args.seed}' "
            f"--result-json '{result_json}'"
        ),
    ]
    render_gpu = _render_device_for_slot(args, slot)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"command: {shlex.join(command)}\n")
    log_handle.write(f"infer_gpu={slot.gpu} render_gpu={render_gpu} port={slot.port} slot={slot.replica}\n")
    log_handle.flush()

    env = _client_env(
        os.environ,
        robocasa_path=args.robocasa_path,
        render_gpu=render_gpu,
        slot_id=slot.replica,
    )

    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    registry.add(process)
    print(
        f"[start] {job.env_id} infer={slot.gpu} sim={render_gpu} port={slot.port} "
        f"slot={slot.replica} attempt={attempt}",
        flush=True,
    )
    try:
        while process.poll() is None and not stop_event.wait(1):
            pass
        rc = process.poll()
        if rc is None:
            return False
        if rc != 0:
            print(f"[FAIL] {job.env_id} exit={rc} log={log_path}", flush=True)
            return False
        if not _result_done(output_dir, job, args.num_episodes):
            print(f"[FAIL] {job.env_id} exit=0 but result missing: {result_json}", flush=True)
            return False
        print(f"[done] {job.env_id}", flush=True)
        return True
    finally:
        log_handle.close()


def _dynamic_worker(
    args: argparse.Namespace,
    *,
    slot: ReplicaSlot,
    work_queue: queue.Queue[QueuedTask | None],
    output_dir: Path,
    registry: ProcessRegistry,
    stop_event: threading.Event,
) -> list[TaskJob]:
    failed: list[TaskJob] = []
    stagger = max(0, slot.replica) * float(args.client_start_stagger)
    if stagger > 0 and stop_event.wait(stagger):
        return failed
    while True:
        if stop_event.is_set():
            return failed
        try:
            queued = work_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if queued is None:
                return failed
            job = queued.job
            attempt = queued.attempt
            ok = _run_client(
                args,
                slot=slot,
                job=job,
                output_dir=output_dir,
                registry=registry,
                stop_event=stop_event,
                attempt=attempt,
            )
            if ok:
                continue
            if stop_event.is_set():
                failed.append(job)
                continue
            if attempt >= args.client_max_attempts:
                print(f"[exhausted] {job.env_id} attempts={attempt}", flush=True)
                failed.append(job)
                continue
            if stop_event.wait(args.client_retry_delay):
                failed.append(job)
                continue
            work_queue.put(QueuedTask(job=job, attempt=attempt + 1))
            print(f"[requeue] {job.env_id} next_attempt={attempt + 1}", flush=True)
        finally:
            work_queue.task_done()


def _build_jobs(args: argparse.Namespace) -> list[TaskJob]:
    if args.smoke:
        ids: Iterable[str] = smoke_env_ids()
    elif args.env_ids:
        ids = args.env_ids
    else:
        ids = OFFICIAL_ENV_IDS
    return [TaskJob(env_id=eid) for eid in ids]


def _jobs_sha(jobs: list[TaskJob]) -> str:
    blob = "\n".join(j.env_id for j in jobs).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--ckpt-name", required=True)
    p.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("2"))
    p.add_argument("--render-gpus", type=_parse_gpus, default=None)
    p.add_argument("--infer-gpus", type=_parse_gpus, default=None)
    p.add_argument("--sim-gpus", type=_parse_gpus, default=None)
    p.add_argument("--n-sims", type=int, default=1)
    p.add_argument("--max-infer-batch", type=int, default=2)
    p.add_argument("--encoder-device", default="cuda:0")
    p.add_argument("--wan-path", type=Path, default=DEFAULT_WAN_PATH)
    p.add_argument("--base-port", type=int, default=9300)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--server-python", type=Path, default=DEFAULT_SERVER_PYTHON)
    p.add_argument("--client-python", type=Path, default=DEFAULT_CLIENT_PYTHON)
    p.add_argument("--robocasa-path", type=Path, default=DEFAULT_ROBOCASA_PATH)
    p.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument(
        "--inference-horizon",
        type=int,
        default=None,
        help="Actions returned per denoise on batched_server (default: deploy.yaml / full chunk)",
    )
    p.add_argument("--server-start-timeout", type=int, default=900)
    p.add_argument("--client-start-stagger", type=float, default=0.5)
    p.add_argument("--client-max-attempts", type=int, default=2)
    p.add_argument("--client-retry-delay", type=float, default=5.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--save-video",
        action="store_true",
        help="record ego-view mp4 per episode under output_dir/videos/",
    )
    p.add_argument(
        "--env-ids",
        type=lambda s: _csv_items(s),
        default=None,
        help="comma-separated env ids (default: all 24 official)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.sim_gpus is None and args.render_gpus is None:
        raise SystemExit("batched GR1 eval requires --sim-gpus (or --render-gpus)")
    _normalize_batched_topology(args)

    ckpt_file = args.ckpt_dir / args.ckpt_name
    if not ckpt_file.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_file}")
    if not (args.ckpt_dir / "config.yaml").is_file():
        raise SystemExit(f"missing config.yaml under {args.ckpt_dir}")
    if not args.wan_path.is_dir():
        raise SystemExit(f"WAN_PATH missing: {args.wan_path}")
    if not args.robocasa_path.is_dir():
        raise SystemExit(f"ROBOCASA_GR1_PATH missing: {args.robocasa_path}")
    if not args.client_python.is_file():
        raise SystemExit(f"client python missing: {args.client_python}")
    if not args.server_python.is_file():
        raise SystemExit(f"server python missing: {args.server_python}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args)
    slots = _build_batched_slots(args.gpus, args.render_gpus, args.base_port, args.n_sims)
    pending = [j for j in jobs if not _result_done(output_dir, j, args.num_episodes)]
    active_slots = slots[: min(len(slots), max(len(pending), 1))] if pending else []

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_name": args.ckpt_name,
        "infer_gpus": args.infer_gpus,
        "sim_gpus": args.sim_gpus,
        "n_sims": args.n_sims,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "jobs_sha256": _jobs_sha(jobs),
        "jobs": [j.env_id for j in jobs],
        "pending": [j.env_id for j in pending],
        "slots": [
            {
                "infer_gpu": s.gpu,
                "replica": s.replica,
                "port": s.port,
                "render_gpu": _render_device_for_slot(args, s),
            }
            for s in slots
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"[plan] jobs={len(jobs)} pending={len(pending)} "
        f"infer={args.infer_gpus} sim={args.sim_gpus} n_sims={args.n_sims} "
        f"width={len(slots)} episodes={args.num_episodes}",
        flush=True,
    )
    if not pending:
        print("[plan] nothing pending — summarizing", flush=True)
        summarize(output_dir, write_files=True)
        return 0

    registry = ProcessRegistry()
    stop_event = threading.Event()

    def _on_signal(signum, _frame):
        print(f"[signal] received {signum}, stopping...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    work_queue: queue.Queue[QueuedTask | None] = queue.Queue()
    for job in pending:
        work_queue.put(QueuedTask(job=job))

    try:
        active_ports = {slot.port for slot in active_slots}
        server_slots = [slot for slot in slots if slot.port in active_ports]
        _start_batched_servers(args, slots=server_slots, output_dir=output_dir, registry=registry)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(active_slots))
        futures = [
            executor.submit(
                _dynamic_worker,
                args,
                slot=slot,
                work_queue=work_queue,
                output_dir=output_dir,
                registry=registry,
                stop_event=stop_event,
            )
            for slot in active_slots
        ]
        for _ in active_slots:
            work_queue.put(None)
        failed: list[TaskJob] = []
        for fut in concurrent.futures.as_completed(futures):
            failed.extend(fut.result())
        executor.shutdown(wait=True)
        summarize(output_dir, write_files=True)
        if failed:
            print(f"[FAIL] {len(failed)} task(s) exhausted retries", flush=True)
            return 1
        print("[ok] all pending tasks finished", flush=True)
        return 0
    finally:
        stop_event.set()
        registry.terminate_all()


if __name__ == "__main__":
    # Ensure local imports resolve when invoked as a script.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    raise SystemExit(main())
