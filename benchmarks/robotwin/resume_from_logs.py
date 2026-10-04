#!/usr/bin/env python3
"""Parse a batched_eval log directory and emit per-task resume plans.

Reads ``{mode}_{task}_gpu*_slot*.log`` files, takes the last
``Success rate: S/E => …, current seed: SEED`` line (ANSI stripped), and
computes how to continue to ``--target`` episodes starting at ``seed + 1``
(seed-sequential resume; scheduling width is decided by batched_eval).

Output (stdout): one JSON object per line, plus a final summary object with
``"_summary": true``. Suitable for bash ``while read``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RATE_RE = re.compile(
    r"Success rate:\s*(?P<suc>\d+)/(?P<done>\d+)\s*=>.*?current seed:\s*(?P<seed>\d+)",
    re.IGNORECASE | re.DOTALL,
)
HEADER_RE = re.compile(
    r"task_name\s*:\s*(?P<task>\S+)\s*"
    r"task_config\s*:\s*(?P<mode>\S+)\s*"
    r"ckpt_setting\s*:\s*(?P<name>\S+).*?"
    r"seed\s*:\s*(?P<header_seed>\S+)",
    re.DOTALL,
)
LOG_NAME_RE = re.compile(
    r"^(?P<mode>demo_clean|demo_randomized)_(?P<task>.+)_gpu\d+_slot\d+\.log$"
)
CKPT_RE = re.compile(r"Loading checkpoint:\s*(\S+\.safetensors)")
DIR_NAME_RE = re.compile(
    r"^batched_eval_(?P<name>.+)_(?P<modes>demo_clean(?:\+demo_randomized)?|demo_randomized(?:\+demo_clean)?)_\d{8}_\d{6}$"
)


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_log(path: Path) -> dict | None:
    raw = path.read_text(errors="replace")
    text = _strip_ansi(raw)
    m_name = LOG_NAME_RE.match(path.name)
    if not m_name:
        return None
    mode = m_name.group("mode")
    task = m_name.group("task")

    header = HEADER_RE.search(text)
    if header:
        mode = header.group("mode")
        task = header.group("task")
        ckpt_setting = header.group("name")
        header_seed = header.group("header_seed")
    else:
        ckpt_setting = None
        header_seed = "0"

    last = None
    for last in RATE_RE.finditer(text):
        pass

    if last is None:
        return {
            "mode": mode,
            "task": task,
            "ckpt_setting": ckpt_setting,
            "header_seed": header_seed,
            "suc": 0,
            "done": 0,
            "last_seed": None,
            "complete": False,
            "log": str(path),
        }

    suc = int(last.group("suc"))
    done = int(last.group("done"))
    seed = int(last.group("seed"))
    return {
        "mode": mode,
        "task": task,
        "ckpt_setting": ckpt_setting,
        "header_seed": header_seed,
        "suc": suc,
        "done": done,
        "last_seed": seed,
        "complete": False,  # filled by caller with target
        "log": str(path),
    }


def _latest_log_per_job(log_dir: Path) -> dict[tuple[str, str], Path]:
    """If a job was relaunched into a new slot, keep the newest mtime log."""
    best: dict[tuple[str, str], Path] = {}
    for path in log_dir.glob("demo_*_gpu*_slot*.log"):
        m = LOG_NAME_RE.match(path.name)
        if not m:
            continue
        key = (m.group("mode"), m.group("task"))
        prev = best.get(key)
        if prev is None or path.stat().st_mtime >= prev.stat().st_mtime:
            best[key] = path
    return best


def _infer_ckpt(log_dir: Path) -> str | None:
    for path in sorted(log_dir.glob("infer_gpu*.log")):
        text = path.read_text(errors="replace")
        m = CKPT_RE.search(text)
        if m:
            ckpt_file = Path(m.group(1))
            return str(ckpt_file.parent if ckpt_file.suffix == ".safetensors" else ckpt_file)
    return None


def _infer_meta(log_dir: Path) -> dict:
    meta = {"policy_name": None, "modes": None, "ckpt_dir": _infer_ckpt(log_dir)}
    m = DIR_NAME_RE.match(log_dir.name)
    if m:
        meta["policy_name"] = m.group("name")
        modes = m.group("modes")
        if "+" in modes:
            meta["modes"] = "both"
        else:
            meta["modes"] = modes
    # Prefer ckpt_setting from any task log header.
    for path in log_dir.glob("demo_*_gpu*_slot*.log"):
        text = _strip_ansi(path.read_text(errors="replace")[:2000])
        h = HEADER_RE.search(text)
        if h and h.group("name"):
            meta["policy_name"] = h.group("name")
            break
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir", type=Path, help="batched_eval log directory")
    ap.add_argument("--target", type=int, default=100, help="total episodes per task")
    ap.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="RoboTwin config seed when a task has no progress yet (st_seed=100000*(1+base))",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default="",
        help="optional comma-separated task filter",
    )
    args = ap.parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        print(f"[ERROR] log dir not found: {log_dir}", file=sys.stderr)
        return 2
    if args.target <= 0:
        print("[ERROR] --target must be > 0", file=sys.stderr)
        return 2

    task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}
    jobs = _latest_log_per_job(log_dir)
    meta = _infer_meta(log_dir)

    plans = []
    for (mode, task), path in sorted(jobs.items()):
        if task_filter and task not in task_filter:
            continue
        info = _parse_log(path)
        if info is None:
            continue
        done = int(info["done"])
        suc = int(info["suc"])
        complete = done >= args.target
        if info["last_seed"] is None:
            next_seed = 100000 * (1 + args.base_seed)
        else:
            next_seed = int(info["last_seed"]) + 1
        remaining = max(0, args.target - done)
        plan = {
            "mode": mode,
            "task": task,
            "ckpt_setting": info.get("ckpt_setting") or meta.get("policy_name"),
            "suc": suc,
            "done": done,
            "last_seed": info["last_seed"],
            "next_seed": next_seed,
            "remaining": remaining,
            "target": args.target,
            "complete": complete,
            "skip": complete or remaining == 0,
            "log": info["log"],
        }
        plans.append(plan)
        print(json.dumps(plan, ensure_ascii=False), flush=True)

    summary = {
        "_summary": True,
        "log_dir": str(log_dir),
        "ckpt_dir": meta.get("ckpt_dir"),
        "policy_name": meta.get("policy_name"),
        "modes": meta.get("modes"),
        "target": args.target,
        "n_jobs": len(plans),
        "n_skip": sum(1 for p in plans if p["skip"]),
        "n_resume": sum(1 for p in plans if not p["skip"]),
        "remaining_episodes": sum(p["remaining"] for p in plans if not p["skip"]),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
