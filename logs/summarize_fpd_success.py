#!/usr/bin/env python3
"""Merge FPD RoboTwin eval logs -> CSV in the same shape as fpd_full_success.csv.

Output columns:
  task name,clean_success,random_success
  ...
  avg,<macro clean>,<macro random>

``clean_success`` / ``random_success`` are integer success rates in percent
(success count when the run finished 100 episodes). Incomplete / missing
cells are left blank. ``avg`` is the macro mean over tasks that have a value.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-9;]*m")
rate_pat = re.compile(r"Success rate:\s*(\d+)/(\d+)")
saved_pat = re.compile(r"Data has been saved to")

ROOT = Path(__file__).resolve().parent
SKIP = ROOT / "fpd_colocate6_skip_done.txt"
OUT_CSV = ROOT / "fpd_latest_success_merged.csv"
OUT_SUMMARY = ROOT / "fpd_latest_success_summary.csv"

# Prefer newer dirs when merging the same mode:task key (done > partial, then higher total).
LOGDIRS = [
    # ROOT / "batched_eval_fpd_full_colocate_6_demo_clean+demo_randomized_20260925_122013",
    ROOT / "batched_eval_fpd_colocate6_norender_demo_clean+demo_randomized_20260926_160915",
    # ROOT / "batched_eval_fpd_full_colocate_6_demo_clean+demo_randomized_20260927_215041",
    # ROOT / "batched_eval_fpd_colocate6_remaining_skipfront_demo_randomized_20260927_224957",
]

TASK_ORDER_FILE = (
    ROOT.parent / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
)


def _load_task_order() -> list[str]:
    if not TASK_ORDER_FILE.is_file():
        return []
    # Lightweight YAML key scrape (file is a flat map).
    tasks: list[str] = []
    for line in TASK_ORDER_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        key = s.split(":", 1)[0].strip().strip("'\"")
        if key and key not in tasks:
            tasks.append(key)
    return tasks


def parse_logdir(logdir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not logdir.exists():
        return out
    for p in sorted(logdir.glob("*.log")):
        m = re.match(r"(demo_clean|demo_randomized)_(.+)_gpu\d+_slot\d+\.log", p.name)
        if not m:
            continue
        mode, task = m.group(1), m.group(2)
        text = ansi.sub("", p.read_text(errors="ignore"))
        rates = rate_pat.findall(text)
        succ, total = (int(rates[-1][0]), int(rates[-1][1])) if rates else (0, 0)
        done = bool(saved_pat.search(text)) or total >= 100
        key = f"{mode}:{task}"
        cur = dict(succ=succ, total=total, done=done, source=logdir.name)
        prev = out.get(key)
        if (
            prev is None
            or (cur["done"] and not prev["done"])
            or (cur["done"] == prev["done"] and cur["total"] >= prev["total"])
        ):
            out[key] = cur
    return out


def _better(a: dict | None, b: dict) -> dict:
    if a is None:
        return b
    if b["done"] and not a["done"]:
        return b
    if a["done"] and not b["done"]:
        return a
    return b if b["total"] >= a["total"] else a


def _rate_int(v: dict | None) -> str:
    """Integer percent for the wide CSV; blank if missing / empty."""
    if v is None or v["total"] <= 0:
        return ""
    return str(int(round(100.0 * v["succ"] / v["total"])))


def main() -> None:
    skip: set[str] = set()
    if SKIP.is_file():
        for line in SKIP.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and ":" in line:
                skip.add(line)

    # Parse all log dirs (oldest -> newest).
    per_dir = [(d, parse_logdir(d)) for d in LOGDIRS if d.exists()]
    # Seed from oldest "OLD" dir for skip reuse.
    old = per_dir[0][1] if per_dir else {}

    merged: dict[str, dict] = {}
    for _d, parsed in per_dir:
        for k, v in parsed.items():
            merged[k] = _better(merged.get(k), v)

    # If skip-file marks a task and the old colocate run finished it, prefer that
    # (same rule as before), unless a newer done result already beat it via _better.
    for k in skip:
        if k in old and old[k]["done"]:
            cur = merged.get(k)
            if cur is None or not cur["done"]:
                merged[k] = {**old[k], "source": "skip/old_colocate6"}

    task_order = _load_task_order()
    tasks_seen = {k.split(":", 1)[1] for k in merged}
    if task_order:
        tasks = [t for t in task_order if t in tasks_seen] + sorted(tasks_seen - set(task_order))
    else:
        tasks = sorted(tasks_seen)

    wide_rows: list[dict[str, str]] = []
    clean_vals: list[float] = []
    random_vals: list[float] = []
    for task in tasks:
        clean = merged.get(f"demo_clean:{task}")
        rand = merged.get(f"demo_randomized:{task}")
        c = _rate_int(clean)
        r = _rate_int(rand)
        wide_rows.append({"task name": task, "clean_success": c, "random_success": r})
        if c != "":
            clean_vals.append(float(c))
        if r != "":
            random_vals.append(float(r))

    clean_avg = f"{sum(clean_vals) / len(clean_vals):.2f}" if clean_vals else ""
    random_avg = f"{sum(random_vals) / len(random_vals):.2f}" if random_vals else ""
    wide_rows.append({"task name": "avg", "clean_success": clean_avg, "random_success": random_avg})

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task name", "clean_success", "random_success"])
        w.writeheader()
        w.writerows(wide_rows)

    # Compact status summary (still useful while random is incomplete).
    def _status(v: dict | None) -> str:
        if v is None or v["total"] <= 0:
            return "missing"
        return "done" if v["done"] else "partial"

    n_clean_done = sum(1 for t in tasks if _status(merged.get(f"demo_clean:{t}")) == "done")
    n_rand_done = sum(1 for t in tasks if _status(merged.get(f"demo_randomized:{t}")) == "done")
    n_rand_partial = sum(1 for t in tasks if _status(merged.get(f"demo_randomized:{t}")) == "partial")
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "tasks_done",
                "tasks_partial",
                "tasks_with_score",
                "macro_avg_rate_pct",
            ],
        )
        w.writeheader()
        w.writerows(
            [
                {
                    "mode": "demo_clean",
                    "tasks_done": n_clean_done,
                    "tasks_partial": sum(
                        1 for t in tasks if _status(merged.get(f"demo_clean:{t}")) == "partial"
                    ),
                    "tasks_with_score": len(clean_vals),
                    "macro_avg_rate_pct": clean_avg,
                },
                {
                    "mode": "demo_randomized",
                    "tasks_done": n_rand_done,
                    "tasks_partial": n_rand_partial,
                    "tasks_with_score": len(random_vals),
                    "macro_avg_rate_pct": random_avg,
                },
            ]
        )

    print(f"wrote {OUT_CSV} ({len(wide_rows)} rows incl. avg)")
    print(f"wrote {OUT_SUMMARY}")
    print(f"  demo_clean:      macro={clean_avg}%  scored={len(clean_vals)} done={n_clean_done}")
    print(
        f"  demo_randomized: macro={random_avg}%  scored={len(random_vals)} "
        f"done={n_rand_done} partial={n_rand_partial}"
    )


if __name__ == "__main__":
    main()
