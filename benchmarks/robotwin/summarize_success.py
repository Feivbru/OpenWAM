#!/usr/bin/env python3
"""Summarize finished RoboTwin task success rates in one folder.

Pass one inner folder, not the parent ``logs/`` directory:

  # one batched-eval log directory
  python benchmarks/robotwin/summarize_success.py \\
      logs/batched_eval_fpd_full_demo_clean+demo_randomized_20260919_221723

  # one CSV per checkpoint, written into the given directory
  python benchmarks/robotwin/summarize_success.py \\
      third_party/RoboTwin/eval_result -o logs

A log counts as finished only when it contains ``Data has been saved``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

LOG_NAME = re.compile(
    r"^(?P<mode>demo_clean|demo_randomized)_(?P<task>.+)_gpu\d+_slot\d+\.log$"
)
RATE_IN_LOG = re.compile(r"Success rate:.*?(?P<success>\d+)/(?P<episodes>\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _is_logs_parent(folder: str) -> bool:
    names = os.listdir(folder)
    has_run_dir = any(
        name.startswith("batched_eval_") and os.path.isdir(os.path.join(folder, name))
        for name in names
    )
    has_task_log = any(LOG_NAME.match(name) for name in names)
    has_result = os.path.isfile(os.path.join(folder, "_result.txt"))
    return has_run_dir and not has_task_log and not has_result


def _rows_from_logs(folder: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(folder)):
        match = LOG_NAME.match(name)
        if match is None:
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        text = ANSI.sub("", open(path, encoding="utf-8", errors="replace").read())
        if "Data has been saved" not in text:
            continue
        found = list(RATE_IN_LOG.finditer(text))
        if not found:
            continue
        last = found[-1]
        success = int(last.group("success"))
        episodes = int(last.group("episodes"))
        rows.append(
            {
                "mode": match.group("mode"),
                "task": match.group("task"),
                "successes": success,
                "episodes": episodes,
                "success_rate": success / episodes if episodes else "",
                "ckpt": "",
                "timestamp": "",
            }
        )
    return rows


def _rate_from_result(path: str) -> float:
    lines = [
        line.strip()
        for line in open(path, encoding="utf-8", errors="replace")
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"empty result file: {path}")
    return float(lines[-1])


def _rows_from_results(folder: str) -> list[dict]:
    latest: dict[tuple[str, str, str], tuple[str, str]] = {}
    for dirpath, _dirnames, filenames in os.walk(folder):
        if "_result.txt" not in filenames:
            continue
        path = os.path.join(dirpath, "_result.txt")
        rel = os.path.relpath(dirpath, folder)
        parts = rel.split(os.sep)
        # eval_result/<task>/<policy>/<mode>/<ckpt>/<timestamp>
        if len(parts) < 5:
            continue
        task, _policy, mode, ckpt, timestamp = parts[-5:]
        key = (mode, task, ckpt)
        previous = latest.get(key)
        if previous is None or timestamp > previous[0]:
            latest[key] = (timestamp, path)

    rows = []
    for (mode, task, ckpt), (timestamp, path) in sorted(latest.items()):
        rate = _rate_from_result(path)
        episodes = 100
        successes = int(round(rate * episodes))
        rows.append(
            {
                "mode": mode,
                "task": task,
                "successes": successes,
                "episodes": episodes,
                "success_rate": rate,
                "ckpt": ckpt,
                "timestamp": timestamp,
            }
        )
    return rows


def collect(folder: str) -> list[dict]:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        _fail(f"not a directory: {folder}")
    if _is_logs_parent(folder):
        _fail(
            f"{folder} looks like the parent logs directory. "
            "Pass one eval run folder, or an eval_result directory."
        )
    names = os.listdir(folder)
    if any(LOG_NAME.match(name) for name in names):
        return _rows_from_logs(folder)
    return _rows_from_results(folder)


def _pivot(rows: list[dict]) -> list[dict]:
    """One row per task. Values are success counts (rate * 100, rounded).

    Unfinished modes stay blank. The last row is the mean of those counts.
    """
    by_task: dict[str, dict[str, tuple]] = {}
    for row in rows:
        mode = row["mode"]
        if mode not in ("demo_clean", "demo_randomized"):
            continue
        slot = by_task.setdefault(row["task"], {})
        count = int(round(float(row["success_rate"]) * 100))
        timestamp = row.get("timestamp") or ""
        previous = slot.get(mode)
        if previous is None or timestamp >= previous[0]:
            slot[mode] = (timestamp, count)

    table = []
    clean_counts = []
    random_counts = []
    for task in sorted(by_task):
        slot = by_task[task]
        clean = slot.get("demo_clean", (None, None))[1]
        random = slot.get("demo_randomized", (None, None))[1]
        if clean is not None:
            clean_counts.append(clean)
        if random is not None:
            random_counts.append(random)
        table.append({"task name": task, "clean_success": clean, "random_success": random})

    table.append(
        {
            "task name": "avg",
            "clean_success": sum(clean_counts) / len(clean_counts) if clean_counts else None,
            "random_success": sum(random_counts) / len(random_counts) if random_counts else None,
        }
    )
    return table


def _fmt(value, avg: bool = False) -> str:
    if value is None:
        return ""
    if avg:
        return f"{float(value):.2f}"
    return str(int(value))


def write_csv(rows: list[dict], output) -> None:
    fields = ["task name", "clean_success", "random_success"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in _pivot(rows):
        is_avg = row["task name"] == "avg"
        writer.writerow(
            {
                "task name": row["task name"],
                "clean_success": _fmt(row["clean_success"], is_avg),
                "random_success": _fmt(row["random_success"], is_avg),
            }
        )


def write_ckpt_csvs(rows: list[dict], out_dir: str) -> list[str]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("ckpt") or "unknown", []).append(row)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ckpt, group in sorted(groups.items()):
        path = os.path.join(out_dir, f"{ckpt}_success.csv")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            write_csv(group, handle)
        paths.append(path)
        print(f"wrote {len(group)} finished runs to {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="one eval log directory, or an eval_result directory")
    parser.add_argument("-o", "--output", help="directory for <ckpt>_success.csv (default: stdout)")
    args = parser.parse_args()

    rows = collect(args.folder)
    if not rows:
        _fail(f"no finished tasks in {args.folder}")

    if args.output:
        out_dir = args.output
        if out_dir.endswith(".csv"):
            out_dir = os.path.dirname(os.path.abspath(out_dir)) or "."
        write_ckpt_csvs(rows, out_dir)
    else:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row.get("ckpt") or "unknown", []).append(row)
        for ckpt, group in sorted(groups.items()):
            print(f"# {ckpt}", file=sys.stderr)
            write_csv(group, sys.stdout)

    by_mode: dict[str, list[float]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(float(row["success_rate"]))
    print(f"finished tasks: {len(rows)}", file=sys.stderr)
    for mode in sorted(by_mode):
        rates = by_mode[mode]
        mean = sum(rates) / len(rates)
        print(f"  {mode}: {len(rates)} tasks, mean {mean * 100:.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
