#!/usr/bin/env python3
"""Summarize RoboCasa GR1 batched eval outputs under an OUTPUT_DIR."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from task_list import OFFICIAL_ENV_IDS, env_short_name


def _load_results(output_dir: Path) -> list[dict]:
    rows = []
    results_root = output_dir / "results"
    if results_root.is_dir():
        for path in sorted(results_root.glob("*/result.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    # Also accept flat result_*.json
    for path in sorted(output_dir.glob("**/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload not in rows:
            rows.append(payload)
    return rows


def summarize(output_dir: Path, *, write_files: bool = True) -> dict:
    rows = _load_results(output_dir)
    by_env = {r["env_id"]: r for r in rows if "env_id" in r}
    task_rows = []
    total_s = total_n = 0
    missing = []
    for env_id in OFFICIAL_ENV_IDS:
        payload = by_env.get(env_id)
        if payload is None:
            missing.append(env_id)
            successes = episodes = 0
            rate = None
        else:
            successes = int(payload.get("successes", 0))
            episodes = int(payload.get("num_episodes", 0))
            rate = float(payload.get("success_rate", successes / max(episodes, 1)))
            total_s += successes
            total_n += episodes
        task_rows.append(
            {
                "env_id": env_id,
                "short_name": env_short_name(env_id),
                "successes": successes,
                "num_episodes": episodes,
                "success_rate": rate,
                "done": payload is not None,
            }
        )
    mean_sr = (total_s / total_n) if total_n else 0.0
    summary = {
        "n_tasks_official": len(OFFICIAL_ENV_IDS),
        "n_tasks_done": sum(1 for r in task_rows if r["done"]),
        "n_tasks_missing": len(missing),
        "total_successes": total_s,
        "total_episodes": total_n,
        "mean_success_rate": mean_sr,
        "tasks": task_rows,
        "missing_env_ids": missing,
    }
    if write_files:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["short_name", "env_id", "successes", "num_episodes", "success_rate", "done"],
            )
            writer.writeheader()
            for row in task_rows:
                writer.writerow(row)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.output_dir, write_files=not args.no_write)
    print(
        f"done={summary['n_tasks_done']}/{summary['n_tasks_official']} "
        f"mean_SR={summary['mean_success_rate'] * 100:.1f}% "
        f"({summary['total_successes']}/{summary['total_episodes']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
