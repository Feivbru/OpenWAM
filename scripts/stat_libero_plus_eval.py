#!/usr/bin/env python3
"""Summarize one LIBERO-plus eval run: progress + success by suite / category.

Edit ``EVAL_DIR`` below to point at the run you care about, then:

  python scripts/stat_libero_plus_eval.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---- edit this for the run you want ----
EVAL_DIR = Path(
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/outputs/libero-plus/"
    # "fpd_with_umt5_chunk_all_nsims1_full_20261001_204033"
    # "fpd_with_umt5_libero10_sample0.2_no67_20261002_135517"
    # "fpd_with_umt5_libero10_full_20261003_054825"
    "fpd_with_umt5_no10_sample0.2_20261003_131256"
)

# Fallback if manifest has no libero_path.
DEFAULT_LIBERO_PLUS = Path(
    "/data/zixian_guo/projects/haoming/project/PI/ImageWAM/third_party/LIBERO-plus"
)

SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORY_ORDER = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_category_map(libero_path: Path) -> dict[tuple[str, int], str]:
    """Map (suite, task_id) -> category. task_id is 0-based (OpenWAM / results.json)."""
    cls_path = libero_path / "libero" / "libero" / "benchmark" / "task_classification.json"
    payload = _load_json(cls_path)
    out: dict[tuple[str, int], str] = {}
    for suite, records in payload.items():
        for item in records:
            out[(suite, int(item["id"]) - 1)] = str(item["category"])
    return out


def _fmt_rate(succ: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * succ / total:5.1f}%  ({succ}/{total})"


def _rate(succ: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(succ) / float(total)


def _row_dict(name: str, finished_or_succ: int, planned_or_total: int, *, kind: str) -> dict:
    """kind='progress' → finished/planned; kind='success' → successes/finished."""
    if kind == "progress":
        finished, planned = finished_or_succ, planned_or_total
        return {
            "name": name,
            "finished": finished,
            "planned": planned,
            "remaining": max(0, planned - finished),
            "progress_rate": _rate(finished, planned),
        }
    successes, finished = finished_or_succ, planned_or_total
    return {
        "name": name,
        "successes": successes,
        "finished": finished,
        "success_rate": _rate(successes, finished),
    }


def _print_table(title: str, rows: list[tuple[str, int, int]], *, indent: str = "") -> None:
    print(f"{indent}{title}")
    if not rows:
        print(f"{indent}  (empty)")
        return
    name_w = max(len(r[0]) for r in rows)
    for name, succ, total in rows:
        print(f"{indent}  {name:<{name_w}}  {_fmt_rate(succ, total)}")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_outputs(
    eval_dir: Path,
    *,
    payload: dict,
    suite_progress: list[dict],
    suite_success: list[dict],
    cat_progress: list[dict],
    cat_success: list[dict],
    suite_cat_rows: list[dict],
    suites: list[str],
    categories: list[str],
) -> tuple[Path, ...]:
    json_path = eval_dir / "eval_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    suite_csv = eval_dir / "eval_stats_by_suite.csv"
    _write_csv(
        suite_csv,
        ["suite", "finished", "planned", "remaining", "progress_rate", "successes", "success_rate"],
        [
            {
                "suite": p["name"],
                "finished": p["finished"],
                "planned": p["planned"],
                "remaining": p["remaining"],
                "progress_rate": p["progress_rate"],
                "successes": s["successes"],
                "success_rate": s["success_rate"],
            }
            for p, s in zip(suite_progress, suite_success)
        ],
    )

    cat_csv = eval_dir / "eval_stats_by_category.csv"
    _write_csv(
        cat_csv,
        ["category", "finished", "planned", "remaining", "progress_rate", "successes", "success_rate"],
        [
            {
                "category": p["name"],
                "finished": p["finished"],
                "planned": p["planned"],
                "remaining": p["remaining"],
                "progress_rate": p["progress_rate"],
                "successes": s["successes"],
                "success_rate": s["success_rate"],
            }
            for p, s in zip(cat_progress, cat_success)
        ],
    )

    matrix_csv = eval_dir / "eval_stats_suite_category.csv"
    _write_csv(
        matrix_csv,
        ["suite", "category", "successes", "finished", "planned", "success_rate", "progress_rate"],
        suite_cat_rows,
    )

    # Pivot: rows=suite, columns=category, values=success_rate (finished-only).
    rate_lookup = {(r["suite"], r["category"]): r["success_rate"] for r in suite_cat_rows}
    pivot_csv = eval_dir / "eval_stats_success_matrix.csv"
    pivot_fields = ["suite", *categories]
    pivot_rows = []
    for suite in suites:
        row: dict = {"suite": suite}
        for cat in categories:
            row[cat] = rate_lookup.get((suite, cat))
        pivot_rows.append(row)
    _write_csv(pivot_csv, pivot_fields, pivot_rows)

    return json_path, suite_csv, cat_csv, matrix_csv, pivot_csv


def main() -> None:
    eval_dir = EVAL_DIR
    if not eval_dir.is_dir():
        raise SystemExit(f"EVAL_DIR not found: {eval_dir}")

    manifest_path = eval_dir / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    libero_path = Path(manifest.get("libero_path") or DEFAULT_LIBERO_PLUS)
    cat_map = _load_category_map(libero_path)

    planned = [(str(j["suite"]), int(j["task_id"])) for j in manifest.get("initial_queue", [])]
    planned_set = set(planned)
    planned_by_suite: dict[str, int] = defaultdict(int)
    planned_by_cat: dict[str, int] = defaultdict(int)
    planned_by_suite_cat: dict[tuple[str, str], int] = defaultdict(int)
    for suite, tid in planned:
        planned_by_suite[suite] += 1
        cat = cat_map.get((suite, tid), "UNKNOWN")
        planned_by_cat[cat] += 1
        planned_by_suite_cat[(suite, cat)] += 1

    results_files = sorted(eval_dir.glob("videos/**/results.json"))
    done_succ: dict[tuple[str, int], int] = {}
    done_meta: dict[tuple[str, int], dict] = {}
    mtimes: list[float] = []
    for path in results_files:
        data = _load_json(path)
        suite = str(data["suite"])
        tid = int(data["task_id"])
        succ = int(data.get("successes", 0))
        trials = int(data.get("trial_stop", 1)) - int(data.get("trial_start", 0))
        # official protocol is 1 trial → treat as 0/1 success count
        done_succ[(suite, tid)] = succ
        done_meta[(suite, tid)] = {
            "successes": succ,
            "trials": max(1, trials),
            "category": cat_map.get((suite, tid), "UNKNOWN"),
            "path": path,
        }
        mtimes.append(path.stat().st_mtime)

    # Prefer planned queue as denominator; fall back to finished-only if no manifest queue.
    universe = planned_set if planned_set else set(done_succ)
    n_planned = len(universe) if universe else len(done_succ)
    n_done = sum(1 for k in universe if k in done_succ) if universe else len(done_succ)
    # Also count finished tasks not in the planned set (shouldn't happen).
    extra = [k for k in done_succ if k not in planned_set] if planned_set else []

    total_succ = sum(done_succ[k] for k in done_succ if (not planned_set) or k in planned_set)
    total_done = n_done

    print("=" * 72)
    print(f"eval_dir : {eval_dir}")
    print(f"ckpt     : {manifest.get('checkpoint', 'n/a')}")
    print(f"created  : {manifest.get('created_at', 'n/a')}")
    ratio = manifest.get("task_sample_ratio")
    if ratio is not None:
        print(f"sample   : ratio={ratio} seed={manifest.get('task_sample_seed')}")
    print("-" * 72)
    print(f"progress : {n_done}/{n_planned} tasks  ({(100.0 * n_done / n_planned) if n_planned else 0:.1f}%)")
    print(f"success  : {_fmt_rate(total_succ, total_done)}   (among finished tasks)")
    if extra:
        print(f"warning  : {len(extra)} finished task(s) not in manifest.initial_queue")

    speed_tasks_per_min = None
    eta_hours = None
    eta_finish_local = None
    if mtimes and n_done >= 2:
        span = max(1e-6, max(mtimes) - min(mtimes))
        rate = (n_done - 1) / span
        rem = max(0, n_planned - n_done)
        eta_s = rem / rate if rate > 0 else float("inf")
        speed_tasks_per_min = rate * 60.0
        eta_hours = (eta_s / 3600.0) if eta_s < 1e12 else None
        eta_finish_local = (
            datetime.fromtimestamp(max(mtimes) + eta_s).strftime("%Y-%m-%d %H:%M:%S")
            if eta_s < 1e12
            else None
        )
        print(
            f"speed    : {speed_tasks_per_min:.2f} tasks/min  "
            f"(first→last finished); ETA≈{eta_hours:.2f}h "
            f"→ ~{datetime.fromtimestamp(max(mtimes) + eta_s).strftime('%H:%M') if eta_s < 1e12 else 'n/a'}"
        )
    print("=" * 72)

    # Per-suite progress + success
    suite_rows_prog = []
    suite_rows_succ = []
    for suite in list(SUITE_ORDER) + sorted(
        {s for s, _ in universe} - set(SUITE_ORDER),
    ):
        keys = [(suite, tid) for s, tid in universe if s == suite] if universe else [
            (s, tid) for (s, tid) in done_succ if s == suite
        ]
        if not keys and suite not in planned_by_suite and suite not in {s for s, _ in done_succ}:
            continue
        planned_n = planned_by_suite.get(suite, len(keys))
        finished = [k for k in keys if k in done_succ]
        succ = sum(done_succ[k] for k in finished)
        suite_rows_prog.append((suite, len(finished), planned_n))
        suite_rows_succ.append((suite, succ, len(finished)))
    _print_table("Progress by suite", suite_rows_prog)
    print()
    _print_table("Success by suite (finished only)", suite_rows_succ)
    print()

    # Per-category success (finished only) + planned progress
    cat_done: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k, meta in done_meta.items():
        if planned_set and k not in planned_set:
            continue
        cat_done[meta["category"]].append(k)

    cat_rows_prog = []
    cat_rows_succ = []
    cats = [c for c in CATEGORY_ORDER if c in planned_by_cat or c in cat_done] + sorted(
        (set(planned_by_cat) | set(cat_done)) - set(CATEGORY_ORDER)
    )
    for cat in cats:
        planned_n = planned_by_cat.get(cat, 0)
        finished_keys = cat_done.get(cat, [])
        # if no planned map, planned_n = finished
        if planned_n == 0 and finished_keys:
            planned_n = len(finished_keys)
        succ = sum(done_succ[k] for k in finished_keys)
        cat_rows_prog.append((cat, len(finished_keys), planned_n))
        cat_rows_succ.append((cat, succ, len(finished_keys)))
    _print_table("Progress by category", cat_rows_prog)
    print()
    _print_table("Success by category (finished only)", cat_rows_succ)
    print()

    # Suite × category success matrix (finished only)
    print("Success by suite × category (finished only)")
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # succ, total
    for (suite, tid), succ in done_succ.items():
        if planned_set and (suite, tid) not in planned_set:
            continue
        cat = cat_map.get((suite, tid), "UNKNOWN")
        cells[(suite, cat)][0] += succ
        cells[(suite, cat)][1] += 1

    suites = [s for s, _, _ in suite_rows_succ]
    cats_present = [c for c, _, _ in cat_rows_succ]
    cat_short = {
        "Camera Viewpoints": "Camera",
        "Robot Initial States": "Robot",
        "Language Instructions": "Language",
        "Light Conditions": "Light",
        "Background Textures": "Background",
        "Sensor Noise": "Noise",
        "Objects Layout": "Layout",
        "UNKNOWN": "UNKNOWN",
    }
    headers = [cat_short.get(c, c[:10]) for c in cats_present]
    col_w = max(10, max((len(h) for h in headers), default=10))
    suite_w = max((len(s) for s in suites), default=12)
    print("  " + " " * suite_w + "  " + "  ".join(f"{h:>{col_w}}" for h in headers))
    for suite in suites:
        pretty = []
        for cat in cats_present:
            succ, total = cells[(suite, cat)]
            if total:
                pretty.append(f"{100.0 * succ / total:5.1f}%({total})".rjust(col_w))
            else:
                pretty.append("—".rjust(col_w))
        print(f"  {suite:<{suite_w}}  " + "  ".join(pretty))
    print("=" * 72)

    suite_progress = [_row_dict(n, a, b, kind="progress") for n, a, b in suite_rows_prog]
    suite_success = [_row_dict(n, a, b, kind="success") for n, a, b in suite_rows_succ]
    cat_progress = [_row_dict(n, a, b, kind="progress") for n, a, b in cat_rows_prog]
    cat_success = [_row_dict(n, a, b, kind="success") for n, a, b in cat_rows_succ]

    suite_cat_rows: list[dict] = []
    for suite in suites:
        for cat in cats_present:
            succ, finished = cells[(suite, cat)]
            planned_n = planned_by_suite_cat.get((suite, cat), 0)
            suite_cat_rows.append(
                {
                    "suite": suite,
                    "category": cat,
                    "successes": succ,
                    "finished": finished,
                    "planned": planned_n,
                    "success_rate": _rate(succ, finished),
                    "progress_rate": _rate(finished, planned_n) if planned_n else None,
                }
            )

    tasks_payload = []
    for suite, tid in sorted(universe if universe else done_succ, key=lambda x: (x[0], x[1])):
        cat = cat_map.get((suite, tid), "UNKNOWN")
        if (suite, tid) in done_succ:
            status = "done"
            successes = done_succ[(suite, tid)]
        else:
            status = "pending"
            successes = None
        tasks_payload.append(
            {
                "suite": suite,
                "task_id": tid,
                "category": cat,
                "status": status,
                "successes": successes,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(eval_dir),
        "checkpoint": manifest.get("checkpoint"),
        "created_at": manifest.get("created_at"),
        "task_sample_ratio": manifest.get("task_sample_ratio"),
        "task_sample_seed": manifest.get("task_sample_seed"),
        "overview": {
            "planned": n_planned,
            "finished": n_done,
            "remaining": max(0, n_planned - n_done),
            "progress_rate": _rate(n_done, n_planned),
            "successes": total_succ,
            "success_rate": _rate(total_succ, total_done),
            "speed_tasks_per_min": speed_tasks_per_min,
            "eta_hours": eta_hours,
            "eta_finish_local": eta_finish_local,
            "extra_finished_not_in_queue": len(extra),
        },
        "by_suite": {
            "progress": suite_progress,
            "success": suite_success,
        },
        "by_category": {
            "progress": cat_progress,
            "success": cat_success,
        },
        "by_suite_category": suite_cat_rows,
        "tasks": tasks_payload,
    }

    written = _save_outputs(
        eval_dir,
        payload=payload,
        suite_progress=suite_progress,
        suite_success=suite_success,
        cat_progress=cat_progress,
        cat_success=cat_success,
        suite_cat_rows=suite_cat_rows,
        suites=suites,
        categories=cats_present,
    )
    print("wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
