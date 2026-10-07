#!/usr/bin/env python3
"""VRTC cube-prefetch topology toy experiment (no real model / robot).

Wire rules (frozen for this experiment)
---------------------------------------
1. Server always returns **one cube** of actions (``video_stride`` floats).
2. Client ``open_loop_horizon`` == cube action length (executes the whole cube,
   then requests again with the new real observation).
3. Replan is cube-granular with prefetch: when ``len(wait_pool) <= replan_cubes``
   after a delivery (or after a merge), start a background inference whose clear
   condition ends at the latest real frame. When the job finishes, **overwrite**
   every cube still sitting after the last client delivery so the kept head sits
   immediately after that real frame (skipping cubes the client already consumed).

Toy dynamics
------------
- Environment observation is an integer ``n`` that grows by 1 per cube executed.
- Clear condition is the last ``pool_warmup`` real integers.
- Model (instant math, wall-clock sleep simulated as ``t_infer``) predicts the
  next ``predict_cubes`` frames and their actions::

      frame k  ->  actions (k-0.4, k-0.3, k-0.2, k-0.1)

Timings (seconds)
-----------------
- model inference: 0.24
- client↔env one cube: 0.13
- one-way client↔server comm: 0.001

Run::

    python examples/vrtc_topology_sim.py
    python examples/vrtc_topology_sim.py --n-max 30 --csv /tmp/vrtc_topo.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants / toy model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    pool_warmup: int = 5
    predict_cubes: int = 4
    video_stride: int = 4
    replan_cubes: int = 2
    t_infer: float = 0.24
    t_env: float = 0.13
    t_comm: float = 0.001
    n_max: int = 100


def cube_actions(frame_id: int) -> Tuple[float, ...]:
    return tuple(round(frame_id - 0.4 + 0.1 * i, 1) for i in range(4))


@dataclass
class Cube:
    frame_id: int
    actions: Tuple[float, ...]
    source: str = "pred"  # "warmup" | "pred" | "real"

    def short(self) -> str:
        return f"{self.frame_id}{self.source[0]}"


def toy_predict(clear_frames: Sequence[int], predict_cubes: int) -> List[Cube]:
    """Linear teacher: next frames are last_clear+1 .. +predict_cubes."""
    if not clear_frames:
        raise ValueError("empty clear condition")
    last = clear_frames[-1]
    out: List[Cube] = []
    for i in range(1, predict_cubes + 1):
        fid = last + i
        out.append(Cube(frame_id=fid, actions=cube_actions(fid), source="pred"))
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@dataclass
class LogRow:
    t: float
    event: str
    n: Optional[int]
    detail: str
    wait: str
    clear: str
    infer: str
    delivered: str
    notes: str = ""


class Timeline:
    def __init__(self) -> None:
        self.rows: List[LogRow] = []

    def add(self, row: LogRow) -> None:
        self.rows.append(row)

    def dump_text(self) -> str:
        lines = [
            f"{'t':>8}  {'event':<18}  {'n':>4}  {'delivered':<10}  "
            f"{'wait':<22}  {'clear':<18}  {'infer':<28}  notes"
        ]
        lines.append("-" * 140)
        for r in self.rows:
            n = "-" if r.n is None else str(r.n)
            lines.append(
                f"{r.t:8.3f}  {r.event:<18}  {n:>4}  {r.delivered:<10}  "
                f"{r.wait:<22}  {r.clear:<18}  {r.infer:<28}  {r.notes or r.detail}"
            )
        return "\n".join(lines)

    def dump_csv(self, path: str) -> None:
        fields = [
            "t",
            "event",
            "n",
            "detail",
            "wait",
            "clear",
            "infer",
            "delivered",
            "notes",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(
                    {
                        "t": f"{r.t:.6f}",
                        "event": r.event,
                        "n": "" if r.n is None else r.n,
                        "detail": r.detail,
                        "wait": r.wait,
                        "clear": r.clear,
                        "infer": r.infer,
                        "delivered": r.delivered,
                        "notes": r.notes,
                    }
                )


# ---------------------------------------------------------------------------
# Server with cube prefetch
# ---------------------------------------------------------------------------


@dataclass
class InferJob:
    ready_at: float
    clear_snapshot: Tuple[int, ...]
    origin_last_clear: int
    job_id: int


@dataclass
class Server:
    cfg: SimConfig
    log: Timeline
    clear_pool: List[int] = field(default_factory=list)
    wait_pool: List[Cube] = field(default_factory=list)
    last_delivered_frame: Optional[int] = None
    pending: Optional[InferJob] = None
    _job_seq: int = 0

    def _wait_str(self) -> str:
        if not self.wait_pool:
            return "[]"
        return "[" + ",".join(c.short() for c in self.wait_pool) + "]"

    def _clear_str(self) -> str:
        if not self.clear_pool:
            return "[]"
        return "[" + ",".join(str(x) for x in self.clear_pool) + "]"

    def _infer_str(self, now: float) -> str:
        if self.pending is None:
            return "idle"
        left = max(0.0, self.pending.ready_at - now)
        snap = ",".join(str(x) for x in self.pending.clear_snapshot)
        return f"job#{self.pending.job_id}@{self.pending.ready_at:.3f}(-{left:.3f}) snap=[{snap}]"

    def _row(
        self,
        t: float,
        event: str,
        n: Optional[int],
        detail: str,
        delivered: str = "-",
        notes: str = "",
    ) -> None:
        self.log.add(
            LogRow(
                t=t,
                event=event,
                n=n,
                detail=detail,
                wait=self._wait_str(),
                clear=self._clear_str(),
                infer=self._infer_str(t),
                delivered=delivered,
                notes=notes,
            )
        )

    def _maybe_merge(self, now: float) -> None:
        """Apply finished inference: overwrite wait after last delivery."""
        if self.pending is None or now + 1e-12 < self.pending.ready_at:
            return
        job = self.pending
        self.pending = None
        predicted = toy_predict(job.clear_snapshot, self.cfg.predict_cubes)

        # How many predicted frames were already committed to the client
        # after the snapshot's last clear frame?
        if self.last_delivered_frame is None:
            skip = 0
        else:
            skip = max(0, self.last_delivered_frame - job.origin_last_clear)

        kept = predicted[skip:]
        old_wait = self._wait_str()
        self.wait_pool = kept
        self._row(
            now,
            "INFER_MERGE",
            n=None,
            detail=f"job#{job.job_id}",
            notes=(
                f"pred={[c.frame_id for c in predicted]} skip={skip} "
                f"old_wait={old_wait} -> wait={self._wait_str()} "
                f"(head after real {job.origin_last_clear})"
            ),
        )
        # If merge left a short buffer, arm the next prefetch immediately.
        self._maybe_start_replan(now)

    def _start_infer(self, now: float, reason: str) -> InferJob:
        if len(self.clear_pool) < self.cfg.pool_warmup:
            raise RuntimeError("cannot infer before warmup")
        if self.pending is not None:
            raise RuntimeError("infer already pending")
        snap = tuple(self.clear_pool[-self.cfg.pool_warmup :])
        self._job_seq += 1
        job = InferJob(
            ready_at=now + self.cfg.t_infer,
            clear_snapshot=snap,
            origin_last_clear=snap[-1],
            job_id=self._job_seq,
        )
        self.pending = job
        self._row(
            now,
            "INFER_START",
            n=snap[-1],
            detail=reason,
            notes=f"job#{job.job_id} clear={list(snap)} -> expect {[snap[-1]+i for i in range(1, self.cfg.predict_cubes+1)]}",
        )
        return job

    def handle_request(self, obs_n: int, arrive_t: float) -> Tuple[Cube, float]:
        """Process one client request that arrives at ``arrive_t``.

        Returns (cube, reply_depart_t). Reply needs +t_comm to reach client.
        May block (advance reply time) if wait is empty and inference in flight.
        """
        now = arrive_t
        self._maybe_merge(now)
        self._row(now, "REQ_ARRIVE", obs_n, detail=f"obs={obs_n}")

        # ---- Warmup: collect real frames, return synthetic cube ----
        if len(self.clear_pool) < self.cfg.pool_warmup:
            self.clear_pool.append(obs_n)
            cube = Cube(frame_id=obs_n, actions=cube_actions(obs_n), source="warmup")
            self.last_delivered_frame = obs_n
            self._row(
                now,
                "DELIVER",
                obs_n,
                detail="warmup",
                delivered=cube.short(),
                notes=f"pool {len(self.clear_pool)}/{self.cfg.pool_warmup}",
            )
            # After filling warmup, kick the first inference immediately so the
            # next request can be served from wait (or wait for it).
            if len(self.clear_pool) == self.cfg.pool_warmup and self.pending is None:
                self._start_infer(now, reason="post-warmup")
            return cube, now

        # ---- Steady state: obs overwrites the cube we just executed ----
        # Roll clear pool: append latest real, keep last warmup frames.
        # (The obs is the true image at the butt of the last executed cube.)
        if not self.clear_pool or self.clear_pool[-1] != obs_n:
            self.clear_pool.append(obs_n)
            if len(self.clear_pool) > self.cfg.pool_warmup + self.cfg.predict_cubes:
                # Keep a little history; condition always uses last warmup.
                self.clear_pool = self.clear_pool[-(self.cfg.pool_warmup + self.cfg.predict_cubes) :]

        # If nothing to send, we must wait for in-flight (or start) inference.
        if not self.wait_pool:
            if self.pending is None:
                self._start_infer(now, reason="wait-empty")
            assert self.pending is not None
            if now < self.pending.ready_at:
                block_until = self.pending.ready_at
                self._row(
                    now,
                    "CLIENT_BLOCK",
                    obs_n,
                    detail="wait empty",
                    notes=f"block until {block_until:.3f}",
                )
                now = block_until
            self._maybe_merge(now)

        if not self.wait_pool:
            raise RuntimeError("wait_pool still empty after merge")

        cube = self.wait_pool.pop(0)
        self.last_delivered_frame = cube.frame_id
        self._row(
            now,
            "DELIVER",
            obs_n,
            detail="steady",
            delivered=cube.short(),
            notes=f"remaining={len(self.wait_pool)}",
        )

        # Prefetch trigger: after delivery, remaining <= replan_cubes.
        # Use <= (not ==): after a merge that already leaves fewer than
        # replan_cubes, a later delivery would otherwise miss the threshold.
        self._maybe_start_replan(now)

        return cube, now

    def _maybe_start_replan(self, now: float) -> None:
        if (
            len(self.wait_pool) <= self.cfg.replan_cubes
            and self.pending is None
            and len(self.clear_pool) >= self.cfg.pool_warmup
            and len(self.wait_pool) > 0  # empty path starts infer via wait-empty
        ):
            self._start_infer(now, reason=f"replan_cubes<={self.cfg.replan_cubes}")


# ---------------------------------------------------------------------------
# Client + Environment + discrete orchestration
# ---------------------------------------------------------------------------


def run_sim(cfg: SimConfig) -> Timeline:
    log = Timeline()
    server = Server(cfg=cfg, log=log)

    t = 0.0
    n = 0  # environment observation
    interactions = 0

    log.add(
        LogRow(
            t=0.0,
            event="SIM_START",
            n=0,
            detail="",
            wait="[]",
            clear="[]",
            infer="idle",
            delivered="-",
            notes=(
                f"warmup={cfg.pool_warmup} predict={cfg.predict_cubes} "
                f"replan={cfg.replan_cubes} t_infer={cfg.t_infer} "
                f"t_env={cfg.t_env} t_comm={cfg.t_comm} n_max={cfg.n_max}"
            ),
        )
    )

    while n < cfg.n_max:
        # Client sends request
        send_t = t
        log.add(
            LogRow(
                t=send_t,
                event="CLIENT_SEND",
                n=n,
                detail=f"obs={n}",
                wait=server._wait_str(),
                clear=server._clear_str(),
                infer=server._infer_str(send_t),
                delivered="-",
                notes="",
            )
        )
        arrive_t = send_t + cfg.t_comm
        cube, reply_depart = server.handle_request(n, arrive_t)
        # If server blocked, reply_depart may be >> arrive_t
        recv_t = reply_depart + cfg.t_comm
        t = recv_t
        log.add(
            LogRow(
                t=recv_t,
                event="CLIENT_RECV",
                n=n,
                detail=f"actions={cube.actions}",
                wait=server._wait_str(),
                clear=server._clear_str(),
                infer=server._infer_str(recv_t),
                delivered=cube.short(),
                notes=f"source={cube.source}",
            )
        )

        # Execute one cube open-loop (horizon == video_stride)
        env_start = t
        # Inference may finish during env step — merge at end of env for logging
        # (server merges lazily on next request / explicit check).
        t = env_start + cfg.t_env
        server._maybe_merge(t)

        n_before = n
        n = n + 1  # fixed growth per interaction
        interactions += 1
        log.add(
            LogRow(
                t=t,
                event="ENV_STEP",
                n=n,
                detail=f"{n_before} -> {n}",
                wait=server._wait_str(),
                clear=server._clear_str(),
                infer=server._infer_str(t),
                delivered=cube.short(),
                notes=f"exec {cube.actions}",
            )
        )

    log.add(
        LogRow(
            t=t,
            event="SIM_END",
            n=n,
            detail="",
            wait=server._wait_str(),
            clear=server._clear_str(),
            infer=server._infer_str(t),
            delivered="-",
            notes=f"interactions={interactions} wall={t:.3f}s",
        )
    )
    return log


def summarize(log: Timeline, cfg: SimConfig) -> str:
    events = {}
    for r in log.rows:
        events[r.event] = events.get(r.event, 0) + 1
    blocks = [r for r in log.rows if r.event == "CLIENT_BLOCK"]
    merges = [r for r in log.rows if r.event == "INFER_MERGE"]
    starts = [r for r in log.rows if r.event == "INFER_START"]
    end = log.rows[-1]
    lines = [
        "=== Summary ===",
        f"wall_time={end.t:.3f}s  final_n={end.n}  event_counts={events}",
        f"infer_starts={len(starts)}  infer_merges={len(merges)}  client_blocks={len(blocks)}",
        "",
        "First few INFER_START / INFER_MERGE:",
    ]
    shown = 0
    for r in log.rows:
        if r.event in ("INFER_START", "INFER_MERGE", "CLIENT_BLOCK"):
            lines.append(
                f"  t={r.t:.3f} {r.event}: {r.notes or r.detail} | wait={r.wait}"
            )
            shown += 1
            if shown >= 24:
                lines.append("  ...")
                break
    if blocks:
        lines.append(
            f"\nClient blocked {len(blocks)} time(s) — wait emptied before prefetch finished "
            f"(t_infer={cfg.t_infer} vs t_env={cfg.t_env}, replan={cfg.replan_cubes})."
        )
    else:
        lines.append(
            "\nNo client blocks — prefetch always refilled wait before it drained."
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-max", type=int, default=100)
    p.add_argument("--replan-cubes", type=int, default=2)
    p.add_argument("--t-infer", type=float, default=0.24)
    p.add_argument("--t-env", type=float, default=0.13)
    p.add_argument("--t-comm", type=float, default=0.001)
    p.add_argument("--pool-warmup", type=int, default=5)
    p.add_argument("--predict-cubes", type=int, default=4)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--quiet", action="store_true", help="only print summary")
    args = p.parse_args()

    cfg = SimConfig(
        pool_warmup=args.pool_warmup,
        predict_cubes=args.predict_cubes,
        replan_cubes=args.replan_cubes,
        t_infer=args.t_infer,
        t_env=args.t_env,
        t_comm=args.t_comm,
        n_max=args.n_max,
    )
    log = run_sim(cfg)
    if not args.quiet:
        print(log.dump_text())
        print()
    print(summarize(log, cfg))
    if args.csv:
        log.dump_csv(args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
