#!/usr/bin/env python3
"""VRTC cube-prefetch: **real** three-party wall-clock interaction.

Unlike ``vrtc_topology_sim.py`` (virtual clock, no sleep), this script runs
three concurrent actors that really ``time.sleep``:

* **Environment** — receives one cube of actions, sleeps ``t_env``, ``n += 1``
* **Server**     — cube pool + prefetch; inference thread sleeps ``t_infer``
* **Client**     — obs → server (``t_comm`` each way) → exec cube on env

Wall-clock timestamps come from ``time.perf_counter()`` relative to start.

Run::

    python examples/vrtc_topology_realtime.py --n-max 20
    python examples/vrtc_topology_realtime.py --n-max 12 --quiet
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Shared toy bits (same semantics as the discrete sim)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cfg:
    pool_warmup: int = 5
    predict_cubes: int = 4
    video_stride: int = 4
    replan_cubes: int = 2
    t_infer: float = 0.24
    t_env: float = 0.13
    t_comm: float = 0.001
    n_max: int = 20


def cube_actions(frame_id: int) -> Tuple[float, ...]:
    return tuple(round(frame_id - 0.4 + 0.1 * i, 1) for i in range(4))


@dataclass
class Cube:
    frame_id: int
    actions: Tuple[float, ...]
    source: str = "pred"

    def short(self) -> str:
        return f"{self.frame_id}{self.source[0]}"


def toy_predict(clear_frames: Sequence[int], predict_cubes: int) -> List[Cube]:
    last = clear_frames[-1]
    return [
        Cube(frame_id=last + i, actions=cube_actions(last + i), source="pred")
        for i in range(1, predict_cubes + 1)
    ]


# ---------------------------------------------------------------------------
# Wall-clock log
# ---------------------------------------------------------------------------


@dataclass
class LogRow:
    t: float
    who: str
    event: str
    n: Optional[int]
    detail: str


class ClockLog:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.rows: List[LogRow] = []

    def now(self) -> float:
        return time.perf_counter() - self._t0

    def add(self, who: str, event: str, n: Optional[int] = None, detail: str = "") -> float:
        t = self.now()
        with self._lock:
            self.rows.append(LogRow(t=t, who=who, event=event, n=n, detail=detail))
        return t

    def dump_text(self) -> str:
        with self._lock:
            rows = list(self.rows)
        lines = [
            f"{'t_wall':>8}  {'who':<6}  {'event':<16}  {'n':>4}  detail",
            "-" * 100,
        ]
        for r in rows:
            nn = "-" if r.n is None else str(r.n)
            lines.append(f"{r.t:8.3f}  {r.who:<6}  {r.event:<16}  {nn:>4}  {r.detail}")
        return "\n".join(lines)

    def dump_csv(self, path: str) -> None:
        with self._lock:
            rows = list(self.rows)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_wall", "who", "event", "n", "detail"])
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "t_wall": f"{r.t:.6f}",
                        "who": r.who,
                        "event": r.event,
                        "n": "" if r.n is None else r.n,
                        "detail": r.detail,
                    }
                )


# ---------------------------------------------------------------------------
# Environment actor
# ---------------------------------------------------------------------------


class Environment(threading.Thread):
    def __init__(self, cfg: Cfg, log: ClockLog, action_q: queue.Queue, obs_q: queue.Queue):
        super().__init__(name="env", daemon=True)
        self.cfg = cfg
        self.log = log
        self.action_q = action_q
        self.obs_q = obs_q
        self.n = 0
        self._halt = threading.Event()

    def run(self) -> None:
        # Initial observation available immediately.
        self.obs_q.put(self.n)
        self.log.add("env", "INIT_OBS", self.n, "n=0 ready")
        while not self._halt.is_set():
            try:
                item = self.action_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            cube: Cube = item
            self.log.add("env", "EXEC_BEGIN", self.n, f"cube={cube.short()} actions={cube.actions}")
            time.sleep(self.cfg.t_env)  # REAL sleep
            self.n += 1
            self.log.add("env", "EXEC_DONE", self.n, f"after {cube.short()}")
            self.obs_q.put(self.n)

    def stop(self) -> None:
        self._halt.set()
        self.action_q.put(None)


# ---------------------------------------------------------------------------
# Server actor (request loop + background infer threads)
# ---------------------------------------------------------------------------


@dataclass
class _InferSpec:
    job_id: int
    clear_snapshot: Tuple[int, ...]
    origin_last_clear: int


class Server(threading.Thread):
    def __init__(self, cfg: Cfg, log: ClockLog, req_q: queue.Queue, resp_q: queue.Queue):
        super().__init__(name="server", daemon=True)
        self.cfg = cfg
        self.log = log
        self.req_q = req_q
        self.resp_q = resp_q
        self._halt = threading.Event()

        self._lock = threading.Lock()
        self.clear_pool: List[int] = []
        self.wait_pool: List[Cube] = []
        self.last_delivered_frame: Optional[int] = None
        self._job_seq = 0
        self._pending: Optional[_InferSpec] = None
        self._infer_cv = threading.Condition(self._lock)
        self._done_q: queue.Queue = queue.Queue()  # finished InferSpec + cubes

    def run(self) -> None:
        # Drain completed inferences in a helper watcher? We merge lazily on request
        # and also via a small poller so merges happen during client/env sleep.
        poller = threading.Thread(target=self._poll_done, name="infer-poller", daemon=True)
        poller.start()
        while not self._halt.is_set():
            try:
                item = self.req_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            obs_n, req_id = item
            cube = self._handle(obs_n, req_id)
            self.resp_q.put((req_id, cube))

    def stop(self) -> None:
        self._halt.set()
        self.req_q.put(None)

    def _poll_done(self) -> None:
        while not self._halt.is_set():
            try:
                job, predicted = self._done_q.get(timeout=0.05)
            except queue.Empty:
                continue
            with self._infer_cv:
                self._apply_merge(job, predicted)
                self._infer_cv.notify_all()

    def _apply_merge(self, job: _InferSpec, predicted: List[Cube]) -> None:
        # Caller holds self._lock
        if self._pending is None or self._pending.job_id != job.job_id:
            # Stale / already cleared
            return
        self._pending = None
        if self.last_delivered_frame is None:
            skip = 0
        else:
            skip = max(0, self.last_delivered_frame - job.origin_last_clear)
        kept = predicted[skip:]
        old = "[" + ",".join(c.short() for c in self.wait_pool) + "]"
        self.wait_pool = kept
        new = "[" + ",".join(c.short() for c in self.wait_pool) + "]"
        self.log.add(
            "server",
            "INFER_MERGE",
            None,
            f"job#{job.job_id} pred={[c.frame_id for c in predicted]} skip={skip} "
            f"{old}->{new} (after real {job.origin_last_clear})",
        )
        self._maybe_start_replan_locked()

    def _start_infer_locked(self, reason: str) -> None:
        assert self._pending is None
        snap = tuple(self.clear_pool[-self.cfg.pool_warmup :])
        self._job_seq += 1
        job = _InferSpec(
            job_id=self._job_seq,
            clear_snapshot=snap,
            origin_last_clear=snap[-1],
        )
        self._pending = job
        self.log.add(
            "server",
            "INFER_START",
            snap[-1],
            f"job#{job.job_id} reason={reason} clear={list(snap)} "
            f"expect={[snap[-1]+i for i in range(1, self.cfg.predict_cubes+1)]}",
        )

        def worker() -> None:
            t0 = time.perf_counter()
            time.sleep(self.cfg.t_infer)  # REAL sleep — model latency
            predicted = toy_predict(job.clear_snapshot, self.cfg.predict_cubes)
            elapsed = time.perf_counter() - t0
            self.log.add(
                "server",
                "INFER_DONE",
                None,
                f"job#{job.job_id} slept={elapsed:.3f}s frames={[c.frame_id for c in predicted]}",
            )
            self._done_q.put((job, predicted))

        threading.Thread(target=worker, name=f"infer-{job.job_id}", daemon=True).start()

    def _maybe_start_replan_locked(self) -> None:
        if (
            self._pending is None
            and len(self.clear_pool) >= self.cfg.pool_warmup
            and 0 < len(self.wait_pool) <= self.cfg.replan_cubes
        ):
            self._start_infer_locked(reason=f"replan_cubes<={self.cfg.replan_cubes}")

    def _handle(self, obs_n: int, req_id: int) -> Cube:
        with self._infer_cv:
            self.log.add("server", "REQ_ARRIVE", obs_n, f"req_id={req_id}")

            # Warmup
            if len(self.clear_pool) < self.cfg.pool_warmup:
                self.clear_pool.append(obs_n)
                cube = Cube(frame_id=obs_n, actions=cube_actions(obs_n), source="warmup")
                self.last_delivered_frame = obs_n
                self.log.add(
                    "server",
                    "DELIVER",
                    obs_n,
                    f"warmup {cube.short()} pool={len(self.clear_pool)}/{self.cfg.pool_warmup}",
                )
                if len(self.clear_pool) == self.cfg.pool_warmup and self._pending is None:
                    self._start_infer_locked(reason="post-warmup")
                return cube

            if not self.clear_pool or self.clear_pool[-1] != obs_n:
                self.clear_pool.append(obs_n)
                cap = self.cfg.pool_warmup + self.cfg.predict_cubes
                if len(self.clear_pool) > cap:
                    self.clear_pool = self.clear_pool[-cap:]

            # Block for real until wait has a cube (infer thread wakes us).
            blocked = False
            while not self.wait_pool:
                if self._pending is None:
                    self._start_infer_locked(reason="wait-empty")
                if not blocked:
                    self.log.add("server", "CLIENT_BLOCK", obs_n, "wait empty — parking on CV")
                    blocked = True
                self._infer_cv.wait(timeout=0.05)

            cube = self.wait_pool.pop(0)
            self.last_delivered_frame = cube.frame_id
            self.log.add(
                "server",
                "DELIVER",
                obs_n,
                f"{cube.short()} remaining={len(self.wait_pool)}",
            )
            self._maybe_start_replan_locked()
            return cube


# ---------------------------------------------------------------------------
# Client actor
# ---------------------------------------------------------------------------


class Client(threading.Thread):
    def __init__(
        self,
        cfg: Cfg,
        log: ClockLog,
        obs_q: queue.Queue,
        action_q: queue.Queue,
        req_q: queue.Queue,
        resp_q: queue.Queue,
        done_event: threading.Event,
    ):
        super().__init__(name="client", daemon=True)
        self.cfg = cfg
        self.log = log
        self.obs_q = obs_q
        self.action_q = action_q
        self.req_q = req_q
        self.resp_q = resp_q
        self.done_event = done_event
        self.interactions = 0
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            req_id = 0
            while True:
                obs = self.obs_q.get()
                if obs >= self.cfg.n_max:
                    self.log.add("client", "STOP", obs, f"n_max={self.cfg.n_max}")
                    break

                self.log.add("client", "GOT_OBS", obs, "")
                time.sleep(self.cfg.t_comm)  # REAL one-way to server
                req_id += 1
                self.req_q.put((obs, req_id))
                self.log.add("client", "SEND_REQ", obs, f"req_id={req_id}")

                rid, cube = self.resp_q.get()
                assert rid == req_id
                time.sleep(self.cfg.t_comm)  # REAL one-way back
                self.log.add(
                    "client",
                    "GOT_CUBE",
                    obs,
                    f"{cube.short()} actions={cube.actions} source={cube.source}",
                )

                self.action_q.put(cube)
                self.interactions += 1
        except BaseException as e:
            self.error = e
            self.log.add("client", "ERROR", None, repr(e))
        finally:
            self.done_event.set()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_realtime(cfg: Cfg) -> ClockLog:
    log = ClockLog()
    log.add("main", "START", None, f"cfg={cfg}")

    obs_q: queue.Queue = queue.Queue()
    action_q: queue.Queue = queue.Queue()
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    done = threading.Event()

    env = Environment(cfg, log, action_q, obs_q)
    server = Server(cfg, log, req_q, resp_q)
    client = Client(cfg, log, obs_q, action_q, req_q, resp_q, done)

    wall0 = time.perf_counter()
    env.start()
    server.start()
    client.start()

    # Wait until client finishes (or timeout: generous bound).
    # Worst case ~ n_max * (t_env + 2*t_comm) + a few t_infer blocks.
    timeout = cfg.n_max * (cfg.t_env + 2 * cfg.t_comm + cfg.t_infer) + 5.0
    ok = done.wait(timeout=timeout)
    wall = time.perf_counter() - wall0

    client.join(timeout=1.0)
    server.stop()
    env.stop()
    server.join(timeout=1.0)
    env.join(timeout=1.0)

    if client.error is not None:
        raise RuntimeError(f"client failed: {client.error}") from client.error
    if not ok:
        raise TimeoutError(f"realtime run exceeded {timeout:.1f}s")

    log.add(
        "main",
        "END",
        None,
        f"interactions={client.interactions} wall_sleep_time={wall:.3f}s",
    )
    return log


def summarize(log: ClockLog, cfg: Cfg) -> str:
    rows = list(log.rows)
    counts = {}
    for r in rows:
        counts[r.event] = counts.get(r.event, 0) + 1
    end = next(r for r in reversed(rows) if r.event == "END")
    blocks = counts.get("CLIENT_BLOCK", 0)
    infer_s = counts.get("INFER_START", 0)
    infer_d = counts.get("INFER_DONE", 0)
    merges = counts.get("INFER_MERGE", 0)
    # Measure actual sleep span from first to last env EXEC
    execs = [r for r in rows if r.event == "EXEC_DONE"]
    span = (execs[-1].t - execs[0].t) if len(execs) >= 2 else 0.0
    lines = [
        "=== Realtime summary ===",
        f"wall_clock={end.detail}  event_counts={counts}",
        f"infer_start={infer_s} infer_done={infer_d} merge={merges} client_block_logs={blocks}",
        f"env EXEC_DONE span={span:.3f}s over {len(execs)} steps "
        f"(expect ~{(len(execs)-1)*cfg.t_env:.3f}s of pure env sleep if contiguous)",
        "",
        "INFER / BLOCK excerpts:",
    ]
    n = 0
    for r in rows:
        if r.event in ("INFER_START", "INFER_DONE", "INFER_MERGE", "CLIENT_BLOCK"):
            lines.append(f"  t={r.t:.3f} [{r.who}] {r.event}: {r.detail}")
            n += 1
            if n >= 20:
                lines.append("  ...")
                break
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--replan-cubes", type=int, default=2)
    p.add_argument("--t-infer", type=float, default=0.24)
    p.add_argument("--t-env", type=float, default=0.13)
    p.add_argument("--t-comm", type=float, default=0.001)
    p.add_argument("--pool-warmup", type=int, default=5)
    p.add_argument("--predict-cubes", type=int, default=4)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = Cfg(
        pool_warmup=args.pool_warmup,
        predict_cubes=args.predict_cubes,
        replan_cubes=args.replan_cubes,
        t_infer=args.t_infer,
        t_env=args.t_env,
        t_comm=args.t_comm,
        n_max=args.n_max,
    )
    print(f"[realtime] starting 3-party wall-clock run n_max={cfg.n_max} ...", flush=True)
    t0 = time.perf_counter()
    log = run_realtime(cfg)
    elapsed = time.perf_counter() - t0
    print(f"[realtime] finished in {elapsed:.3f}s wall clock\n", flush=True)

    if not args.quiet:
        print(log.dump_text())
        print()
    print(summarize(log, cfg))
    if args.csv:
        log.dump_csv(args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
