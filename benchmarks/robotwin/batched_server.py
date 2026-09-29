"""Batched RoboTwin policy server: business thread + inference thread.

One process serves ``n`` simulation clients on one GPU. The inference thread
blocks until at least one slot has a fresh condition frame, then calls
``generate_batch`` on those slots only. Action chunks are consumed the same
way as the sync executor: a new denoise runs only when a slot's buffer is empty.

The original ``PolicyServer`` / ``generate()`` path is not used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("batched_server")


class EncoderClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.create_connection((self.host, self.port), timeout=600)
        sock.settimeout(600)
        self._sock = sock
        return sock

    def _reset(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    def encode(self, prompt: str):
        import torch

        with self._lock:
            for attempt in range(2):
                try:
                    sock = self._connect()
                    _write_message(sock, {"type": "encode", "prompt": prompt})
                    header, payload = _read_message(sock)
                    break
                except (ConnectionError, OSError, TimeoutError):
                    self._reset()
                    if attempt == 1:
                        raise
            if header.get("type") != "embedding":
                raise RuntimeError(f"encoder error: {header}")
            context = payload["context"]
            seq_lens = payload["seq_lens"]
            if not torch.is_tensor(context):
                raise TypeError(f"encoder context is {type(context)}")
            return context, seq_lens

    def ping(self) -> bool:
        with self._lock:
            try:
                sock = self._connect()
                _write_message(sock, {"type": "ping"})
                header, _payload = _read_message(sock)
                return header.get("type") == "pong"
            except (ConnectionError, OSError, TimeoutError):
                self._reset()
                return False


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("encoder socket closed")
        buf += chunk
    return buf


def _read_message(conn: socket.socket):
    header_len = struct.unpack("<I", _recv_exact(conn, 4))[0]
    header = json.loads(_recv_exact(conn, header_len).decode("utf-8"))
    payload_len = struct.unpack("<I", _recv_exact(conn, 4))[0]
    payload = pickle.loads(_recv_exact(conn, payload_len)) if payload_len else None
    return header, payload


def _write_message(conn: socket.socket, header: dict, payload=None) -> None:
    body = b"" if payload is None else pickle.dumps(payload, protocol=4)
    raw = json.dumps(header).encode("utf-8")
    conn.sendall(struct.pack("<I", len(raw)) + raw + struct.pack("<I", len(body)) + body)


class Slot:
    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.prompt = None
        self.context = None
        self.seq_lens = None
        self.image = None
        self.proprio = None
        self.dirty = False
        self.actions: deque = deque()
        self.waiter = None
        self.step = 0
        self.t0 = 0.0


class BatchedPolicy:
    def __init__(
        self,
        architecture,
        cfg,
        encoder: EncoderClient,
        n_slots: int,
        loop: asyncio.AbstractEventLoop,
        max_batch: int = 0,
    ):
        from openwam.deploy.denoise_schedule import make_schedule
        from openwam.deploy.obs_preprocess import ObsPreprocessor

        self.arch = architecture
        self.cfg = cfg
        self.encoder = encoder
        self.loop = loop
        self.max_batch = int(max_batch) if max_batch else 0
        self.obs = ObsPreprocessor.from_cfg(cfg)
        self.slots = [Slot(i) for i in range(n_slots)]
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = False
        self._free_slots = deque(range(n_slots))
        self._conn_slot: dict[int, int] = {}

        inf = cfg.inference
        self.denoise_steps = int(inf.denoise_steps)
        self.action_num_frames = int(inf.num_frames)
        self.video_num_frames = int(getattr(inf, "video_num_frames", self.action_num_frames))
        self.height = int(inf.height)
        self.width = int(inf.width)
        horizon = getattr(inf, "inference_horizon", None)
        self.inference_horizon = None if horizon is None else int(horizon)
        opt = getattr(cfg, "optimization", None)
        dc = getattr(opt, "dit_cache", None) if opt is not None else None
        self.dit_cache_cfg = None
        if dc is not None and bool(getattr(dc, "enabled", False)):
            self.dit_cache_cfg = {
                "enabled": True,
                "cosine_threshold": float(getattr(dc, "cosine_threshold", 0.99)),
                "max_skips": int(getattr(dc, "max_skips", 3)),
            }
        ab = getattr(architecture, "action_backbone", None)
        self.shift = float(getattr(ab, "shift_action", None) or 5.0)
        vb = getattr(architecture, "video_backbone", None)
        self.shift_video = getattr(vb, "shift_video", None) if vb is not None else None
        self.schedule = make_schedule(
            getattr(inf, "denoise_mode", "sync"),
            video_scheduler=architecture.video_scheduler,
            action_scheduler=architecture.action_scheduler,
            num_steps=self.denoise_steps,
            shift=self.shift,
            shift_video=self.shift_video,
            lead=getattr(inf, "lead_modality", "video"),
            alpha=float(getattr(inf, "variance_shift_alpha", 1.0)),
            offset=float(getattr(inf, "linear_offset", 0.0)),
        )
        self.binary_dims = tuple(getattr(architecture, "binary_command_dims", ()) or ())
        self.thread = threading.Thread(target=self._infer_loop, name="infer", daemon=True)
        self.thread.start()
        logger.info(
            "batched policy n=%d max_batch=%s denoise_steps=%d horizon=%s dit_cache=%s",
            n_slots,
            self.max_batch or "all",
            self.denoise_steps,
            self.inference_horizon,
            bool(self.dit_cache_cfg),
        )

    def bind_slot(self, conn_id: int, requested) -> int:
        with self.lock:
            if conn_id in self._conn_slot:
                bound = self._conn_slot[conn_id]
                if requested is not None and int(requested) != bound:
                    raise ValueError(f"connection already bound to slot {bound}, got {requested}")
                return bound
            if requested is None:
                if not self._free_slots:
                    raise ValueError("no free simulation slots")
                slot_id = self._free_slots.popleft()
            else:
                # Fixed ROBOTWIN_SLOT is exclusive to one sim process. On WS
                # timeout/reconnect the old handler may still hold the slot while
                # infer is in flight — always transfer ownership to the new conn
                # instead of rejecting with "already taken".
                slot_id = int(requested)
                if slot_id < 0 or slot_id >= len(self.slots):
                    raise ValueError(f"slot_id {slot_id} out of range 0..{len(self.slots) - 1}")
                stale = [cid for cid, sid in self._conn_slot.items() if sid == slot_id]
                for cid in stale:
                    del self._conn_slot[cid]
                    logger.warning(
                        "reclaimed slot %d from stale conn id=%s for conn id=%s",
                        slot_id,
                        cid,
                        conn_id,
                    )
                try:
                    self._free_slots.remove(slot_id)
                except ValueError:
                    if not stale:
                        logger.warning(
                            "force-claimed slot %d (not free, no owner) for conn id=%s",
                            slot_id,
                            conn_id,
                        )
                if stale or slot_id not in self._free_slots:
                    # Clear any in-flight waiter left by the previous owner.
                    self._clear_slot(self.slots[slot_id])
            self._conn_slot[conn_id] = slot_id
            return slot_id

    def release_conn(self, conn_id: int) -> None:
        with self.lock:
            slot_id = self._conn_slot.pop(conn_id, None)
            if slot_id is None:
                # Slot may already have been reclaimed by a reconnecting client.
                return
            slot = self.slots[slot_id]
            self._clear_slot(slot)
            if slot_id not in self._free_slots:
                self._free_slots.append(slot_id)

    def _clear_slot(self, slot: Slot) -> None:
        slot.prompt = None
        slot.context = None
        slot.seq_lens = None
        slot.image = None
        slot.proprio = None
        slot.dirty = False
        slot.actions.clear()
        waiter = slot.waiter
        slot.waiter = None
        if waiter is not None and not waiter.done():
            self.loop.call_soon_threadsafe(waiter.cancel)

    def reset_slot(self, slot_id: int) -> None:
        with self.lock:
            self._clear_slot(self.slots[slot_id])

    async def predict(self, slot_id: int, obs: dict) -> dict:
        t0 = time.monotonic()
        obs = self.obs.preprocess(obs)
        prompt = obs.get("prompt") or ""
        slot = self.slots[slot_id]
        if slot.prompt != prompt or slot.context is None:
            context, seq_lens = await asyncio.to_thread(self.encoder.encode, prompt)
            with self.lock:
                if slot.prompt != prompt:
                    slot.actions.clear()
                slot.prompt = prompt
                slot.context = context
                slot.seq_lens = seq_lens

        loop = asyncio.get_running_loop()
        with self.lock:
            if slot.actions:
                chunk = list(slot.actions)
                slot.actions.clear()
                slot.step += len(chunk)
                return self._pack_chunk(slot, chunk, t0)
            fut = loop.create_future()
            slot.waiter = fut
            slot.image = obs["image"]
            state = obs.get("state")
            slot.proprio = None if state is None else self.arch.normalize_deploy_proprio(state)
            slot.dirty = True
            slot.t0 = t0
            self.wake.set()
        try:
            chunk = await fut
        except asyncio.CancelledError:
            # Slot was reclaimed by a reconnecting client; surface as a normal
            # request error so the old handler can exit without killing the loop.
            raise RuntimeError(f"slot {slot_id} reclaimed during inference") from None
        return self._pack_chunk(slot, chunk, t0)

    def _pack_chunk(self, slot: Slot, actions, t0: float) -> dict:
        projected = [self._project(np.asarray(action)).reshape(-1).tolist() for action in actions]
        if not projected:
            raise RuntimeError("empty action chunk")
        return {
            "action": projected[0],
            "actions": projected,
            "step": slot.step,
            "latency_ms": round((time.monotonic() - t0) * 1000, 2),
        }

    def _project(self, action: np.ndarray) -> np.ndarray:
        if not self.binary_dims:
            return action
        action = np.array(action)
        for dim in self.binary_dims:
            action[..., dim] = np.where(action[..., dim] > 0.5, 1.0, -1.0)
        return action

    def _infer_loop(self) -> None:
        while not self.stop:
            with self.lock:
                ready = [s for s in self.slots if s.dirty and s.image is not None and s.context is not None]
                # Cap concurrent denoise width so encoder+policy fit one 96GB card.
                if self.max_batch > 0 and len(ready) > self.max_batch:
                    ready = ready[: self.max_batch]
                for slot in ready:
                    slot.dirty = False
                snapshot = [
                    {
                        "slot": slot,
                        "waiter": slot.waiter,
                        "first_frame_image": [slot.image],
                        "context": slot.context,
                        "seq_lens": slot.seq_lens,
                        "proprio": slot.proprio,
                        "prompt": slot.prompt or "",
                        "seed": 42,
                    }
                    for slot in ready
                ]
            if not snapshot:
                self.wake.wait()
                self.wake.clear()
                continue
            slot_ids = ",".join(str(item["slot"].slot_id) for item in snapshot)
            n_slots = len(snapshot)
            t_infer = time.perf_counter()
            logger.info("[timing] infer start n=%d slots=%s", n_slots, slot_ids)
            try:
                result = self.arch.generate_batch(
                    snapshot,
                    schedule=self.schedule,
                    action_num_frames=self.action_num_frames,
                    video_num_frames=self.video_num_frames,
                    height=self.height,
                    width=self.width,
                    denoise_steps=self.denoise_steps,
                    shift=self.shift,
                    dit_cache_cfg=self.dit_cache_cfg,
                )
                actions = result["actions"]
                logger.info(
                    "[timing] infer end n=%d slots=%s elapsed_s=%.3f ok=1",
                    n_slots,
                    slot_ids,
                    time.perf_counter() - t_infer,
                )
            except Exception as exc:  # noqa: BLE001 — deliver to the waiting sims
                logger.info(
                    "[timing] infer end n=%d slots=%s elapsed_s=%.3f ok=0",
                    n_slots,
                    slot_ids,
                    time.perf_counter() - t_infer,
                )
                logger.exception("batched generate failed")
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                # OOM on a wide batch: shrink max_batch and re-queue waiters instead of
                # killing every sim (which aborts the whole eval fan-out).
                is_oom = "out of memory" in str(exc).lower()
                if is_oom and n_slots > 1:
                    new_cap = max(1, n_slots // 2)
                    with self.lock:
                        if self.max_batch <= 0 or self.max_batch > new_cap:
                            self.max_batch = new_cap
                        for item in snapshot:
                            slot = item["slot"]
                            if slot.waiter is item["waiter"]:
                                slot.dirty = True
                    logger.warning(
                        "OOM on n=%d; lowered max_batch=%d and re-queued slots=%s",
                        n_slots,
                        self.max_batch,
                        slot_ids,
                    )
                    continue
                for item in snapshot:
                    self._fail_waiter(item["slot"], exc, expected=item["waiter"])
                continue

            for row, chunk_row in enumerate(actions):
                meta = snapshot[row]
                slot = meta["slot"]
                chunk = np.asarray(chunk_row)
                horizon = self.inference_horizon if self.inference_horizon is not None else len(chunk)
                if horizon > len(chunk):
                    self._fail_waiter(
                        slot,
                        ValueError(f"inference_horizon ({horizon}) > action chunk ({len(chunk)})"),
                        expected=meta["waiter"],
                    )
                    continue
                horizon_chunk = [np.asarray(step) for step in chunk[:horizon]]
                with self.lock:
                    if slot.waiter is not meta["waiter"]:
                        continue
                    slot.actions.clear()
                    slot.step += len(horizon_chunk)
                    waiter = slot.waiter
                    slot.waiter = None
                if waiter is not None and not waiter.done():
                    self.loop.call_soon_threadsafe(waiter.set_result, horizon_chunk)

    def _fail_waiter(self, slot: Slot, exc: BaseException, expected=None) -> None:
        with self.lock:
            # Skip if the slot was reclaimed / rebound to a newer waiter.
            if expected is not None and slot.waiter is not expected:
                return
            waiter = slot.waiter
            slot.waiter = None
            slot.dirty = False
        if waiter is not None and not waiter.done():
            self.loop.call_soon_threadsafe(waiter.set_exception, exc)


def _ckpt_contract(architecture) -> dict:
    contract = getattr(architecture, "repr_contract", None)
    return dict(contract) if contract else {}


async def _run(host: str, port: int, policy: BatchedPolicy, contract: dict) -> None:
    import websockets

    from openwam.deploy.server import (
        ACTION,
        ERROR,
        ERR_INTERNAL,
        ERR_OBS_VALIDATION,
        ERR_UNKNOWN_TYPE,
        MAX_MESSAGE_BYTES,
        OBS,
        PING,
        PONG,
        RESET,
        RESET_ACK,
    )
    from openwam.deploy.obs_preprocess import ObsValidationError

    conn_ids = itertools_count()

    async def handler(websocket):
        conn_id = next(conn_ids)
        logger.info("client connected %s id=%s", websocket.remote_address, conn_id)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    kind = data.get("type", OBS)
                    if kind == PING:
                        await websocket.send(json.dumps({"type": PONG, **contract}))
                        continue
                    slot_id = policy.bind_slot(conn_id, data.get("slot_id"))
                    if kind == RESET:
                        policy.reset_slot(slot_id)
                        await websocket.send(json.dumps({"type": RESET_ACK}))
                    elif kind == OBS:
                        result = await policy.predict(slot_id, data)
                        result["type"] = ACTION
                        await websocket.send(json.dumps(result))
                    else:
                        await websocket.send(
                            json.dumps({"type": ERROR, "code": ERR_UNKNOWN_TYPE, "message": kind})
                        )
                except ObsValidationError as exc:
                    await websocket.send(
                        json.dumps({"type": ERROR, "code": ERR_OBS_VALIDATION, "message": str(exc)})
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("request failed")
                    await websocket.send(
                        json.dumps({"type": ERROR, "code": ERR_INTERNAL, "message": str(exc)})
                    )
        except websockets.exceptions.ConnectionClosed:
            logger.info("client disconnected id=%s", conn_id)
        finally:
            policy.release_conn(conn_id)

    async with websockets.serve(handler, host, port, max_size=MAX_MESSAGE_BYTES, ping_interval=None):
        logger.info("batched policy server ws://%s:%d", host, port)
        await asyncio.Future()


def itertools_count():
    import itertools

    return itertools.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenWAM batched RoboTwin policy server")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--encoder-host", default="127.0.0.1")
    parser.add_argument("--encoder-port", type=int, required=True)
    parser.add_argument("--n-slots", type=int, required=True)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=0,
        help="Max slots per generate_batch call (0 = all ready). Use 1-4 when sharing GPU with encoder.",
    )
    parser.add_argument("--denoise-steps", type=int, default=None)
    args = parser.parse_args()
    if args.n_slots < 1:
        parser.error("--n-slots must be >= 1")
    if args.max_batch < 0:
        parser.error("--max-batch must be >= 0")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from omegaconf import OmegaConf

    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import (
        _apply_compile_enabled_override,
        _load_deploy_yaml,
        _normalize_compile_enabled_in_cfg,
        _validate_inference_config,
        merge_deploy_cfg,
    )
    from openwam.deploy import JointInferenceEngine

    cfg = _load_deploy_yaml(args.config)
    if args.denoise_steps is not None:
        OmegaConf.update(cfg, "inference.denoise_steps", int(args.denoise_steps), merge=False)
    _validate_inference_config(cfg)
    _normalize_compile_enabled_in_cfg(cfg)
    _apply_compile_enabled_override(cfg, None)

    encoder = EncoderClient(args.encoder_host, args.encoder_port)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if encoder.ping():
            break
        time.sleep(1)
    else:
        raise SystemExit(f"encoder at {args.encoder_host}:{args.encoder_port} did not answer ping")

    training_cfg, architecture = load_from_checkpoint_dir(
        args.ckpt_dir,
        device=args.device,
        ckpt_name=args.ckpt_name,
        skip_text_encoder=True,
    )
    merged = merge_deploy_cfg(training_cfg, cfg)
    # Compile / dtype path matches the single-sample server. generate() is not called.
    JointInferenceEngine(cfg=merged, architecture=architecture)

    loop = asyncio.new_event_loop()
    policy = BatchedPolicy(architecture, merged, encoder, args.n_slots, loop, max_batch=args.max_batch)
    try:
        loop.run_until_complete(_run(args.host, args.port, policy, _ckpt_contract(architecture)))
    finally:
        policy.stop = True
        policy.wake.set()


if __name__ == "__main__":
    main()
