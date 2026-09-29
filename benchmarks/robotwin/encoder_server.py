"""UMT5-only encoder process for batched RoboTwin eval.

One process, one GPU. The inference process does not load T5; it asks this
process for ``(context, seq_lens)`` when a slot's instruction changes.

Wire format (TCP, little-endian lengths, localhost):
    uint32 json_len | json header | uint32 payload_len | pickle payload

Requests:
    {"type": "encode", "prompt": "..."}
    {"type": "ping"}
Responses:
    {"type": "embedding", "seq_len": int}
        payload = {"context": Tensor [L, D], "seq_lens": Tensor scalar}
    {"type": "pong"}
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import socket
import struct
import sys
import threading
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from openwam.model.video_backbone.wan.encode import encode_text  # noqa: E402
from precompute_robotwin_t5_embeds import DEFAULT_WAN_PATH, _load_text_stack  # noqa: E402

logger = logging.getLogger("encoder_server")


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("encoder socket closed")
        buf += chunk
    return buf


def _read_message(conn: socket.socket) -> tuple[dict, object]:
    header_len = struct.unpack("<I", _recv_exact(conn, 4))[0]
    header = json.loads(_recv_exact(conn, header_len).decode("utf-8"))
    payload_len = struct.unpack("<I", _recv_exact(conn, 4))[0]
    payload = pickle.loads(_recv_exact(conn, payload_len)) if payload_len else None
    return header, payload


def _write_message(conn: socket.socket, header: dict, payload: object = None) -> None:
    body = b"" if payload is None else pickle.dumps(payload, protocol=4)
    raw_header = json.dumps(header).encode("utf-8")
    conn.sendall(struct.pack("<I", len(raw_header)) + raw_header + struct.pack("<I", len(body)) + body)


class EncoderServer:
    def __init__(self, tokenizer, text_encoder, device: torch.device):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()

    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            hit = self._cache.get(prompt)
            if hit is not None:
                return hit
            context, seq_lens = encode_text(
                [prompt],
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
            )
            # [1, L, D] -> [L, D] on CPU so the inference process owns the GPU copy.
            context_cpu = context[0].detach().to("cpu").contiguous()
            seq_cpu = seq_lens[0].detach().to("cpu")
            self._cache[prompt] = (context_cpu, seq_cpu)
            return context_cpu, seq_cpu

    def handle(self, conn: socket.socket) -> None:
        try:
            while True:
                header, _payload = _read_message(conn)
                kind = header.get("type")
                if kind == "ping":
                    _write_message(conn, {"type": "pong"})
                elif kind == "encode":
                    prompt = str(header.get("prompt") or "")
                    context, seq_lens = self.encode(prompt)
                    _write_message(
                        conn,
                        {"type": "embedding", "seq_len": int(seq_lens.item())},
                        {"context": context, "seq_lens": seq_lens},
                    )
                else:
                    _write_message(conn, {"type": "error", "message": f"unknown type {kind!r}"})
        except ConnectionError:
            return


def serve(host: str, port: int, server: EncoderServer) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(8)
    logger.info("T5 encoder listening on %s:%d", host, port)
    while True:
        conn, addr = sock.accept()
        logger.info("encoder client %s", addr)
        threading.Thread(target=_serve_conn, args=(conn, server), daemon=True).start()


def _serve_conn(conn: socket.socket, server: EncoderServer) -> None:
    with conn:
        server.handle(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenWAM UMT5 encoder process")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ckpt-dir", required=True, help="OpenWAM checkpoint dir (tokenizer fallback)")
    parser.add_argument("--wan-path", default=DEFAULT_WAN_PATH)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    device = torch.device(args.device)
    tokenizer, text_encoder = _load_text_stack(args.wan_path, args.ckpt_dir, device)
    serve(args.host, args.port, EncoderServer(tokenizer, text_encoder, device))


if __name__ == "__main__":
    main()
