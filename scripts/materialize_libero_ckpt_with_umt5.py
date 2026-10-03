#!/usr/bin/env python3
"""Materialize a LIBERO deploy ckpt that includes UMT5 (text_encoder).

Training with ``dataloader.use_t5_cache=true`` saves checkpoints without
``video_backbone.text_encoder.*`` (UMT5 is skipped). The WebSocket deploy
server still needs UMT5 to encode prompts at inference time.

This script copies a train run dir and writes a new safetensors that merges
frozen UMT5 weights from a donor checkpoint (default: OpenWAM-Alpha-Sim-LIBERO).

Example:
  python scripts/materialize_libero_ckpt_with_umt5.py \\
    --src outputs/openwam_fpd_libero/2026-10-01_12-42-09

  # then eval against the printed --dst dir
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_DONOR = (
    "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO/"
    "checkpoint_step_10690.safetensors"
)
TE_PREFIX = "video_backbone.text_encoder."


def _latest_step_ckpt(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("checkpoint_step_*.safetensors"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint_step_*.safetensors under {run_dir}")
    return cands[-1]


def _count_te_keys(path: Path) -> int:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return sum(1 for k in handle.keys() if k.startswith(TE_PREFIX))


def materialize(*, src_dir: Path, dst_dir: Path, donor_ckpt: Path, force: bool) -> Path:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    donor_ckpt = donor_ckpt.resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"src run dir missing: {src_dir}")
    if not (src_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"missing {src_dir / 'config.yaml'}")
    if not donor_ckpt.is_file():
        raise FileNotFoundError(f"donor ckpt missing: {donor_ckpt}")

    src_ckpt = _latest_step_ckpt(src_dir)
    out_ckpt = dst_dir / src_ckpt.name

    if out_ckpt.is_file() and not force:
        n_te = _count_te_keys(out_ckpt)
        if n_te > 0:
            print(f"[materialize] reuse existing {out_ckpt} (te_keys={n_te})")
            return out_ckpt
        print(f"[materialize] existing {out_ckpt} has te_keys=0; rebuilding")

    src_te = _count_te_keys(src_ckpt)
    if src_te > 0:
        print(f"[materialize] src already has te_keys={src_te}; copying run dir as-is")
        if dst_dir.exists() and force:
            shutil.rmtree(dst_dir)
        if not dst_dir.exists():
            shutil.copytree(
                src_dir,
                dst_dir,
                ignore=shutil.ignore_patterns("checkpoint_step_*.safetensors"),
                dirs_exist_ok=False,
            )
            shutil.copy2(src_ckpt, out_ckpt)
        return out_ckpt

    donor_te = _count_te_keys(donor_ckpt)
    if donor_te <= 0:
        raise RuntimeError(f"donor has no text_encoder keys: {donor_ckpt}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy sidecar files (config / tokenizer / stats); replace weight file below.
    for name in ("config.yaml", "normalization_stats.npy"):
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
    tok = src_dir / "tokenizer"
    if tok.is_dir():
        dst_tok = dst_dir / "tokenizer"
        if dst_tok.exists():
            shutil.rmtree(dst_tok)
        shutil.copytree(tok, dst_tok)

    print(f"[materialize] loading student tensors from {src_ckpt}")
    state = {}
    with safe_open(str(src_ckpt), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)

    print(f"[materialize] injecting {donor_te} text_encoder keys from {donor_ckpt}")
    with safe_open(str(donor_ckpt), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(TE_PREFIX):
                state[key] = handle.get_tensor(key)

    n_te = sum(1 for k in state if k.startswith(TE_PREFIX))
    print(f"[materialize] writing {out_ckpt} (total_keys={len(state)} te_keys={n_te})")
    tmp = out_ckpt.with_suffix(".safetensors.tmp")
    if tmp.exists():
        tmp.unlink()
    save_file(state, str(tmp))
    tmp.replace(out_ckpt)
    print(f"[materialize] done -> {dst_dir}")
    return out_ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="train run dir (has config.yaml + ckpt)")
    parser.add_argument(
        "--dst",
        default=None,
        help="output deploy dir (default: <src>_with_umt5)",
    )
    parser.add_argument(
        "--donor",
        default=DEFAULT_DONOR,
        help="safetensors that still contains frozen UMT5 weights",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = Path(args.src)
    if not src.is_absolute():
        src = (root / src).resolve()
    dst = Path(args.dst) if args.dst else Path(str(src) + "_with_umt5")
    if not dst.is_absolute():
        dst = (root / dst).resolve()
    donor = Path(args.donor)
    if not donor.is_absolute():
        donor = (root / donor).resolve()

    out = materialize(src_dir=src, dst_dir=dst, donor_ckpt=donor, force=bool(args.force))
    print(f"CKPT_DIR={out.parent}")
    print(f"CKPT_NAME={out.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[materialize] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
