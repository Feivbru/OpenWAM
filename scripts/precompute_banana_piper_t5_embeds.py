#!/usr/bin/env python3
"""Precompute UMT5 embeddings for banana Piper (one file per task_index).

Writes ``{dataset_dir}/meta/umt5_openwam/task_{N}.pt``. Banana currently has a
single instruction, so this produces one cache file. Prompt text matches
``BananaPiperDataset`` (raw task string, no RoboTwin wrap).

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/precompute_banana_piper_t5_embeds.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openwam.model.video_backbone.wan.encode import encode_text
from precompute_robotwin_t5_embeds import _load_text_stack  # noqa: E402

logger = logging.getLogger("precompute_banana_piper_t5")

DEFAULT_DATASET_DIR = "/data/zixian_guo/projects/haoming/project/PI/riri/data/banana"
DEFAULT_WAN_PATH = (
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/assets/video_backbone_ckpt/Wan2.2-TI2V-5B"
)
DEFAULT_OPENWAM_CKPT = (
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/assets/openwam_ckpt/"
    "openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
)
CACHE_DIRNAME = "umt5_openwam"


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _load_task_prompts(dataset_dir: Path) -> list[tuple[int, str]]:
    """Mirror BananaPiperDataset._load_prompts (tasks.jsonl / tasks.parquet)."""
    tasks_parquet = dataset_dir / "meta" / "tasks.parquet"
    if tasks_parquet.is_file():
        import pandas as pd

        tasks_df = pd.read_parquet(tasks_parquet)
        pairs = [
            (int(task_idx), str(text).strip())
            for text, task_idx in zip(tasks_df.index.to_numpy(), tasks_df["task_index"].to_numpy())
        ]
        pairs = [(i, t) for i, t in pairs if t]
        pairs.sort(key=lambda x: x[0])
        if pairs:
            return pairs

    tasks_jsonl = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_jsonl.is_file():
        pairs = []
        with tasks_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = str(row.get("task") or "").strip()
                if not prompt:
                    continue
                pairs.append((int(row["task_index"]), prompt))
        pairs.sort(key=lambda x: x[0])
        if pairs:
            return pairs

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        with episodes_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                tasks = ep.get("tasks") or []
                if tasks and str(tasks[0]).strip():
                    return [(0, str(tasks[0]).strip())]
    raise FileNotFoundError(f"No task prompts found under {dataset_dir}/meta")


def _encode_one(tokenizer, text_encoder, device, prompt: str) -> dict:
    with torch.no_grad():
        context, sl = encode_text(
            [prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device
        )
        L = int(sl[0].item())
        ctx = context[0, :L].detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    return {
        "prompt": prompt,
        "context": ctx,
        "seq_len": L,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--wan_path", default=DEFAULT_WAN_PATH)
    parser.add_argument("--openwam_ckpt", default=DEFAULT_OPENWAM_CKPT)
    parser.add_argument("--cache_dirname", default=CACHE_DIRNAME)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    if args.overwrite:
        args.skip_existing = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    is_dist, rank, world_size, local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(args.dataset_dir)
    jobs = _load_task_prompts(dataset_dir)
    jobs = jobs[rank::world_size]

    out_dir = dataset_dir / "meta" / args.cache_dirname
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "n_jobs_this_rank=%d world_size=%d out=%s skip_existing=%s",
            len(jobs),
            world_size,
            out_dir,
            args.skip_existing,
        )
    if is_dist:
        dist.barrier()

    tokenizer, text_encoder = _load_text_stack(args.wan_path, args.openwam_ckpt, device)

    done = skipped = errors = 0
    pbar = tqdm(jobs, disable=(rank != 0), desc=f"rank{rank}")
    for task_idx, prompt in pbar:
        out_path = out_dir / f"task_{task_idx}.pt"
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        try:
            obj = _encode_one(tokenizer, text_encoder, device, prompt)
            obj["task_index"] = int(task_idx)
            torch.save(obj, out_path)
            done += 1
            logger.info("wrote %s seq_len=%d prompt=%r", out_path.name, obj["seq_len"], prompt)
        except Exception as e:
            errors += 1
            logger.exception("FAIL task_%d: %s", task_idx, e)

    if is_dist:
        dist.barrier()
    logger.info("rank=%d done=%d skipped=%d errors=%d", rank, done, skipped, errors)
    if is_dist:
        dist.destroy_process_group()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
