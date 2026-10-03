#!/usr/bin/env python3
"""Precompute UMT5 embeddings for LIBERO task prompts (one file per task_index).

Writes ``{dataset_dir}/meta/umt5_openwam/task_{N}.pt``. Each file stores the
exact prompt string used by ``LiberoDataset`` (raw ``tasks.parquet`` index text,
no RoboTwin wrap) plus the truncated bf16 context.

LIBERO has only ~40 unique tasks, so a single-GPU run finishes in minutes.

Example:
  CUDA_VISIBLE_DEVICES=2 python scripts/precompute_libero_t5_embeds.py
  CUDA_VISIBLE_DEVICES=2,3 bash scripts/precompute_libero_t5_embeds.sh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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

logger = logging.getLogger("precompute_libero_t5")

DEFAULT_DATASET_DIR = (
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/assets/benchmark_data/libero"
)
DEFAULT_WAN_PATH = (
    "/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B"
)
DEFAULT_OPENWAM_CKPT = (
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/assets/openwam_ckpt/"
    "openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
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
    """Mirror ``LeRobotV3Reader._load_prompts``: index=text, column=task_index."""
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing {tasks_path}")
    tasks_df = pd.read_parquet(tasks_path)
    if "task_index" not in tasks_df.columns:
        raise KeyError(f"{tasks_path} missing task_index column")
    pairs = []
    for text, task_idx in zip(tasks_df.index.to_numpy(), tasks_df["task_index"].to_numpy()):
        prompt = str(text).strip()
        if not prompt:
            raise ValueError(f"task_index={int(task_idx)} has empty prompt text")
        pairs.append((int(task_idx), prompt))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _encode_one(tokenizer, text_encoder, device, prompt: str) -> dict:
    # Batch size 1 to match online train encodes (same as RoboTwin precompute).
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
    parser.add_argument("--max_tasks", type=int, default=None, help="Cap tasks (smoke)")
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
    if args.max_tasks is not None:
        jobs = jobs[: int(args.max_tasks)]
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
