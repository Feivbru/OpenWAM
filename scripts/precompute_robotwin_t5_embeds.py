#!/usr/bin/env python3
"""Precompute OpenWAM-wrap T5 embeddings for RoboTwin (seen[:10] per episode).

Writes ``{task}/{embodiment}_{variant}/umt5_openwam/episode{N}.pt`` next to
``instructions/``. Each file stores 10 wrapped prompts + truncated contexts.

Example (smoke on GPUs 6,7):
  CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \\
    scripts/precompute_robotwin_t5_embeds.py --tasks adjust_bottle --max_episodes 4
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

# Ensure repo root is importable when launched as scripts/...
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from openwam.model.video_backbone.wan.encode import encode_text
from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer, WanTextEncoder

logger = logging.getLogger("precompute_robotwin_t5")

DEFAULT_DATASET_DIR = (
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/assets/benchmark_data/robotwin2.0/dataset"
)
DEFAULT_WAN_PATH = (
    "/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B"
)
DEFAULT_OPENWAM_CKPT = (
    "/data/zixian_guo/projects/haoming/project/PI/OpenWAM/assets/openwam_ckpt/"
    "openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"
)
CACHE_DIRNAME = "umt5_openwam"
WRAP_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "
TOP_K = 10
VARIANTS = ("clean_50", "randomized_500")


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


def _load_text_stack(wan_path: str, openwam_ckpt: str, device: torch.device):
    """Load tokenizer + frozen UMT5 (same as OpenWAM train; T5 is never finetuned)."""
    tok_dir = Path(openwam_ckpt) / "tokenizer" / "google" / "umt5-xxl"
    if not tok_dir.is_dir():
        tok_dir = Path(wan_path) / "google" / "umt5-xxl"
    if not tok_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer not found under {openwam_ckpt} or {wan_path}")

    tokenizer = HuggingfaceTokenizer(name=str(tok_dir), seq_len=512, clean="whitespace")
    text_encoder = WanTextEncoder()

    # Prefer Wan UMT5 pth (identical to OpenWAM's frozen text_encoder init).
    pth = Path(wan_path) / "models_t5_umt5-xxl-enc-bf16.pth"
    if pth.is_file():
        logger.info("Loading text_encoder from %s", pth)
        raw = torch.load(str(pth), map_location="cpu", weights_only=True)
        text_encoder.load_state_dict(raw, strict=True)
    else:
        from safetensors import safe_open

        ckpt_dir = Path(openwam_ckpt)
        cands = sorted(ckpt_dir.glob("checkpoint_step_*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"No Wan T5 pth at {pth} and no OpenWAM safetensors in {ckpt_dir}")
        ckpt_file = cands[-1]
        logger.info("Loading text_encoder from %s", ckpt_file)
        te_sd = {}
        with safe_open(str(ckpt_file), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("video_backbone.text_encoder."):
                    te_sd[k[len("video_backbone.text_encoder.") :]] = f.get_tensor(k)
        missing, unexpected = text_encoder.load_state_dict(te_sd, strict=False)
        if missing:
            raise RuntimeError(f"text_encoder missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning("text_encoder unexpected keys: %d", len(unexpected))

    text_encoder = text_encoder.to(device=device, dtype=torch.bfloat16).eval()
    return tokenizer, text_encoder


def _iter_jobs(dataset_dir: Path, embodiment: str, tasks: list[str] | None, variants: list[str]):
    task_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "meta"])
    if tasks:
        want = set(tasks)
        task_dirs = [p for p in task_dirs if p.name in want]
    for task_dir in task_dirs:
        for variant in variants:
            root = task_dir / f"{embodiment}_{variant}"
            instr = root / "instructions"
            if not instr.is_dir():
                continue
            for jf in sorted(instr.glob("episode*.json")):
                ep_num = int(jf.stem.replace("episode", ""))
                yield task_dir.name, variant, root, ep_num, jf


def _encode_episode(tokenizer, text_encoder, device, base_prompts: list[str]):
    # Encode ONE prompt at a time. Batched UMT5 (pad-to-512) is not bitwise
    # equal to single-prompt encodes; training prepare_inputs typically sees
    # B≈1–4, so cache must be produced with batch size 1 to match online T5.
    wrapped = [format_prompt_for_inference(p) for p in base_prompts]
    assert all(w.startswith(WRAP_PREFIX) for w in wrapped)
    contexts = []
    seq_lens = []
    with torch.no_grad():
        for w in wrapped:
            context, sl = encode_text(
                [w], tokenizer=tokenizer, text_encoder=text_encoder, device=device
            )
            L = int(sl[0].item())
            contexts.append(context[0, :L].detach().to(dtype=torch.bfloat16, device="cpu").contiguous())
            seq_lens.append(L)
    return {
        "wrap_prefix": WRAP_PREFIX,
        "base_prompts": list(base_prompts),
        "wrapped_prompts": wrapped,
        "contexts": contexts,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument("--wan_path", default=DEFAULT_WAN_PATH)
    parser.add_argument("--openwam_ckpt", default=DEFAULT_OPENWAM_CKPT)
    parser.add_argument("--cache_dirname", default=CACHE_DIRNAME)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max_episodes", type=int, default=None, help="Cap total jobs (smoke)")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--batch_episodes", type=int, default=1, help="Encode this many eps per forward (1=safe)")
    args = parser.parse_args()
    if args.overwrite:
        args.skip_existing = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    is_dist, rank, world_size, local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    jobs = list(_iter_jobs(Path(args.dataset_dir), args.embodiment, args.tasks, args.variants))
    if args.max_episodes is not None:
        jobs = jobs[: args.max_episodes]
    jobs = jobs[rank::world_size]
    if rank == 0:
        logger.info(
            "jobs_total_approx_per_rank=%d world_size=%d cache=%s skip_existing=%s",
            len(jobs),
            world_size,
            args.cache_dirname,
            args.skip_existing,
        )

    tokenizer, text_encoder = _load_text_stack(args.wan_path, args.openwam_ckpt, device)

    done = skipped = errors = 0
    pbar = tqdm(jobs, disable=(rank != 0), desc=f"rank{rank}")
    for task, variant, root, ep_num, jf in pbar:
        out_dir = root / args.cache_dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode{ep_num}.pt"
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        try:
            payload = json.loads(jf.read_text())
            seen = payload.get("seen") or []
            if len(seen) < TOP_K:
                raise ValueError(f"{jf} has only {len(seen)} seen (<{TOP_K})")
            base = seen[:TOP_K]
            obj = _encode_episode(tokenizer, text_encoder, device, base)
            torch.save(obj, out_path)
            done += 1
        except Exception as e:
            errors += 1
            logger.exception("FAIL %s/%s episode%s: %s", task, variant, ep_num, e)

    if is_dist:
        dist.barrier()
    logger.info("rank=%d done=%d skipped=%d errors=%d", rank, done, skipped, errors)
    if is_dist:
        dist.destroy_process_group()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
