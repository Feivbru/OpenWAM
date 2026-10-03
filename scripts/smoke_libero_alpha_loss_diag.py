#!/usr/bin/env python3
"""Smoke: Alpha-Sim-LIBERO forward action-FM loss under OpenWAM LIBERO dataloader.

No optimizer — measures the *native* train-time loss the teacher sees at step 0,
to check whether ~0.05 is already present on the release ckpt (vs a train bug).

Also supports normalize_mode ablation (min-max vs null vs z-score).

Example:
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_libero_alpha_loss_diag.py --steps 20
  CUDA_VISIBLE_DEVICES=3 python scripts/smoke_libero_alpha_loss_diag.py --steps 20 --normalize_mode null
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_libero_alpha_loss")

DEFAULT_CKPT = str(
    _ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
)
DEFAULT_WAN = (
    "/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    p.add_argument("--wan_path", default=DEFAULT_WAN)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--normalize_mode",
        default="min-max",
        choices=["min-max", "null", "z-score", "quantile"],
    )
    p.add_argument("--use_t5_cache", action="store_true", default=True)
    p.add_argument("--no_t5_cache", action="store_true", default=False)
    p.add_argument("--video_mode", default="clean", choices=["clean", "noise"])
    args = p.parse_args()
    if args.no_t5_cache:
        args.use_t5_cache = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataloader (same yaml contract as teacher train) ────────────────────
    from openwam.dataloader.registry import build_dataset

    dl_cfg = OmegaConf.load(_ROOT / "configs/dataloader/libero.yaml")
    dl_cfg.use_t5_cache = bool(args.use_t5_cache)
    if args.normalize_mode == "null":
        dl_cfg.normalize_mode = None
    else:
        dl_cfg.normalize_mode = args.normalize_mode
    dataset = build_dataset(dl_cfg, split="train")
    logger.info(
        "dataset len=%d normalize_mode=%r use_t5_cache=%s stats=%s",
        len(dataset),
        getattr(dataset, "_normalize_mode", None),
        getattr(dataset, "use_t5_cache", False),
        getattr(dataset, "normalization_stats_path", None),
    )

    # Peek one batch action range (active dims only)
    s0 = dataset[0]
    act = s0["action"].numpy()
    mask = s0["action_mask"].numpy().astype(bool)
    active = act[mask[:, 0]][:, mask[0]] if act.ndim == 2 else act
    # simpler: first timestep active dims
    a1 = act[0, mask[0]]
    logger.info(
        "sample0 active_action dim=%d min=%.4f max=%.4f mean=%.4f std=%.4f E[a^2]=%.4f t5=%s",
        a1.shape[0],
        float(a1.min()),
        float(a1.max()),
        float(a1.mean()),
        float(a1.std()),
        float((a1**2).mean()),
        "t5_context" in s0,
    )

    # ── load Alpha architecture (skip UMT5 when using disk embeds) ──────────
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

    # Minimal override so Wan path resolves on this machine.
    override = OmegaConf.create(
        {
            "model": {
                "video_backbone": {"model_path": args.wan_path},
                "freeze_video_dit": True,  # irrelevant for forward-only
            },
            "dataloader": {"use_t5_cache": bool(args.use_t5_cache)},
        }
    )
    _, arch, _ = build_architecture_from_ckpt_dir(
        args.ckpt_dir,
        weights_required=True,
        override_cfg=override,
        skip_text_encoder=bool(args.use_t5_cache),
    )
    arch.set_dtype_device(torch.bfloat16, device)
    arch.init_training_schedulers(num_timesteps=1000)
    arch.eval()
    logger.info("architecture on %s dtype=%s", device, arch.dtype)

    # ── forward-only loss loop ──────────────────────────────────────────────
    losses = []
    with torch.no_grad():
        for step in range(args.steps):
            # Independent seed per step (matches trainer's per-step noise seed intent)
            torch.manual_seed(args.seed + step)
            batch = [dataset[(step * args.batch_size + b) % len(dataset)] for b in range(args.batch_size)]
            inputs = arch.prepare_inputs(batch)
            # Match teacher yaml defaults (also injected by OpenWAMTrainer.prepare_inputs path).
            inputs.setdefault("max_timestep_boundary", 1.0)
            inputs.setdefault("min_timestep_boundary", 0.0)
            result = arch.compute_loss(
                **inputs,
                lambda_video=0.0,
                lambda_action=1.0,
                video_mode=args.video_mode,
            )
            la = float(result["loss_action"].detach().float().cpu())
            lt = float(result["loss"].detach().float().cpu())
            losses.append(la)
            logger.info(
                "step=%02d loss=%.6f loss_action=%.6f loss_video=%.6f",
                step + 1,
                lt,
                la,
                float(result.get("loss_video", torch.tensor(0.0)).detach().float().cpu()),
            )

    arr = np.asarray(losses, dtype=np.float64)
    print("\n========== SUMMARY ==========")
    print(f"ckpt:            {args.ckpt_dir}")
    print(f"normalize_mode:  {args.normalize_mode}")
    print(f"use_t5_cache:    {args.use_t5_cache}")
    print(f"video_mode:      {args.video_mode}")
    print(f"steps/batch:     {args.steps}/{args.batch_size}")
    print(
        f"loss_action:     mean={arr.mean():.6f}  std={arr.std():.6f}  "
        f"min={arr.min():.6f}  max={arr.max():.6f}  median={np.median(arr):.6f}"
    )
    print(
        "interpretation:  ~1e-2 healthy for OpenWAM-LIBERO 10D; "
        "~O(1) suggests broken norm / wrong space; "
        "Alpha already ~0.05 ⇒ not introduced by teacher train loop"
    )
    print("==============================\n")


if __name__ == "__main__":
    main()
