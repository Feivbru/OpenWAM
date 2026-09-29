"""Fixed-width batched denoising for multi-slot eval.

``BaseWAMArchitecture.generate`` is left untouched. This module is the only
caller of the new ``generate_batch`` path: active slots share one schedule,
each slot keeps its own DiT velocity cache, and slots that are allowed to
skip are left out of that step's forward.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Leading axis is the sample axis for these preprocess outputs.
_BATCH_KEYS = (
    "latents",
    "noise",
    "context",
    "seq_lens",
    "first_frame_latents",
    "proprio",
    "context_mask",
    "y",
    "clip_feature",
    "condition_mask",
    "input_latents",
    "vace_context",
)


def generate_batch(
    arch,
    samples: list[dict],
    *,
    schedule,
    action_num_frames: int,
    video_num_frames: int,
    height: int,
    width: int,
    denoise_steps: int,
    shift: float,
    dit_cache_cfg: Optional[dict] = None,
    active_action_mask: Optional[Tensor] = None,
) -> dict:
    """Denoise ``samples`` together. Every entry is active.

    Each sample dict:
        first_frame_image: list[PIL] or one PIL
        context: Tensor [L, D] or [1, L, D]
        seq_lens: Tensor scalar or [1]
        proprio: Tensor or None
        seed: int
        prompt: str, optional, only stored for debugging

    Returns ``{"actions": np.ndarray [B, T, action_dim]}`` in sample order.
    """
    if not samples:
        raise ValueError("generate_batch requires at least one active sample")

    arch.eval()
    device = arch.device
    dtype = arch.dtype
    vb = arch.video_backbone
    if not hasattr(vb, "preprocess_external_context_for_inference"):
        raise TypeError(
            f"{type(vb).__name__} has no preprocess_external_context_for_inference; "
            "batched eval currently supports the Wan backbone."
        )

    prepared = []
    for sample in samples:
        image = sample["first_frame_image"]
        if not isinstance(image, list):
            image = [image]
        one = vb.preprocess_external_context_for_inference(
            context=sample["context"],
            seq_lens=sample["seq_lens"] if torch.is_tensor(sample["seq_lens"]) else torch.tensor(sample["seq_lens"]),
            prompt=str(sample.get("prompt") or ""),
            first_frame_image=image,
            num_frames=int(video_num_frames),
            height=int(height),
            width=int(width),
            seed=int(sample.get("seed", 42)),
            num_inference_steps=int(denoise_steps),
            shift=float(shift),
            tiled=True,
        )
        proprio = sample.get("proprio")
        if arch.uses_proprioception:
            if proprio is None:
                raise ValueError("use_proprioception=True requires proprio on every batched sample")
            one["proprio"] = proprio.to(device=device, dtype=dtype)
        prepared.append(one)

    inputs = _stack_inputs(prepared)
    ref_latents = inputs.get("first_frame_latents")
    if ref_latents is not None:
        latents = inputs["latents"].clone()
        latents[:, :, : ref_latents.shape[2]] = ref_latents
        inputs["latents"] = latents

    batch = inputs["latents"].shape[0]
    action_num_frames = int(action_num_frames)
    action_chunks = []
    for sample in samples:
        generator = torch.Generator(device=device).manual_seed(int(sample.get("seed", 42)))
        action_chunks.append(
            torch.randn(
                1,
                action_num_frames - 1,
                arch.action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    action_latents = torch.cat(action_chunks, dim=0)

    inactive_action_dims = arch._resolve_inactive_action_dims(active_action_mask, device)
    inactive_action_noise = None
    caches = _make_caches(batch, dit_cache_cfg)

    num_train_ts_v = float(arch.video_scheduler.num_train_timesteps)
    num_train_ts_a = float(arch.action_scheduler.num_train_timesteps)

    for i in range(len(schedule) - 1):
        t_v, t_a = schedule[i]
        t_v_next, t_a_next = schedule[i + 1]
        sigma_v = t_v / num_train_ts_v
        sigma_a = t_a / num_train_ts_a
        sigma_v_next = t_v_next / num_train_ts_v
        sigma_a_next = t_a_next / num_train_ts_a
        video_stepping = sigma_v != sigma_v_next
        action_stepping = sigma_a != sigma_a_next
        if not video_stepping and not action_stepping:
            continue

        need = []
        for slot in range(batch):
            cache = caches[slot]
            if (
                cache is not None
                and video_stepping
                and not cache.should_recompute(sigma_v, require_action=action_stepping)
            ):
                continue
            need.append(slot)

        video_pred = torch.empty_like(inputs["latents"])
        action_pred = torch.empty_like(action_latents) if action_stepping else None
        if need:
            idx = torch.tensor(need, device=device, dtype=torch.long)
            sub_inputs = _index_inputs(inputs, idx, batch)
            sub_actions = action_latents.index_select(0, idx)
            v_timestep = torch.full((len(need),), float(t_v), dtype=dtype, device=device)
            a_timestep = torch.full((len(need),), float(t_a), dtype=dtype, device=device)
            torch.compiler.cudagraph_mark_step_begin()
            noise_pred, action_noise_pred = arch.forward(
                sub_actions,
                a_timestep,
                **sub_inputs,
                timestep=v_timestep,
            )
            for local, slot in enumerate(need):
                sl_v = noise_pred[local : local + 1]
                sl_a = None if action_noise_pred is None else action_noise_pred[local : local + 1]
                video_pred[slot : slot + 1] = sl_v
                if action_pred is not None and sl_a is not None:
                    action_pred[slot : slot + 1] = sl_a
                if caches[slot] is not None and video_stepping:
                    caches[slot].update(sl_v, float(sigma_v), sl_a)

        if len(need) != batch:
            have = set(need)
            for slot in range(batch):
                if slot in have:
                    continue
                video_pred[slot : slot + 1] = caches[slot].get_cached()
                if action_stepping:
                    cached_action = caches[slot].get_cached_action()
                    if cached_action is None:
                        raise RuntimeError(f"slot {slot} skipped a joint step without a cached action velocity")
                    action_pred[slot : slot + 1] = cached_action

        if video_stepping:
            new_latents = inputs["latents"] + video_pred * (sigma_v_next - sigma_v)
            if ref_latents is not None:
                new_latents = new_latents.clone()
                new_latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs["latents"] = new_latents

        if action_stepping and action_pred is not None:
            if inactive_action_dims is not None and inactive_action_noise is None:
                sigma_a_f = float(sigma_a)
                if sigma_a_f <= 0.0:
                    raise ValueError("Cannot initialize inactive action noise from a non-positive sigma.")
                inactive_action_noise = action_latents[..., inactive_action_dims].detach().clone() / sigma_a_f
            action_latents = arch.action_scheduler.flow_step(
                action_pred, sigma_a, sigma_a_next, action_latents
            )
            if inactive_action_dims is not None:
                action_latents[..., inactive_action_dims] = inactive_action_noise * float(sigma_a_next)

    actions = action_latents.detach().float().cpu().numpy()
    normalizer = getattr(arch, "normalizer", None)
    if normalizer is not None:
        actions = normalizer.unnormalize(actions)
    return {"actions": actions}


def _make_caches(batch: int, dit_cache_cfg: Optional[dict]):
    if not dit_cache_cfg or not dit_cache_cfg.get("enabled", False):
        return [None] * batch
    from openwam.deploy.optimizations.dit_cache import DiTVelocityCache

    return [
        DiTVelocityCache(
            cosine_threshold=float(dit_cache_cfg.get("cosine_threshold", 0.99)),
            max_consecutive_skips=int(dit_cache_cfg.get("max_skips", 3)),
        )
        for _ in range(batch)
    ]


def _stack_inputs(items: list[dict]) -> dict:
    keys = list(items[0].keys())
    stacked = {}
    for key in keys:
        values = [item[key] for item in items]
        if all(isinstance(value, Tensor) for value in values):
            normed = [_as_batch_row(key, value) for value in values]
            if key == "context":
                normed = _pad_context(normed)
            if key == "context_mask":
                width = max(value.shape[-1] for value in normed)
                padded = []
                for value in normed:
                    if value.shape[-1] < width:
                        pad = torch.zeros(
                            *value.shape[:-1],
                            width - value.shape[-1],
                            dtype=value.dtype,
                            device=value.device,
                        )
                        value = torch.cat([value, pad], dim=-1)
                    padded.append(value)
                normed = padded
            try:
                stacked[key] = torch.cat(normed, dim=0)
            except RuntimeError as exc:
                shapes = [tuple(value.shape) for value in normed]
                raise RuntimeError(f"cannot stack preprocess key {key!r}, shapes={shapes}") from exc
            continue
        if all(value == values[0] for value in values):
            stacked[key] = values[0]
    return stacked


def _as_batch_row(key: str, value: Tensor) -> Tensor:
    if key in _BATCH_KEYS and (value.ndim == 0 or value.shape[0] != 1):
        if key == "seq_lens" and value.ndim == 0:
            return value.reshape(1)
        if key == "proprio" and value.ndim == 1:
            return value.unsqueeze(0)
        if key == "context" and value.ndim == 2:
            return value.unsqueeze(0)
    return value


def _pad_context(rows: list[Tensor]) -> list[Tensor]:
    width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < width:
            pad = row.new_zeros(row.shape[0], width - row.shape[1], row.shape[2])
            row = torch.cat([row, pad], dim=1)
        padded.append(row)
    return padded


def _index_inputs(inputs: dict, idx: Tensor, batch: int) -> dict:
    out = {}
    for key, value in inputs.items():
        if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == batch:
            out[key] = value.index_select(0, idx)
        else:
            out[key] = value
    return out
