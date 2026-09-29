"""Shared action flow-matching noise bundle for FPD teacher/student."""

from __future__ import annotations

from typing import Optional

import torch


def sample_action_noise_bundle(
    scheduler,
    actions: torch.Tensor,
    *,
    noise: Optional[torch.Tensor] = None,
    timestep_ids: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Sample a shared action FM noise bundle.

    Args:
        scheduler: Action flow-matching scheduler (``add_noise`` / ``training_target``).
        actions: (B, T, D) clean actions.
        noise: Optional pre-sampled noise (same shape as ``actions``).
        timestep_ids: Optional integer timestep ids on CPU (shape ``[B]``).

    Returns:
        Dict with ``noise``, ``timestep_ids``, ``timesteps``, ``sigmas``,
        ``noisy_actions``, ``target``.
    """
    if actions.dim() == 2:
        actions = actions.unsqueeze(0)
    B = actions.shape[0]
    device = actions.device
    dtype = actions.dtype

    if timestep_ids is None:
        timestep_ids = torch.randint(0, len(scheduler.timesteps), (B,))
    else:
        timestep_ids = timestep_ids.to(device="cpu")

    timesteps = scheduler.timesteps[timestep_ids].to(dtype=dtype, device=device)
    sigmas = scheduler.sigmas[timestep_ids].to(dtype=dtype, device=device)

    if noise is None:
        noise = torch.randn_like(actions)
    else:
        noise = noise.to(dtype=dtype, device=device)

    if sigmas.dim() == 1:
        sigma_bc = sigmas.view(B, 1, 1)
    else:
        sigma_bc = sigmas.unsqueeze(-1)

    noisy_actions = scheduler.add_noise(actions, noise, sigma_bc)
    target = scheduler.training_target(actions, noise)
    return {
        "noise": noise,
        "timestep_ids": timestep_ids,
        "timesteps": timesteps,
        "sigmas": sigmas,
        "noisy_actions": noisy_actions,
        "target": target,
    }
