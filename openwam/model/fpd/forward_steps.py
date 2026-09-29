"""Teacher / student single-step forwards for OpenWAM FPD."""

from __future__ import annotations

from typing import Any

import torch


def forward_fpd_teacher_step(
    teacher,
    inputs: dict[str, Any],
    action_noise_bundle: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Teacher forward: clean GT video + shared noisy actions → action velocity."""
    # Shallow copy so latents / pops inside compute_loss do not mutate caller's dict.
    inputs = dict(inputs)
    out = teacher.compute_loss(
        action_noise_bundle=action_noise_bundle,
        video_mode="clean",
        lambda_video=0.0,
        lambda_action=1.0,
        return_details=True,
        **inputs,
    )
    return {
        "action_pred": out["action_pred"],
        "action_target": out["action_target"],
        "action_gt_mse_per_sample": out["action_gt_mse_per_sample"],
        "action_timestep_ids": out["action_timestep_ids"],
    }


def forward_fpd_student_step(
    student,
    inputs: dict[str, Any],
    action_noise_bundle: dict[str, torch.Tensor],
    *,
    lambda_video: float,
    lambda_action: float,
) -> dict[str, torch.Tensor]:
    """Student forward: noisy video + shared noisy actions → action (+ video) loss."""
    inputs = dict(inputs)
    out = student.compute_loss(
        action_noise_bundle=action_noise_bundle,
        video_mode="train",
        lambda_video=lambda_video,
        lambda_action=lambda_action,
        return_details=True,
        **inputs,
    )
    return out
