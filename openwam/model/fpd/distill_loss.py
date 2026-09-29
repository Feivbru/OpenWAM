"""Per-sample action distill / GT MSE helpers for FPD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def action_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unweighted per-sample action MSE. Returns ``[B]``."""
    per_element = F.mse_loss(pred.float(), target.float(), reduction="none")
    if action_is_pad is None:
        return per_element.mean(dim=(1, 2))

    action_is_pad = action_is_pad.to(device=per_element.device, dtype=torch.bool)
    valid = (~action_is_pad).float()
    if valid.shape == per_element.shape:
        weighted = per_element * valid
        return weighted.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)
    if valid.ndim == 3:
        valid = (valid > 0).any(dim=-1).float()
    per_step = per_element.mean(dim=2) * valid
    return per_step.sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def distill_gate(
    student_gt_mse: torch.Tensor,
    teacher_gt_mse: torch.Tensor,
    lambda_action: float,
    lambda_distill: float,
    use_distill_gate: bool = False,
) -> torch.Tensor:
    """Per-sample bool gate: True => keep distill.

    Zero distill when student is closer to GT than the mix target
    ``m = (λ_a * gt + λ_d * teacher) / (λ_a + λ_d)``, i.e. when
    ``student_gt < (λ_d / (λ_a + λ_d))^2 * teacher_gt``.

    Gate is detached. If gating is disabled or either λ ≤ 0, gate ≡ 1.
    """
    student_gt = student_gt_mse.detach()
    teacher_gt = teacher_gt_mse.detach()
    if (not use_distill_gate) or lambda_action <= 0.0 or lambda_distill <= 0.0:
        return torch.ones_like(student_gt, dtype=torch.bool)

    ratio = (lambda_distill / (lambda_action + lambda_distill)) ** 2
    gate = student_gt >= (ratio * teacher_gt)
    bad = ~(torch.isfinite(student_gt) & torch.isfinite(teacher_gt))
    return gate | bad
