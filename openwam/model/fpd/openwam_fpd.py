"""OpenWAM FPD wrapper: frozen teacher distills into trainable student."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .action_noise import sample_action_noise_bundle
from .distill_loss import action_mse_per_sample, distill_gate
from .forward_steps import forward_fpd_student_step, forward_fpd_teacher_step

logger = logging.getLogger(__name__)


class OpenWAMFPD(nn.Module):
    """FPD wrapper around a trainable student + frozen teacher architecture.

    Both sides use the architecture's default training attention mask.
    Asymmetry is only clean video (teacher) vs noisy video (student) plus
    shared action noise.

    Loss:
      L = λ_a * L_action + λ_v * L_video + λ_d * g * L_distill

    Per-sample distill gate (when ``use_distill_gate`` and λ_a, λ_d > 0):
      zero distill if student_gt_mse < (λ_d / (λ_a + λ_d))^2 * teacher_gt_mse

    Action / distill terms are multiplied by the action scheduler's
    ``training_weight(t)`` (same as OpenWAM's built-in action loss weighting).
    Gate decisions and logged ``*_gt_mse`` stay unweighted.
    """

    training_mode = "fpd"

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        *,
        lambda_action: float = 1.0,
        lambda_video: float = 0.01,
        lambda_distill: float = 1.0,
        use_distill_gate: bool = True,
    ):
        super().__init__()
        self.student = student
        # Keep teacher out of the nn.Module tree so DeepSpeed / optimizer
        # never see its parameters.
        object.__setattr__(self, "_teacher", teacher)
        self.lambda_action = float(lambda_action)
        self.lambda_video = float(lambda_video)
        self.lambda_distill = float(lambda_distill)
        self.use_distill_gate = bool(use_distill_gate)

    @property
    def teacher(self) -> nn.Module:
        return object.__getattribute__(self, "_teacher")

    def prepare_inputs(self, batch):
        return self.student.prepare_inputs(batch)

    def compute_loss(self, batch) -> dict:
        """Run shared-noise teacher/student forwards and return FPD losses."""
        if not isinstance(batch, list):
            batch = [batch]

        inputs = self.student.prepare_inputs(batch)
        actions = inputs.get("actions")
        if actions is None:
            raise ValueError("FPD requires actions in the batch.")

        scheduler = self.student.action_backbone.scheduler
        bundle = sample_action_noise_bundle(scheduler, actions)

        self.teacher.eval()
        with torch.no_grad():
            teacher_out = forward_fpd_teacher_step(self.teacher, inputs, bundle)

        student_out = forward_fpd_student_step(
            self.student,
            inputs,
            bundle,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
        )

        action_is_pad = inputs.get("action_is_pad")
        student_pred = student_out["action_pred"]
        teacher_pred = teacher_out["action_pred"]
        action_target = bundle["target"].to(dtype=student_pred.dtype, device=student_pred.device)

        student_gt = student_out["action_gt_mse_per_sample"]
        with torch.no_grad():
            teacher_gt = teacher_out["action_gt_mse_per_sample"].to(device=student_gt.device)

        distill_mse = action_mse_per_sample(student_pred, teacher_pred.detach(), action_is_pad)

        with torch.no_grad():
            gate = distill_gate(
                student_gt_mse=student_gt,
                teacher_gt_mse=teacher_gt,
                lambda_action=self.lambda_action,
                lambda_distill=self.lambda_distill,
                use_distill_gate=self.use_distill_gate,
            )
            gate_f = gate.to(dtype=distill_mse.dtype)

        # Timestep weight (always-on in OpenWAM action loss).
        tw = scheduler.training_weight(bundle["timestep_ids"]).to(
            device=distill_mse.device, dtype=distill_mse.dtype
        )
        loss_action = (student_gt * tw).mean()
        loss_distill = (distill_mse * gate_f * tw).mean()
        # student_out["loss_video"] is already scaled by lambda_video inside compute_loss
        # when lambda_video was passed; peel that so we can re-apply consistently below.
        # Actually forward_fpd_student_step passes lambda_video into compute_loss, so
        # loss_video is already λ_v * L_v. Use raw path: recompute from returned value.
        loss_video_scaled = student_out["loss_video"]
        if not isinstance(loss_video_scaled, torch.Tensor):
            loss_video_scaled = torch.tensor(0.0, device=loss_action.device)
        else:
            loss_video_scaled = loss_video_scaled.to(device=loss_action.device)

        # student_out loss_action is also already λ_a-scaled; we recompute weighted
        # action/distill ourselves for gate + shared weight consistency.
        loss_total = (
            self.lambda_action * loss_action
            + loss_video_scaled  # already includes lambda_video
            + self.lambda_distill * loss_distill
        )

        student_gt_mean = float(student_gt.detach().mean().item())
        teacher_gt_mean = float(teacher_gt.mean().item())
        return {
            "loss": loss_total,
            "loss_action": self.lambda_action * loss_action.detach(),
            "loss_video": loss_video_scaled.detach(),
            "loss_distill": self.lambda_distill * loss_distill.detach(),
            "teacher_action_gt_mse": teacher_gt_mean,
            "student_action_gt_mse": student_gt_mean,
            "student_minus_teacher": student_gt_mean - teacher_gt_mean,
            "distill_gate_frac": float(gate_f.mean().item()),
        }
