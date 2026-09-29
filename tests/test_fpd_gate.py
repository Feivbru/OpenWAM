"""Unit tests for OpenWAM FPD distill gate math."""

import torch

from openwam.model.fpd.distill_loss import action_mse_per_sample, distill_gate


def test_distill_gate_disabled_is_all_true():
    s = torch.tensor([0.1, 0.2])
    t = torch.tensor([1.0, 1.0])
    gate = distill_gate(s, t, lambda_action=1.0, lambda_distill=1.0, use_distill_gate=False)
    assert bool(gate.all())


def test_distill_gate_zeros_when_student_much_closer():
    # λ_a = λ_d = 1 → ratio = 0.25; student_gt < 0.25 * teacher_gt → gate False
    s = torch.tensor([0.1, 0.5])
    t = torch.tensor([1.0, 1.0])
    gate = distill_gate(s, t, lambda_action=1.0, lambda_distill=1.0, use_distill_gate=True)
    assert gate.tolist() == [False, True]


def test_action_mse_per_sample_respects_pad():
    pred = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)
    pad = torch.zeros(2, 3, dtype=torch.bool)
    pad[0, 2] = True
    mse = action_mse_per_sample(pred, target, pad)
    assert mse.shape == (2,)
    assert torch.isfinite(mse).all()
