"""Step 3 DepthHead unit tests — skip when torch absent (Mac may lack it).

The torch-free ④ collector-wiring test lives in ``test_collector_depth_shield``
so it still runs on GPU-less hosts.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments.aerial.rl.dynamics_torch import (
    _DepthHead,
    depth_delta_scale_loss,
    depth_head_loss,
)


def test_depth_head_forward_shapes():
    model = _DepthHead(image_size=16, n_frames=2, base=8)
    rgb = torch.randint(0, 256, (2, 3, 16, 16, 3), dtype=torch.uint8)
    depth, log_sigma = model.predict_from_window(rgb)
    assert depth.shape == (2, 16, 16)
    assert log_sigma.shape == (2, 16, 16)
    assert torch.all(depth > 0)


def test_depth_head_loss_finite_and_improves_on_identity():
    gt = torch.ones(2, 16, 16) * 4.0
    pred_good = gt.clone()
    log_sigma = torch.zeros_like(gt)
    loss_good, stats_good = depth_head_loss(pred_good, log_sigma, gt)
    pred_bad = gt * 2.0
    loss_bad, stats_bad = depth_head_loss(pred_bad, log_sigma, gt)
    assert torch.isfinite(loss_good)
    assert stats_good["absrel"] < stats_bad["absrel"]


def test_delta_scale_loss_approach_gate_skips_flat_windows():
    """Flat GT Δ must not contribute (prevents AbsRel-killing noise)."""
    B, H, W = 2, 8, 8
    gt0 = torch.ones(B, H, W) * 10.0
    gt1 = gt0.clone()  # Δ = 0
    pred0 = gt0.clone().requires_grad_(True)
    pred1 = (gt0 * 1.1).requires_grad_(True)
    loss, stats = depth_delta_scale_loss(
        pred0, pred1, gt0, gt1, min_gt_delta_m=0.5, min_depth_m=1.0, max_depth_m=40.0
    )
    assert stats["n_delta"] == 0
    assert loss.item() == 0.0


def test_delta_scale_loss_penalizes_wrong_approach_delta():
    B, H, W = 2, 8, 8
    gt0 = torch.ones(B, H, W) * 20.0
    gt1 = torch.ones(B, H, W) * 10.0  # |Δ| = 10 m (approach)
    pred_good0 = gt0.clone()
    pred_good1 = gt1.clone()
    pred_bad0 = gt0.clone()
    pred_bad1 = gt0.clone()  # predicted Δ ≈ 0
    good, s_good = depth_delta_scale_loss(
        pred_good0, pred_good1, gt0, gt1, min_gt_delta_m=0.5
    )
    bad, s_bad = depth_delta_scale_loss(
        pred_bad0, pred_bad1, gt0, gt1, min_gt_delta_m=0.5
    )
    assert s_good["n_delta"] == B and s_bad["n_delta"] == B
    assert float(good.item()) < float(bad.item())
    # Gradients must flow through band-mean (regression vs nanmedian collapse).
    pred = gt0.clone().requires_grad_(True)
    loss, _ = depth_delta_scale_loss(pred, gt1.clone(), gt0, gt1, min_gt_delta_m=0.5)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_delta_scale_loss_respects_motion_support_ratio():
    B, H, W = 2, 8, 8
    gt0 = torch.ones(B, H, W) * 12.0
    gt1 = torch.ones(B, H, W) * 10.0  # |Δ| = 2
    pred0, pred1 = gt0.clone(), gt1.clone()
    # motion=10 → need ŝ ≥ 0.6*10=6; 2 < 6 → gated out
    loss, stats = depth_delta_scale_loss(
        pred0,
        pred1,
        gt0,
        gt1,
        min_gt_delta_m=0.5,
        motion_m=torch.tensor([10.0, 10.0]),
        support_ratio=0.6,
    )
    assert stats["n_delta"] == 0
    assert loss.item() == 0.0
    # motion=2 → need ŝ ≥ 1.2; 2 ≥ 1.2 → kept
    loss2, stats2 = depth_delta_scale_loss(
        pred0,
        pred1,
        gt0,
        gt1,
        min_gt_delta_m=0.5,
        motion_m=torch.tensor([2.0, 2.0]),
        support_ratio=0.6,
    )
    assert stats2["n_delta"] == B
    assert torch.isfinite(loss2)
