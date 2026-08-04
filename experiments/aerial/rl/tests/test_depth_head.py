"""Step 3 DepthHead unit tests — skip when torch absent (Mac may lack it).

The torch-free ④ collector-wiring test lives in ``test_collector_depth_shield``
so it still runs on GPU-less hosts.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments.aerial.rl.dynamics_torch import _DepthHead, depth_head_loss


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
