"""Step 3 DepthHead unit tests — skip when torch absent (Mac may lack it)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.aerial.rl.buffer import ReplayBuffer, Transition
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.dynamics_torch import _DepthHead, depth_head_loss
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.safety import ThresholdSafetyShield


def _obs(size=16, depth_val=5.0, info=None):
    state = np.zeros(7, dtype=np.float32)
    rgb = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    depth = np.full((size, size), depth_val, dtype=np.float32)
    return Observation(rgb=rgb, state=state, depth=depth, info=info or {})


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


def test_collector_writes_depth_min_pred_before_shield():
    """Frozen §4 ④: predictor fills info before should_override sees the obs."""

    class _StubEnv:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([10.0, 0.0, 0.0])

        def reset(self, episode=None):
            return _obs(depth_val=10.0)

        def step(self, action):
            return _obs(depth_val=10.0), {"cmd": action.tolist()}

    class _Pred:
        def reset(self):
            pass

        def predict_min(self, obs):
            return 0.5  # below ThresholdSafetyShield.min_depth_m=1.5

    class _Policy:
        def act(self, view):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    buf = ReplayBuffer(capacity_episodes=2, seed=0)
    shield = ThresholdSafetyShield(min_depth_m=1.5)
    col = RolloutCollector(
        _StubEnv(), _Policy(), buf,
        safety=shield, max_steps=3, target_hz=0.0,
        depth_predictor=_Pred(),
        skip_reset_collision=False,
    )
    ep, stats = col.collect_episode()
    assert stats.interventions >= 1
    # The stored transition's obs.info must carry the prediction.
    assert ep[0].obs.info.get("depth_min_pred") == 0.5
