"""Step 3 DepthHead unit tests — skip when torch absent (Mac may lack it).

The torch-free ④ collector-wiring test lives in ``test_collector_depth_shield``
so it still runs on GPU-less hosts.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.aerial.rl.buffer import ReplayBuffer, Transition
from experiments.aerial.rl.dynamics_torch import (
    _DepthHead,
    depth_delta_scale_loss,
    depth_head_loss,
)
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.train_depth_head import (
    _apply_freeze_encoder,
    _load_depth_cfg,
    _sample_approach_biased_windows,
    main as train_depth_main,
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


def _const_depth_window(
    depths_m: list[float], *, hw: int = 4, dx: float = 1.0
) -> list[Transition]:
    """Build a length-L window with constant per-frame depth and forward motion."""
    out: list[Transition] = []
    for i, d in enumerate(depths_m):
        depth = np.full((hw, hw), float(d), dtype=np.float32)
        rgb = np.zeros((hw, hw, 3), dtype=np.uint8)
        state = np.array([i * dx, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs = Observation(rgb=rgb, state=state, depth=depth)
        out.append(
            Transition(obs=obs, action=np.zeros(4, np.float32), reward=0.0, done=False)
        )
    return out


def test_approach_sampler_scores_loss_interval_not_window_start():
    """With n_f>1, rank on depth[:, n_f-1] vs [:, -1] (Δ-loss endpoints), not [0, -1].

    Window A: large |Δ| on [0, L-1] but flat on [n_f-1, L-1] → must lose.
    Window B: approach alive on [n_f-1, L-1] → must win.
    """
    L, n_f = 8, 4
    # A: early approach then flat from frame 3..7 → |Δ[0,7]|=20, |Δ[3,7]|=0
    win_a = _const_depth_window([30.0, 20.0, 15.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    # B: flat until n_f-1, then approach → |Δ[0,7]|=10, |Δ[3,7]|=10
    win_b = _const_depth_window([20.0, 20.0, 20.0, 20.0, 17.0, 14.0, 12.0, 10.0])
    buf = ReplayBuffer(capacity_episodes=2, seed=0)
    # sample_windows is stubbed below; episodes only need to exist.
    buf.add_episode(win_a)
    buf.add_episode(win_b)

    candidates = [win_a, win_b, win_a, win_b]  # oversample=4, batch=1 → n_cand=4
    buf.sample_windows = lambda n, length: candidates[: int(n)]  # type: ignore[method-assign]

    picked = _sample_approach_biased_windows(
        buf,
        batch=1,
        window=L,
        oversample=4,
        min_depth_m=1.0,
        max_depth_m=40.0,
        min_gt_delta_m=0.5,
        support_ratio=0.0,
        n_frames=n_f,
    )
    assert len(picked) == 1
    # Identify by first-frame depth: A starts at 30, B at 20.
    assert float(picked[0][0].obs.depth.mean()) == pytest.approx(20.0)


def test_freeze_encoder_no_grad_and_optimizer_excludes_encoder():
    """Decoder-only FT: encoder requires_grad=False and absent from AdamW."""
    model = _DepthHead(image_size=16, n_frames=2, base=8)
    trainable = _apply_freeze_encoder(model, freeze=True)
    assert trainable, "decoder must still expose trainable params"
    for p in model.encoder.parameters():
        assert p.requires_grad is False
    for p in model.decoder.parameters():
        assert p.requires_grad is True
    train_ids = {id(p) for p in trainable}
    for p in model.encoder.parameters():
        assert id(p) not in train_ids
    for p in model.decoder.parameters():
        assert id(p) in train_ids
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == train_ids
    # Gradients must not accumulate on frozen encoder after a backward.
    rgb = torch.rand(1, 2, 16, 16, 3)
    pred, log_sigma = model.predict_from_window(rgb)
    loss = pred.mean() + log_sigma.mean()
    loss.backward()
    for p in model.encoder.parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_unfreeze_encoder_restores_full_trainable_set():
    model = _DepthHead(image_size=16, n_frames=2, base=8)
    _apply_freeze_encoder(model, freeze=True)
    trainable = _apply_freeze_encoder(model, freeze=False)
    assert all(p.requires_grad for p in model.parameters())
    assert {id(p) for p in trainable} == {id(p) for p in model.parameters()}


def test_depth_cfg_defaults_to_effective_grad_clip(tmp_path):
    config = tmp_path / "minimal.yaml"
    config.write_text("world_model:\n  depth_head: {}\n")
    assert _load_depth_cfg(config)["grad_clip"] == pytest.approx(5.0)


def test_depth_head_base64_forward_shapes():
    """Capacity-lift width (base=64) must keep [1b] D̂/logσ spatial contract."""
    model = _DepthHead(image_size=16, n_frames=2, base=64)
    rgb = torch.randint(0, 256, (1, 3, 16, 16, 3), dtype=torch.uint8)
    depth, log_sigma = model.predict_from_window(rgb)
    assert depth.shape == (1, 16, 16)
    assert log_sigma.shape == (1, 16, 16)
    assert torch.all(depth > 0)
    # Wider net should expose more params than base=32 at same spatial size.
    n64 = sum(p.numel() for p in model.parameters())
    n32 = sum(p.numel() for p in _DepthHead(image_size=16, n_frames=2, base=32).parameters())
    assert n64 > n32


def test_base_cli_wins_over_yaml(monkeypatch, tmp_path):
    cfg = {
        "n_frames": 4,
        "base": 32,
        "delta_weight": 0.0,
        "approach_oversample": 1,
        "enable": False,
        "image_size": 16,
        "lr": 1.0e-4,
        "grad_clip": 5.0,
        "absrel_weight": 1.0,
        "nll_weight": 0.1,
        "max_depth_m": 200.0,
        "scale_depth_min_m": 1.0,
        "scale_depth_max_m": 40.0,
        "freeze_encoder": False,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    monkeypatch.setattr(
        "experiments.aerial.rl.train_depth_head._refuse_bad_corpus",
        lambda root, allow: None,
    )
    monkeypatch.setattr(
        "experiments.aerial.rl.train_depth_head._load_depth_cfg",
        lambda path: cfg,
    )

    def stop_after_overrides(root, window):
        assert cfg["base"] == 64
        raise RuntimeError("base override observed")

    monkeypatch.setattr(
        "experiments.aerial.rl.train_depth_head._usable_episodes",
        stop_after_overrides,
    )
    with pytest.raises(RuntimeError, match="base override observed"):
        train_depth_main(
            [
                "--dataset",
                str(tmp_path),
                "--device",
                "cpu",
                "--base",
                "64",
                "--approach-oversample",
                "1",
            ]
        )


def test_approach_oversample_cli_wins_over_yaml(monkeypatch, tmp_path):
    cfg = {
        "n_frames": 4,
        "delta_weight": 0.0,
        "approach_oversample": 4,
        "enable": False,
    }
    monkeypatch.setattr(
        "experiments.aerial.rl.train_depth_head._refuse_bad_corpus",
        lambda root, allow: None,
    )
    monkeypatch.setattr(
        "experiments.aerial.rl.train_depth_head._load_depth_cfg",
        lambda path: cfg,
    )

    def stop_after_overrides(root, window):
        assert cfg["approach_oversample"] == 1
        raise RuntimeError("override observed")

    monkeypatch.setattr(
        "experiments.aerial.rl.train_depth_head._usable_episodes",
        stop_after_overrides,
    )
    with pytest.raises(RuntimeError, match="override observed"):
        train_depth_main(
            [
                "--dataset",
                str(tmp_path),
                "--device",
                "cpu",
                "--approach-oversample",
                "1",
                "--eval-every",
                "50",
            ]
        )
