"""Unit tests for local VIO math + V0 numeric gate scorers (torch-free)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.aerial.rl import v0_metrics as m
from experiments.aerial.rl import vio


def test_integrate_velocity_constant():
    L = 5
    ts = np.linspace(0.0, 1.0, L, dtype=np.float32)
    vel = np.zeros((L, 3), dtype=np.float32)
    vel[:, 0] = 2.0  # 2 m/s
    pos, dt = vio.integrate_velocity(vel, ts)
    assert pos.shape == (L, 3)
    # ~2 m over 1 s
    assert abs(float(pos[-1, 0]) - 2.0) < 0.05
    assert np.allclose(pos[:, 1:], 0.0)
    assert np.all(dt[1:] > 0)


def test_scale_relative_error_masks_low_motion():
    s_d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    s_v = np.array([1.0, 0.1, 3.0], dtype=np.float32)  # middle below min_motion
    out = vio.scale_relative_error(s_d, s_v, min_motion_m=0.5)
    assert out["n_valid"] == 2
    assert not out["valid"][1]
    assert abs(float(out["median_rel_err"])) < 1e-5


def test_window_scale_report_aligned_depth_motion():
    B, L, H, W = 2, 8, 4, 4
    ts = np.linspace(0.0, 1.0, L, dtype=np.float32)
    timestamps = np.stack([ts, ts], axis=0)
    vel = np.zeros((B, L, 3), dtype=np.float32)
    vel[..., 0] = 1.0
    depth = np.ones((B, L, H, W), dtype=np.float32)
    for t in range(L):
        depth[:, t] = 5.0 - ts[t]
    report = vio.window_scale_report(vel, timestamps, depth, min_motion_m=0.5)
    assert report["n_valid"] == 2
    assert float(report["median_rel_err"]) <= 0.25


def test_depth_median_masks_outdoor_far_plane():
    """Sky/far-plane fill must not dominate ŝ_D (AbsRel uses the same 200 m cap)."""
    H, W = 4, 4
    # ≥50% sky pixels so an unmasked median is pulled to the far plane; masked
    # keeps the navigational near field (same 200 m cap as AbsRel / DepthHead).
    near = np.full((H, W), 10.0, dtype=np.float32)
    near[:, W // 2 :] = 5000.0
    near_next = np.full((H, W), 9.0, dtype=np.float32)
    near_next[:, W // 2 :] = 4000.0
    depth = np.stack([near, near_next], axis=0)[None, ...]  # [1, 2, H, W]
    unmasked = vio.depth_median(depth, max_depth_m=None)
    masked = vio.depth_median(depth, max_depth_m=200.0)
    assert float(unmasked[0, 0]) > 100.0
    assert abs(float(masked[0, 0]) - 10.0) < 1e-3
    assert abs(float(masked[0, 1]) - 9.0) < 1e-3
    s_d = vio.scale_from_depth_change(masked)
    assert abs(float(s_d[0]) - 1.0) < 1e-3


def test_sample_scale_windows_forward_filter():
    from experiments.aerial.rl._v0_gate import _window_forward_frac
    from experiments.aerial.rl.buffer import Transition
    from experiments.aerial.rl.env.obs import Observation

    def _tr(pos):
        state = np.array([pos[0], pos[1], pos[2], 0, 0, 0, 0], dtype=np.float32)
        obs = Observation(
            rgb=np.zeros((4, 4, 3), np.uint8),
            state=state,
            collided=False,
            depth=np.ones((4, 4), np.float32),
            imu={},
            t=0.0,
        )
        return Transition(obs=obs, action=np.zeros(4, np.float32), reward=0.0, done=False)

    forward = [_tr([0, 0, 0]), _tr([1.0, 0.1, 0])]
    sideways = [_tr([0, 0, 0]), _tr([0.1, 1.0, 0])]
    assert _window_forward_frac(forward) > 0.5
    assert _window_forward_frac(sideways) < 0.5


def test_learning_curves_pass_and_fail():
    thr = m.DEFAULT_THRESHOLDS
    ok = m.check_learning_curves(
        [10.0] * 50 + [4.0] * 50,
        [0.2] * 50 + [0.05] * 50,
        [0.5] * 100,
        thr=thr,
    )
    assert ok["ok"]
    bad = m.check_learning_curves(
        [1.0] * 100,  # no drop
        [0.1] * 100,
        [0.5] * 100,
        thr=thr,
    )
    assert not bad["ok"]
    collapsed = m.check_learning_curves(
        [10.0] * 50 + [4.0] * 50,
        [0.2] * 50 + [0.05] * 50,
        [0.01] * 100,  # below 0.10 floor
        thr=thr,
    )
    assert not collapsed["ok"]


def test_progress_vs_random_or_logic():
    thr = m.DEFAULT_THRESHOLDS
    # progress margin only
    r = m.check_progress_vs_random(
        [20.0] * 16, [10.0] * 16, [40.0] * 16, [40.0] * 16, thr=thr
    )
    assert r["ok"] and r["progress_ok"]
    # distance margin only
    r2 = m.check_progress_vs_random(
        [1.0] * 16, [1.0] * 16, [10.0] * 16, [20.0] * 16, thr=thr
    )
    assert r2["ok"] and r2["dist_ok"]


def test_depth_absrel_threshold():
    gt = np.ones((4, 4), dtype=np.float32) * 10.0
    pred = gt * 1.1  # AbsRel = 0.1
    assert m.check_depth_absrel(pred, gt)["ok"]
    pred_bad = gt * 2.0  # AbsRel = 1.0
    assert not m.check_depth_absrel(pred_bad, gt)["ok"]


def test_shield_effectiveness():
    thr = m.DEFAULT_THRESHOLDS
    r = m.check_shield_effectiveness(
        interventions_on=[[False, True, False, False]],
        collided_on=[[False, False, False, True]],
        near_coll_on=[[False, False, False, False]],
        near_coll_off=[[True, True, True, False]],
        thr=thr,
    )
    assert r["ok"]
    assert r["before_ok"]
    assert r["ratio_ok"]


def test_aggregate_requires_all_four():
    base = {"ok": True}
    assert not m.aggregate_v0_verdict({"1": base, "2": base})["ok"]
    v = m.aggregate_v0_verdict({k: base for k in ("1", "2", "3", "4")})
    assert v["ok"]


def test_v0_gate_self_check_entrypoint():
    from experiments.aerial.rl._v0_gate import main
    assert main(["--self-check"]) == 0
