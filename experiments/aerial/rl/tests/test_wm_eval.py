"""Imagination-fidelity harness + verdict tests (pure numpy, green on the Mac).

Two jobs, mirroring the 2a/2b split:
  1. Pin the metric primitives (AUROC / growth-bounded / masked MAE) to
     hand-computed values — this is the single source of truth the torch H100
     eval (``_wm_fidelity_eval``) reuses.
  2. Exercise the harness end-to-end on ``StubLatentDynamics`` and the
     PASS/FAIL verdict logic on synthetic traces (perfect predictor passes,
     constant predictor fails, super-linear error is caught).
"""
import numpy as np
import pytest

from experiments.aerial.rl import wm_eval
from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.dynamics import StubLatentDynamics
from experiments.aerial.rl.env.obs import Observation


def _obs(pos=(0.0, 0.0, 0.0), yaw=0.0, collided=False):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, yaw], dtype=np.float32)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    return Observation(rgb=rgb, state=state, collided=collided)


def _tr(pos, reward, done=False, collided=False, action=(0.5, 0.0, 0.0, 0.0)):
    return Transition(
        obs=_obs(pos=pos, collided=collided),
        action=np.array(action, dtype=np.float32),
        reward=float(reward),
        done=bool(done),
    )


def _trace(reward_pred, reward_real, p_coll_pred, collided_real,
           done_pred, done_real, latent_norm=None):
    H = len(reward_pred)
    lat = latent_norm if latent_norm is not None else np.ones(H + 1)
    return wm_eval.RolloutTrace(
        reward_pred=np.asarray(reward_pred, float),
        p_coll_pred=np.asarray(p_coll_pred, float),
        done_pred=np.asarray(done_pred, bool),
        reward_real=np.asarray(reward_real, float),
        collided_real=np.asarray(collided_real, bool),
        done_real=np.asarray(done_real, bool),
        latent_norm=np.asarray(lat, float),
        valid=np.ones(H, dtype=bool),
    )


# -- metric primitives (hand-computed) ---------------------------------------
def test_auroc_matches_known_value():
    # classic example: sklearn roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.75
    au = wm_eval.auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1]))
    assert au == pytest.approx(0.75)


def test_auroc_perfect_and_ties():
    assert wm_eval.auroc(np.array([0.1, 0.2, 0.9, 0.95]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    # all-tied scores -> 0.5 (chance)
    assert wm_eval.auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_auroc_undefined_when_class_absent():
    assert np.isnan(wm_eval.auroc(np.array([0.2, 0.7]), np.array([0, 0])))


def test_growth_bounded_flat_linear_pass_superlinear_fail():
    assert wm_eval.growth_bounded(np.array([1.0, 1.0, 1.0]))          # flat
    assert wm_eval.growth_bounded(np.array([1.0, 2.0, 3.0]))          # linear (<= ceiling 3)
    assert not wm_eval.growth_bounded(np.array([1.0, 2.0, 10.0]))     # super-linear
    assert wm_eval.growth_bounded(np.array([1e-12, 1e-12]))           # near-zero base -> pass


def test_masked_mae_curve_ignores_invalid():
    pred = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 9.0]])
    real = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    valid = np.array([[True, True, True], [True, True, False]])
    curve = wm_eval._masked_mae_curve(pred, real, valid)
    # h0: |0|,|0| -> 0 ; h1: |1|,|1| -> 1 ; h2: only traj0 valid -> |2| = 2
    np.testing.assert_allclose(curve, [0.0, 1.0, 2.0])


# -- harness plumbing on the real stub ---------------------------------------
def test_open_loop_rollout_shapes_and_alignment():
    dyn = StubLatentDynamics(goal=np.array([10.0, 0.0, 0.0]), latent_dim=8)
    window = [_tr((float(i), 0.0, 0.0), reward=float(i)) for i in range(5)]
    tr = wm_eval.open_loop_rollout(dyn, window)
    assert tr.reward_pred.shape == (5,)
    assert tr.latent_norm.shape == (6,)
    # real signals are read straight off the window (alignment contract).
    np.testing.assert_allclose(tr.reward_real, [0, 1, 2, 3, 4])
    assert tr.valid.all()


def test_open_loop_rollout_valid_mask_stops_after_done():
    dyn = StubLatentDynamics(goal=np.array([10.0, 0.0, 0.0]), latent_dim=8)
    window = [
        _tr((0.0, 0.0, 0.0), 1.0),
        _tr((1.0, 0.0, 0.0), 1.0),
        _tr((2.0, 0.0, 0.0), 1.0, done=True),   # first real done at t=2
        _tr((3.0, 0.0, 0.0), 1.0),
        _tr((4.0, 0.0, 0.0), 1.0),
    ]
    tr = wm_eval.open_loop_rollout(dyn, window)
    # the done step itself stays valid; everything after is masked out.
    np.testing.assert_array_equal(tr.valid, [True, True, True, False, False])


def test_open_loop_rollout_caps_horizon():
    dyn = StubLatentDynamics(goal=np.array([10.0, 0.0, 0.0]), latent_dim=8)
    window = [_tr((float(i), 0.0, 0.0), 1.0) for i in range(30)]
    tr = wm_eval.open_loop_rollout(dyn, window, max_horizon=15)
    assert tr.reward_pred.shape == (15,)


# -- verdict logic on synthetic traces ---------------------------------------
def _perfect_traces():
    # rewards vary (so the constant-mean baseline has nonzero error), predictions
    # are exact, one colliding + one clean trajectory, done predicted correctly.
    rr = [0.0, 1.0, 2.0, 1.0, 0.0]
    t_clean = _trace(
        reward_pred=rr, reward_real=rr,
        p_coll_pred=[0.0, 0.0, 0.1, 0.0, 0.0], collided_real=[False] * 5,
        done_pred=[False, False, False, False, True], done_real=[False, False, False, False, True],
    )
    t_coll = _trace(
        reward_pred=rr, reward_real=rr,
        p_coll_pred=[0.1, 0.3, 0.9, 0.0, 0.0], collided_real=[False, False, True, False, False],
        done_pred=[False, False, True, False, False], done_real=[False, False, True, False, False],
    )
    return [t_clean, t_coll]


def test_verdict_passes_on_perfect_predictor():
    agg = wm_eval.aggregate(_perfect_traces())
    verdict = wm_eval.fidelity_verdict(agg)
    assert verdict["reward_ok"] and verdict["coll_ok"] and verdict["done_ok"]
    assert verdict["passed"]
    assert agg["coll_auroc"] == pytest.approx(1.0)


def test_verdict_reward_fails_on_constant_mean_predictor():
    # predict the global mean everywhere == the baseline -> cannot beat it.
    traces = _perfect_traces()
    mean = float(np.mean([0.0, 1.0, 2.0, 1.0, 0.0]))
    for t in traces:
        t.reward_pred[:] = mean
    agg = wm_eval.aggregate(traces)
    verdict = wm_eval.fidelity_verdict(agg)
    assert not verdict["reward_ok"]
    assert not verdict["passed"]


def test_verdict_reward_fails_on_superlinear_error():
    rr = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    # small 1-step error but blowing up with horizon
    bad = rr + np.array([0.05, 0.1, 0.5, 3.0, 20.0])
    t = _trace(
        reward_pred=bad, reward_real=rr,
        p_coll_pred=[0.1, 0.3, 0.9, 0.0, 0.0], collided_real=[False, False, True, False, False],
        done_pred=[False] * 5, done_real=[False] * 5,
    )
    t2 = _trace(
        reward_pred=rr, reward_real=rr,
        p_coll_pred=[0.0] * 5, collided_real=[False] * 5,
        done_pred=[False] * 5, done_real=[False] * 5,
    )
    agg = wm_eval.aggregate([t, t2])
    verdict = wm_eval.fidelity_verdict(agg)
    assert not verdict["reward_growth_ok"]
    assert not verdict["reward_ok"]


def test_verdict_coll_fails_when_no_collisions_present():
    rr = [0.0, 1.0, 2.0]
    t = _trace(
        reward_pred=rr, reward_real=rr,
        p_coll_pred=[0.1, 0.2, 0.3], collided_real=[False, False, False],
        done_pred=[False, False, False], done_real=[False, False, False],
    )
    agg = wm_eval.aggregate([t, t])
    assert np.isnan(agg["coll_auroc"])
    verdict = wm_eval.fidelity_verdict(agg)
    assert not verdict["coll_ok"]


def test_evaluate_end_to_end_on_stub():
    dyn = StubLatentDynamics(goal=np.array([10.0, 0.0, 0.0]), latent_dim=8)
    windows = [[_tr((float(i), 0.0, 0.0), 1.0) for i in range(6)] for _ in range(3)]
    out = wm_eval.evaluate(dyn, windows)
    assert "agg" in out and "verdict" in out
    assert out["agg"]["horizon"] == 6
    assert set(out["verdict"]) >= {"reward_ok", "coll_ok", "done_ok", "passed"}
