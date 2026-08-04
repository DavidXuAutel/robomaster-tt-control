import numpy as np
import pytest

from experiments.aerial.rl.reward import (
    DEFAULT_ONLINE_SUCCESS_DIST_M,
    EVAL_SUCCESS_DIST_M,
    NavigationReward,
    RewardConfig,
    maneuver_weight_at,
    reward_terms,
)
from experiments.aerial.rl.env.obs import Observation


def test_bare_config_default_is_tight_online_radius_not_eval():
    # A RewardConfig()/NavigationReward() built WITHOUT the YAML path must still
    # use the tight online radius (3 m), never the loose 20 m eval SR radius.
    assert RewardConfig().success_dist_m == pytest.approx(DEFAULT_ONLINE_SUCCESS_DIST_M)
    assert DEFAULT_ONLINE_SUCCESS_DIST_M < EVAL_SUCCESS_DIST_M
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]))
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.zeros(3))
    # 5 m out: arrived under a 20 m radius, NOT under the 3 m default.
    _, done, terms = r.step(_obs([5.0, 0.0, 0.0]), np.zeros(4))
    assert not done
    assert terms["arrived"] == 0.0


def _obs(pos, collided=False):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=state,
        collided=collided,
    )


def test_reward_terms_signs():
    cfg = RewardConfig(w_progress=1.0, w_collision=10.0, w_maneuver=0.1)
    t = reward_terms(progress=2.0, collision_risk=0.0, maneuver_cost=0.0, cfg=cfg)
    assert t["reward"] == pytest.approx(2.0)
    # collision subtracts; maneuver subtracts
    t2 = reward_terms(progress=0.0, collision_risk=1.0, maneuver_cost=3.0, cfg=cfg)
    assert t2["reward"] == pytest.approx(-10.0 - 0.3)


def test_progress_positive_when_approaching_goal():
    r = NavigationReward(goal=None, cfg=RewardConfig(success_dist_m=1.0))
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.array([0.0, 0.0, 0.0]))
    reward, done, terms = r.step(_obs([3.0, 0.0, 0.0]), np.zeros(4))
    assert terms["progress"] == pytest.approx(3.0)
    assert reward > 0
    assert not done


def test_collision_makes_done_and_penalizes():
    cfg = RewardConfig(w_progress=1.0, w_collision=10.0, w_maneuver=0.0, success_dist_m=1.0)
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.array([0.0, 0.0, 0.0]))
    reward, done, terms = r.step(_obs([1.0, 0.0, 0.0], collided=True), np.zeros(4))
    assert done
    assert terms["collision_risk"] == 1.0
    assert reward < 0  # collision penalty dominates 1m of progress


def test_arrival_bonus_and_done():
    cfg = RewardConfig(success_dist_m=1.0, success_bonus=10.0, w_maneuver=0.0)
    r = NavigationReward(goal=np.array([5.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([5.0, 0.0, 0.0]), start_pos=np.array([0.0, 0.0, 0.0]))
    reward, done, terms = r.step(_obs([4.8, 0.0, 0.0]), np.zeros(4))
    assert done
    assert terms["arrived"] == 1.0
    assert reward >= cfg.success_bonus  # bonus applied on top of progress


def test_maneuver_cost_is_action_norm():
    cfg = RewardConfig(w_progress=0.0, w_collision=0.0, w_maneuver=1.0)
    r = NavigationReward(goal=None, cfg=cfg)
    r.reset(goal=None, start_pos=np.zeros(3))
    action = np.array([3.0, 4.0, 0.0, 0.0])  # norm 5
    reward, done, terms = r.step(_obs([0.0, 0.0, 0.0]), action)
    assert terms["maneuver_cost"] == pytest.approx(5.0)
    assert reward == pytest.approx(-5.0)


def test_no_goal_yields_zero_progress():
    r = NavigationReward(goal=None)
    r.reset(goal=None, start_pos=np.zeros(3))
    _, _, terms = r.step(_obs([9.0, 9.0, 9.0]), np.zeros(4))
    assert terms["progress"] == 0.0


# --- maneuver-penalty curriculum --------------------------------------------

def test_curriculum_is_noop_when_final_equals_start():
    # default config: final == start -> flat regardless of the metric
    cfg = RewardConfig(w_maneuver=0.01)
    for m in (-100.0, 0.0, 5.0, 1000.0):
        assert maneuver_weight_at(m, cfg) == pytest.approx(0.01)


def test_curriculum_flat_below_threshold_then_ramps():
    cfg = RewardConfig(
        w_maneuver=0.01, w_maneuver_final=0.05,
        maneuver_curriculum_threshold=10.0, maneuver_curriculum_ramp=10.0,
    )
    assert maneuver_weight_at(0.0, cfg) == pytest.approx(0.01)   # below threshold
    assert maneuver_weight_at(10.0, cfg) == pytest.approx(0.01)  # at threshold
    assert maneuver_weight_at(15.0, cfg) == pytest.approx(0.03)  # halfway up the ramp
    assert maneuver_weight_at(20.0, cfg) == pytest.approx(0.05)  # top of the ramp
    assert maneuver_weight_at(999.0, cfg) == pytest.approx(0.05)  # never exceeds final


def test_curriculum_monotone_nondecreasing():
    cfg = RewardConfig(
        w_maneuver=0.01, w_maneuver_final=0.05,
        maneuver_curriculum_threshold=0.0, maneuver_curriculum_ramp=20.0,
    )
    ws = [maneuver_weight_at(m, cfg) for m in range(0, 30)]
    assert all(b >= a for a, b in zip(ws, ws[1:]))
    assert all(0.01 <= w <= 0.05 for w in ws)


def test_curriculum_zero_ramp_is_a_step():
    cfg = RewardConfig(
        w_maneuver=0.01, w_maneuver_final=0.05,
        maneuver_curriculum_threshold=10.0, maneuver_curriculum_ramp=0.0,
    )
    assert maneuver_weight_at(9.9, cfg) == pytest.approx(0.01)
    assert maneuver_weight_at(10.0, cfg) == pytest.approx(0.05)  # jumps at threshold


def test_curriculum_w_start_override_prevents_feedback():
    # simulate the corrector mutating cfg.w_maneuver: passing the snapshot as
    # w_start keeps the schedule anchored to the base, not the last output.
    cfg = RewardConfig(
        w_maneuver=0.03,  # already ramped up in a prior iter
        w_maneuver_final=0.05,
        maneuver_curriculum_threshold=0.0, maneuver_curriculum_ramp=10.0,
    )
    # metric=5 -> halfway; anchored to the true start 0.01, not the mutated 0.03
    assert maneuver_weight_at(5.0, cfg, w_start=0.01) == pytest.approx(0.03)
