import numpy as np
import pytest

from experiments.aerial.rl.reward import NavigationReward, RewardConfig, reward_terms
from experiments.aerial.rl.env.obs import Observation


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
