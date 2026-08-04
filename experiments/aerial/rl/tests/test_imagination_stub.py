import numpy as np
import pytest

from experiments.aerial.rl.dynamics import StubLatentDynamics, DynamicsOutput
from experiments.aerial.rl.imagination import (
    MAX_IMAGINATION_HORIZON,
    imagine,
)
from experiments.aerial.rl.env.obs import Observation


class _ForwardPolicy:
    """Always drive +3 m forward in body frame."""

    def act_latent(self, z):
        return np.array([3.0, 0.0, 0.0, 0.0])


def _obs(pos, yaw=0.0):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, yaw], dtype=np.float32)
    return Observation(rgb=np.zeros((4, 4, 3), np.uint8), state=state)


def test_encode_carries_proprio():
    dyn = StubLatentDynamics(latent_dim=8)
    z = dyn.encode(_obs([1.0, 2.0, 3.0], yaw=0.5))
    assert z.shape == (8,)
    assert np.allclose(z[:4], [1.0, 2.0, 3.0, 0.5])


def test_step_integrates_forward():
    dyn = StubLatentDynamics(goal=np.array([100.0, 0.0, 0.0]), latent_dim=8)
    z = dyn.encode(_obs([0.0, 0.0, 0.0]))
    out = dyn.step(z, np.array([3.0, 0.0, 0.0, 0.0]))
    assert isinstance(out, DynamicsOutput)
    assert np.allclose(out.z_next[:3], [3.0, 0.0, 0.0])
    assert out.progress == pytest.approx(3.0)  # moved 3m closer to goal


def test_p_coll_rises_near_obstacle():
    dyn = StubLatentDynamics(obstacle=np.array([3.0, 0.0, 0.0]), latent_dim=8, collide_radius_m=2.0)
    z = dyn.encode(_obs([0.0, 0.0, 0.0]))
    out = dyn.step(z, np.array([3.0, 0.0, 0.0, 0.0]))  # lands exactly on obstacle
    assert out.p_coll == pytest.approx(1.0)
    assert out.done


def test_imagine_shapes_and_returns():
    dyn = StubLatentDynamics(goal=np.array([100.0, 0.0, 0.0]), latent_dim=8)
    z0 = np.stack([dyn.encode(_obs([0.0, 0.0, 0.0])) for _ in range(5)])
    horizon = 6
    roll = imagine(dyn, _ForwardPolicy(), z0, horizon)
    assert roll.z.shape == (5, horizon + 1, 8)
    assert roll.actions.shape == (5, horizon, 4)
    assert roll.rewards.shape == (5, horizon)
    assert roll.p_coll.shape == (5, horizon)
    assert roll.returns.shape == (5,)
    # forward toward goal -> positive returns
    assert np.all(roll.returns > 0)


def test_imagine_done_masking_stops_accrual():
    dyn = StubLatentDynamics(obstacle=np.array([3.0, 0.0, 0.0]), latent_dim=8, collide_radius_m=2.0)
    z0 = dyn.encode(_obs([0.0, 0.0, 0.0]))[None, :]
    roll = imagine(dyn, _ForwardPolicy(), z0, horizon=5)
    # first step lands on obstacle -> done at t=0, all subsequent done
    assert roll.done[0, 0]
    assert np.all(roll.done[0])


def test_imagine_horizon_cap():
    dyn = StubLatentDynamics(latent_dim=8)
    z0 = dyn.encode(_obs([0.0, 0.0, 0.0]))[None, :]
    with pytest.raises(ValueError):
        imagine(dyn, _ForwardPolicy(), z0, horizon=MAX_IMAGINATION_HORIZON + 1)


def test_dynamics_update_is_v1_gated():
    dyn = StubLatentDynamics(latent_dim=8)
    result = dyn.update(windows=[])
    assert result["skipped"] is True
