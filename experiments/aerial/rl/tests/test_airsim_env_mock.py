import sys
import pathlib

import numpy as np
import pytest

from experiments.aerial.rl.env.mock_env import MockAirSimDroneEnv, MockEnvConfig
from experiments.aerial.rl.env.obs import Observation, depth_sanity_detail

# Reuse the already-validated numerical gates from sim_verify.
_SIM_VERIFY = pathlib.Path(__file__).resolve().parents[2] / "sim_verify"
if str(_SIM_VERIFY) not in sys.path:
    sys.path.insert(0, str(_SIM_VERIFY))
from lib import sanity  # noqa: E402


EPISODE = {"pos": [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], "yaw": [0.0, 0.0], "gpt_instruction": "go"}


def test_reset_returns_observation_with_shapes():
    env = MockAirSimDroneEnv(MockEnvConfig(width=224, height=224))
    obs = env.reset(EPISODE)
    assert isinstance(obs, Observation)
    assert obs.rgb.shape == (224, 224, 3)
    assert obs.rgb.dtype == np.uint8
    assert obs.state.shape == (7,)
    assert obs.depth.shape == (224, 224)
    assert obs.proprio4().shape == (4,)
    assert np.allclose(obs.position, [0.0, 0.0, 0.0])
    assert np.allclose(env.goal, [9.0, 0.0, 0.0])


def test_mock_obs_passes_sanity_gates():
    env = MockAirSimDroneEnv()
    obs = env.reset(EPISODE)
    ok_imu, note = sanity.imu_ok(obs.imu)
    assert ok_imu, note
    ok_depth, note = sanity.depth_ok(depth_sanity_detail(obs.depth))
    assert ok_depth, note


def test_step_moves_forward_and_updates_velocity():
    env = MockAirSimDroneEnv(MockEnvConfig(step_hz=30.0))
    env.reset(EPISODE)
    obs, info = env.step(np.array([3.0, 0.0, 0.0, 0.0]))
    assert np.allclose(obs.position, [3.0, 0.0, 0.0], atol=1e-6)
    # velocity = displacement / dt
    assert obs.velocity[0] == pytest.approx(3.0 * 30.0, rel=1e-6)
    assert not obs.collided


def test_collision_flag_when_out_of_bounds():
    env = MockAirSimDroneEnv(MockEnvConfig(bounds_m=5.0))
    env.reset(EPISODE)
    obs, _ = env.step(np.array([9.0, 0.0, 0.0, 0.0]))  # clips to 9 but >5 bound
    assert obs.collided


def test_depth_and_imu_are_supervision_only_shapes():
    # Regression guard for the RGB-only boundary: proprio4 is 4-D (no depth/imu).
    env = MockAirSimDroneEnv()
    obs = env.reset(EPISODE)
    assert obs.proprio4().shape == (4,)
    assert "lin_acc" in obs.imu and "ang_vel" in obs.imu
