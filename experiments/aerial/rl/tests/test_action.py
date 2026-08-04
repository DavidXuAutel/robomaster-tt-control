import math

import numpy as np
import pytest

from experiments.aerial.eval.run_closed_loop import apply_body_delta
from experiments.aerial.rl.env.action import (
    ACTION_DIM,
    DEFAULT_BODY_DELTA_LIMITS,
    body_delta_to_velocity_ned,
    clip_body_delta,
    delta_to_nearest_primitive,
    primitive_to_delta,
)


def test_clip_rejects_non_finite():
    with pytest.raises(ValueError):
        clip_body_delta(np.array([np.nan, 0, 0, 0]))


def test_clip_bounds_per_axis():
    huge = np.array([100.0, 100.0, 100.0, 100.0])
    clipped = clip_body_delta(huge)
    assert np.allclose(clipped, DEFAULT_BODY_DELTA_LIMITS)
    assert clipped.shape == (ACTION_DIM,)


def test_velocity_matches_displacement_over_dt():
    # Forward 3 m at yaw=0 over dt: vx = 3/dt, others 0.
    dt = 1.0 / 30.0
    vx, vy, vz_ned, yaw_rate = body_delta_to_velocity_ned([3.0, 0.0, 0.0, 0.0], 0.0, dt)
    assert vx == pytest.approx(3.0 / dt)
    assert vy == pytest.approx(0.0, abs=1e-9)
    assert vz_ned == pytest.approx(0.0, abs=1e-9)
    assert yaw_rate == pytest.approx(0.0)


def test_velocity_up_is_negative_ned_z():
    dt = 0.1
    _, _, vz_ned, _ = body_delta_to_velocity_ned([0.0, 0.0, 3.0, 0.0], 0.0, dt)
    assert vz_ned < 0  # NED: climb is -z


def test_velocity_heading_consistent_with_apply_body_delta():
    # Integrating the velocity over dt must land where apply_body_delta lands.
    dt = 0.1
    yaw = 0.7
    delta = np.array([2.0, -1.0, 0.5, 0.0])
    vx, vy, vz_ned, _ = body_delta_to_velocity_ned(delta, yaw, dt)
    world_disp = np.array([vx * dt, vy * dt, -vz_ned * dt])  # back to +up
    pos_new, _ = apply_body_delta(np.zeros(3), yaw, delta)
    assert np.allclose(world_disp, pos_new, atol=1e-9)


def test_discrete_fallback_roundtrip():
    for pid in range(10):
        delta = primitive_to_delta(pid)
        assert delta_to_nearest_primitive(delta) == pid
