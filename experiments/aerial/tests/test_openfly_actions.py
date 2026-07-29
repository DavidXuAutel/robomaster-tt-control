import numpy as np
import pytest

from experiments.aerial.openfly_actions import (
    delta_to_nearest_primitive,
    is_padding_action,
    pos_yaw_to_body_delta,
    primitive_to_delta,
)


def test_padding_actions():
    assert is_padding_action(-1) is True
    assert is_padding_action(-2) is True
    assert is_padding_action(1) is False


def test_forward_3m_primitive():
    d = primitive_to_delta(1)
    np.testing.assert_allclose(d, [3.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_turn_left_30_deg():
    d = primitive_to_delta(2)
    np.testing.assert_allclose(d, [0.0, 0.0, 0.0, np.pi / 6], atol=1e-6)


def test_pos_yaw_body_delta_forward():
    # world +X forward when yaw=0
    d = pos_yaw_to_body_delta(
        pos0=np.array([0.0, 0.0, 10.0]),
        yaw0=0.0,
        pos1=np.array([3.0, 0.0, 10.0]),
        yaw1=0.0,
    )
    np.testing.assert_allclose(d, [3.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_nearest_primitive_roundtrip_forward():
    pid = delta_to_nearest_primitive(np.array([3.0, 0.0, 0.0, 0.0]))
    assert pid == 1


def test_no_backward_primitive_in_table():
    from experiments.aerial.openfly_actions import OPENFLY_PRIMITIVES

    for pid, delta in OPENFLY_PRIMITIVES.items():
        if pid == 0:
            continue
        # forbid large negative body-x (backward)
        assert delta[0] >= -1e-6, f"primitive {pid} has backward dx={delta[0]}"
