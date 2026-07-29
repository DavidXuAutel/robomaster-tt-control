from __future__ import annotations

import numpy as np
import pytest

from experiments.aerial.path_expert import PathExpert


def _line_episode(*, yaw: float = 0.0) -> dict[str, list[list[float]] | list[float]]:
    return {
        "pos": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        "yaw": [yaw, yaw, yaw],
    }


def test_projection_onto_straight_polyline():
    expert = PathExpert()
    expert.reset(_line_episode())

    label = expert.label(np.array([2.0, 3.0, 0.0]), 0.0)

    assert label.progress_m == pytest.approx(2.0)
    assert label.cross_track_m == pytest.approx(3.0)
    np.testing.assert_allclose(label.lookahead_pos, [8.0, 0.0, 0.0])
    np.testing.assert_allclose(label.action, [6.0, -3.0, 0.0, 0.0])


def test_monotone_cursor_never_decreases_after_forward_motion():
    expert = PathExpert()
    expert.reset(_line_episode())
    forward = expert.label(np.array([8.0, 0.1, 0.0]), 0.0)

    behind = expert.label(np.array([3.0, -0.1, 0.0]), 0.0)

    assert behind.progress_m >= forward.progress_m
    assert forward.action.shape == (4,)


def test_lookahead_shortens_near_goal():
    expert = PathExpert()
    expert.reset(_line_episode())

    label = expert.label(np.array([17.0, 0.0, 0.0]), 0.0)

    assert label.progress_m == pytest.approx(17.0)
    np.testing.assert_allclose(label.lookahead_pos, [20.0, 0.0, 0.0])
    np.testing.assert_allclose(label.action, [3.0, 0.0, 0.0, 0.0])


def test_returned_action_is_clipped_to_training_ranges():
    expert = PathExpert()
    expert.reset(_line_episode(yaw=2.0))

    label = expert.label(np.array([0.0, -100.0, -100.0]), 0.0)

    np.testing.assert_allclose(
        label.action,
        [6.0, 7.794228553771973, 3.0, 0.5235987901687622],
    )


def test_lookahead_yaw_interpolates_across_wrap_boundary():
    expert = PathExpert()
    expert.reset(
        {
            "pos": [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            "yaw": [np.pi - 0.1, -np.pi + 0.1],
        }
    )

    label = expert.label(np.array([0.0, 0.0, 0.0]), np.pi - 0.1)

    assert label.action[3] == pytest.approx(0.1)


def test_projection_tie_prefers_earliest_repeated_endpoint():
    expert = PathExpert()
    expert.reset(
        {
            "pos": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "yaw": [0.0, 0.0, 0.0],
        }
    )

    label = expert.label(np.array([0.0, 0.0, 0.0]), 0.0)

    assert label.progress_m == pytest.approx(0.0)
    np.testing.assert_allclose(label.lookahead_pos, [6.0, 0.0, 0.0])
