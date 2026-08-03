import math

import numpy as np
import pytest

from experiments.aerial.verify_aerial_source import check_action_sample, DEFAULT_CRITERIA


def test_valid_4d_action_passes():
    action = np.array([3.0, 0.0, 0.0, math.pi / 6])
    ok, notes = check_action_sample(action, DEFAULT_CRITERIA)
    assert ok is True
    assert notes == []


def test_wrong_action_dim_fails():
    action = np.array([1.0, 2.0, 3.0])
    ok, notes = check_action_sample(action, DEFAULT_CRITERIA)
    assert ok is False
    assert any("action dim" in n for n in notes)


def test_translation_out_of_bounds_fails():
    action = np.array([16.0, 0.0, 0.0, 0.0])
    ok, notes = check_action_sample(action, DEFAULT_CRITERIA)
    assert ok is False
    assert any("translation" in n for n in notes)


def test_yaw_out_of_bounds_fails():
    action = np.array([0.0, 0.0, 0.0, math.pi + 0.1])
    ok, notes = check_action_sample(action, DEFAULT_CRITERIA)
    assert ok is False
    assert any("yaw" in n for n in notes)
