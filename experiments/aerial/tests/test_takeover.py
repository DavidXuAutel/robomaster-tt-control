from __future__ import annotations

import inspect

import pytest

from experiments.aerial.takeover import TakeoverConfig, TakeoverController, freeze_thresholds


def test_freeze_thresholds_low_p95():
    config = freeze_thresholds(1.0)

    assert config.takeover_m == pytest.approx(9.0)
    assert config.release_m == pytest.approx(6.0)
    assert config.abort_m == pytest.approx(30.0)


def test_freeze_thresholds_high_p95():
    config = freeze_thresholds(5.0)

    assert config.takeover_m == pytest.approx(15.0)
    assert config.release_m == pytest.approx(10.0)
    assert config.abort_m == pytest.approx(45.0)


def test_controller_enters_expert_when_cross_track_exceeds_takeover():
    config = freeze_thresholds(1.0)
    controller = TakeoverController(config)

    decision = controller.step(cross_track_m=10.0, progress_m=0.0)

    assert decision.mode == "expert"
    assert decision.intervene is True


def test_step_does_not_accept_ne():
    signature = inspect.signature(TakeoverController.step)
    parameter_names = list(signature.parameters)

    assert "ne" not in parameter_names
    assert "cross_track_m" in parameter_names
    assert "progress_m" in parameter_names


def test_abort_after_no_progress_gain_for_twenty_steps():
    config = TakeoverConfig(
        takeover_m=9.0,
        release_m=6.0,
        abort_m=30.0,
        stall_steps=99,
        no_progress_abort_steps=20,
    )
    controller = TakeoverController(config)

    for _ in range(19):
        decision = controller.step(cross_track_m=1.0, progress_m=0.0)
        assert decision.mode != "abort"

    decision = controller.step(cross_track_m=1.0, progress_m=0.0)

    assert decision.mode == "abort"
    assert decision.intervene is False


def test_release_resets_stall_counter_before_policy_resumes():
    controller = TakeoverController(
        TakeoverConfig(
            takeover_m=5.0,
            release_m=4.0,
            abort_m=30.0,
            stall_steps=3,
            release_stable_steps=1,
            no_progress_abort_steps=20,
        )
    )
    assert controller.step(cross_track_m=6.0, progress_m=0.0).mode == "expert"
    assert controller.step(cross_track_m=3.0, progress_m=0.0).mode == "policy"

    decision = controller.step(cross_track_m=3.0, progress_m=0.0)

    assert decision.mode == "policy"


def test_stall_takeover_does_not_reuse_no_progress_budget_after_release():
    controller = TakeoverController(
        TakeoverConfig(
            takeover_m=10.0,
            release_m=4.0,
            abort_m=30.0,
            stall_steps=2,
            release_stable_steps=1,
            no_progress_abort_steps=4,
        )
    )
    assert controller.step(cross_track_m=5.0, progress_m=0.0).mode == "policy"
    assert controller.step(cross_track_m=5.0, progress_m=0.0).mode == "expert"
    assert controller.step(cross_track_m=3.0, progress_m=0.0).mode == "policy"

    decision = controller.step(cross_track_m=3.0, progress_m=0.0)

    assert decision.mode == "policy"
    assert decision.reason == "policy_control"
