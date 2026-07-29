import json
from pathlib import Path

import numpy as np
import pytest

from experiments.aerial.eval.run_closed_loop import (
    MockBridge,
    ReplayPolicy,
    apply_body_delta,
    evaluate_episodes,
    load_annotation,
    run_episode,
    write_metrics,
)
from experiments.aerial.openfly_actions import primitive_to_delta

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_openfly" / "seen_mini.json"


def test_apply_body_delta_forward():
    pos = np.array([0.0, 0.0, 1.0])
    delta = primitive_to_delta(1)
    new_pos, new_yaw = apply_body_delta(pos, 0.0, delta)
    assert np.allclose(new_pos, [3.0, 0.0, 1.0], atol=1e-6)
    assert new_yaw == pytest.approx(0.0)


def test_apply_body_delta_wraps_yaw():
    """Recorded yaw feeds the state normalizer; unwrapped drift wrecks its min/max."""
    yaw = 0.0
    for _ in range(83):
        _, yaw = apply_body_delta(np.zeros(3), yaw, primitive_to_delta(2))
    assert -np.pi < yaw <= np.pi

    _, wrapped = apply_body_delta(np.zeros(3), np.pi - 0.1, np.array([0.0, 0.0, 0.0, 0.3]))
    assert wrapped == pytest.approx(np.pi - 0.1 + 0.3 - 2 * np.pi)


def test_mock_bridge_replay_episode_success():
    episode = load_annotation(FIXTURE)[0]
    bridge = MockBridge(seed=7)
    policy = ReplayPolicy(episode["action"])
    result = run_episode(bridge, policy, episode, max_steps=20)
    assert result.success
    assert result.navigation_error == pytest.approx(0.0, abs=1e-5)
    assert result.path_length == pytest.approx(6.0, abs=1e-5)


def test_evaluate_episodes_mock_writes_finite_metrics(tmp_path):
    episodes = load_annotation(FIXTURE)
    out = tmp_path / "metrics_mock.json"
    metrics = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=17,
    )
    write_metrics(metrics, out)

    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["n"] == float(len(episodes))
    for key in ("SR", "NE", "SPL"):
        assert key in loaded
        assert np.isfinite(float(loaded[key]))


def test_evaluate_episodes_preserves_per_episode_metrics():
    episodes = load_annotation(FIXTURE)[:2]
    metrics = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=42,
    )
    assert len(metrics["episodes"]) == 2
    assert set(metrics["episodes"][0]) == {
        "episode_id",
        "success",
        "NE",
        "path_length",
        "shortest_length",
        "steps",
    }


def test_mock_metrics_reproducible_with_seed():
    episodes = load_annotation(FIXTURE)
    first = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=99,
    )
    second = evaluate_episodes(
        episodes,
        bridge_name="mock",
        policy_name="replay",
        max_steps=50,
        seed=99,
    )
    assert first == second
