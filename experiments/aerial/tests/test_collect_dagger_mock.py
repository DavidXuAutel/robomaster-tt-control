from __future__ import annotations

import json

import numpy as np

import experiments.aerial.eval.collect_dagger as dagger
from experiments.aerial.eval.collect_dagger import collect_episode
from experiments.aerial.eval.run_closed_loop import MockBridge, ReplayPolicy
from experiments.aerial.path_expert import PathExpert
from experiments.aerial.takeover import TakeoverConfig, TakeoverController


class OffsetMockBridge(MockBridge):
    def __init__(self, offset_y: float) -> None:
        super().__init__(seed=0)
        self._offset_y = offset_y

    def reset(self, episode: dict) -> None:
        super().reset(episode)
        self.step_delta(np.array([0.0, self._offset_y, 0.0, 0.0]))


def _episode() -> dict:
    return {
        "pos": [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [24.0, 0.0, 0.0]],
        "yaw": [0.0, 0.0, 0.0],
        "action": [1, 1, 0],
        "gpt_instruction": "fly along the line",
    }


def _records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_collects_finite_expert_labels_during_forced_takeover(tmp_path):
    out_path = tmp_path / "ep000.jsonl"
    result = collect_episode(
        OffsetMockBridge(offset_y=10.0),
        ReplayPolicy([1, 1, 0]),
        PathExpert(),
        TakeoverController(
            TakeoverConfig(takeover_m=9.0, release_m=6.0, abort_m=30.0)
        ),
        _episode(),
        out_path,
        max_steps=3,
    )

    records = _records(out_path)
    assert result.status == "completed"
    assert records
    assert any(record["intervention"] for record in records)
    for record in records:
        action = np.asarray(record["action"])
        assert action.shape == (4,)
        assert np.isfinite(action).all()
        assert isinstance(record["intervention"], bool)
        assert (tmp_path / record["observation.images.ego"]).is_file()


def test_abort_truncates_configured_distorted_tail(tmp_path):
    out_path = tmp_path / "ep001.jsonl"
    result = collect_episode(
        OffsetMockBridge(offset_y=8.0),
        ReplayPolicy([6, 6, 6]),
        PathExpert(),
        TakeoverController(
            TakeoverConfig(
                takeover_m=100.0,
                release_m=50.0,
                abort_m=12.0,
                worsen_steps=99,
                stall_steps=99,
                no_progress_abort_steps=99,
            )
        ),
        _episode(),
        out_path,
        max_steps=3,
        abort_tail_frames=1,
    )

    assert result.status == "failed"
    assert result.reason == "cross_track_abort"
    assert len(_records(out_path)) == 1


def test_saved_action_is_expert_label_when_policy_action_differs(tmp_path):
    out_path = tmp_path / "ep002.jsonl"
    collect_episode(
        MockBridge(seed=0),
        ReplayPolicy([6]),
        PathExpert(),
        TakeoverController(
            TakeoverConfig(
                takeover_m=100.0,
                release_m=50.0,
                abort_m=200.0,
                worsen_steps=99,
                stall_steps=99,
                no_progress_abort_steps=99,
            )
        ),
        _episode(),
        out_path,
        max_steps=1,
    )

    record = _records(out_path)[0]
    np.testing.assert_allclose(record["action"], [6.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(record["executed_action"], [0.0, 3.0, 0.0, 0.0])


def test_policy_setup_failure_closes_bridge_and_updates_manifest(
    tmp_path, monkeypatch
):
    class ClosingBridge:
        closed = False

        def close(self):
            self.closed = True

    bridge = ClosingBridge()
    monkeypatch.setattr(dagger, "build_bridge", lambda *args, **kwargs: bridge)

    def fail_policy(*args, **kwargs):
        raise RuntimeError("policy setup failed")

    monkeypatch.setattr(dagger, "build_policy", fail_policy)

    manifest = dagger.collect_episodes(
        [_episode()],
        tmp_path,
        bridge_name="mock",
        policy_name="replay",
        config=TakeoverConfig(takeover_m=9.0, release_m=6.0, abort_m=30.0),
        max_steps=1,
    )

    assert bridge.closed is True
    assert manifest["failed"][0]["reason"] == "policy setup failed"
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_bridge_setup_failure_updates_manifest(tmp_path, monkeypatch):
    def fail_bridge(*args, **kwargs):
        raise RuntimeError("bridge setup failed")

    monkeypatch.setattr(dagger, "build_bridge", fail_bridge)

    manifest = dagger.collect_episodes(
        [_episode()],
        tmp_path,
        bridge_name="mock",
        policy_name="replay",
        config=TakeoverConfig(takeover_m=9.0, release_m=6.0, abort_m=30.0),
        max_steps=1,
    )

    assert manifest["failed"][0]["reason"] == "bridge setup failed"
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
