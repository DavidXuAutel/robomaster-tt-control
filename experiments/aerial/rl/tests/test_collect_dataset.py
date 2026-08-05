"""Run-level behavior of the collect_dataset driver: the quarantine fraction gate.

Instant-crash episodes are excluded per-episode, but the driver must only fail
the whole run when their fraction exceeds ``MAX_QUARANTINE_FRACTION``. This drives
``main`` with a fake collector so the gate arithmetic is exercised without a
renderer.
"""
import numpy as np

from experiments.aerial.rl import collect_dataset as cd
from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.collector import CollectStats
from experiments.aerial.rl.env.obs import Observation


def _obs(pos, frame_val, collided=False):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], np.float32)
    rgb = np.full((8, 8, 3), int(frame_val) % 256, np.uint8)
    rgb[0, 0, 0] = (int(frame_val) + 40) % 256
    return Observation(rgb=rgb, state=state, collided=collided)


def _healthy():
    return [Transition(obs=_obs([float(i), 0, 2], 10 + i * 20),
                       action=np.ones(4) * 0.5, reward=1.0, done=(i == 4))
            for i in range(5)]


def _crash():
    return [Transition(obs=_obs([0, 0, 2], 30, collided=True),
                       action=np.ones(4) * 0.5, reward=-10.0, done=True)]


class _FakeCollector:
    """Replays a fixed list of episodes through the driver's on_episode sink."""

    def __init__(self, episodes):
        self._episodes = episodes
        self.on_episode = None
        self.env = self  # provides a no-op close()

    def close(self):
        pass

    def collect(self, num_episodes, episodes=None):
        stats = CollectStats()
        for ep in self._episodes:
            self.on_episode(ep, CollectStats(episodes=1, steps=len(ep), seconds=1.0))
            stats.episodes += 1
            stats.steps += len(ep)
        return stats


class _FakeLoop:
    def __init__(self, episodes):
        self.collector = _FakeCollector(episodes)
        self.episodes = [{"pos": [[0, 0, 0]], "yaw": [0]}]  # non-None: skip injection


def _run(monkeypatch, tmp_path, episodes):
    monkeypatch.setattr(cd, "build_from_config", lambda cfg: _FakeLoop(episodes))
    return cd.main(["--backend", "mock", "--episodes", str(len(episodes)),
                    "--out", str(tmp_path)])


def test_few_crashes_pass_but_are_excluded(monkeypatch, tmp_path):
    # 1 crash out of 10 == 10% <= 20% tolerance -> exit 0, but excluded.
    eps = [_healthy() for _ in range(9)] + [_crash()]
    assert _run(monkeypatch, tmp_path, eps) == 0
    import json
    summ = json.loads((tmp_path / "QUALITY_SUMMARY.json").read_text())
    assert summ["quarantined"] == 1 and summ["usable"] == 9


def test_flood_of_crashes_fails_the_run(monkeypatch, tmp_path):
    # 4 crashes out of 8 == 50% > 20% -> exit 1.
    eps = [_healthy() for _ in range(4)] + [_crash() for _ in range(4)]
    assert _run(monkeypatch, tmp_path, eps) == 1


def test_empty_collection_fails(monkeypatch, tmp_path):
    # Every reset skipped on spawn-collision → 0 episodes. Must FAIL, not ship an
    # empty dataset with a spurious "OK: 0/0" exit 0.
    assert _run(monkeypatch, tmp_path, []) == 1


def test_hard_failure_always_fails(monkeypatch, tmp_path):
    # a frozen-renderer episode (identical frames, no motion) is a hard fail
    # regardless of quarantine fraction.
    frozen = [Transition(obs=_obs([0, 0, 2], 30), action=np.zeros(4),
                         reward=0.0, done=(i == 3)) for i in range(4)]
    eps = [_healthy() for _ in range(9)] + [frozen]
    assert _run(monkeypatch, tmp_path, eps) == 1
