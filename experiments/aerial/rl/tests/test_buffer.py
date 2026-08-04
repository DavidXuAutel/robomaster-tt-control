import numpy as np
import pytest

from experiments.aerial.rl.buffer import ReplayBuffer, Transition
from experiments.aerial.rl.env.mock_env import MockAirSimDroneEnv


def _make_episode(n: int):
    env = MockAirSimDroneEnv()
    obs = env.reset({"pos": [[0, 0, 0], [30, 0, 0]], "yaw": [0, 0]})
    ep = []
    for i in range(n):
        nxt, _ = env.step(np.array([3.0, 0.0, 0.0, 0.0]))
        ep.append(Transition(obs=obs, action=[3, 0, 0, 0], reward=float(i), done=(i == n - 1), next_obs=nxt))
        obs = nxt
    return ep


def test_add_and_size():
    buf = ReplayBuffer(capacity_episodes=10, seed=1)
    buf.add_episode(_make_episode(5))
    assert buf.num_episodes == 1
    assert buf.num_transitions == 5
    assert len(buf) == 5


def test_capacity_evicts_fifo():
    buf = ReplayBuffer(capacity_episodes=2, seed=1)
    for _ in range(3):
        buf.add_episode(_make_episode(4))
    assert buf.num_episodes == 2


def test_sample_flat():
    buf = ReplayBuffer(seed=3)
    buf.add_episode(_make_episode(6))
    batch = buf.sample(4)
    assert len(batch) == 4
    assert all(isinstance(t, Transition) for t in batch)


def test_sample_windows_within_episode():
    buf = ReplayBuffer(seed=5)
    buf.add_episode(_make_episode(8))
    windows = buf.sample_windows(3, length=4)
    assert len(windows) == 3
    for w in windows:
        assert len(w) == 4


def test_sample_windows_raises_when_too_short():
    buf = ReplayBuffer(seed=5)
    buf.add_episode(_make_episode(3))
    with pytest.raises(ValueError):
        buf.sample_windows(2, length=8)


def test_sample_empty_raises():
    with pytest.raises(ValueError):
        ReplayBuffer().sample(1)


def test_reproducible_with_seed():
    b1 = ReplayBuffer(seed=7); b1.add_episode(_make_episode(6))
    b2 = ReplayBuffer(seed=7); b2.add_episode(_make_episode(6))
    r1 = [t.reward for t in b1.sample(5)]
    r2 = [t.reward for t in b2.sample(5)]
    assert r1 == r2
