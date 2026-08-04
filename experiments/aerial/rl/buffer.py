"""Episode replay store for the aerial RL skeleton.

Holds whole episodes of ``Transition``s (obs, action, reward, done, info) and
serves two access patterns:

  * ``sample_windows(batch, length)`` — contiguous length-``L`` slices for
    sequence-model / world-model training (V1). Windows never cross an episode
    boundary, so temporal dynamics stay valid.
  * ``sample(batch)`` — flat single transitions for value/critic fitting.

Pure Python + numpy, no external dep. Bounded by ``capacity`` episodes (FIFO
eviction). A deterministic ``rng`` seed keeps tests reproducible.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from experiments.aerial.rl.env.obs import Observation


@dataclass
class Transition:
    obs: Observation
    action: np.ndarray            # [4] body delta
    reward: float
    done: bool
    next_obs: Optional[Observation] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = np.asarray(self.action, dtype=np.float32).reshape(-1)
        self.reward = float(self.reward)
        self.done = bool(self.done)


Episode = List[Transition]


class ReplayBuffer:
    """FIFO episode buffer with window + flat sampling."""

    def __init__(self, capacity_episodes: int = 1000, seed: int = 0) -> None:
        self.capacity = int(capacity_episodes)
        self._episodes: Deque[Episode] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(int(seed))

    # -- writing ----------------------------------------------------------
    def add_episode(self, transitions: Sequence[Transition]) -> None:
        ep = list(transitions)
        if not ep:
            return
        self._episodes.append(ep)

    # -- sizing -----------------------------------------------------------
    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return sum(len(ep) for ep in self._episodes)

    def __len__(self) -> int:
        return self.num_transitions

    # -- reading ----------------------------------------------------------
    def sample(self, batch: int) -> List[Transition]:
        """``batch`` random single transitions (with replacement)."""
        flat = [t for ep in self._episodes for t in ep]
        if not flat:
            raise ValueError("cannot sample from an empty buffer")
        idx = self._rng.integers(0, len(flat), size=int(batch))
        return [flat[i] for i in idx]

    def sample_windows(self, batch: int, length: int) -> List[Episode]:
        """``batch`` contiguous windows of ``length`` transitions.

        Only episodes at least ``length`` long are eligible; a window is a
        within-episode slice so it never straddles a reset.
        """
        if length < 1:
            raise ValueError("window length must be >= 1")
        eligible = [ep for ep in self._episodes if len(ep) >= length]
        if not eligible:
            raise ValueError(
                f"no episode has >= {length} transitions "
                f"(longest={max((len(e) for e in self._episodes), default=0)})"
            )
        windows: List[Episode] = []
        for _ in range(int(batch)):
            ep = eligible[self._rng.integers(0, len(eligible))]
            start = int(self._rng.integers(0, len(ep) - length + 1))
            windows.append(ep[start:start + length])
        return windows

    def clear(self) -> None:
        self._episodes.clear()
