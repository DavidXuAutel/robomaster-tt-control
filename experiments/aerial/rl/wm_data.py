"""Replay-window → stacked-array adapter for world-model training.

``ReplayBuffer.sample_windows(batch, length)`` returns ``List[List[Transition]]``
— a list of equal-length within-episode slices (``buffer.py``). The torch RSSM
trainer (``dynamics_torch.py``, Phase 2b) needs these as dense, batched tensors;
``windows_to_arrays`` is the **torch-free boundary** that produces stacked numpy
arrays the trainer then wraps with ``torch.from_numpy``. Keeping the reshape here
(and unit-testing it against stub episodes on this GPU-less host) de-risks the
trainer's data plumbing before any torch is involved.

Output arrays are obs-aligned along the window axis (index ``t`` in a window is
observation ``t``, the action taken from it, and the resulting reward/done):

    rgb       [B, L, H, W, 3] uint8   policy/WM exteroception (RGB-only, §1.2)
    proprio   [B, L, 4]       float32 (x, y, z, yaw)
    action    [B, L, 4]       float32 body delta
    reward    [B, L]          float32
    done      [B, L]          bool
    collided  [B, L]          bool    contact GT (reward/termination signal)
    depth     [B, L, H, W]    float32 ONLY if every frame carries it (else absent)

``depth`` follows ``dataset.episode_arrays``: present only when *every* frame in
*every* window has it, so a partial-depth batch never yields a ragged channel.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.aerial.rl.buffer import Episode, Transition


def _validate(windows: List[Episode]) -> int:
    if not windows:
        raise ValueError("windows is empty; nothing to stack")
    length = len(windows[0])
    if length < 1:
        raise ValueError("windows contain a zero-length slice")
    for b, w in enumerate(windows):
        if len(w) != length:
            raise ValueError(
                f"ragged windows: window 0 has length {length} but window {b} "
                f"has length {len(w)} — sample_windows must yield equal lengths"
            )
    return length


def windows_to_arrays(windows: List[Episode]) -> Dict[str, np.ndarray]:
    """Stack ``List[List[Transition]]`` into batched, obs-aligned numpy arrays.

    See the module docstring for the emitted keys/shapes. Raises ``ValueError``
    on empty or ragged input (the RSSM trainer assumes a dense ``[B, L, ...]``).
    """
    length = _validate(windows)
    batch = len(windows)

    def _obs(w: Episode, t: int):
        return w[t].obs

    rgb = np.stack(
        [np.stack([_obs(w, t).rgb for t in range(length)], axis=0) for w in windows],
        axis=0,
    ).astype(np.uint8, copy=False)
    proprio = np.stack(
        [np.stack([_obs(w, t).proprio4() for t in range(length)], axis=0) for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    action = np.stack(
        [np.stack([np.asarray(w[t].action, dtype=np.float32).reshape(4)
                   for t in range(length)], axis=0) for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    reward = np.asarray(
        [[float(w[t].reward) for t in range(length)] for w in windows], dtype=np.float32
    )
    done = np.asarray(
        [[bool(w[t].done) for t in range(length)] for w in windows], dtype=np.bool_
    )
    collided = np.asarray(
        [[bool(_obs(w, t).collided) for t in range(length)] for w in windows], dtype=np.bool_
    )

    out: Dict[str, np.ndarray] = {
        "rgb": rgb,
        "proprio": proprio,
        "action": action,
        "reward": reward,
        "done": done,
        "collided": collided,
    }

    # Depth only when EVERY frame has it (mirror dataset.episode_arrays): a
    # partial-depth batch drops the channel rather than emitting a ragged array.
    if all(_obs(w, t).depth is not None for w in windows for t in range(length)):
        out["depth"] = np.stack(
            [np.stack([np.asarray(_obs(w, t).depth, dtype=np.float32)
                       for t in range(length)], axis=0) for w in windows],
            axis=0,
        ).astype(np.float32, copy=False)

    return out
