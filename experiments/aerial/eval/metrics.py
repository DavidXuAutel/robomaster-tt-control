from __future__ import annotations

from typing import Dict, Sequence

OPENFLY_SUCCESS_DIST_M = 20.0


def compute_sr_ne_spl(
    successes: Sequence[bool],
    path_lengths: Sequence[float],
    shortest_lengths: Sequence[float],
    nes: Sequence[float],
) -> Dict[str, float]:
    n = len(successes)
    assert n == len(path_lengths) == len(shortest_lengths) == len(nes)
    sr = sum(1 for s in successes if s) / max(n, 1)
    ne = sum(nes) / max(n, 1)
    spl_sum = 0.0
    for ok, p, sp in zip(successes, path_lengths, shortest_lengths):
        if ok and p > 0:
            spl_sum += sp / max(p, sp)
    return {"SR": sr, "NE": ne, "SPL": spl_sum / max(n, 1)}
