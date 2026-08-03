from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.aerial.eval.run_closed_loop import normalize_episode_poses
from experiments.aerial.openfly_actions import clip_body_delta, pos_yaw_to_body_delta

_LOOKAHEAD_M = 6.0


def _wrap_angle(angle: float) -> float:
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ExpertLabel:
    action: np.ndarray
    progress_m: float
    cross_track_m: float
    lookahead_pos: np.ndarray


class PathExpert:
    def __init__(self) -> None:
        self._positions: np.ndarray | None = None
        self._yaws: np.ndarray | None = None
        self._cumulative: np.ndarray | None = None
        self._progress_m = 0.0

    def reset(self, episode: dict[str, Any]) -> None:
        positions, yaws = normalize_episode_poses(episode)
        segment_lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        self._positions = positions
        self._yaws = yaws
        self._cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self._progress_m = 0.0

    def label(self, pos: np.ndarray, yaw: float) -> ExpertLabel:
        if self._positions is None or self._yaws is None or self._cumulative is None:
            raise RuntimeError("PathExpert.reset must be called before label")

        current_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        projection, progress = self._project_at_or_after_cursor(current_pos)
        self._progress_m = progress
        lookahead_progress = min(progress + _LOOKAHEAD_M, float(self._cumulative[-1]))
        lookahead_pos, lookahead_yaw = self._pose_at_progress(lookahead_progress)
        action = clip_body_delta(
            pos_yaw_to_body_delta(current_pos, yaw, lookahead_pos, lookahead_yaw)
        )
        return ExpertLabel(
            action=action,
            progress_m=progress,
            cross_track_m=float(np.linalg.norm(current_pos - projection)),
            lookahead_pos=lookahead_pos,
        )

    def _project_at_or_after_cursor(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        assert self._positions is not None
        assert self._cumulative is not None

        best_point = self._positions[-1]
        best_progress = float(self._cumulative[-1])
        best_distance = float("inf")

        for index in range(len(self._positions) - 1):
            start_progress = float(self._cumulative[index])
            end_progress = float(self._cumulative[index + 1])
            if end_progress < self._progress_m:
                continue

            start = self._positions[index]
            delta = self._positions[index + 1] - start
            length = end_progress - start_progress
            if length == 0.0:
                continue

            min_t = max(0.0, (self._progress_m - start_progress) / length)
            t = float(np.dot(pos - start, delta) / (length * length))
            t = float(np.clip(t, min_t, 1.0))
            point = start + t * delta
            distance = float(np.linalg.norm(pos - point))
            candidate_progress = start_progress + t * length
            if distance < best_distance or (
                distance == best_distance and candidate_progress < best_progress
            ):
                best_point = point
                best_progress = candidate_progress
                best_distance = distance

        return best_point.copy(), max(self._progress_m, best_progress)

    def _pose_at_progress(self, progress_m: float) -> tuple[np.ndarray, float]:
        assert self._positions is not None
        assert self._yaws is not None
        assert self._cumulative is not None

        if len(self._positions) == 1 or progress_m >= float(self._cumulative[-1]):
            return self._positions[-1].copy(), float(self._yaws[-1])

        index = int(np.searchsorted(self._cumulative, progress_m, side="right") - 1)
        start_progress = float(self._cumulative[index])
        length = float(self._cumulative[index + 1] - start_progress)
        if length == 0.0:
            return self._positions[index + 1].copy(), float(self._yaws[index + 1])

        t = (progress_m - start_progress) / length
        position = self._positions[index] + t * (
            self._positions[index + 1] - self._positions[index]
        )
        yaw_delta = _wrap_angle(float(self._yaws[index + 1] - self._yaws[index]))
        yaw = _wrap_angle(float(self._yaws[index]) + t * yaw_delta)
        return position, yaw
