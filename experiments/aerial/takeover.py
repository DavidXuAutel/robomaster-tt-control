from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TakeoverConfig:
    takeover_m: float
    release_m: float
    abort_m: float
    worsen_steps: int = 3
    stall_steps: int = 8
    release_stable_steps: int = 3
    no_progress_abort_steps: int = 20


@dataclass(frozen=True)
class TakeoverDecision:
    mode: str  # policy | expert | abort
    intervene: bool
    reason: str


def freeze_thresholds(oracle_cross_track_p95: float) -> TakeoverConfig:
    takeover_m = max(9.0, 3.0 * oracle_cross_track_p95)
    release_m = takeover_m * (2.0 / 3.0)
    abort_m = max(30.0, 3.0 * takeover_m)
    return TakeoverConfig(
        takeover_m=takeover_m,
        release_m=release_m,
        abort_m=abort_m,
    )


class TakeoverController:
    def __init__(self, config: TakeoverConfig) -> None:
        self._config = config
        self._mode = "policy"
        self._last_cross_track: float | None = None
        self._last_progress: float | None = None
        self._worsen_steps = 0
        self._stall_steps = 0
        self._release_stable_steps = 0
        self._no_progress_steps = 0

    def _reset_mode_counters(self) -> None:
        self._worsen_steps = 0
        self._stall_steps = 0
        self._release_stable_steps = 0
        self._no_progress_steps = 0

    def step(self, cross_track_m: float, progress_m: float) -> TakeoverDecision:
        if self._mode == "abort":
            return TakeoverDecision(
                mode="abort", intervene=False, reason="already_aborted"
            )

        config = self._config

        if self._last_cross_track is not None and cross_track_m > self._last_cross_track:
            self._worsen_steps += 1
        else:
            self._worsen_steps = 0

        if self._last_progress is None or progress_m <= self._last_progress:
            self._stall_steps += 1
            self._no_progress_steps += 1
        else:
            self._stall_steps = 0
            self._no_progress_steps = 0

        self._last_cross_track = cross_track_m
        self._last_progress = progress_m

        if cross_track_m > config.abort_m:
            self._mode = "abort"
            return TakeoverDecision(
                mode="abort", intervene=False, reason="cross_track_abort"
            )

        if self._no_progress_steps >= config.no_progress_abort_steps:
            self._mode = "abort"
            return TakeoverDecision(
                mode="abort", intervene=False, reason="no_progress_abort"
            )

        if self._mode == "expert":
            if cross_track_m < config.release_m:
                self._release_stable_steps += 1
            else:
                self._release_stable_steps = 0

            if self._release_stable_steps >= config.release_stable_steps:
                self._mode = "policy"
                self._reset_mode_counters()
                return TakeoverDecision(
                    mode="policy", intervene=False, reason="released_to_policy"
                )

            return TakeoverDecision(
                mode="expert", intervene=True, reason="expert_control"
            )

        if (
            cross_track_m > config.takeover_m
            or self._worsen_steps >= config.worsen_steps
            or self._stall_steps >= config.stall_steps
        ):
            self._mode = "expert"
            if cross_track_m > config.takeover_m:
                reason = "cross_track_exceeded"
            elif self._worsen_steps >= config.worsen_steps:
                reason = "cross_track_worsening"
            else:
                reason = "progress_stalled"
            self._reset_mode_counters()
            return TakeoverDecision(mode="expert", intervene=True, reason=reason)

        return TakeoverDecision(mode="policy", intervene=False, reason="policy_control")
