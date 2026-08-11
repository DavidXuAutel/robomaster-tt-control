"""Hard safety shield (spec §2#6, §4.5) — interface + null stub.

The shield sits ABOVE the learned policy: if inflated predicted depth ``D̂``,
time-to-contact ``τ``, or world-model collision probability ``p_coll`` breaches a
threshold, it overrides the policy's action with a conservative one (brake /
hover / retreat). It is a *hard* override, not a learned behaviour — so it lives
outside the RL graph.

Only the contract is fixed here. ``NullSafetyShield`` never overrides (V0/V1
default). A real ``DepthTauShield`` is deferred until the perception heads that
produce ``D̂`` / ``τ`` exist (V2+); ``ThresholdSafetyShield`` shows the intended
trigger wiring against fields that may not be populated yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from experiments.aerial.rl.env.obs import Observation


@runtime_checkable
class SafetyShield(Protocol):
    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool: ...

    def override_action(self, obs: Observation) -> np.ndarray: ...


class NullSafetyShield:
    """No-op shield: never intervenes. Default until D̂/τ heads exist."""

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        return False

    def override_action(self, obs: Observation) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)


@dataclass
class ThresholdSafetyShield:
    """Trigger contract for D̂ ∪ τ ∪ p_coll (fields wired at V2+).

    Reads optional predictions off ``obs.info`` / ``wm_out`` — if none are
    present it degrades to never-override, so it is safe to install early.

    **Latch + continuous retreat** (V2, re-freeze 2026-08-11). Two prior designs
    failed ④ under an *optimistic* depth predictor (approach AbsRel ≈0.167 ⇒ ``D̂``
    reads FARTHER than GT near the band):
      * pure hover (return zeros) parks the vehicle inside the near-collision band
        (goal-seeker keeps pushing forward, shield keeps cancelling) →
        near_coll_rate_on ≫ off, ④'s ratio inverts;
      * "retreat until ``D̂`` ≥ ``safe_depth_m`` then hover" *also* parks it there,
        because ``D̂`` recovers past ``safe`` while GT is still <1.5 m (ratio 6.10,
        observed 2026-08-11).
    Fix: on the first breach we *latch* for the rest of the episode and retreat
    (body −x) **every step, never hovering** — monotonic backward travel drives GT
    clearance strictly up regardless of predictor bias, so near_coll_rate_on ≈ 0 by
    construction. Paired with a *reaction margin* (``min_depth_m`` set ABOVE the
    1.5 m near-collision metric, default 3.0 m) the shield triggers BEFORE the band,
    so it also intervenes before contact (④b). ``reset()`` clears the latch between
    episodes (the eval reuses one instance). See frozen-spec ④a re-freeze note.
    """

    # Reaction standoff (NOT the near-collision metric): trigger when predicted
    # clearance < this. Default 3.0 m > the 1.5 m ④a metric so the shield reacts
    # before entering the band; the metric stays frozen at 1.5 in v0_metrics.
    min_depth_m: float = 3.0
    min_tau_s: float = 1.0            # brake if time-to-contact < this
    max_p_coll: float = 0.5           # brake if WM collision prob > this
    brake_gain: float = 1.0
    retreat_step_m: float = 3.0       # backward body-delta request (collector re-clips to the rate cap)
    _engaged: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        """Clear the per-episode latch (the shield instance is reused across episodes)."""
        self._engaged = False

    def _breached(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        d_hat = obs.info.get("depth_min_pred")
        tau = obs.info.get("tau_pred")
        p_coll = None
        if wm_out is not None:
            p_coll = getattr(wm_out, "p_coll", None)
        if d_hat is not None and float(d_hat) < self.min_depth_m:
            return True
        if tau is not None and float(tau) < self.min_tau_s:
            return True
        if p_coll is not None and float(p_coll) > self.max_p_coll:
            return True
        return False

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        # Latch for the rest of the episode once breached, so we retreat clear of
        # the band and hold — rather than oscillating in/out of it, which would
        # keep sampling near-collision frames on the shield-on arm.
        if self._engaged:
            return True
        if self._breached(obs, wm_out):
            self._engaged = True
            return True
        return False

    def override_action(self, obs: Observation) -> np.ndarray:
        # Latched → retreat every step (body −x) for the rest of the episode; do
        # NOT hover once ``D̂`` "recovers". The predictor is optimistic near the
        # band (reads farther than GT), so any hover-at-safe branch parks the
        # vehicle INSIDE the GT near-collision band and inflates near_coll_rate_on
        # (④ ratio inverted, observed 2026-08-11). Monotonic retreat makes GT
        # clearance strictly increase regardless of predictor bias. The collector
        # re-clips to ``body_delta_limits(1/step_hz)`` so this is a steady per-step
        # backward increment, not a teleport.
        return np.array([-abs(float(self.retreat_step_m)), 0.0, 0.0, 0.0], dtype=np.float64)
