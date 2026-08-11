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

    **Latch + retreat** (V2, 2026-08-11). A pure hover-override (return zeros)
    parks the vehicle *inside* the near-collision band once it drifts there: the
    goal-seeking policy keeps commanding forward, the shield keeps cancelling it,
    and the drone hovers at ~``min_depth_m`` — so the shield-on arm accrues MORE
    near-collision frames than shield-off (near_coll_rate_on > near_coll_rate_off,
    ④'s ratio inverts / NaNs, observed 2026-08-11). Instead: on the first breach
    we *latch* for the rest of the episode and *retreat* (body −x) until predicted
    clearance recovers past ``safe_depth_m``, then hover. This drives the drone OUT
    of the band and keeps it there, so near_coll_rate_on ≈ 0 by construction.
    ``reset()`` clears the latch between episodes (the eval reuses one instance).
    """

    min_depth_m: float = 1.5          # brake if nearest predicted depth < this
    min_tau_s: float = 1.0            # brake if time-to-contact < this
    max_p_coll: float = 0.5           # brake if WM collision prob > this
    brake_gain: float = 1.0
    safe_depth_m: float = 2.5         # retreat until predicted clearance ≥ this
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
        # Retreat (body −x) while still inside the safe standoff, else hover. The
        # collector re-clips to ``body_delta_limits(1/step_hz)`` so a large request
        # becomes a steady per-step backward increment, not a teleport.
        d_hat = obs.info.get("depth_min_pred")
        if d_hat is not None and float(d_hat) < self.safe_depth_m:
            return np.array([-abs(float(self.retreat_step_m)), 0.0, 0.0, 0.0], dtype=np.float64)
        return np.zeros(4, dtype=np.float64)
