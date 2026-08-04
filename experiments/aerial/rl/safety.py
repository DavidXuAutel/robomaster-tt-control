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

from dataclasses import dataclass
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
    """

    min_depth_m: float = 1.5          # brake if nearest predicted depth < this
    min_tau_s: float = 1.0            # brake if time-to-contact < this
    max_p_coll: float = 0.5           # brake if WM collision prob > this
    brake_gain: float = 1.0

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
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

    def override_action(self, obs: Observation) -> np.ndarray:
        # Conservative: cancel forward motion, hold altitude/heading.
        return np.zeros(4, dtype=np.float64)
