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

    **Latch + hold (brake-to-hover)** (V2, re-freeze 2026-08-11 晚¹⁰). History:
      1. pure *pre-latch* hover (return zeros, no latch): the goal-seeker keeps
         pushing forward while the shield cancels it → oscillates in/out of the
         band → near_coll_rate_on ≫ off, ratio inverts.
      2. latch + *continuous retreat* (body −x every step): fixed (1)'s oscillation
         and, under an *optimistic* predictor (approach AbsRel ≈0.167 ⇒ ``D̂`` reads
         FARTHER than GT), drove GT clearance monotonically up regardless of bias.
         BUT the 2026-08-11 晚¹⁰ 4090 rollout showed it *relocates* the crash: with
         the policy latched off, the vehicle retreats body −x **blindly with no rear
         sensing** and, in enclosed scenes, backs into the rear/side wall — telemetry
         ``coll_after_latch=9/9``, ``near_before_latch=0`` (front band never entered
         before the latch), collisions ~33 steps AFTER an immediate latch. A shield
         that survives 3× longer but still crashes 9/10 is not avoidance.
      3. latch + **hold (hover, return zeros)** — current. The whole premise that
         forced (2)'s retreat was the *optimistic* predictor; 晚⁷ eliminated it (DA3
         near-band retrain: forward ``D̂`` 6.4→0.65 m, near-band P(trigger)=1.0, now
         accurate/slightly conservative near the band). With the ``min_depth_m``=3.0
         reaction standoff the shield trips at true ≈3 m (``D̂`` under-reads near the
         band ⇒ trips EARLY, above the band), then the *latch* holds position — the
         policy is ignored so nothing pushes it in, and zero body-delta means no
         blind backward travel. Front clearance stays ≥ standoff (near_coll_rate_on
         ≈ 0), no rear crash (coll_after_latch → 0), and it intervenes before contact
         (④b). Hold beats hover-*only* precisely because of the latch (no re-approach)
         and beats retreat precisely because 晚⁷ removed the optimism that retreat
         compensated for. ``reset()`` clears the latch between episodes (the eval
         reuses one instance). See frozen-spec ④a re-freeze note (2026-08-11 晚¹⁰).
    """

    # Reaction standoff (NOT the near-collision metric): trigger when predicted
    # clearance < this. Default 3.0 m > the 1.5 m ④a metric so the shield reacts
    # before entering the band; the metric stays frozen at 1.5 in v0_metrics.
    min_depth_m: float = 3.0
    min_tau_s: float = 1.0            # brake if time-to-contact < this
    max_p_coll: float = 0.5           # brake if WM collision prob > this
    brake_gain: float = 1.0
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
        # Latch for the rest of the episode once breached, so we hold clear of the
        # band — rather than oscillating in/out of it (which would keep sampling
        # near-collision frames on the shield-on arm) or re-approaching once the
        # predicted clearance nominally recovers.
        if self._engaged:
            return True
        if self._breached(obs, wm_out):
            self._engaged = True
            return True
        return False

    def override_action(self, obs: Observation) -> np.ndarray:
        # Latched → HOLD (hover) for the rest of the episode: zero body-delta. Do
        # NOT retreat body −x. The blind backward retreat had no rear sensing and,
        # in enclosed scenes, backed the vehicle into the rear/side wall (晚¹⁰:
        # coll_after_latch=9/9, crash ~33 steps after latching) — it relocated the
        # crash instead of avoiding it. Holding at the ``min_depth_m`` (3.0 m)
        # reaction standoff is safe now that 晚⁷ removed the near-band predictor
        # optimism that retreat was compensating for: the trip fires at true ≈3 m
        # (D̂ under-reads near the band ⇒ trips early, above the 1.5 m metric), the
        # latch keeps the policy from pushing back in, and zero delta means no
        # forward creep and no blind backward travel. The collector re-clips this
        # through ``body_delta_limits(1/step_hz)`` (0 → 0), so it is a true hold.
        return np.zeros(4, dtype=np.float64)
