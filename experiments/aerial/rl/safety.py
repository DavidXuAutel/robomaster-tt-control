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

    **Latch + bounded state-feedback retreat** (V2, re-freeze 2026-08-11 晚¹²).
    History — each design fixed the prior one's failure and exposed the next:
      1. pure *pre-latch* hover (return zeros, no latch): the goal-seeker keeps
         pushing forward while the shield cancels it → oscillates in/out of the
         band → near_coll_rate_on ≫ off, ratio inverts.
      2. latch + *unbounded continuous retreat* (body −x EVERY step forever): fixed
         (1)'s oscillation, but with the policy latched off the vehicle retreats
         **blindly with no rear sensing** and, in enclosed scenes, backs into the
         rear/side wall — 晚¹⁰ telemetry ``coll_after_latch=9/9``, crash ~33 steps
         after an immediate latch. Surviving 3× longer but still crashing 9/10 is
         not avoidance — it relocates the crash.
      3. latch + *pure hold* (zeros): stops the rear crash, but a zero body-delta
         does NOT arrest forward momentum — the vehicle **coasts into the band**
         after latching and parks there — 晚¹¹ ``near_count_on`` up to 200/200,
         ``near_coll_rate_on`` 0.385, ratio 12.96. The retreat in (2) was doing
         double duty: countering the *optimistic predictor* AND killing forward
         momentum; hold dropped both.
      4. latch + **bounded state-feedback retreat** — current. Retreat body −x only
         WHILE ``D̂`` < ``min_depth_m`` (the reaction standoff); HOLD (zeros) once
         ``D̂`` ≥ standoff. This kills forward momentum and backs out of the band
         (fixes (3)), then STOPS retreating at the standoff instead of reversing
         into the rear wall (fixes (2)). Reliable only because 晚⁷ eliminated the
         near-band predictor optimism (DA3 retrain: forward ``D̂`` 6.4→0.65 m,
         near-band P(trigger)=1.0, now accurate / under-reading near the band): so
         ``D̂`` ≥ standoff ⟺ genuinely ≈3 m clear (the earlier "retreat-until-D̂-safe
         then hover" parked in-band precisely because optimistic ``D̂`` recovered
         past safe while GT was still <1.5 m; that premise is gone). The *latch*
         keeps the policy from re-approaching after the vehicle settles at the
         standoff. Net: front clearance settles at ≥ standoff (near_coll_rate_on→0),
         bounded backward travel (no rear crash), intervention before contact (④b).
         ``reset()`` clears the latch between episodes (the eval reuses one
         instance). See frozen-spec ④a re-freeze note (2026-08-11 晚¹²).
    """

    # Reaction standoff (NOT the near-collision metric): trigger when predicted
    # clearance < this. Default 3.0 m > the 1.5 m ④a metric so the shield reacts
    # before entering the band; the metric stays frozen at 1.5 in v0_metrics.
    min_depth_m: float = 3.0
    min_tau_s: float = 1.0            # brake if time-to-contact < this
    max_p_coll: float = 0.5           # brake if WM collision prob > this
    brake_gain: float = 1.0
    retreat_step_m: float = 3.0       # body −x retreat request while breached (collector re-clips to the rate cap)
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
        # Bounded state-feedback retreat (晚¹²). While still inside the reaction
        # standoff (``D̂`` < ``min_depth_m``) retreat body −x: a zero-delta HOLD does
        # NOT arrest forward momentum, so the vehicle coasts into the band and parks
        # there (晚¹¹: near_count_on up to 200/200, ratio 12.96). Once ``D̂`` ≥ the
        # standoff, HOLD (zeros) — do NOT keep retreating, or the blind body −x drive
        # with no rear sensing backs into the rear/side wall (晚¹⁰: coll_after_latch
        # =9/9, crash ~33 steps after latching). This stops the retreat AT the
        # standoff. Trusting ``D̂`` ≥ standoff as "genuinely clear" is safe only post-
        # 晚⁷ (near-band DA3 retrain: forward D̂ 6.4→0.65 m, under-reads near the band
        # so D̂<standoff holds true while GT<1.5); the pre-晚⁷ optimistic D̂ recovered
        # past standoff while GT was still <1.5 → parked in band. The latch (see
        # should_override) keeps the policy from re-approaching once settled. The
        # collector re-clips −x through ``body_delta_limits(1/step_hz)`` to the rate
        # cap, so retreat is a steady per-step increment, not a teleport.
        d_hat = obs.info.get("depth_min_pred")
        if d_hat is not None and float(d_hat) < float(self.min_depth_m):
            return np.array([-abs(float(self.retreat_step_m)), 0.0, 0.0, 0.0], dtype=np.float64)
        return np.zeros(4, dtype=np.float64)
