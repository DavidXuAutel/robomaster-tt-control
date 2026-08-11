"""Entrypoint wiring the serial-corrector loop from ``configs/aerial_rl.yaml``.

    python -m experiments.aerial.rl.train_rl                     # mock, V0 loop
    python -m experiments.aerial.rl.train_rl corrector.smoke=true env.backend=airsim

``build_from_config`` assembles env / buffer / dynamics / policy / collector /
corrector from a plain (Omega)Conf-like mapping and is directly unit-testable
without Hydra. ``main`` is the Hydra wrapper (config dir = repo ``configs/``).

The default policy is a lightweight goal-seeking heuristic (``HeuristicPolicy``)
so the V0 collection loop runs end-to-end with no checkpoint. Swap in
``FastWAMAerialPolicy`` via ``build_policy`` once a checkpoint is available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.corrector import CorrectorConfig, SerialCorrectorLoop
from experiments.aerial.rl.dynamics import StubLatentDynamics
from experiments.aerial.rl.env.obs import PolicyObservation
from experiments.aerial.rl.reward import DEFAULT_ONLINE_SUCCESS_DIST_M, RewardConfig
from experiments.aerial.rl.safety import NullSafetyShield, ThresholdSafetyShield

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """PRIVILEGED goal-seeking stand-in — NOT an RGB policy.

    Steers toward the (externally supplied) goal using only proprio (x, y, z,
    yaw), so it respects the RGB-only input boundary — it receives a
    ``PolicyObservation`` and never touches depth/IMU. But it reaches the goal
    via a straight-line oracle rather than perception, so it exists only to
    exercise the V0 collection path end-to-end with no checkpoint. Swap in a
    learned ``act(view)`` policy once one exists. With no goal it idles.
    """

    def __init__(self, goal_getter, step_m: float = 3.0) -> None:
        self._goal_getter = goal_getter
        self.step_m = float(step_m)

    def reset(self) -> None:
        return None

    def act(self, obs: PolicyObservation) -> np.ndarray:
        goal = self._goal_getter()
        if goal is None:
            return np.zeros(4, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        d_world = goal - obs.position
        yaw = obs.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        dx = c * d_world[0] + s * d_world[1]      # world -> body
        dy = -s * d_world[0] + c * d_world[1]
        dz = d_world[2]
        vec = np.array([dx, dy, dz], dtype=np.float64)
        n = np.linalg.norm(vec)
        if n > self.step_m:
            vec = vec / n * self.step_m
        # Return the step_m-scaled body delta RAW — do NOT clip to the default
        # 30 Hz cap here. Doing so silently locked every step to
        # ``body_delta_limits(1/30) ≈ [0.167, .067, .067] m`` regardless of
        # ``step_m`` or the env's actual ``step_hz``, so a 5 Hz rollout crawled
        # ~0.167 m/step instead of the physical 1.0 m cap (5 m/s ÷ 5 Hz) and the
        # probe/eval could never reach an obstacle. The collector's ``act_delta``
        # re-clips with the rate-correct ``body_delta_limits(1/step_hz)``, so this
        # command is bounded downstream; ``step_m`` bounds ‖vec‖ here.
        return np.array([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _build_env(env_cfg: Any) -> Any:
    backend = str(_get(env_cfg, "backend", "mock"))
    if backend == "mock":
        from experiments.aerial.rl.env.mock_env import MockAirSimDroneEnv, MockEnvConfig

        return MockAirSimDroneEnv(MockEnvConfig(
            width=int(_get(env_cfg, "width", 224)),
            height=int(_get(env_cfg, "height", 224)),
            step_hz=float(_get(env_cfg, "step_hz", 30.0)),
            seed=int(_get(env_cfg, "seed", 0)),
        ))
    if backend == "airsim":
        from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig

        return AirSimDroneEnv(AirSimEnvConfig(
            host=str(_get(env_cfg, "host", "127.0.0.1")),
            port=int(_get(env_cfg, "port", 41451)),
            camera=str(_get(env_cfg, "camera", "front_custom")),
            vehicle=str(_get(env_cfg, "vehicle", "drone_1")),
            width=int(_get(env_cfg, "width", 224)),
            height=int(_get(env_cfg, "height", 224)),
            step_hz=float(_get(env_cfg, "step_hz", 30.0)),
            health_check=bool(_get(env_cfg, "health_check", True)),
            grab_depth=bool(_get(env_cfg, "grab_depth", True)),
        ))
    raise ValueError(f"unknown env backend {backend!r} (expected mock|airsim)")


def _build_dynamics(dyn_cfg: Any, *, success_dist_m: float, wm_cfg: Any = None) -> Any:
    """Dispatch on ``dynamics.kind``. ``wan`` is an offline distillation source
    (spec §4.4/§11) — it must NOT drive the online corrector, so selecting it
    here is a hard error rather than a silent fall-back to the stub. ``torch``
    is the real DreamerV3 RSSM WM (V1) and needs torch (H100); it reads the
    ``world_model:`` block, imported lazily so the stub/mock path stays torch-free."""
    kind = str(_get(dyn_cfg, "kind", "stub"))
    if kind == "stub":
        return StubLatentDynamics(
            goal=None,  # set per-episode by the corrector before imagination
            latent_dim=int(_get(dyn_cfg, "latent_dim", 8)),
            collide_radius_m=float(_get(dyn_cfg, "collide_radius_m", 2.0)),
            success_dist_m=float(success_dist_m),
        )
    if kind == "wan":
        raise ValueError(
            "dynamics.kind='wan' (WanImaginationDynamics) is an OFFLINE "
            "distillation source only (spec §4.4/§11) — stepping the Wan2.2 "
            "pixel model in the online RL loop is a non-goal. Use kind='stub' "
            "for V0; the real fast latent WM (V1) drops into the stub slot."
        )
    if kind == "torch":
        try:
            from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
        except ImportError as exc:  # torch absent (dev host) -> clear H100 pointer
            raise RuntimeError(
                "dynamics.kind='torch' needs the torch DreamerV3 RSSM WM, which "
                "runs on the H100 (torch 2.7.1+cu128) — it is not importable here. "
                f"Use kind='stub' on the GPU-less host. (import error: {exc})"
            ) from exc
        return TorchRSSMDynamics.from_config(wm_cfg or {})
    raise ValueError(f"unknown dynamics kind {kind!r} (expected stub|wan|torch)")


def _build_safety(safety_cfg: Any) -> Any:
    kind = str(_get(safety_cfg, "kind", "null"))
    if kind in ("null", "none", "None"):
        return NullSafetyShield()
    if kind == "threshold":
        # min_depth_m is the reaction STANDOFF, not the ④a near-collision metric
        # (frozen at 1.5). Default 3.0 m gives the shield room to intervene before
        # the band (frozen-spec ④a re-freeze 2026-08-11).
        return ThresholdSafetyShield(
            min_depth_m=float(_get(safety_cfg, "min_depth_m", 3.0)),
            min_tau_s=float(_get(safety_cfg, "min_tau_s", 1.0)),
            max_p_coll=float(_get(safety_cfg, "max_p_coll", 0.5)),
        )
    raise ValueError(f"unknown safety kind {kind!r}")


def _load_episodes(cfg: Any) -> Optional[List[Dict[str, Any]]]:
    ann = _get(cfg, "annotation", None)
    if not ann:
        return None
    from experiments.aerial.eval.run_closed_loop import load_annotation

    episodes = load_annotation(Path(str(ann)))
    return episodes[: max(0, int(_get(cfg, "max_episodes", 20)))]


def build_from_config(cfg: Any) -> SerialCorrectorLoop:
    env = _build_env(_get(cfg, "env", {}))
    buf_cfg = _get(cfg, "buffer", {})
    buffer = ReplayBuffer(
        capacity_episodes=int(_get(buf_cfg, "capacity_episodes", 1000)),
        seed=int(_get(buf_cfg, "seed", 0)),
    )

    rc = _get(cfg, "reward", {})
    w_maneuver = float(_get(rc, "w_maneuver", 0.01))
    reward_cfg = RewardConfig(
        w_progress=float(_get(rc, "w_progress", 1.0)),
        w_collision=float(_get(rc, "w_collision", 10.0)),
        w_maneuver=w_maneuver,
        # Online arrival/termination radius — tighter than the eval SR metric
        # (EVAL_SUCCESS_DIST_M=20 m); falls back to the tight online default.
        success_dist_m=float(_get(rc, "success_dist_m", DEFAULT_ONLINE_SUCCESS_DIST_M)),
        success_bonus=float(_get(rc, "success_bonus", 10.0)),
        # Maneuver-penalty curriculum (§2.4); defaults leave it a no-op.
        w_maneuver_final=float(_get(rc, "w_maneuver_final", w_maneuver)),
        maneuver_curriculum_threshold=float(_get(rc, "maneuver_curriculum_threshold", 0.0)),
        maneuver_curriculum_ramp=float(_get(rc, "maneuver_curriculum_ramp", 1.0)),
    )

    # Imagined dynamics shares the reward's arrival radius so imagined and real
    # returns agree on when the goal is reached. The world_model block feeds the
    # torch DreamerV3 WM (kind=torch); ignored by stub/wan.
    dynamics = _build_dynamics(
        _get(cfg, "dynamics", {}),
        success_dist_m=reward_cfg.success_dist_m,
        wm_cfg=_get(cfg, "world_model", {}),
    )

    cc = _get(cfg, "corrector", {})
    ic = _get(cfg, "imagination", {})
    corrector_cfg = CorrectorConfig(
        iterations=int(_get(cc, "iterations", 10)),
        episodes_per_iter=int(_get(cc, "episodes_per_iter", 1)),
        enable_wm_update=bool(_get(cc, "enable_wm_update", False)),
        enable_policy_update=bool(_get(cc, "enable_policy_update", False)),
        strict_gates=bool(_get(cc, "strict_gates", False)),
        wm_batch=int(_get(cc, "wm_batch", 32)),
        wm_window=int(_get(cc, "wm_window", 8)),
        imagine_batch=int(_get(ic, "batch", 64)),
        imagine_horizon=int(_get(ic, "horizon", 10)),
        smoke=bool(_get(cc, "smoke", False)),
    )

    policy = HeuristicPolicy(goal_getter=lambda: getattr(env, "goal", None))
    collector = RolloutCollector(
        env, policy, buffer,
        reward_cfg=reward_cfg,
        safety=_build_safety(_get(cfg, "safety", {})),
        max_steps=int(_get(cc, "max_steps", 200)),
        target_hz=float(_get(_get(cfg, "env", {}), "step_hz", 30.0)),
    )
    episodes = _load_episodes(cfg)
    return SerialCorrectorLoop(
        collector, buffer, dynamics,
        config=corrector_cfg, episodes=episodes,
    )


def main() -> None:  # pragma: no cover - Hydra wrapper
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(version_base="1.3", config_path="../../../configs", config_name="aerial_rl")
    def _run(cfg: "DictConfig") -> None:
        logging.basicConfig(level=logging.INFO)
        loop = build_from_config(OmegaConf.to_container(cfg, resolve=True))
        reports = loop.run()
        total_steps = sum(r.collect.steps for r in reports)
        logger.info("done: %d iters, %d env steps", len(reports), total_steps)

    _run()


if __name__ == "__main__":  # pragma: no cover
    main()
