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
from experiments.aerial.rl.env.action import clip_body_delta
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.reward import RewardConfig
from experiments.aerial.rl.safety import NullSafetyShield, ThresholdSafetyShield

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """Goal-seeking body-delta policy: step toward the goal, capped per axis.

    Continuous (``act``) so the collector's RL path is exercised. With no goal it
    idles (zero delta). Not a learned policy — a stand-in so V0 collection runs.
    """

    def __init__(self, goal_getter, step_m: float = 3.0) -> None:
        self._goal_getter = goal_getter
        self.step_m = float(step_m)

    def reset(self) -> None:
        return None

    def act(self, obs: Observation) -> np.ndarray:
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
        return clip_body_delta(np.array([vec[0], vec[1], vec[2], 0.0]))


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
        ))
    raise ValueError(f"unknown env backend {backend!r} (expected mock|airsim)")


def _build_safety(safety_cfg: Any) -> Any:
    kind = str(_get(safety_cfg, "kind", "null"))
    if kind in ("null", "none", "None"):
        return NullSafetyShield()
    if kind == "threshold":
        return ThresholdSafetyShield(
            min_depth_m=float(_get(safety_cfg, "min_depth_m", 1.5)),
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
    reward_cfg = RewardConfig(
        w_progress=float(_get(rc, "w_progress", 1.0)),
        w_collision=float(_get(rc, "w_collision", 10.0)),
        w_maneuver=float(_get(rc, "w_maneuver", 0.01)),
        success_bonus=float(_get(rc, "success_bonus", 10.0)),
    )

    dc = _get(cfg, "dynamics", {})
    dynamics = StubLatentDynamics(
        goal=None,
        latent_dim=int(_get(dc, "latent_dim", 8)),
        collide_radius_m=float(_get(dc, "collide_radius_m", 2.0)),
    )

    cc = _get(cfg, "corrector", {})
    ic = _get(cfg, "imagination", {})
    corrector_cfg = CorrectorConfig(
        iterations=int(_get(cc, "iterations", 10)),
        episodes_per_iter=int(_get(cc, "episodes_per_iter", 1)),
        enable_wm_update=bool(_get(cc, "enable_wm_update", False)),
        enable_policy_update=bool(_get(cc, "enable_policy_update", False)),
        strict_gates=bool(_get(cc, "strict_gates", False)),
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
