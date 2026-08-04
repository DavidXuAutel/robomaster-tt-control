"""``MockAirSimDroneEnv`` — offline/CI stand-in with the AirSimDroneEnv surface.

No AirSim, no torch, no cv2. Kinematics reuse ``apply_body_delta`` (the same
body->world map the real env's velocity command and the eval bridge use), so the
mock and real paths agree on heading conventions. It fabricates a deterministic
pseudo-RGB, a synthetic depth field, and a plausible IMU (|lin_acc| ~ g so the
``sanity.imu_ok`` gate would pass), and flags a collision when the drone leaves a
configurable box — enough to exercise the collector, reward, and corrector
end-to-end without a renderer.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from experiments.aerial.eval.run_closed_loop import apply_body_delta
from experiments.aerial.rl.env.action import body_delta_limits, clip_body_delta
from experiments.aerial.rl.env.obs import Observation


@dataclass
class MockEnvConfig:
    width: int = 224
    height: int = 224
    step_hz: float = 30.0
    seed: int = 0
    bounds_m: float = 500.0   # collide if |pos| exceeds this on any axis


class MockAirSimDroneEnv:
    """Kinematic mock mirroring ``AirSimDroneEnv`` (reset/step/observe/close)."""

    def __init__(self, config: Optional[MockEnvConfig] = None, **kwargs: Any) -> None:
        self.config = config or MockEnvConfig(**kwargs)
        self._pos = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0
        self._vel = np.zeros(3, dtype=np.float64)
        self._goal: Optional[np.ndarray] = None
        self._collided = False
        self._t0 = time.perf_counter()

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        if episode is not None:
            positions = np.asarray(episode["pos"], dtype=np.float64)
            yaws = np.asarray(episode["yaw"], dtype=np.float64).reshape(-1)
            self._pos = positions[0].copy()
            self._yaw = float(yaws[0])
            self._goal = positions[-1].copy()
        else:
            self._pos = np.zeros(3, dtype=np.float64)
            self._yaw = 0.0
            self._goal = None
        self._vel = np.zeros(3, dtype=np.float64)
        self._collided = False
        return self.observe()

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        dt = 1.0 / float(self.config.step_hz)
        cmd = clip_body_delta(action, body_delta_limits(dt))  # same cap as real env
        prev = self._pos.copy()
        self._pos, self._yaw = apply_body_delta(self._pos, self._yaw, cmd)
        self._vel = (self._pos - prev) / dt
        self._collided = bool(np.any(np.abs(self._pos) > self.config.bounds_m))
        return self.observe(), {"cmd": cmd.tolist()}

    def observe(self) -> Observation:
        return Observation(
            rgb=self._render(),
            state=self.observe_state(),
            collided=self._collided,
            depth=self._render_depth(),
            imu=self._fake_imu(),
            t=time.perf_counter() - self._t0,
            info={"goal": None if self._goal is None else self._goal.tolist()},
        )

    def observe_state(self) -> np.ndarray:
        return np.array(
            [self._pos[0], self._pos[1], self._pos[2],
             self._vel[0], self._vel[1], self._vel[2], self._yaw],
            dtype=np.float32,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "MockAirSimDroneEnv":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    # -- synthetic sensors ------------------------------------------------
    def _render(self) -> np.ndarray:
        x, y, z = self._pos
        s = self.config.seed
        r = int((math.sin(x * 0.1 + s) * 0.5 + 0.5) * 255) % 256
        g = int((math.sin(y * 0.1 + s) * 0.5 + 0.5) * 255) % 256
        b = int((math.sin(z * 0.1 + s) * 0.5 + 0.5) * 255) % 256
        return np.full((self.config.height, self.config.width, 3), [r, g, b], dtype=np.uint8)

    def _render_depth(self) -> np.ndarray:
        # A smooth ramp so depth_ok's dynamic-range check would pass.
        h, w = self.config.height, self.config.width
        col = np.linspace(1.0, 50.0, w, dtype=np.float32)
        return np.broadcast_to(col, (h, w)).copy()

    def _fake_imu(self) -> Dict[str, Any]:
        # Stationary-ish: |lin_acc| ~ g so sanity.imu_ok passes.
        return {"ang_vel": [0.0, 0.0, 0.0], "lin_acc": [0.0, 0.0, 9.807]}
