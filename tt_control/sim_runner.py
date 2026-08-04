"""无头离线会话:SimDrone + SimVideo + (可选)MuJoCo 孪生 + 策略 的闭环。

不依赖 cv2 窗口,供:
- 实机不在时端到端跑通/演示;
- 自动化测试(tests/test_integration_sim.py)。

用法:
    python -m tt_control.sim_runner [steps] [policy]
    # 例: python -m tt_control.sim_runner 200 mock
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from typing import Optional

from tt_control.policy import Policy, create_policy
from tt_control.sim_drone import SimDrone, SimVideo

logger = logging.getLogger(__name__)


def run_sim_session(
    policy: Optional[Policy] = None,
    steps: int = 200,
    dt: float = 0.05,
    enable_twin: bool = True,
    do_takeoff: bool = True,
    traj_dir: Optional[pathlib.Path] = None,
) -> dict:
    """跑一段无头仿真,返回摘要 dict。"""
    drone = SimDrone()
    drone.connect()
    drone.start_state_listener()
    video = SimVideo()
    video.start()

    twin = None
    if enable_twin:
        # 延迟导入:mujoco 较重且仅孪生需要
        from tt_control.mujoco_twin import MujocoPadTwin
        twin = MujocoPadTwin(
            get_state=lambda: drone.state,
            traj_dir=traj_dir,
            headless=True,
        )
        if not twin.start():
            logger.warning("twin start failed: %s", twin.status)
            twin = None

    if do_takeoff:
        drone.takeoff()

    if policy is not None:
        policy.reset()

    time.sleep(0.2)  # 让状态线程先产出数据
    for _ in range(steps):
        frame = video.read()
        state = drone.state
        if policy is not None and frame is not None:
            axes = policy.decide(frame, state)
            drone.rc(*axes.as_tuple())
        time.sleep(dt)

    final_state = dict(drone.state)
    traj_count = twin.traj_count if twin else 0

    drone.land()
    csv_path = None
    if twin:
        twin.stop()
        csv_path = getattr(twin, "_saved_path", None)
    video.stop()
    drone.close()

    summary = {
        "steps": steps,
        "traj_count": traj_count,
        "csv": str(csv_path) if csv_path else None,
        "cmd_count": drone.cmd_count,
        "final_state": final_state,
    }
    logger.info("sim session summary: %s", summary)
    return summary


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    steps = int(argv[0]) if argv else 200
    pol_name = argv[1] if len(argv) > 1 else "mock"
    policy = create_policy(pol_name)
    s = run_sim_session(policy=policy, steps=steps)
    print("=== sim session ===")
    for k, v in s.items():
        if k == "final_state":
            print(f"  {k}: mid={v.get('mid')} x={v.get('x')} y={v.get('y')} "
                  f"z={v.get('z')} yaw={v.get('yaw')} bat={v.get('bat')}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
