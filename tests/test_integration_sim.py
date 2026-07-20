"""端到端离线闭环:SimDrone + SimVideo + 孪生 + MockAvoidPolicy。"""
from tt_control.policy import MockAvoidPolicy
from tt_control.sim_runner import run_sim_session


def test_headless_closed_loop(tmp_path):
    s = run_sim_session(
        policy=MockAvoidPolicy(),
        steps=80,
        dt=0.03,
        enable_twin=True,
        traj_dir=tmp_path,
    )
    assert s["cmd_count"] > 80, s          # takeoff + 每步 rc + land
    assert s["traj_count"] >= 3, s         # 孪生记到多点
    assert s["csv"] is not None
    # 机体确实移动(x 偏离 0)
    assert abs(float(s["final_state"]["x"])) > 3, s["final_state"]
