"""定点航线仿真端到端:闭环飞行 + MuJoCo 孪生记录 + 出图。

需要 mujoco / matplotlib(见 requirements-sim.txt);未装则跳过。
"""
import json
import pathlib

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("matplotlib")

from sim_mission import run_mission, run_square_mission


def test_mission_closed_loop_and_record(tmp_path):
    # 用较短距离缩短测试耗时,断言仍覆盖全链路
    s = run_mission(
        forward_cm=30.0,
        height_cm=50.0,
        cruise=40,
        traj_dir=tmp_path,
        do_plot=True,
    )

    # 闭环定位精度(容差 5cm)
    assert abs(s["forward_err_cm"]) <= 5, s
    assert s["return_err_cm"] <= 5, s          # 回原点
    assert abs(s["cruise_z_err_cm"]) <= 5, s   # 巡航高 50cm

    # 轨迹已记录且导出
    assert s["traj_count"] >= 20, s
    csv_path = pathlib.Path(s["csv"])
    assert csv_path.is_file()
    json_path = csv_path.with_suffix(".json")
    assert json_path.is_file()

    # 出图
    assert s["png"] is not None
    assert pathlib.Path(s["png"]).stat().st_size > 5000

    # JSON 元数据:起飞抬升 + 回落
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["count"] == s["traj_count"]
    assert meta["frame"] == "mission_pad_local_m"
    assert meta["start"]["z"] > 0.4          # 起飞已离地
    assert meta["end"]["z"] < 0.2            # 已降落


def test_square_mission_yaw_and_return(tmp_path):
    # 小边长缩短耗时;方形巡航含 4 次偏航,验证转向 + 回原点 + 航向复位
    s = run_square_mission(
        side_cm=40.0,
        height_cm=50.0,
        cruise=40,
        traj_dir=tmp_path,
        do_plot=True,
    )

    assert s["mission"] == "square"
    assert s["return_err_cm"] <= 8, s              # 绕一圈回起点
    assert abs(s["heading_err_deg"]) <= 5, s       # 航向复位到 ~0°
    assert abs(s["cruise_z_err_cm"]) <= 5, s

    # 四个角都被记录,且走出二维方形(x、y 都明显展开)
    m = s["marks_cm"]
    assert {"corner1_x", "corner2_y", "corner3_x", "corner4_y"} <= set(m), m
    assert m["corner1_x"] >= 30 and m["corner2_y"] >= 30, m   # 展开到 +x、+y

    assert s["traj_count"] >= 30, s
    assert pathlib.Path(s["png"]).stat().st_size > 5000
