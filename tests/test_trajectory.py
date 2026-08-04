"""轨迹记录(无头孪生)+ 实线绘图测试。"""
import time
import math
import pathlib

from tt_control.mujoco_twin import MujocoPadTwin
from tt_control.trajectory_plot import plot_trajectory


def test_headless_twin_records_and_exports(tmp_path):
    # 提供一个随时间沿 x 前进的位姿(cm),间距足够触发记点
    t0 = time.time()

    def get_state():
        dx = (time.time() - t0) * 40.0   # 40 cm/s
        return {"mid": "1", "x": f"{dx:.0f}", "y": "0", "z": "100",
                "yaw": "0", "h": "100", "bat": "88"}

    twin = MujocoPadTwin(get_state=get_state, traj_dir=tmp_path, headless=True)
    assert twin.start()
    time.sleep(1.2)
    twin.stop()
    assert twin.traj_count >= 3, twin.traj_count
    csvs = list(tmp_path.glob("*.csv"))
    assert csvs, "no csv exported"


def test_plot_from_csv(tmp_path):
    csv = tmp_path / "traj_demo.csv"
    header = "t,mid,x,y,z,yaw,pitch,roll,vgx,vgy,vgz,h,bat,pad_locked\n"
    rows = []
    for i in range(40):
        x = 0.02 * i
        y = 0.3 * math.sin(i * 0.2)
        rows.append(f"{i*0.05:.2f},1,{x:.3f},{y:.3f},1.0,0,0,0,0,0,0,100,80,True")
    csv.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    out = plot_trajectory(csv)
    assert pathlib.Path(out).is_file()
    assert pathlib.Path(out).stat().st_size > 5000
