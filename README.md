# RoboMaster TT Control

远端服务器上的**统一界面**：实时图传 + 可插拔推理 + 键盘操控（Tello SDK 3.0 UDP）。

不修改服务器有线网络；仅通过 Wi-Fi（`192.168.10.x`）连接飞机。

## 环境

- Python 3.9+
- 服务器 Wi-Fi 已连接 `TELLO-*` / `RMTT-*`
- 有图形显示（本机桌面 / VNC / X11 转发）

```bash
cd ~/Projects/robomaster-tt-control   # 或你的部署路径
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键组网起飞（macOS，推荐）

```bash
python auto_fly.py
```

**首次运行**会自动进入配置向导（只需一次）：路由器 Wi-Fi 名自动探测、密码输入不回显，
配置保存在本地 `wifi_config.json`（已被 `.gitignore` 忽略，**不会提交到仓库**）。
不想走向导也可以手动配置：

```bash
cp wifi_config.example.json wifi_config.json   # 然后填入自己的 Wi-Fi 信息
```

之后每次零参数运行。脚本自动完成：连接飞机热点 → 发送 `ap` 组网指令 →
Mac 切回路由器 → 扫描飞机局域网 IP → 拉起控制界面。
组网完成后 Mac 一直待在路由器 Wi-Fi 上（**可正常上网**），通过局域网控制飞机。
想重新配置：删掉 `wifi_config.json` 再运行即可；`--ssid/--password` 参数可临时覆盖配置。

分步执行（调试用）：`python station_mode.py setup`（直连热点时发组网指令）、
`python station_mode.py find`（回路由器后扫描飞机 IP）。

## 启动（直连飞机热点模式）

```bash
# 自动检测 192.168.10.x
python main.py

# 或手动指定 Wi-Fi IP
python main.py --local-ip 192.168.10.2 -v
```

## 键位 / 界面

| 操作 | 说明 |
|------|------|
| **CONNECT 按钮** / `C` | 连接或断开无人机 |
| 右上状态灯 | `OFFLINE` / `ONLINE` / `CONNECTING` / `CONNECTED` / `ERROR` |
| T / L / Esc | 起飞 / 降落 / 紧急停桨 |
| W/S A/D R/F Q/E | 前后、左右、升降、偏航 |
| Space | 悬停 |
| H | 帮助 |
| X | 退出 |

启动后先显示界面与在线状态；点 **CONNECT** 再进入 SDK 并开图传。

## Mission Pad → MuJoCo 孪生

启用后，连接飞机会自动 `mon`（下视垫子检测）。看到 Mission Pad 时，用 SDK 的 `x/y/z`（相对垫子，cm）更新 MuJoCo 中机体位姿。

```bash
pip install mujoco
python main.py --mujoco -v
```

流程：CONNECT → 起飞 → 飞到垫子上方锁定 → MuJoCo 窗口跟随。  
控制界面 HUD 会显示 `MuJoCo: pad m1 xyz=... traj=N`；未看到垫子时为 `no pad lock`。

### 轨迹记录

- MuJoCo 中**绿色**轨迹=垫子锁定段；**黄色**=丢垫后水平保持最后锁定位置、高度继续更新  
- **蓝点**=起点，**红点**=终点  
- 关闭 MuJoCo / DISCONNECT 时自动保存：  
  `logs/trajectories/traj_YYYYMMDD_HHMMSS.csv` + 同名 `.json` 摘要  
- CSV 字段：`t, mid, x, y, z, yaw, pitch, roll, vgx, vgy, vgz, h, bat, pad_locked`（米，垫子局部系）

垫子放在 MuJoCo 世界原点（红色方块）。无垫子时水平位置不可靠。

## 接入实时推理

实现 `InferenceBackend.infer(frame) -> frame`，例如：

```python
# my_infer.py
import cv2
import numpy as np
from tt_control.inference import InferenceBackend

class MyDetector(InferenceBackend):
    def infer(self, frame: np.ndarray) -> np.ndarray:
        # TODO: 你的模型推理
        cv2.putText(frame, "infer ok", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame
```

在 `main.py` 或启动脚本里：

```python
from my_infer import MyDetector
App(cfg, inference=MyDetector()).run()
```

也可在 `tt_control/inference.py` 的 `create_backend()` 中注册新名称。

## 目录

```
main.py
tt_control/
  app.py            # 统一界面主循环
  tello_client.py   # UDP 控制/状态
  video_stream.py   # H.264 图传
  inference.py      # 推理占位
  control.py        # 键位
  config.py
docs/superpowers/specs/  # 设计说明
```

## 安全

- SDK 模式约 15 秒无指令会自动降落；程序会周期性发送 `rc` 心跳
- 退出时若已起飞会尝试 `land`
- 紧急情况按 **Esc**
