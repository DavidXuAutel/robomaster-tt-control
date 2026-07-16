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

## 启动

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
