# RoboMaster TT 统一控制界面 — 设计说明

日期：2026-07-16  
状态：已确认

## 目标

在远端服务器上运行单一 OpenCV 窗口，同时完成：

1. 实时相机图传显示  
2. 可插拔实时推理（默认透传）  
3. 键盘操控无人机  

约束：仅使用 Wi-Fi 与 Tello 通信，不修改服务器有线网络配置。

## 架构

```
main.py
  └─ App
       ├─ TelloClient     UDP 8889 控制 / 8890 状态
       ├─ VideoStream     UDP 11111 H.264 → BGR
       ├─ InferenceBackend  infer(frame) → frame
       └─ KeyboardMapper  按键 → takeoff/land/rc
```

## 网络

| 用途 | 地址 |
|------|------|
| 飞机 | `192.168.10.1` |
| 本机 Wi-Fi | `192.168.10.x`（自动或 `--local-ip`） |
| 控制 | UDP `8889` |
| 状态 | UDP `8890` |
| 视频 | UDP `11111` |

## 界面

单窗口：图传全屏区域 + 左上 HUD（电量/高度/FPS/IP）+ 右下键位提示；推理结果叠加在画面上。

## 键位

| 键 | 动作 |
|----|------|
| T | 起飞 |
| L | 降落 |
| Esc | 紧急停桨 |
| W/S | 前后 |
| A/D | 左右 |
| R/F | 升降 |
| Q/E | 偏航 |
| Space | 悬停（rc 清零） |
| H | 显示/隐藏帮助 |

## 推理扩展

实现 `InferenceBackend.infer(frame: np.ndarray) -> np.ndarray`，在配置中替换默认 `PassthroughBackend`。
