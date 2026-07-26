# CLAUDE.md

RoboMaster TT 视觉避障项目的 Agent 入口文档。每次新 Agent 进入本项目，先读此文件。

## 项目简介

基于 Depth Anything V2 深度估计的 RoboMaster TT 无人机半自动视觉避障系统。运行在 Mac (Apple M5) 上，通过 Wi-Fi 与 Tello 无人机通信。

## 当前进展（2026-07-25）

### 已实现 & 真机验证

| 功能 | 状态 | 关键文件 |
|------|------|----------|
| 手动飞行 (键盘/GUI) | ✅ | `tt_control/control.py`, `tt_control/app.py` |
| 半自动视觉避障 (遇障横移绕行) | ✅ | `tt_control/avoidance.py` (AvoidanceController) |
| POI 环绕飞行 (视觉伺服居中+绕圈) | ✅ | `tt_control/avoidance.py` (OrbitController) |
| FSM 状态机 (APPROACH→AVOID_TURN→...→ORBIT) | ✅ | `tt_control/avoidance_fsm.py` |
| 深度推理 (Depth Anything V2, 本地 MPS) | ✅ | `server/da_v2_service.py`, `tt_control/depth_backend.py` |
| 手势控制 (手掌起飞+打响指降落) | ✅ | `tt_control/gesture_control.py` |
| Episode 录制 (RGB+深度+动作+状态) | ✅ | `tt_control/episode_recorder.py` |
| 安全保护 (看门狗/低电量/高度/超时/死胡同) | ✅ | `tt_control/auto_safety.py` |
| 仿真模式 (SimDrone/SimVideo) | ✅ | `tt_control/sim_drone.py` |
| Station 组网 (飞机加入路由器) | ✅ | `station_mode.py` |

### 下一步计划

1. **P0 - 随机探索采集**：在 FSM 上叠加随机动作 + 安全约束，最大化工况覆盖
2. **P0 - 固定路线重复采集**：预设几条路线，不同光照/布局下重复采数
3. **P1 - 走廊跟随 / 动态目标跟踪 / 贴墙飞行**：复用现有视觉伺服逻辑

目标：为 WAM 世界模型训练积累多样化飞行数据。

## 快速启动

```bash
# 1. 找飞机
python3 station_mode.py find

# 2. 启动本地深度推理（后台）
server/.venv/bin/python server/da_v2_service.py --host 0.0.0.0 --port 8890 --grid 96x128 &

# 3. 启动控制界面
.venv/bin/python main.py \
  --tello-ip 192.168.0.100 \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --depth-service http://127.0.0.1:8890/depth \
  -v

# 可选：录制飞行数据
.venv/bin/python main.py ... --record --record-hz 10
```

## 飞行操作

| 按键 | 功能 |
|------|------|
| `C` | 连接/断开 |
| `T` | 起飞 |
| `L` | 降落 |
| `SPACE` | 悬停（同时解除 AUTO） |
| `V` | AUTO 切换: OFF → ARMED → ON → ARMED |
| `ESC` | 急停 |
| `WASD` / 方向键 | 手动飞行 |
| `H` | 帮助面板 |
| `X` | 退出 |

## 环境

- **Mac**: Apple M5, IP `192.168.0.103`
- **Tello**: IP `192.168.0.100`（station mode，同路由器）
- **Python**: `.venv/bin/python` (3.11, 所有依赖已安装)
- **推理 venv**: `server/.venv/bin/python` (torch 2.13 + MPS)

## 代码结构速览

```
tt_control/
├── app.py              # GUI 主程序 (OpenCV 界面)
├── avoidance.py        # 控制律: AvoidanceController + OrbitController
├── avoidance_fsm.py    # 状态机: AvoidanceFSM (9 状态)
├── control.py          # RcAxes + 键盘映射
├── tello_client.py     # Tello UDP SDK
├── depth_backend.py    # 深度推理客户端
├── async_infer.py      # 异步推理线程
├── auto_safety.py      # 看门狗
├── episode_recorder.py # 飞行数据录制
├── sim_drone.py        # 仿真模式
└── ...
server/
└── da_v2_service.py    # 深度推理 HTTP 服务
```

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/handover/2026-07-25-capability-inventory.md` | 全部 11 个模块的详细能力清单 |
| `docs/handover/2026-07-25-orbit-avoidance-handover.md` | 环绕飞行开发交接（参数/调试/排障） |
| `docs/design/2026-07-25-wam-training-scenarios.md` | WAM 训练 21 个场景规划 + 数据采集设计 |
| `docs/design/2026-07-17-tt-visual-avoidance-design.md` | 避障系统设计说明 |
| `docs/README.md` | 完整文档索引 |
