# RoboMaster TT 能力清单

日期：2026-07-25
状态：✅ 所有模块均已真机验证

## 一、总览

| 模块 | 状态 | 成熟度 | 说明 |
|------|------|--------|------|
| 手动飞行 | ✅ | ⭐⭐⭐⭐⭐ | 键盘 / GUI 按钮全操控，生产可用 |
| 半自动视觉避障 (Cruise) | ✅ | ⭐⭐⭐⭐ | 直飞遇障自动绕行，真机验证通过 |
| POI 环绕飞行 (Orbit) | ✅ | ⭐⭐⭐⭐ | 视觉伺服居中 + 环绕，真机验证通过 |
| 深度估计推理 | ✅ | ⭐⭐⭐⭐ | Depth Anything V2，本地 MPS + 远程 API |
| 手势控制 | ✅ | ⭐⭐⭐⭐ | 手掌起飞 + 打响指降落 |
| 安全保护系统 | ✅ | ⭐⭐⭐⭐⭐ | 多层看门狗 + 急停 + 电量/高度保护 |
| Episode 录制 | ✅ | ⭐⭐⭐ | RGB+深度+动作+状态同步落盘 |
| MuJoCo 数字孪生 | ✅ | ⭐⭐⭐ | Mission Pad 坐标驱动仿真机体 |
| 仿真模式 (Sim) | ✅ | ⭐⭐⭐ | SimDrone/SimVideo 离线开发调试 |
| Station 组网 | ✅ | ⭐⭐⭐⭐⭐ | 飞机加入路由器，局域网内操控 |
| 单元测试 | ✅ | ⭐⭐⭐ | 控制律单元测试 |

---

## 二、各模块详细能力

### 2.1 手动飞行控制

**入口**：GUI 界面 + 键盘

| 操作 | 按键 | 说明 |
|------|------|------|
| 连接/断开 | `C` / CONNECT 按钮 | 自动探测飞机在线状态 |
| 起飞 | `T` / TAKEOFF 按钮 | 自动定高悬停 |
| 降落 | `L` / LAND 按钮 | 自动落地停桨 |
| 悬停 | `SPACE` / HOVER 按钮 | 立即回中悬停，同时解除 AUTO |
| 急停 | `ESC` | 立即停桨（危险，仅紧急用） |
| 前进/后退 | `W` / `S` 或 ↑↓ | pitch 杆量 ±40（可配 --rc-speed） |
| 左移/右移 | `A` / `D` 或 ←→ | roll 杆量 ±40 |
| 上升/下降 | `R` / `F` | throttle 杆量 ±40 |
| 左转/右转 | `Q` / `E` | yaw 杆量 ±40 |
| 切换帮助 | `H` | 显示/隐藏快捷键面板 |
| 退出 | `X` / QUIT 按钮 | 安全降落并退出 |

**特点**：
- RC 杆量保持：按住不放持续发送（15Hz），松开自动回中
- 手动操作优先级高于 AUTO：手动推杆立即覆盖避障指令
- 未起飞时手动杆量被忽略（防止地面误操作）

---

### 2.2 半自动视觉避障 (Cruise Mode)

**入口**：起飞后按 `V` → ARMED，再按 `V` → ON

**FSM 状态流转**：
```
PREFLIGHT → HOVER → APPROACH → AVOID_TURN → PASS_OBSTACLE
                                         → RECOVER_HEADING
                                         → POST_CLEAR → 回到 APPROACH
```

**控制律特点**：
- 基于深度图分区（左/中/右三等分，中位近度）
- 三区通畅 → 直飞前进 (cruise_speed=30)
- 前方有障碍 → 判定左右哪侧更开阔，横移 (roll) 绕开
- 方向锁定滞回：一旦选定绕行方向，不因障碍暂时消失而翻转
- 接近区前进量线性递减：越靠近障碍越少前进，形成自然弧线
- 被围住 → 悬停保护
- 死胡同检测：连续 10 帧 danger 单调上升 → 判定死胡同，悬停

**可配置参数** (`AvoidParams`, `FsmParams`)：见 `tt_control/avoidance.py`

---

### 2.3 POI 环绕飞行 (Orbit Mode)

**入口**：启动时 `orbit_mode=True`（默认开启），FSM 检测到目标后自动切入

**控制策略（视觉伺服优先）**：
```
每帧:
  chair_pos = 重心法算椅子水平位置 [-1, 1]

  |pos| < 0.12 (居中) → yaw 微调 + roll=10 慢速横移环绕
  |pos| ≥ 0.12 (偏了) → yaw 全力拉回 (±50) + roll=0 停横移
  pos = None (丢了)   → 悬停等待, 5s 超时→HOVER
```

**核心算法**：
- **椅子检测**：中部水平带(30%-80%)逐列中位近度，减去 0.10 背景噪声，加权重心归一化到 [-1,1]
- **视觉伺服**：`yaw = chair_pos × 200`，钳位 ±50
- **距离保持**：`dist_err = target_nearness - M`，pitch 修正，死区 ±0.05
- **安全保护**：M > 0.78 → DANGER 悬停；M < 0.10 持续 5s → LOST 悬停

**可配置参数** (`OrbitParams`)：见 `tt_control/avoidance.py`

---

### 2.4 深度估计推理

| 后端 | 状态 | 说明 |
|------|------|------|
| `passthrough` | 默认 | 不做推理，仅透传帧 |
| `depth-anything` | ✅ | Depth Anything V2-Small，HTTP API |
| `gestures` | ✅ | 手势识别分类器 |

**推理架构**：
```
VideoStream (H.264 解码, 30fps)
  → AsyncInferWorker (独立线程, 异步推理)
    → POST JPEG → Depth Service → nearness grid (float16, H×W)
  → AvoidanceFSM / OrbitController (控制律消费 nearness)
```

**服务端** (`server/da_v2_service.py`)：
- 模型：`depth-anything/Depth-Anything-V2-Small-hf`
- 设备：Apple M5 MPS（本地）/ NVIDIA 4090（远程 `depth.david-x.com`）
- 端口：8890（本地）/ 8899（远程）
- 协议：POST JPEG → binary nearness grid
- 无 Flask 依赖，标准库 `http.server`

**启动命令**：
```bash
# 本地真机（推荐）：受管子进程，退出 main 自动停（默认 :8899）
.venv/bin/python main.py --inference depth-anything --start-depth-service -v

# 指定远程/外挂服务（main 不会代为清理）
.venv/bin/python main.py --inference depth-anything --depth-service http://10.229.20.125:8899/depth -v
```

---

### 2.5 手势控制

**入口**：`--inference gestures` + 可选 `--gesture-dry-run` / `--gesture-flight-test`

| 手势 | 动作 | 说明 |
|------|------|------|
| 手掌向上 (palm-up) | 起飞 | 需电池 >30%，未起飞状态 |
| 打响指 (finger-snap) | 降落 | 需已起飞状态 |

**模式**：
- **干跑模式** (`--gesture-dry-run`)：只识别不控制，用于验证手势检测
- **真机测试** (`--gesture-flight-test`)：需手动 ARM → 手势起飞 → 悬停等待 → 手势降落 → PASS/FAIL

**训练**：GUI 提供 TRAIN TAKEOFF / TRAIN LAND / TRAIN NONE / SAVE PROFILE 按钮，支持自定义手势训练

---

### 2.6 安全保护系统

| 保护机制 | 触发条件 | 行为 |
|----------|----------|------|
| **AutoWatchdog 深度失联** | 深度超过 `depth_stale_s`(20s) 未更新 | 悬停 + 解除 AUTO |
| **AutoWatchdog 挂载超时** | AUTO ON 超过 `max_engaged_s`(120s) | 悬停 + 解除 AUTO |
| **低电量保护** | 电量 < `min_battery_pct`(10%, 可配) | ABORT → HOVER |
| **高度异常** | 高度 < 0 或 > `max_height_cm`(500) | ABORT → HOVER |
| **急停按钮** | ESC 键 | 立即停桨 |
| **SPACE 悬停** | 任意时刻 | 回中悬停 + 解除 AUTO |
| **FSM 状态超时** | APPROACH/TURN/PASS/RECOVER 各态超时 | ABORT → HOVER |
| **死胡同检测** | 连续 10 帧 danger 单调上升 | ABORT → HOVER |
| **DANGER 保护 (Orbit)** | 中区 nearness > 0.78 | 悬停保护 |
| **LOST 超时 (Orbit)** | 椅子丢失 > 5s | 悬停保护 |

---

### 2.7 Episode 录制

**入口**：`--record` 参数

**录制内容**：
- RGB 原始图传帧（10Hz 限流采样）
- 深度 nearness grid
- 当前动作 (RcAxes: roll/pitch/throttle/yaw)
- 控制状态 (FSM state + sub_state)
- 无人机遥测 (电池/高度/姿态/速度等)
- 时间戳 (Python monotonic + 系统时间)

**输出**：`logs/episodes/<timestamp>/` 目录，每帧一个 `.npz` + `meta.json`

**用途**：WAM 世界模型训练数据、离线回放分析

---

### 2.8 MuJoCo 数字孪生

**入口**：`--mujoco` 参数

**功能**：
- 读取 Mission Pad 下视定位 (x/y/z)
- 驱动 MuJoCo 仿真中的虚拟机体同步运动
- 轨迹保存到 `logs/trajectories/`
- 需要飞机飞越 Mission Pad 获取初始定位

---

### 2.9 仿真模式

**入口**：`--sim` 参数

**功能**：
- SimDrone：模拟 Tello UDP 协议，接受所有 SDK 命令并更新模拟状态
- SimVideo：读取本地视频文件 / 生成测试图案替代图传
- 无需真机即可开发调试控制律、GUI、FSM

---

### 2.10 Station 组网

**入口**：`python station_mode.py <setup|find>`

| 命令 | 说明 |
|------|------|
| `setup` | Mac 连接飞机热点 RMTT-xxxx → 发送路由器 SSID/密码 → 飞机重启加入路由器 |
| `find` | Mac 连接路由器 → 广播扫描 /24 网段 → 找到飞机局域网 IP |

**特点**：
- 纯标准库，无需额外依赖
- Wi-Fi 配置持久化 `wifi_config.json`
- 飞机加入路由器后，操控距离远超直连模式（AP 模式 ~50m，路由器模式覆盖整个房间）

---

### 2.11 单元测试

**文件**：`tests/test_avoidance.py`

**覆盖**：
- `AvoidanceController.zone_nearness()` 分区正确性
- `AvoidanceController.decide()` 各种场景决策
- `OrbitController._chair_horizontal()` 重心计算
- `OrbitController.decide()` 环绕控制律

**运行**：`.venv/bin/pytest tests/test_avoidance.py -v`

---

## 三、代码结构一览

```
robomaster-tt-control/
├── main.py                 # 入口 + CLI 参数
├── station_mode.py         # Tello 组网工具 (setup / find)
├── tt_control/
│   ├── app.py              # GUI 主程序 (OpenCV 界面 + 事件分发)
│   ├── config.py           # AppConfig, IP 检测
│   ├── control.py          # RcAxes, 键盘映射, HELP_TEXT
│   ├── tello_client.py     # Tello UDP SDK 客户端
│   ├── video_stream.py     # H.264 图传解码
│   ├── status.py           # 飞机在线探测 (ping + UDP)
│   ├── avoidance.py        # 控制律: AvoidanceController + OrbitController
│   ├── avoidance_fsm.py    # 状态机: AvoidanceFSM
│   ├── depth_backend.py    # 深度推理客户端
│   ├── async_infer.py      # 异步推理工作线程
│   ├── inference.py        # 推理后端抽象 + PassthroughBackend
│   ├── auto_safety.py      # AutoWatchdog 看门狗
│   ├── episode_recorder.py # Episode 录制器
│   ├── mujoco_twin.py      # MuJoCo 数字孪生
│   ├── flight_test.py      # 手势真机测试录制器
│   └── sim_drone.py        # 仿真模式 (SimDrone/SimVideo)
├── server/
│   └── da_v2_service.py    # 深度推理 HTTP 服务
├── tests/
│   └── test_avoidance.py   # 控制律单元测试
└── docs/
    ├── README.md           # 文档索引
    ├── design/             # 设计文档
    ├── dev-notes/          # 开发笔记
    ├── handover/           # 交接文档 (面向大众/服务器)
    └── references/         # 外部参考
```
