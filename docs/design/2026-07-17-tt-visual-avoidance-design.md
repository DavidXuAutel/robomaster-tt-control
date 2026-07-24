# RoboMaster TT 半自动视觉避障 — 设计说明

日期：2026-07-17（2026-07-20 修订，对齐当前实现）
状态：已实现
前置：[`./2026-07-16-tt-control-design.md`](./2026-07-16-tt-control-design.md)
相关：[`../dev-notes/2026-07-18-avoidance-dev-notes.md`](../dev-notes/2026-07-18-avoidance-dev-notes.md)

> 说明：本文最初为 07-17 的方案设计（"待确认"）。07-18 之后落地并经离线/真机验证，方案有几处演进（感知改为远端 GPU 微服务、深度统一为"近度"语义、参数与控制律增强）。本次修订已将正文对齐当前代码，可直接作为后续开发的权威背景。

## 背景与决策

| 决策项 | 选择 |
|--------|------|
| 目标形态 | 半自动避障：按键确认后沿开阔方向缓慢前飞并偏航绕障 |
| 感知模型 | Depth Anything V2 Small |
| 感知部署 | **远端 GPU HTTP 微服务**（推理在 GPU 机，控制端只做瘦客户端） |
| 安全策略 | 起飞后可开自动，但首次需按键确认；近距急停；键盘随时可覆盖 |
| 实现路线 | 感知后端 + 独立 `AvoidanceController`（与 `InferenceBackend` 解耦） |

**为何拆成远端微服务**：控制端（`main.py`）跑在连飞机热点的机器上，若本地加载 DA-V2 需引入 torch/transformers 重依赖且吃 GPU。改为把推理独立成一个 HTTP 服务部署在 GPU 机（默认 4090），控制端只用标准库 `urllib` + 已有的 `opencv/numpy` 发帧收网格，主环境保持轻量、可无 GPU 运行。

## 目标

1. 起飞后按 `V` 确认开启半自动。
2. 用 Depth Anything V2 Small 估计深度，向开阔方向缓慢前飞并偏航绕障。
3. 前方过近强制悬停；键盘 RC / Space / Esc / L 可随时打断。

## 非目标（本版不做）

- 全局建图 / SLAM / 航点导航
- ToF 融合、机载推理
- 默认自动起飞或无人值守飞行
- 横移绕障（`roll=0`）、原地 yaw 扫描找路

## 架构

```
[GPU 机] server/da_v2_service.py
    DA-V2 Small 推理 + 帧内分位数归一化 → 近度网格(96×128, float16)
    协议：POST /depth (JPEG) → struct("<II",H,W)+float16 ；GET /health
                       ▲ HTTP（JPEG 上行 / 近度网格下行）
                       │
[控制机] VideoStream → DepthAnythingBackend.infer()   # 瘦客户端
                    ├─ 请求/缓存最新 DepthFrame（线程安全）
                    └─ 热力图 + 左中右分区 + 决策 HUD 叠图 → 显示
           App._update_rc_stream()
             ├─ 键盘 RC 优先（持有中，覆盖避障）
             ├─ 否则若 auto==ON → AvoidanceController.decide(nearness) → rc
             └─ 三区被围 → RcAxes(0,0,0,0) 悬停
```

| 模块 | 职责 |
|------|------|
| `server/da_v2_service.py` | GPU 机上加载 DA-V2 Small，推理，帧内分位数归一化为近度网格，HTTP 暴露 |
| `tt_control/depth_backend.py` | 瘦 HTTP 客户端：发 JPEG、收/缓存近度网格、叠图；容错复用上一帧 |
| `tt_control/avoidance.py` | 近度图 → 分区启发式 → `RcAxes`（`AvoidanceController.decide`） |
| `tt_control/inference.py` | 注册 `depth-anything` / `da-v2` / `depth` 后端 |
| `tt_control/app.py` / `control.py` | `V`/`G` 三态开关、RC 仲裁、HUD 状态 |
| `requirements-avoidance.txt` | `torch`、`transformers`、`pillow`（**仅感知服务端**需要） |

与现有约定一致：`InferenceBackend.infer(frame) → frame` 只负责感知与叠图；飞控决策在 `App` + `AvoidanceController`。

### 近度约定（关键）

服务端不直接返回原始深度，而是返回**近度图 `nearness`**：对 DA-V2 的逆深度输出做**帧内分位数（2%~98%）归一化**到 `[0,1]`，**值越大表示越近 / 越挡路**，再降采样成固定 `96×128` 小网格传输。这样阈值语义单一（"越大越危险"），传输量小，也便于单测。

## 控制律（`AvoidParams` 当前默认值，可 CLI 调）

从近度图中部水平带 `[band_top, band_bottom] = [0.30, 0.80]`（忽略天花板/地面）取出，按**左 / 中 / 右**三等分，各取中位近度 `left / mid / right`。

**危险度 `danger = max(left, mid, right)`** —— 取全视场最大而非只看中区：障碍常只占某一侧，只看中区会"斜插进侧向障碍"。

| 条件 | 动作 | 状态 |
|------|------|------|
| `danger ≤ clear_thresh(0.45)` | 全通畅，释放方向锁，`pitch = +cruise_speed(25)` | `CRUISE` |
| `danger > clear_thresh` | 锁定"远离更挡一侧"转向：`yaw = ±yaw_speed(35)` | `TURN_L/TURN_R` |
| `mid > stop_thresh(0.70)` 且 `min(left,right) > stop_thresh − side_margin(0.08)` | 被围住，悬停 | `BLOCKED` |

绕障时前进量随正前方近度线性递减，保证"中区远则绕弧推进、中区近则刹到近零原地转"：

```
frac  = (mid − clear_thresh) / (stop_thresh − clear_thresh)   # clip 到 [0,1]
pitch = round(approach_pitch(16) × (1 − frac))
```

附加规则：

- **高度**：半自动时 `throttle = 0`（锁高，靠下视 VPS）；**不横移** `roll = 0`，仅用 `yaw + 小 pitch` 绕障。
- **转向滞回（commit）**：首次进入避障时锁定绕行方向（`left ≥ right` 取右转 `+1`，否则左转 `−1`；对称居中默认右），保持不翻转，直到整个前方重新通畅（`danger ≤ clear_thresh`）才释放，避免左右反复横跳。
- **键盘覆盖**：`RC_HOLD_TIMEOUT` 内有手动键 → 走键盘链路，不用避障输出；松手超时后交回避障（若 AUTO ON）或悬停。
- **降落 / 急停 / 断开 / 悬停**：自动关闭 AUTO。

`AvoidParams` 全量默认：`cruise_speed=25, yaw_speed=35, turn_pitch=10, approach_pitch=16, stop_thresh=0.70, clear_thresh=0.45, side_margin=0.08, band_top=0.30, band_bottom=0.80`。

> 参数演进记录：`cruise_speed` 20→25、`yaw_speed` 20→35（离线验证 20 转向权限太弱、来不及绕开）；新增 `clear_thresh`（接近区提前转向）与 `approach_pitch/turn_pitch`（绕弧推进）。

## 交互与安全

三态状态机（`app.py::_toggle_auto`）：`OFF → ARMED（首次确认）→ ON`。

| 键 | 行为 |
|----|------|
| `V` / `G` | 三态切换：`OFF`→`ARMED`(确认)→`ON`(接管)；再按回 `ARMED` 暂停 |
| WASD 等 | 覆盖自动，松手经超时后再交回自动 |
| Space | 悬停并**关闭**半自动 |
| Esc / L | 急停或降落，关闭半自动 |

开 AUTO 的前置校验：**必须已起飞** 且 **深度后端在线**（`inference` 具备 `latest_depth`），否则忽略并提示。

**AUTO 看门狗**（`tt_control/auto_safety.py::AutoWatchdog`，已接入控制循环）：AUTO ON 期间每次下发前判定，命中即自动悬停并解除 AUTO、HUD 显示原因——
1. 单次挂载超时（`max_engaged_s=30s`）；
2. 感知失联（距最近一帧有效深度超 `depth_stale_s=1.5s`，含挂载后迟迟无深度）。
纯逻辑无 I/O，见 `tests/test_auto_safety.py`。人工接管（WASD/SPACE/ESC/L）仍为第一保障。

HUD 显示：`AUTO: OFF | ARMED | ON`、左/中/右近度、决策状态与杆量、推理耗时。

启动示例：

```bash
# 1) GPU 机：起感知服务（独立 venv，装 requirements-avoidance.txt）
python server/da_v2_service.py --host 0.0.0.0 --port 8899

# 2) 控制机：启用深度后端（--depth-service 留空则用内置默认地址）
python main.py --inference depth-anything --depth-service http://<gpu-ip>:8899/depth
```

服务端无 GPU 时自动回退 CPU（延迟更高）。控制端首帧连不上服务直接报错、不静默失败；之后偶发错误只记日志并复用上一帧深度。

## 离线验证（真机前）

方案与感知/控制解耦，提供两级离线验证工具：

- `sim_avoidance.py`：运动学闭环。几何真值合成近度图 → `decide()` → 积分机体运动，验证"控制律本身会不会撞"（`slalom` / `wall` / `single` 场景，输出航迹 mp4/png + 最小净空）。
- `offline_avoidance.py`：真感知离线回放。把录像/图片序列/摄像头喂给真 DA-V2 服务 + 控制律，输出标注视频、逐帧决策日志、感知 RTT，用于调阈值。
- `tests/test_avoidance.py`：合成近度图单测（直行 / 左障右转 / 被围悬停 / 滞回保持与复位）。

## 配置扩展

| 参数 | 默认 | 含义 |
|------|------|------|
| `--inference depth-anything` | `passthrough` | 启用深度后端 |
| `--depth-service <url>` | 内置 4090 地址 | 远端感知服务地址 |
| `cruise_speed` | 25 | 半自动前进杆量 |
| `yaw_speed` | 35 | 转向杆量 |
| `stop_thresh` | 0.70 | 中区过近阈值（近度） |
| `clear_thresh` | 0.45 | 全通畅阈值（近度） |

（`offline_avoidance.py` / `sim_avoidance.py` 已暴露 `--cruise / --yaw / --stop-thresh` 供离线调参。）

## 验收标准

1. `--inference passthrough` 行为与现有一致，不受避障代码影响。
2. 有服务时可看到近度热力图叠图；服务缺失/连不上时给出明确错误提示，不静默失败。
3. 地面按 `V` 不能开启自动驱动；起飞并确认（ARMED→ON）后才输出避障 `rc`。
4. 模拟"近"近度时输出悬停；键盘可随时覆盖自动输出。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| Wi-Fi + 远端推理延迟导致反应慢 | 限制 `cruise_speed`；近距阈值偏保守；服务端降采样近度网格；`min_interval` 限流复用 |
| 相对深度尺度不稳定 | 服务端帧内分位数归一化为近度再分区比较；阈值可配置 |
| 左右反复横跳 | 转向滞回 `commit`，通畅前不翻转方向 |
| 侧向障碍被斜插 | 危险度取三区 `max` 而非只看中区 |
| 自动误触 | 三态：`OFF → ARMED → ON`；Space/Esc/L 立即关闭 |
| 依赖体积大 | 重依赖只装在感知服务端；控制端主 `requirements.txt` 保持轻量 |
