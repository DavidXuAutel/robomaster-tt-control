# CLAUDE.md

RoboMaster TT 视觉避障项目的 Agent 入口文档。每次新 Agent 进入本项目，先读此文件。

**要飞真机或采数据？** 读完本文件后接着读
`docs/handover/2026-07-28-flight-modes-and-data-collection-runbook.md`
（模块选型 × 配置档、启动命令、采数 SOP、交付规范、故障速查）。

## 项目简介

基于 Depth Anything V2 深度估计的 RoboMaster TT 无人机半自动视觉避障系统。运行在 Mac (Apple M5) 上，通过 Wi-Fi 与 Tello 无人机通信。

## 当前进展（2026-07-28）

### 已实现 & 真机验证

| 功能 | 状态 | 关键文件 |
|------|------|----------|
| 手动飞行 (键盘/GUI) | ✅ | `tt_control/control.py`, `tt_control/app.py` |
| 半自动视觉避障 (遇障横移绕行) | ✅ | `tt_control/avoidance.py` (AvoidanceController) |
| POI 环绕飞行 (视觉伺服居中+绕圈) | ✅ 07-28 椅子绕飞采数 4min43s 无 abort | `tt_control/avoidance.py` (OrbitController) |
| 随机漫游探索 (Wander) | ✅ 07-28 笼内真机采数 | `tt_control/wander.py` |
| FSM 状态机 (APPROACH→AVOID_TURN→...→ORBIT) | ✅ | `tt_control/avoidance_fsm.py` |
| 深度推理 (Depth Anything V2, 本地 MPS) | ✅ | `server/da_v2_service.py`, `tt_control/depth_backend.py` |
| 手势控制 (手掌起飞+打响指降落) | ✅ | `tt_control/gesture_control.py` |
| Episode 录制 (RGB+深度+动作+状态) | ✅ | `tt_control/episode_recorder.py` |
| 安全保护 (看门狗/低电量/高度/超时/死胡同) | ✅ | `tt_control/auto_safety.py` |
| 仿真模式 (SimDrone/SimVideo) | ✅ | `tt_control/sim_drone.py` |
| Station 组网 (飞机加入路由器) | ✅ | `station_mode.py` |

### 下一步计划

1. **P0 - 扩大采集规模**：已交付 2 份（漫游 `ep_20260727_204010`、
   椅子绕飞 `ep_20260727_205833`），继续在不同布局/光照下累积
2. **P0 - 固定路线重复采集**：预设几条路线，不同光照/布局下重复采数
3. **P1 - 走廊跟随 / 动态目标跟踪 / 贴墙飞行**：复用现有视觉伺服逻辑
4. **P2 - 拆除最后一处模块耦合**：`WanderPolicy` 借用
   `AvoidanceController.zone_nearness()`，使 `avoid.band_top/band_bottom`
   被漫游与绕飞共享，见 `docs/dev-notes/2026-07-28-orbit-wander-decoupling.md`

目标：为 WAM 世界模型训练积累多样化飞行数据。

## Agent 真机测试铁律（必须遵守）

1. **深度服务必须受管启动**：用 `--start-depth-service`（或省略 `--depth-service`，main 会自动启用）。退出 UI / `X` / Ctrl-C 后由 atexit 停子进程。
2. **禁止** 自行 `da_v2_service.py &` 再 `--depth-service http://127.0.0.1:8890/...` 做本地真机测试——外挂进程不会随 main 退出，易 residual 占 GPU/内存。
3. 仅当主人明确要求连远端/已有常驻服务时，才许用 `--depth-service URL`；结束测试后必须确认对应端口已释放。
4. 测试结束自检：`lsof -nP -iTCP:8899,8890 -sTCP:LISTEN` 应无本项目深度服务。

## 环绕（Orbit）模块修改守则（必须遵守）

POI 环绕模块（`tt_control/avoidance.py` 的 OrbitController、
`tt_control/avoidance_fsm.py` 的 orbit 路径、`configs/default.json` 的
orbit/fsm 节）经过 5 天真机调试，每个参数和结构都对应一次真机事故。

- **改这些文件前必读**：`docs/design/2026-07-27-orbit-control-principles.md`
  （安全不变量 + 历史坑 + 参数安全范围 + 修改守则）。
- **该原则文件只允许 Claude (Claude Code) 修改**，其他 Agent 一律只读。
  文件已 chmod 444，不要改回可写。
- 改动必须逐条对照原则文件的「安全不变量」，并保持
  `tests/test_orbit.py` + `tests/test_avoidance.py` 全绿。

## 快速启动

```bash
# 1. 找飞机（飞机重启后 IP 会变，每次飞前必扫；主程序占着 8889 时需先关）
python3 station_mode.py find

# 2. 启动控制界面（自动拉起本地深度子进程，退出时自动停）
#    --config 决定飞哪个模块：orbit-chair=椅子绕飞 / wander-cage=笼内漫游
.venv/bin/python main.py \
  --config configs/orbit-chair.json \
  --tello-ip <上一步扫到的 IP> \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --start-depth-service \
  -v

# 采数时追加录制（一次起飞 = 一个 episode）
.venv/bin/python main.py --config <档> ... --start-depth-service --record --record-hz 10
```

模块选型、采数 SOP 与交付规范见
`docs/handover/2026-07-28-flight-modes-and-data-collection-runbook.md`。

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
| `docs/handover/2026-07-28-flight-modes-and-data-collection-runbook.md` | **飞行/采数必读**：模块×配置档、启动命令、采数 SOP、交付规范、故障速查 |
| `docs/dev-notes/2026-07-28-orbit-wander-decoupling.md` | 绕飞与漫游的模块边界、耦合点核查与约定 |
| `docs/handover/2026-07-25-capability-inventory.md` | 全部 11 个模块的详细能力清单 |
| `docs/handover/2026-07-25-orbit-avoidance-handover.md` | 环绕飞行开发交接（参数/调试/排障） |
| `docs/design/2026-07-25-wam-training-scenarios.md` | WAM 训练 21 个场景规划 + 数据采集设计 |
| `docs/design/2026-07-27-wander-explore-design.md` | 随机漫游（Wander）实现规格书（仅 Claude 可改） |
| `docs/design/2026-07-17-tt-visual-avoidance-design.md` | 避障系统设计说明 |
| `docs/README.md` | 完整文档索引 |
