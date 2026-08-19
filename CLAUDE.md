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
| POI 环绕飞行 (视觉伺服居中+绕圈) | ✅ 07-27 真机长时验证 (2min+ 无 abort) | `tt_control/avoidance.py` (OrbitController) |
| FSM 状态机 (APPROACH→AVOID_TURN→...→ORBIT) | ✅ | `tt_control/avoidance_fsm.py` |
| 深度推理 (Depth Anything V2, 本地 MPS) | ✅ | `server/da_v2_service.py`, `tt_control/depth_backend.py` |
| 手势控制 (手掌起飞+打响指降落) | ✅ | `tt_control/gesture_control.py` |
| Episode 录制 (RGB+深度+动作+状态) | ✅ | `tt_control/episode_recorder.py` |
| 安全保护 (看门狗/低电量/高度/超时/死胡同) | ✅ | `tt_control/auto_safety.py` |
| 仿真模式 (SimDrone/SimVideo) | ✅ | `tt_control/sim_drone.py` |
| Station 组网 (飞机加入路由器) | ✅ | `station_mode.py` |

### 下一步计划

1. **P0 - 随机探索采集（Wander）**：设计已定稿 →
   `docs/design/2026-07-27-wander-explore-design.md`（实现规格书，
   只允许 Claude 修改；实现方 DeepSeek/Grok 照规格执行，含分工与验收阶梯）
2. **P0 - 固定路线重复采集**：预设几条路线，不同光照/布局下重复采数
3. **P1 - 走廊跟随 / 动态目标跟踪 / 贴墙飞行**：复用现有视觉伺服逻辑

目标：为 WAM 世界模型训练积累多样化飞行数据。

## Agent 真机测试铁律（必须遵守）

1. **深度服务必须受管启动**：用 `--start-depth-service`（或省略 `--depth-service`，main 会自动启用）。退出 UI / `X` / Ctrl-C 后由 atexit 停子进程。
2. **禁止** 自行 `da_v2_service.py &` 再 `--depth-service http://127.0.0.1:8890/...` 做本地真机测试——外挂进程不会随 main 退出，易 residual 占 GPU/内存。
3. 仅当项目负责人明确要求连接远端或已有常驻服务时，才可使用 `--depth-service URL`；结束测试后必须确认对应端口已释放。
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
# 1. 找飞机
python3 station_mode.py find

# 2. 启动控制界面（自动拉起本地深度子进程，退出时自动停）
.venv/bin/python main.py \
  --tello-ip 192.168.0.100 \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --start-depth-service \
  -v

# 可选：录制飞行数据
.venv/bin/python main.py ... --start-depth-service --record --record-hz 10
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
| `docs/design/2026-07-27-wander-explore-design.md` | 随机漫游（Wander）实现规格书（仅 Claude 可改） |
| `docs/design/2026-07-17-tt-visual-avoidance-design.md` | 避障系统设计说明 |
| `docs/README.md` | 完整文档索引 |

## Aerial WAM 训练线（2026-07-29）

本仓库还承载 aerial WAM 导航训练（OpenFly/AirSim）的设计与交接文档，代码在 FastWAM worktree `aerial-b0-b1-orchestration`。

- **当前状态**：B0→B1 编排；v1 的 B1 FT 停在 **step_002250/5000**（`ft.status=FAILED`，correction-rate 门禁），且 B1 整体未超过 B0（B0 SR=0/NE≈134）。
- **v2 重设计（从头训 B0）**：`docs/design/2026-07-29-aerial-nav-wam-redesign.md`（模型/数据/评测/训练全规格 + 主机 bring-up/启动 SOP）。
- **v1 交接**：`docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md`。
- 训练主机 `:31126`（1 机 2×H100，从头训），评测 `:30682`（1×H100 + AirSim）。
- Artifacts：`artifacts/b1_seen20_metrics_*.json`、`artifacts/b1_loss_history_through_2400.json`、`artifacts/loss_curve_b1_*.png`。

## 历史决策检索

项目历史决策只以本仓库内的设计文档、交接文档和 Git 记录为依据。公开仓库文档不得引用私人对话档案、本机私人目录或仓库外规划系统。

## 公开仓库脱敏准则（强制）

本仓库会上传 GitHub，对公众与合作方可见。**事情可以做，文档与元数据不得暴露仓库外的私人规划意图。**

1. 文档只写本项目的工程目标、事实、约束、决策、风险与验收标准；不写个人生活策略、职业意图、私人优先级或外部规划代号。
2. 禁止出现私人目录、用户名、本机历史档案、私人知识库、战略台账及其文件名或链接；引用依据必须位于本仓库或经批准的公开来源。
3. 代码注释只解释机制、安全约束和必要取舍，不记录个人动机或与本项目无关的计划。
4. Commit / PR / Issue / 评审记录使用工程语言，不出现「练兵、作品集、保个人项目、离开准备」等私人目的表述。
5. 文件名、artifact 名与报告元数据用功能/阶段/日期/版本命名，不嵌入私人项目代号、姓名、主机用户名或规划体系编号。
6. 配置只提交脱敏示例；不得提交凭据、真实账号、私人绝对路径、未经批准的内部地址。
7. 本仓库资产不得流入无关仓库、个人设备或未经批准的外部系统；外部资产须有来源、许可与审批记录。
8. 发布前扫描私人代号、绝对路径、凭据与敏感元数据，并核对 Markdown / HTML 等派生版本口径一致。

狗侧当前开发权威：`docs/design/2026-08-05-dog-deployment-loop-plan.md`（**v3.7**；当前只授权 D0；见该文 §0.3 / §0.4 / §3.0）。
近期窗口：V1 取流尽量赶在 **08-30** 充电房现场调试前；E7 只认本场景实测端口，禁止采信「一路四口」公式。

狗侧真机实测交接（**动手前必读**）：`docs/handover/2026-08-07-dog-first-navigate-loop-handover.md`
——已验证的派单/重定位 SOP、10 个坑，以及对上述方案接口基线的实测修正（F8/E1 已有结论、
F6 不可用、`relocate()` API 不生效、`result` 字段不可单独判成败）。该文 §2 界定了授权边界：
E1 已实测不等于 `Move`／D7 闭环解禁。

## 狗侧接口检索顺序（强制，先查再问）

集成商已交付一手接口材料，**任何「这个功能怎么调」的问题一律先本地检索，不要先问厂商**：

| 顺序 | 材料 | 说明 |
|---|---|---|
| 0 | `docs/references/2026-08-08-topsee-interface-inventory.md` | **接口清单总索引**，含前端逆向补充的未文档化接口（运控 WS、STOMP 推送、`model/*`） |
| 1 | `docs/handover/2026-08-07-dog-first-navigate-loop-handover.md` | 真机实测修正，**与下列文档冲突时以实测为准** |
| 2 | `docs/references/机器狗文档/*.openapi.json` | 平台 OpenAPI（机器人模块 275 接口 + 登录模块 5 接口） |
| 3 | `docs/references/机器人巡检平台用户说明书/*.md` | 业务语义、操作约束、安全口径、故障处理 |
| 4 | 问厂商 | 仅限前 4 项确实无法回答、且属授权/契约性质的问题 |

已由本地材料自证、**不必再问**的事项：上位机地址（`archivesMan/getDateById.localhostIp`）、
机器人 SN/型号、MQ 主题清单、平台是否存在连续运控接口（存在，见清单 §3.1）。
