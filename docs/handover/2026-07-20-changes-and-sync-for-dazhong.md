# 代码变更与服务器同步说明（2026-07-20）

> 面向：大众。
> 一句话：手势控制以云端 git（`ouxuedong/auto-fly`）为准，已合并进本地并与避障/仿真整合，回归 **44 项全绿**；本文列清我们改了/加了什么，以及往服务器同步的计划与冲突处理。

---

## 一、我们做了什么（本地代码）

### 1. 新增的模块

**手势控制（来自 git，权威版）**

| 文件 | 内容 |
|---|---|
| `tt_control/gesture_control.py` | 手势后端主体：MediaPipe Gesture Recognizer + 动态轨迹 + DTW 判断 + 引导式录制 |
| `tt_control/gesture_profile.py` | 少样本动态手势模板：关键点归一化 / DTW / 自动阈值 / 本地 profile 读写 |
| `tt_control/flight_test.py` | `FlightTestRecorder`：真机手势测试逐事件 JSONL 记录 |
| `tt_control/assets/gesture_recognizer.task` | MediaPipe 手势识别模型（8MB） |
| `tests/test_gesture_control.py` 等 4 个 | 手势相关测试（control/profile/app 事件/flight_test） |

**深度避障 / 离线仿真（本地在做的工作）**

| 文件 | 内容 |
|---|---|
| `tt_control/avoidance.py` | 深度视觉避障控制律 |
| `tt_control/depth_backend.py` | 连 GPU 深度服务（Depth Anything V2）的推理后端 |
| `tt_control/policy.py` | 策略协议 + Mock/Scripted/External 适配点 |
| `tt_control/sim_drone.py` | 仿真无人机 + 合成图传（兼容真机 IO；支持起降/平移/转向） |
| `tt_control/sim_runner.py` | 无头闭环仿真会话 |
| `tt_control/trajectory_plot.py` | 轨迹 CSV → 实线 PNG |
| `server/da_v2_service.py` | Depth Anything V2 推理微服务（部署在 4090） |
| `sim_mission.py` / `sim_avoidance.py` / `sim_replay.py` / `offline_avoidance.py` / `fly_real_mission.py` / `diag_tello.py` | 仿真/避障/真机任务与诊断脚本 |
| `tests/` 下 7 个 | 避障/仿真/控制/策略/轨迹/集成/任务测试 |
| `requirements-avoidance.txt` / `requirements-sim.txt` | 拆分依赖 |

### 2. 修改的模块

| 文件 | 改了什么 |
|---|---|
| `tt_control/app.py` | 统一主循环里让**三条控制路径共存**：git 手势（事件流 `_handle_inference_event` + ARM/dry-run/flight-test 状态机 + 训练按钮）、深度避障（`V` 键 OFF→ARMED→ON）、离线仿真（`--sim`）；`_cleanup_session` 同时清理测试记录与避障状态 |
| `tt_control/inference.py` | `InferenceBackend` 协议加 `InferenceEvent` / `drain_events`；`create_backend` 同时注册 `gestures`(git) 与 `depth-anything` 两个后端 |
| `tt_control/config.py` | 手势参数（`gesture_commands_enabled`/`gesture_flight_test`）+ 仿真开关（`sim`） |
| `main.py` | 手势用 `--inference gestures` + `--gesture-dry-run`/`--gesture-flight-test`；避障/仿真加 `--sim`/`--depth-service` |
| `tt_control/control.py` | 加按键 `V` = 避障自动切换 |
| `tt_control/mujoco_twin.py` | 无头记录模式 + 修复慢速/悬停时轨迹采样锚点漂移 bug |
| `tt_control/status.py` / `tello_client.py` | 跨平台 ping / 客户端小改 |
| `auto_fly.py` / `station_mode.py` / `requirements.txt` / `.gitignore` / `README.md` | 手势入口与依赖、文档、网络脚本（git 7-17 基线）对齐 |

### 3. 删除的（整理时清理，统一以 git 手势为准）

本地曾有过**另一套**手势原型（`tt_control/gesture.py`，MediaPipe Hands 关键点+规则判断）。按"手势以 git 为准"已删除：`gesture.py`、`gesture_dryrun.py`、`sim_gesture_flight.py`、旧模型 `hand_landmarker.task`。

### 4. 状态

- 回归测试：`pytest tests/ -q` → **44 passed**（手势 + 避障 + 仿真 + 控制/轨迹合在一起跑通）。
- **git**：手势 4 个提交已并入**本地** `main`（HEAD `0d4542a`）；避障/仿真为工作区未提交改动，与手势整合共存。
- **GitHub 远端 `origin/main` 未改动**（仍 `ae404f1`，全程无 push）。手势提交本就在 `origin/ouxuedong/auto-fly` 上。

---

## 二、服务器现状与同步计划

**服务器项目**：`yao@10.229.20.125:/home/yao/Projects/robomaster-tt-control`（**非 git**，手工拷贝）。

经内容级比对（已消除 CRLF/LF 干扰）：

- **两边完全一致、无需动**（约 19 个）：避障/仿真层与基础模块全部相同——`avoidance.py`、`depth_backend.py`、`policy.py`、`sim_runner.py`、`trajectory_plot.py`、`mujoco_twin.py`、`tello_client.py`、`control.py`、`status.py`、`video_stream.py`、`da_v2_service.py`、`offline_avoidance.py`、`sim_avoidance.py`、`sim_replay.py`、`fly_real_mission.py`、`diag_tello.py` 及共享测试。
- **差异几乎全由手势引起**：服务器跑的是**老手势** `gesture.py`（MediaPipe Hands）＋一套调参工具（`gesture_capture/shots*/fire/landing_shots`）和样本目录 `shots*/`；git 用的是 `gesture_control.py`（Gesture Recognizer + DTW）。

### 同步动作（计划）

**新增到服务器**：`gesture_control.py`、`gesture_profile.py`、`flight_test.py`、`assets/gesture_recognizer.task`、4 个手势测试；外加 `sim_mission.py`、`test_sim_mission.py`（服务器缺）。

**覆盖服务器（写入前自动备份 `.bak-20260720`）**：`main.py`、`tt_control/app.py`、`config.py`、`inference.py`、`sim_drone.py`（我们是超集）、`auto_fly.py`、`station_mode.py`、`requirements.txt`、`.gitignore`、`README.md`。

**归档并从活动树移除（备份到 `_old_gesture_20260720/`）**：老手势 `gesture.py`、`assets/hand_landmarker.task`、调参脚本（`gesture_capture.py`/`gesture_dryrun.py`/`gesture_fire_shots.py`/`gesture_shots.py`/`gesture_shots_both.py`/`landing_shots.py`/`sim_gesture_flight.py`）、样本目录 `shots/`、`shots_fire/`、`shots_land/`。

**不动**：`.venv/`、`logs/`、`wifi_config.json`、既有 `*.orig*` 备份、`.merge-backup-mine/`、服务器独有文档 `docs/2026-07-19-realdrone-verify-gesture.md`。

同步后会在服务器 `.venv` 里跑一遍 `pytest tests/ -q` 验证。

---

## 附：改动目录结构总览

图例：🟢 新增·git手势　🔵 新增·本地避障/仿真　🟡 修改（整合接触点）　🔴 已删除（旧手势）　⚪ 未改动

```text
robomaster-tt-control/
├── main.py                     🟡 手势 --inference gestures / --gesture-dry-run / --gesture-flight-test；避障 --sim / --depth-service
├── auto_fly.py                 🟡 手势启动入口 + 组网脚本对齐 git 基线
├── station_mode.py             🟡 组网脚本对齐 git 基线
├── wifi_config.py              ⚪
├── requirements.txt            🟡 +mediapipe
├── requirements-avoidance.txt  🔵 避障依赖
├── requirements-sim.txt        🔵 仿真依赖
├── .gitignore                  🟡 +gesture_profiles/
├── README.md                   🟡 手势/组网文档
├── diag_tello.py               🔵 真机诊断
├── offline_avoidance.py        🔵 离线避障
├── sim_avoidance.py            🔵 避障 2D 验证
├── sim_mission.py              🔵 仿真任务（服务器缺）
├── sim_replay.py               🔵 轨迹回放
├── fly_real_mission.py         🔵 真机任务
├── gesture_dryrun.py           🔴 旧手势演示（已删）
├── sim_gesture_flight.py       🔴 旧手势仿真（已删）
│
├── tt_control/
│   ├── app.py                  🟡 ★核心：手势事件流 + 避障 AUTO(V) + 离线 sim 三路径共存
│   ├── inference.py            🟡 InferenceEvent/drain_events；注册 gestures(git)+depth-anything
│   ├── config.py               🟡 手势参数 + sim 开关
│   ├── control.py              🟡 加按键 V=避障切换
│   ├── mujoco_twin.py          🟡 无头记录 + 修复轨迹采样漂移 bug
│   ├── status.py               🟡 跨平台 ping
│   ├── tello_client.py         🟡 客户端小改
│   ├── video_stream.py         ⚪
│   ├── __init__.py             ⚪
│   ├── gesture_control.py      🟢 手势后端主体（Gesture Recognizer + 动态轨迹 + DTW + 录制）
│   ├── gesture_profile.py      🟢 少样本手势模板（归一化 / DTW / 自动阈值 / profile）
│   ├── flight_test.py          🟢 真机手势测试 JSONL 记录器
│   ├── gesture.py              🔴 旧手势（MediaPipe Hands 规则版，已删）
│   ├── avoidance.py            🔵 深度避障控制律
│   ├── depth_backend.py        🔵 连 GPU 深度服务的推理后端
│   ├── policy.py               🔵 策略协议 + External 适配点
│   ├── sim_drone.py            🔵 仿真无人机（超集：起降/平移/转向）
│   ├── sim_runner.py           🔵 无头闭环仿真
│   ├── trajectory_plot.py      🔵 轨迹 CSV → 实线 PNG
│   └── assets/
│       ├── gesture_recognizer.task   🟢 手势模型（8MB）
│       ├── hand_landmarker.task      🔴 旧手势模型（已删）
│       └── tello_pad_twin.xml        ⚪
│
├── tests/
│   ├── test_gesture_control.py       🟢
│   ├── test_gesture_profile.py       🟢
│   ├── test_app_gesture_events.py    🟢
│   ├── test_flight_test.py           🟢
│   ├── test_avoidance.py             🔵
│   ├── test_control.py               🔵
│   ├── test_policy.py                🔵
│   ├── test_trajectory.py            🔵
│   ├── test_integration_sim.py       🔵
│   ├── test_sim_drone.py             🔵
│   └── test_sim_mission.py           🔵（服务器缺）
│
├── server/
│   └── da_v2_service.py        🔵 Depth Anything V2 推理微服务（4090）
│
├── docs/
│   ├── 2026-07-20-changes-and-sync-for-dazhong.md    🔵 本文
│   ├── 2026-07-20-gesture-control-handover.md        🔵 手势模块交接
│   ├── 2026-07-18-merge-notes.md                     🔵
│   ├── 2026-07-18-avoidance-dev-notes.md             🔵
│   ├── 2026-07-17-simulation-plan.html               🔵
│   └── superpowers/specs/
│       ├── 2026-07-16-tt-control-design.md           🟡
│       └── 2026-07-17-tt-simulation-plan.md          🔵
│
└── examples/
    └── demo_inference.py       ⚪
```

- **🟡 修改（8）**：全是"让 git 手势 + 本地避障/仿真共存"的接触点，核心 `tt_control/app.py`。
- **🟢 新增·手势（9）**：`gesture_control.py`/`gesture_profile.py`/`flight_test.py`/模型/4 测试——以云端 git 为准的手势那套。
- **🔵 新增·避障+仿真（约 20）**：`avoidance.py`/`depth_backend.py`/`policy.py`/`sim_*`/`da_v2_service.py`/脚本/测试/文档。
- **🔴 删除（4）**：本地旧手势及模型、演示脚本，被 git 版取代。

---

## 三、手势模块怎么用（接手速查）

```bash
# 本机摄像头离线自测手势识别（无需飞机）
python -m tt_control.gesture_control

# 真机 + 手势（指定飞机/本机 IP）
python main.py --tello-ip 192.168.0.100 --local-ip 192.168.0.102 --inference gestures -v

# 现场标定：dry-run（只显示不飞）
python auto_fly.py --gesture-dry-run

# 真机严格测试：需手动 TEST ARM
python auto_fly.py --gesture-flight-test

# 回归测试
python -m pytest tests/ -q
```

**手势语义**：张开手掌上抬=起飞；拇指中指捏合再快速弹开（响指）=降落。
**安全门控**（都集中在 `App._handle_inference_event`）：连接成功 + 电量≥30% + 在地面才接受起飞；检测到已离地才接受降落；有冷却；键盘 `L`/界面 `LAND`/`Esc` 始终是人工备份。
**接你自己的模型**：实现 `InferenceBackend`，在 `drain_events()` 吐 `InferenceEvent(kind="takeoff"|"land", confidence, detail)`，在 `create_backend()` 注册名字即可用 `--inference 你的名字` 启用；后端只管识别报事件，飞控安全判断全在 App 一层。
**个人手势 profile**：面板 `TRAIN TAKEOFF/LAND/NONE` 各录 10 次 → `SAVE PROFILE`，存 `gesture_profiles/`（不入库）。
