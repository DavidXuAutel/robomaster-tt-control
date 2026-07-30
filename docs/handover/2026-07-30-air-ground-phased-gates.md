# 空地协同演示 · 分阶段验收门（G0–G5）

日期：2026-07-30  
状态：代码骨架已合入；真机门禁待场地执行  
相关：`mission_brain/`、`adapters/`、`configs/mission/shared_map.example.json`

## 架构一句话

狗吃 LiDAR SLAM（权威米制地图）→ 无人机吃侦察 + 区域级定位 → MissionBrain 做任务编排。  
共享的是语义/拓扑地图（region / waypoint / anchor），不是稠密 SLAM。

## 模块入口

| 路径 | 作用 |
|---|---|
| `mission_brain/events.py` | v1 事件契约 |
| `mission_brain/map_model.py` | 共享地图 |
| `mission_brain/brain.py` | 任务 FSM |
| `adapters/drone_tello.py` | Tello Scout |
| `adapters/drone_autel.py` | 道通尖刺（同契约） |
| `adapters/dog_stub.py` / `dog_sdk.py` | 狗 stub → SDK |
| `configs/mission/shared_map.example.json` | 地图样例 |

离线冒烟：

```bash
.venv/bin/python -c "
from mission_brain.map_model import SharedMap
from mission_brain.runner import run_demo_mission
print(run_demo_mission(SharedMap.load('configs/mission/shared_map.example.json')))
"

.venv/bin/python -m pytest tests/test_mission_*.py tests/test_drone_scout_adapters.py tests/test_dog_stub.py -q
```

---

## G0 — 契约回放（无硬件）✅ 自动化

**通过标准**

- 重复 `event_id` 不导致第二次 `dog.inspect` / `gas.sample`
- 乱序 `dog.arrived`（尚在 SCOUTING）被忽略
- 任意状态 `mission.abort` → `SAFE_FAILED`
- 事件含 `global_pose` / 点云等禁止字段 → 校验失败

**命令**：`pytest tests/test_mission_events.py tests/test_mission_replay.py tests/test_mission_brain.py -q`

---

## G1 — 狗单独（场地）

**通过标准**

- 同一狗地图连续 **10 次** 导航到 `dog_goal_id`（staging）并安全停
- 断网/取消任务时狗安全停，不盲走

**操作**：用厂商导航栈 + `DogSdkAdapter` 注入 `NavBackend`；先可用遥控确认路点 ID 与 `shared_map` 一致。

---

## G2 — 无人机单独

**通过标准**

- 脚本场景 **20/20** 正确 `region_id` + `target_label`；零次误报导致派狗
- Tello：`TelloScoutAdapter` + 颜色/AprilTag 区域确认
- 若演示用道通：同期完成尖刺清单（`AutelScoutAdapter.spike`）
  - connect / takeoff / land / abort_rth
  - waypoint_mission / live_frame / telemetry
  - 室外可选 rtk

**命令（离线部分）**：`pytest tests/test_drone_scout_adapters.py -q`

---

## G3 — 交接（机→狗）

**通过标准**

- Brain 收到 `drone.target_found` 后只派一次 `dog.inspect`
- 狗到集结点后 **独立重找** 物体 A，**10/10**
- 禁止用无人机估的米制坐标当狗目标（只许 `dog_goal_id`）

**命令（stub）**：`pytest tests/test_mission_e2e.py -q`

---

## G4 — 气检

**通过标准**

- **10/10** 有效采样窗 → `gas.completed` 含 CH4/H2S 等 readings
- 传感器断开 → `gas.failed` reason=`sensor_disconnected`
- 标定过期 → `gas.failed` reason=`calibration_stale`

**命令（stub）**：`pytest tests/test_dog_stub.py -q`

---

## G5 — 端到端（大场地）

**通过标准**

- 含正常完成、NOT_FOUND、通信中断、abort 的 **10 连跑**
- 除安全干预外无人工改任务状态
- 证据图与事件日志可追溯（`evidence_uri`）

场地节奏：有限场地跑通 G0–G4 → 大场地 G5。Tello 先验算法；道通承大场地/室外 RTK。

---

## 红线

- 不改 orbit 控制律 / `configs/default.json` 基线快照
- 不把多机逻辑塞进 `AvoidanceFSM`
- WAM 不替代 MissionBrain；可后期作 Scout/Dog 内部策略插件
- v1 不做共享稠密 SLAM / 跨视角 VPR / LLM 任务规划
