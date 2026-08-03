# 大众空地协同案例 · 架构说明（实现对照）

日期：2026-07-30  
实现目录：`mission_brain/`、`adapters/`、`configs/mission/`  
验收门：`docs/handover/2026-07-30-air-ground-phased-gates.md`

## 分工

| 角色 | 负责 |
|---|---|
| 无人机 Scout | 漫游/巡航侦察、区域锚点确认、`drone.target_found` |
| MissionBrain | FSM 编排、超时/abort、幂等派单 |
| 机器狗 | 权威 SLAM 路点导航、本地重找 A、RS485 气检 |
| 共享地图 | `region_id → dog_goal_id / drone_route_id / anchor_ids` |

## FSM

`IDLE → SCOUTING → DOG_NAV → DOG_SEARCH → GAS_SAMPLE → COMPLETE`  
失败路径 → `SAFE_FAILED`

## 适配器

- `TelloScoutAdapter`：颜色标记 + 可选 AprilTag；真机控制仍用现有 `tt_control`
- `AutelScoutAdapter`：同契约 tip 刺（dry_run 默认真）；`spike` 记录能力验收
- `DogStubAdapter` / `DogSdkAdapter`：stub 联调 → 注入 Nav/Perception/Gas backend

## 明确不做（v1）

共享稠密 SLAM、无人机伪全局坐标下发、WAM 作多机协调器、Hydra/LLM 规划。
