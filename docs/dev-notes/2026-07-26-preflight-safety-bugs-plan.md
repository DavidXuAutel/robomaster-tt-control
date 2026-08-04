# 飞前安全缺陷审查与修复（第二轮）

日期：2026-07-26  
状态：✅ 飞前最小集已实施  
范围：深度新鲜度 / 急停解锁 / 禁公网回落 / approach hold 对齐 / L1 契约测

## 1. 背景

在 orbit 飞前 P0（LOST 前进、abort_reason、暂停复位、侧向 DANGER、假 POI）落地后，
对全栈做第二轮缺陷审查（探索 agent + Claude 对抗审阅）。  
目标：堵住「时间语义 / 急停 / 静默失败」类安全洞，并建立分模块测试骨架。

相关代码：
- `tt_control/async_infer.py` / `depth_backend.py` / `video_stream.py`
- `tt_control/tello_client.py`（emergency）
- `tt_control/avoidance_fsm.py`（approach_hold）
- `tt_control/app.py`（depth_stale_s）
- `main.py`（深度启动失败路径）

## 2. 飞前已修（本轮）

1. **冻结图传伪装新鲜深度**  
   - `DepthResult.ts` 改为帧采集墙钟；推理完成后消费帧。  
   - `VideoStream.frame_ts` / `SimVideo.frame_ts`；App 喂入 `infer(..., frame_ts=)`。  
   - `depth_stale_s`：20s → **3s**（本地推理，禁公网回落后安全）。

2. **ESC 急停被 takeoff/land 锁堵住**  
   - `emergency()` 独立临时 socket 直发，不抢 `_lock`。

3. **本地深度失败静默回落公网**  
   - `--start-depth-service` 失败/超时 → `return 2`。  
   - 深度后端无 `--depth-service` → `return 2`。  
   - `DepthAnythingBackend` 要求显式 `service_url`；HUD 常显生效 URL。

4. **approach hold 与 orbit danger 门限对齐**  
   - 侧障 hold 用 `OrbitParams.danger_thresh`（0.78），消除 0.78–0.82 侧刮带。  
   - `fsm.reset()` 顺手清 `_last_depth_ts`。

5. **`tello_provision.py` 明文密码**  
   - `ROUTER_PASSWORD=""`，走 CLI / getpass。

## 3. 未修 / 延后

| 项 | 说明 |
|----|------|
| 近度无绝对尺度 | 分位归一化固有限制；飞前用 `offline_avoidance.py` 本场地标定 |
| 遥测龄未接入 abort | `state_age_s` 已有，下一轮接 FSM |
| CRUISE+`_commit` 死代码 | 关 `orbit_mode` 前必修 |
| overlay 共享 controller | orbit 主路径影响小，P2 |
| 录制 `ctrl_state` 塞 HUD 串 | 训练元数据，不涉安全 |

## 4. 测试策略（分模块）

- **L1 契约单测**（本轮已补）：新鲜度 / emergency 不堵锁 / 禁公网回落 / approach hold 0.79  
- **L1.5 离线真回放**：`offline_avoidance.py` + 历史 episode（人工标定，不自动化）  
- **L2 仿真集成**：冻帧 → 看门狗解除 AUTO（下一轮）  
- **L3 真机 ≤5 条**：见下

## 5. 真机 Checklist（≤5）

1. 遮挡摄像头 / 停图传数秒 → AUTO 应在 ~3s 内解除（非盲飞）  
2. land 过程中按 ESC → 急停立即生效（不卡 UI）  
3. 故意让本地深度起不来 → 进程退出码 2，不连公网  
4. 侧障靠近（未进 orbit）→ HUD `approach_hold`，不 cruise  
5. 本场地 `offline_avoidance.py` 确认 danger≈0.78 / target≈0.55 合理

## 6. 新增/更新测试

- `tests/test_depth_freshness.py`
- `tests/test_tello_emergency.py`
- `tests/test_main_depth_guard.py`
- `tests/test_orbit.py`（hold 中间带 + reset 清 depth_ts）
