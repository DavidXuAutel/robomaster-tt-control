# RoboMaster TT 视觉避障 — 真机测试现场 Checklist

日期：2026-07-20
适用：`main.py`（避障半自动）/ `fly_real_mission.py`（定点航线）
设计依据：[`../design/2026-07-17-tt-visual-avoidance-design.md`](../design/2026-07-17-tt-visual-avoidance-design.md)

> 原则：分层递进，前两层不飞。每层通过后再进下一层。首飞务必保守、开阔场地、有人守急停。

## 0. 出发前（不接硬件，公司网即可）

- [ ] 控制律离线自证：
  - `python sim_avoidance.py --scenario slalom --out logs/sim_slalom.mp4`
  - `python sim_avoidance.py --scenario wall`
  - 通过标准：`outcome=CLEARED`、`min_clearance>0`
- [ ] 单测全绿：`python -m pytest tests/ -q`
- [ ] 确认电池充满、备用电池、螺旋桨/桨叶保护圈完好
- [ ] 确认现场场地：开阔、软地面、无人、无强光直射镜头

## 1. 感知服务 + 真感知离线回放（需能连 GPU 机）

- [ ] GPU 机起服务（独立 venv）：
  - `pip install -r requirements-avoidance.txt`
  - `python server/da_v2_service.py --host 0.0.0.0 --port 8899`
- [ ] 健康检查：`curl http://<gpu-ip>:8899/health` 返回 `ok:true`
- [ ] 离线回放调参：
  - `python offline_avoidance.py --source clip.mp4 --service http://<gpu-ip>:8899/depth --out out.mp4 --show`
  - 通过标准：近度热力图与左/中/右分区合理；决策直方图符合预期
- [ ] 记录感知 RTT（mean / p50 / p95）：______ ms
- [ ] 据延迟定首飞参数（延迟高→调低 `cruise_speed`、调低 `stop_thresh` 更保守）

## 2. 真机

### 2.0 网络拓扑（最易卡，务必先通）

- [ ] 控制机能连飞机热点 `TELLO-xxxx`（本机进入 `192.168.10.x`）
- [ ] 控制机**同时**能访问 GPU 服务（建议双网卡：有线保 SSH + WiFi 连飞机）
- [ ] 起飞前先验证 `--inference depth-anything` 后端能真正连上服务（看 HUD 无 "waiting depth service"）

### 2.A 定点航线（不含避障，验证真机↔仿真匹配）

- [ ] 离线自检：`python fly_real_mission.py --sim`
- [ ] 真机：`python fly_real_mission.py --speed-cms 30`
- [ ] 起飞前电量 ≥ 25%（脚本已内置拦截）
- [ ] 降落后核对遥测 CSV / 轨迹 PNG 已生成于 `logs/`

### 2.B 视觉避障半自动

- [ ] 启动：`python main.py --inference depth-anything --depth-service http://<gpu-ip>:8899/depth --mujoco`
- [ ] 时序：`T` 起飞稳住 → `V` 到 `ARMED` → 再 `V` 到 `ON` 接管
- [ ] 全程有人手放键盘，随时可 `WASD` 接管 / `SPACE` 悬停 / `ESC` 急停 / `L` 降落
- [ ] 首飞先在单个障碍前验证"减速+偏航绕开"，再上多障碍

## 安全与急停（牢记）

| 操作 | 效果 |
|---|---|
| `WASD/QE/RF` | 立即接管，覆盖避障输出 |
| `SPACE` | 悬停并关闭 AUTO |
| `ESC` | 急停（停桨），关闭 AUTO |
| `L` | 降落，关闭 AUTO |

- AUTO 仅在**已起飞 + 深度后端在线**时可挂载。
- **AUTO 看门狗（已接入）**：`tt_control/auto_safety.py` 的 `AutoWatchdog` 已接入 `app.py` 控制循环——单次 AUTO 挂载超 30s、或感知失联超 1.5s（含挂载后迟迟无深度），会**自动悬停并解除 AUTO**，HUD 显示解除原因。首飞仍需盯 HUD 感知状态，人工接管为第一保障。

## 异常处置

- 图传卡死 / 感知长时间无返回 → `SPACE` 悬停，`L` 降落，排查网络。
- 位移指令返回非 `ok`（如 `error Not joystick`）→ `fly_real_mission.py` 已内置悬停重试；手动飞行则悬停后重发。
- 任一异常无法恢复 → `ESC` 急停兜底。

## 记录归档

- [ ] `logs/` 下日志 / 遥测 CSV / 轨迹 PNG / 标注视频已保存
- [ ] 现场参数与结论回填本 checklist 或 dev-notes
