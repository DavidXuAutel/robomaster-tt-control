# 差异摘要：远端服务器 → 本机整理提交（2026-07-20）

> 相对 GitHub `main`（`ae404f1`）的增量说明。  
> 来源：`yao@10.229.20.125:~/Projects/robomaster-tt-control`（rsync 拉取后整理入库）。

---

## 一句话

在原有「图传 + 键盘 + MuJoCo 轨迹」基础上，合并了 **手势控制（MediaPipe + DTW）**、**深度避障**、**离线仿真** 三条路径，并补齐测试与交接文档；备份/旧手势原型不入库。

---

## 相对 `ae404f1` 的主要差异

### 1. 新增能力

| 能力 | 关键入口 | 说明 |
|------|----------|------|
| 手势飞控 | `--inference gestures` | 张开手掌上抬起飞；响指/捏合弹开降落；可录个人 profile |
| 手势干跑 / 真机测试 | `--gesture-dry-run` / `--gesture-flight-test` | 只识别不飞；或需手动 ARM 的测试流 |
| 深度避障 | `--inference depth-anything` + 按键 `V` | OFF→ARMED→ON；连 GPU 深度服务 |
| 离线仿真 | `--sim` | `SimDrone` 替代真机，无需飞机 |
| 任务/诊断脚本 | `fly_real_mission.py` 等 | 真机任务、避障仿真、轨迹回放、Tello 诊断 |
| 回归测试 | `tests/` | 服务器验证 **44 passed** |

### 2. 核心改动文件

| 文件 | 变更性质 |
|------|----------|
| `tt_control/app.py` | 大幅扩展：手势事件流 + 避障 AUTO + `--sim` 三路径共存 |
| `tt_control/inference.py` | `InferenceEvent` / `drain_events`；注册 `gestures`、`depth-anything` |
| `tt_control/config.py` / `control.py` | 手势/仿真配置；`V`=避障切换 |
| `tt_control/mujoco_twin.py` | 无头记录 + 轨迹采样锚点修复 |
| `main.py` / `auto_fly.py` / `station_mode.py` | CLI 与组网入口对齐手势/仿真 |
| `requirements.txt` | 增加 `mediapipe` |
| `README.md` | 手势 / 避障 / 仿真用法 |

### 3. 新增模块（入库）

```
tt_control/gesture_control.py
tt_control/gesture_profile.py
tt_control/flight_test.py
tt_control/assets/gesture_recognizer.task   # ~8MB MediaPipe 模型
tt_control/avoidance.py
tt_control/depth_backend.py
tt_control/policy.py
tt_control/sim_drone.py
tt_control/sim_runner.py
tt_control/trajectory_plot.py
server/da_v2_service.py
diag_tello.py / offline_avoidance.py / sim_*.py / fly_real_mission.py
tests/*（12 个测试文件）
docs/2026-07-18*.md / docs/2026-07-19*.md / docs/2026-07-20*.md
requirements-avoidance.txt
```

### 4. 明确不入库

| 路径 | 原因 |
|------|------|
| `*.bak-*` / `*.orig-*` | 服务器覆盖前备份 |
| `_old_gesture_20260720/` | 已归档的旧 MediaPipe Hands 手势原型 |
| `.merge-backup-mine/` | 合并过程临时备份 |
| `wifi_config.json` / `gesture_profiles/` / `logs/` | 本地密钥与运行产物（已在 `.gitignore`） |

---

## 与 GitHub / 服务器关系

| 位置 | 状态 |
|------|------|
| GitHub `main`（整理前提交前） | `ae404f1`（组网脚本 + MuJoCo 轨迹） |
| 服务器工作树 | 已含手势权威版 + 避障/仿真整合，pytest 44 绿 |
| 本机整理后提交 | 将服务器工作树的「可维护源码」写入 git（不含备份） |

---

## 快速验证

```bash
cd ~/Projects/robomaster-tt-control
python -m pytest tests/ -q

# 手势干跑（本机摄像头）
python -m tt_control.gesture_control

# 统一界面 + 手势
python main.py --inference gestures --gesture-dry-run -v

# 离线仿真
python main.py --sim -v
```

---

## 后续建议

1. `git push github main`（及内网 bare，若 SSH 可用）让远端与服务器能力对齐。  
2. 服务器目录改为 git clone / pull，减少手工 rsync 漂移。  
3. 大模型 `gesture_recognizer.task` 若仓库过大，可考虑 Git LFS。
