# 手势控制上云同步 · 执行报告（2026-07-20）

**服务器**：`yao@10.229.20.125:/home/yao/Projects/robomaster-tt-control`（非 git，手工拷贝）
**状态**：✅ 已完成同步，服务器回归 **44 passed**（与本地一致）

## 1. 做了什么

把**云端 git（`ouxuedong/auto-fly`）的权威手势控制**下发到服务器，替换服务器上原先那套老手势原型（MediaPipe Hands 规则版），补齐避障/仿真的少量缺失文件。避障/仿真主体两边本就完全一致，未改动。

服务器现在跑的手势 = git 版 `gesture_control.py`（MediaPipe Gesture Recognizer + 动态轨迹 + DTW + 可录个人 profile）；老 `gesture.py` 已归档下线。

## 2. 实际变更（已执行）

**新增（12）**
- `tt_control/gesture_control.py`、`gesture_profile.py`、`flight_test.py`
- `tt_control/assets/gesture_recognizer.task`（8MB 模型）
- `tests/test_gesture_control.py`、`test_gesture_profile.py`、`test_app_gesture_events.py`、`test_flight_test.py`
- `sim_mission.py`、`tests/test_sim_mission.py`（服务器原先缺）
- 文档：`docs/2026-07-20-changes-and-sync-for-dazhong.md`、`2026-07-20-gesture-control-handover.md`（本报告 `2026-07-20-sync-execution-report.md`）

**覆盖（10，写入前已备份为 `*.bak-20260720`）**
- `main.py`、`tt_control/app.py`、`config.py`、`inference.py`、`sim_drone.py`
- `auto_fly.py`、`station_mode.py`、`requirements.txt`、`.gitignore`、`README.md`

**归档下线（移入 `_old_gesture_20260720/`）**
- 老手势 `tt_control/gesture.py`（+ `gesture.py.orig-20260719`）、`assets/hand_landmarker.task`
- 调参脚本 `gesture_capture.py`/`gesture_dryrun.py`/`gesture_fire_shots.py`/`gesture_shots.py`/`gesture_shots_both.py`/`landing_shots.py`/`sim_gesture_flight.py`
- 样本目录 `shots/`、`shots_fire/`、`shots_land/`

**环境修复（1）**
- 服务器 `.venv` 补装了缺失的 `typing_extensions`（mujoco 的传递依赖，**同步前就缺**，与本次代码无关）。补装后 mujoco 无头孪生恢复，`test_integration_sim` / `test_trajectory` 两项转绿。

**未触碰**：`logs/`、`wifi_config.json`、既有 `*.orig*` 备份、`.merge-backup-mine/`、服务器独有文档 `docs/2026-07-19-realdrone-verify-gesture.md`。

## 3. 验证结果

上传后内容级比对：原本差异的 10 个文件全部变为 `[SAME]`，新增文件到位。

```
$ cd ~/Projects/robomaster-tt-control && .venv/bin/python -m pytest tests/ -q
............................................  [100%]
44 passed in 59.92s
```

（补 typing_extensions 前为 40 passed / 1 skipped / 2 failed，失败均为 mujoco 缺依赖；补装后全绿。）

## 4. 大众怎么接手（速查）

```bash
cd ~/Projects/robomaster-tt-control && source .venv/bin/activate

# 本机摄像头离线自测手势识别（不接飞机）
python -m tt_control.gesture_control
# 真机 + 手势
python main.py --tello-ip <飞机IP> --local-ip <本机IP> --inference gestures -v
# 现场标定，只识别不飞
python auto_fly.py --gesture-dry-run
# 需手动 ARM 的真机测试
python auto_fly.py --gesture-flight-test
```

- **手势**：张开手掌上抬=起飞；拇指中指捏合再快速弹开（响指）=降落。
- **安全门控**都在 `App._handle_inference_event`（连接成功 + 电量≥30% + 地面才起飞；离地才降落；有冷却；`L`/`LAND`/`Esc` 永远是人工备份）。
- **接你自己的模型**：实现 `InferenceBackend`，`drain_events()` 吐 `InferenceEvent(kind, confidence, detail)`，在 `create_backend()` 注册名字即可 `--inference 你的名字`；后端只管识别报事件，飞控安全判断全在 App。
- **录个人手势**：面板 `TRAIN TAKEOFF/LAND/NONE` 各 10 次 → `SAVE PROFILE`，存 `gesture_profiles/`（不入库）。

## 5. 回滚办法

- 被覆盖文件都有 `*.bak-20260720`，`mv` 回去即可。
- 老手势整套在 `_old_gesture_20260720/`，需要时可整体恢复。
- 全程未动 `logs/` / `wifi_config.json`；`.venv` 仅新增一个 `typing_extensions`。

## 6. 待确认 / 后续

- 本地 git：手势 4 提交已并入本地 `main`（`0d4542a`）；避障/仿真仍为本地工作区未提交改动。**GitHub 远端 `origin/main` 未改动**。是否需要把这批整理成提交并 push，请示下。
- 服务器为手工拷贝、非 git；如需长期可维护，建议后续在服务器上也纳入 git 跟踪。
