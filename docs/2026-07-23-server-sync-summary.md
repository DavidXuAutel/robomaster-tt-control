# 服务器同步摘要（2026-07-23）

> 来源：`yao@10.229.20.125:~/Projects/robomaster-tt-control` → 本机 git → GitHub `main`。

## 相对上一提交 `0cf12de` 的实质增量

1. **Episode 录制**（`--record` / `--record-hz`）  
   - `tt_control/episode_recorder.py`：起飞后同步记录 RGB + 深度 + 动作 + 状态到 `logs/episodes/`  
   - `App` 集成：起飞开录、降落/清理结案

2. **避障安全加固**  
   - `tt_control/auto_safety.py`：`AutoWatchdog`（感知失联 / 超时等自动解除 AUTO）  
   - `AvoidParams` 可调：`--cruise` / `--approach-pitch` / `--yaw`  
   - `avoid_preview.py`：悬停观测或 `--engage` 短时接管

3. **标定与匹配**  
   - `calibrate_flight.py` / `calib_vmax.py`：rc 杆量标定飞行  
   - `sim_match_report.py`：真机轨迹 vs SimDrone 运动学对比

4. **演示脚本**  
   - `auto_wall_demo.py`：保守「前进至墙/障碍即停」自动演示（含 `--dry-run`）

5. **测试**  
   - 新增 `tests/test_auto_safety.py`、`test_episode_recorder.py`、`test_sim_match_report.py`  
   - `requirements-sim.txt` 补齐仿真依赖声明

6. **工程整理**  
   - 远端 CRLF 统一为 LF；恢复 bak/old 目录的 `.gitignore` 规则  
   - 不入库：`*.bak-*`、`_old_gesture_*`、`.merge-backup-*`、`wifi_config.json`、`logs/`

## 未改动的主轴

手势控制、深度避障主循环、MuJoCo 孪生、组网脚本仍在；本次主要是**录制 / 看门狗 / 标定 / 预览演示**层。
