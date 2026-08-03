# Phase-1 sim capability on 10.229.20.125 (2026-08-03)

## Host
- `yao@10.229.20.125` RTX 4090 D, AirSim **Multirotor** (`drone_1`), RPC `127.0.0.1:41451`
- Scene：`/home/yao/aerial_airsim_persistent/scene/env_airsim_16`（OpenFly 导出 AirVLN/UE，`recover_renderer.sh` 真起）

## Spec 门禁（不变）

- **`L2F_MIN_FPS = 5.0` 保持为目标硬门禁**。不因本机实测下调。

## Phase 1 首次探测（非独占 / stub 路径）

| 项 | 结果 |
|----|------|
| L1 / L2a–e | PASS |
| L2f 顺序抓帧 @ 1920×1080 | 实测 **~2.6 fps** |
| 默认门禁 `L2F_MIN_FPS=5` | **Fork A⁻** |

该 ~2.6 曾标 **待干净复测**（可能非独占 / 非批量）。

## 干净独占 L2f 复测 @ 1920×1080（已完成）

条件：停干净 AirVLN → `recover_renderer.sh` 真起 `env_airsim_16` → 无其它 client → `probes/l2f_clean_retest.py`。

报告：`artifacts/l2f_clean_retest.json`

| 测量 | 结果 |
|------|------|
| 静止 IMU `\|lin_acc\|` | **9.807 ± 0.045** → **≈9.8 m/s²** ✅（单位已定死） |
| 顺序抓帧 fps | **~3.12**（仍 **&lt; 5**） |
| 批量 `simGetImages([req]*N)` fps | **~0.72**（更差；批量不能抬帧率） |
| 相对 `L2F_MIN_FPS=5` | **仍不达标** → 1080p 下 **A⁻ 瓶颈在 L2f** |

## WAM 工作分辨率复测 @ 224×224（已完成）

训练/评测默认 `IMAGE_SIZE_WH = 224×224`（非 512×288）。

步骤：`patch_capture_res.py --w 224 --h 224` 必须同时改 **Documents/AirSim + persistent +
`…/AirVLN/Binaries/Linux/settings.json`**（后者是本机真正生效的路径），重启 renderer，再
`L2F_W=224 L2F_H=224 L2F_MIN_FPS=5` 跑 `l2f_clean_retest.py`。

报告：`artifacts/l2f_clean_retest_224.json`

| 测量 | 结果 |
|------|------|
| `response.width×height` | **224×224**（res_match ✅） |
| 顺序抓帧 fps | **~36.3**（**≥ 5**） |
| 批量 fps | **~90.3** |
| IMU 静止 | **~9.84 m/s²** ✅ |
| 相对 `L2F_MIN_FPS=5` | **达标** → **`fork_a_at_working_res: true`** |

### 结论（定稿）

1. **惯性/高度/深度/物理：可用**（Multirotor 真信号；静止重力正常）。
2. **连续 RGB @ 1080p**：干净独占上限约 **~3 fps**；不是 stub 假象，也不是批量能救的。
3. **连续 RGB @ WAM 工作分辨率 224×224**：顺序 **~36 fps ≫ 5** → **L2f 门禁在模型真实工作分辨率下满足**。
4. **正式记为 Fork A（工作分辨率）**：吞吐不再是 Phase-2 / pure-vision WAM 的硬阻塞；1080p ~3 fps 仅作可视化/高清导出约束。
5. Spec **`L2F_MIN_FPS=5` 仍为目标**；Phase-1 在该门禁 + 224 工作分辨率下记 **Fork A**。

## 产物

| 文件 | 含义 |
|------|------|
| `phase1_125_sim_capability_report_default_l2f5_Aminus.json` | 首次默认门禁（1080p）Fork A⁻ |
| `phase1_125_sim_capability_report.json` | 曾用 `L2F_MIN_FPS=2.5` 的对照（非正式） |
| `l2f_clean_retest.json` | 干净独占复测 @1080p（IMU≈9.8 + ~3.1 seq fps） |
| `l2f_clean_retest_224.json` | 工作分辨率复测 @224（seq ~36 fps → Fork A） |
