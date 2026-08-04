# V0 — 4090 本地采集 runbook（depth + IMU schema v2）

**Date:** 2026-08-04 · **Branch:** `aerial-rl-skeleton` · **契约：**
[`docs/superpowers/specs/2026-08-04-aerial-wam-v2-frozen-spec.md`](../superpowers/specs/2026-08-04-aerial-wam-v2-frozen-spec.md) §2 / §6

> **本机 Mac 沙箱做不到这一步。** 采集必须在 **4090 渲染主机**上跑
> （collector → `127.0.0.1:41451` loopback）。H100 只收 `rsync` 过来的 npz。
> **禁止 scp 热补丁代码**——4090 上 `git checkout aerial-rl-skeleton`。

## 目标

产出 schema-v2 语料（每帧 RGB + DepthPlanar + IMU + timestamps），使 V0 四信号
（§4.1）可测。跨网 `dataset_v1_rgb` / `dataset_v1_depth@0.7Hz` **不算** V0 训练集。

## 1. V-1：Fork A + depth_rate（4090 loopback）

```bash
cd <repo>
git checkout aerial-rl-skeleton && git pull
# AirSim / env_bridge 已起，RPC 监听 127.0.0.1:41451
cd experiments/aerial/sim_verify
cp -n config.env.example config.env   # AIRSIM_HOST=127.0.0.1
./run_all.sh
```

**PASS：** `verdict` = **Fork A**，且 capability 含 `depth_rate (L2d-rate)=true`
（`fps ≥ L2F_DEPTH_MIN_FPS`，默认应 ≥ 目标采集 `step_hz`）。  
**FAIL（Fork A-/B）：** 停，不进采集。

## 2. 重测闭环 `step_hz`（loopback + `grab_depth=true`）

**实测（4090 loopback, 2026-08-04, commit `ea83f22`+）：**

| 探针 | 结果 |
|---|---|
| DepthPlanar  alone | ~96–101 ms（depth_rate ≈ **9.8 Hz**） |
| observe(RGB+depth+IMU) | ~120 ms |
| closed-loop ceiling (`step_hz=30`, depth) | **≈ 6.2 Hz** |
| commanded **5.0** + depth | achieved **4.99**（rate-lock OK） |
| commanded **6.0** + depth | achieved 5.6–6.0（贴天花板，偶发掉点） |
| RGB-only ceiling | ≈ 14 Hz |

将 `configs/aerial_rl.yaml` 的 `env.step_hz` 钉在 **5.0**（实测地板之下）。
`_refuse_v0` / `_v0_gate`：`grab_depth` 且 `step_hz>6.5` → 拒；另保留 `step_hz>8.5`
拒跨网 RGB 遗产。短探针示例：

```bash
$PYTHON_BIN -m experiments.aerial.rl.collect_dataset \
  --backend airsim --host 127.0.0.1 --port 41451 \
  --camera front_custom --vehicle drone_1 \
  --annotation "$ANNOTATION" \
  --episodes 2 --max-steps 100 --step-hz 5.0 \
  --grab-depth --out experiments/aerial/rl/artifacts/hz_probe_depth
```

若仍使用 rate-lock（async+sleep），墙钟应贴合 commanded `dt`；depth 观察预算
不够时先降 `step_hz`，不要抬高命令 Hz。（`observe_budget` 在 `grab_depth` 时为
150 ms，RGB-only 为 40 ms。）

## 3. 采集

```bash
# 4090
export ANNOTATION=/path/to/seen_airsim16_m1a20.json
export PYTHON_BIN=$(which python)   # 含 airsim + cv2 的 venv
export STEP_HZ_RGB=5.0
# 单档 depth 语料（V0 主集）：grab_depth=true，写本地盘
OUT=experiments/aerial/rl/artifacts/dataset_v0_local_depth
$PYTHON_BIN -m experiments.aerial.rl.collect_dataset \
  --backend airsim --host 127.0.0.1 --port 41451 \
  --camera front_custom --vehicle drone_1 \
  --annotation "$ANNOTATION" \
  --episodes 20 --max-steps 200 --step-hz "$STEP_HZ_RGB" \
  --grab-depth --out "$OUT"
```

**PASS 粗检：**

| 项 | 要求 |
|---|---|
| exit | 0；quarantine 比例 ≤ 20% |
| `manifest.meta.grab_depth` | true |
| `manifest.meta.step_hz` | = 命令值；achieved ≈ 命令（rate-lock） |
| npz keys | 含 `depth`、`imu_*`、`timestamps`、`vel`（schema v2） |
| `path_length_m.mean` / `rgb_frame_variation` | 非零 |

## 4. 同步到 H100（只传数据）

```bash
# 从 4090 → H100
rsync -avP experiments/aerial/rl/artifacts/dataset_v0_local_depth/ \
  a25689@10.239.121.21:<repo>/experiments/aerial/rl/artifacts/dataset_v0_local_depth/
```

代码侧 H100 用 git 同步同一 commit，不要 scp `.py`。

## 5. 下游（采集完成后）

按冻结 §6：**先 Step 3 DepthHead，再 Step 4 清洁 WM + `_v0_gate`**。不要跳过 DepthHead。

### 5.1 Step 3 — DepthHead [1b]（H100）

```bash
cd <repo>   # git checkout aerial-rl-skeleton（与 4090/Mac 同 commit）
export PYTHONPATH="$PWD"
export PYTHON_BIN=/home/a25689/aerial_wam_runtime/env/bin/python
DATA=experiments/aerial/rl/artifacts/dataset_v0_local_depth

$PYTHON_BIN -m experiments.aerial.rl.train_depth_head \
  --dataset "$DATA" --config configs/aerial_rl.yaml \
  --steps 2000 --wm-batch 8 --window 8 --device cuda --save-ckpt
```

产物：`experiments/aerial/rl/artifacts/depth_ckpt/depth_step_*.pt` + `depth_train.jsonl`。  
`world_model.depth_head.enable` **保持 false** 直到 `_v0_gate` 四信号全过。  
VIO [1c] 数学在 `vio.py`（③ 用 vel 积分）；学习 VIO 头可选，非本步硬门槛。

### 5.2 Step 4 — 清洁 WM + `_v0_gate`

1. 清洁 WM co-train（随机初始化，**不用** `wm_step_5000.pt`）：

```bash
$PYTHON_BIN -m experiments.aerial.rl._wm_train_validate \
  --dataset "$DATA" --config configs/aerial_rl.yaml \
  --steps 5000 --wm-batch 8 --window 8 --horizon 15 --device cuda --save-ckpt
```

2. `python -m experiments.aerial.rl._v0_gate --dataset ... --learning-log ...`
   — 四信号全过才允许改 yaml：`world_model.depth_head.enable` /
   `safety.kind: threshold`。

Collector 已接线：挂上 `DepthMinPredictor.from_checkpoint(...)` 时，在
`safety.should_override` **之前**写入 `obs.info["depth_min_pred"]`（④）。

## 回滚

默认 `configs/aerial_rl.yaml` 保持 V0 姿态（`dynamics.kind: stub`、
`enable_wm_update: false`、`safety.kind: null`）。评测 shield 只在 `_v0_gate`
进程内开关。
