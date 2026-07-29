---
title: Aerial B0→B1 编排与增训交接文档
type: handover
date: 2026-07-29
status: 训练中断待续（step≈2250/5000）
stamp: 20260727-072347-5k-2gpu-b0-to-joint-video
hosts:
  train: "10.239.121.22:31103 (2×H100)"
  eval: "10.239.121.22:30682 (AirSim)"
code:
  worktree: /Users/xudazhong/Projects/FastWAM/.worktrees/aerial-b0-b1-orchestration
  branch: feat/aerial-b0-b1-orchestration
  deployed_runtime: /home/a25689/aerial_wam_runtime/FastWAM
  deployed_ft_repo: /home/a25689/aerial_ft_cache/repo
related:
  - docs/design/2026-07-24-aerial-b0-failure-replay-finetune-design.md
  - artifacts/b1_seen20_metrics_20260727-072347.json
  - artifacts/b1_loss_history_through_2400.json
  - artifacts/loss_curve_b1_20260727-072347.png
---

# Aerial B0→B1 编排与增训交接文档

## 0. 一句话

B0 baseline（step_004000，seen-20 NE≈134）已锁定；B1 DAgger FT 已跑到 **step_002250** 后因 **correction-rate 硬门禁**失败停住；评测链路发现并修复了「缺 `dataset_stats.json` → 飞机原地转向」和「yaw 不 wrap → normalizer 被污染」两个工程缺陷。当前 **不得把早期 path_length=0 的评测数字当真**；带 stats 的真值见下文与 `artifacts/b1_seen20_metrics_*.json`。

---

## 1. 当前状态（2026-07-29 UTC ~01:43）

| 项 | 状态 |
|---|---|
| `ft.status` | **FAILED** |
| 最后落盘权重 | `step_002250.pt`（本地 + shared 均有） |
| 目标 | `max_steps=5000`，`save_every=250` |
| 失败点 | ~step 2400：`FT correction rate 0.1700 outside [0.2000, 0.3000]`（窗口 2201–2400） |
| GPU (:31103) | 空闲 |
| ckpt watcher | 仍在跑（`b1_discover`，steps=250..5000） |
| eval queue | pending/running/failed 空；B1 done 到 step_002250 |
| 评测 worker (:30682) | 队列空闲 |

**Stamp**：`20260727-072347-5k-2gpu-b0-to-joint-video`

**关键路径**

```
# 训练 run
/home/a25689/aerial_ft_cache/runs/b1-<STAMP>/
  checkpoints/weights/step_XXXXXX.pt
  dataset_stats.json
  config.yaml

# 共享评测权重（eval 读这里）
/home/a25689/aerial_cache_shared/runs/aerial_b1_ft/m1b-<STAMP>/checkpoints/weights/

# 评测结果
/home/a25689/aerial_cache_shared/orchestration/results/b1_<STAMP>/step_XXXXXX_seen20/metrics.json

# 编排状态
/home/a25689/aerial_cache/orchestration/status.json
/home/a25689/aerial_ft_cache/ft.status
/home/a25689/aerial_ft_cache/logs/ft/b1-5000-step.log
```

持久化注意：`/tmp/aerial_*` 已迁到 `/home/a25689/aerial_*`，旧路径保留 symlink。

---

## 2. 时间线（本轮工作）

1. **B0→B1 门禁通过**：baseline lock → DAgger 采集 → FT sync → smoke（含 NaN fix）→ `phase=RUN_B1_TRAIN`。
2. **B1 首轮 1000 步**完成；每 250 步落盘；ckpt watcher 同步入队 seen-20 评测。
3. **诊断「结果不理想」**：发现 19/20 episode `path_length=0` —— 根因是 mirror 缺 `dataset_stats.json`，评测 `processor=None`，动作为归一化尺度，nearest-primitive 几乎只出 stop/left30/right30。
4. **修复评测 mirror + 补 stats**，作废无效 metrics，重跑评测。
5. **发现 yaw 不 wrap**：`apply_body_delta` 累积 yaw 到 43 rad；与 DAgger 集 max yaw 完全一致；污染 B1 min/max normalizer。已在部署机打上 wrap 补丁。
6. **`/tmp` → `/home/a25689` 迁移**（:31103 与 :30682），脚本落盘到 worktree。
7. **增训至 5000**：从 `step_001000.pt` 权重续训（ZeRO-2 full state resume 因 optimizer param-group 不匹配失败）；trainer 从文件名恢复 `global_step`。
8. **停于 ~2400**：`ft_source_monitor` correction-rate 硬门禁触发 → `ft.status=FAILED`。

---

## 3. 已确认的具体问题

### 3.1 评测缺 normalizer（已修）

- `build_policy` 只在 `checkpoint.parent` / `parent.parent` 找 `dataset_stats.json`。
- `_mirror_checkpoint` 最初只镜像 `.pt` + `.sha256`。
- `processor=None` → 不反归一化 → primitive 映射几乎纯 yaw。
- **修复**：`b1_discover._mirror_dataset_stats`；缺文件则 enqueue 直接报错；已有权重旁已补上 B1 `dataset_stats.json`。

### 3.2 yaw 不 wrap（代码已修；数据未重生）

- `apply_body_delta`：`new_yaw = yaw + dyaw`（无 wrap）。
- 83 次 left30 → yaw≈**43.4587**（与 correction 集 max 一致）。
- 后果：B1 state yaw 的 min/max 被撑开，航向归一化有效分辨率骤降。
- **修复**：公开 `wrap_angle`，`apply_body_delta` 使用 wrap；测试 `test_apply_body_delta_wraps_yaw`。
- **未做**：按 wrap 后逻辑重采 DAgger / 重算 stats / 重训。

### 3.3 ZeRO-2 state resume 失败（已绕过）

- `accelerate.load_state(state/step_001000)` → `ValueError: loaded state dict has a different number of parameter groups`。
- **绕过**：权重-only resume `step_XXXXXX.pt` + trainer 从文件名设 `global_step`（optimizer/scheduler 重新初始化）。

### 3.4 correction-rate 硬门禁（当前阻塞）

```
RuntimeError: FT correction rate 0.1700 outside [0.2000, 0.3000] for steps 2201-2400
```

来源：`experiments/aerial/ft_source_monitor.py`。目标 mix 为 original:correction ≈ 0.75:0.25，200-step 窗口允许 [0.20, 0.30]。续训到 5000 必须放宽、关闭或修采样，否则还会再次炸掉。

### 3.5 代码树分叉

| 位置 | 说明 |
|---|---|
| 本地 worktree `feat/aerial-b0-b1-orchestration` | 开发与测试源 |
| `:31103` `/home/a25689/aerial_wam_runtime/FastWAM` | **实际 import 的 trainer / 多数 eval 补丁**（历史上是 master 快照） |
| `:31103` `/home/a25689/aerial_ft_cache/repo` | train 脚本与 orchestration 入口 |

部署时需两边同步；勿假设 worktree `git push` 等于远端已更新。

---

## 4. 评测结果（带 stats，可信）

Baseline（B0 step_004000）：**NE=133.99，SR=0，SPL=0**。S1 门槛 NE≈**107.19**（未达到）。

| B1 ckpt | NE | SR | SPL |
|---|---:|---:|---:|
| 250 | 327.41 | 0.00 | 0.00 |
| 500 | 214.12 | 0.00 | 0.00 |
| 750 | 199.06 | 0.00 | 0.00 |
| **1000** | **156.08** | **0.05** | **0.05** |
| 1250 | 327.41 | 0.00 | 0.00 |
| 1500 | 207.82 | 0.00 | 0.00 |
| 1750 | 239.71 | 0.00 | 0.00 |
| 2000 | 228.50 | 0.00 | 0.00 |
| 2250 | 321.64 | 0.00 | 0.00 |

解读：

- 相对 B0，B1 整体仍更差；step_1000 最接近且唯一出现 SR>0。
- 250 与 1250 的 NE 完全相同（327.4128）值得怀疑（偶然或链路残留），优先抽查这两个 metrics 的 episode 明细。
- **早期「B1 NE≈122 且 path_length=0」的数字已作废**，勿再引用。

本地落盘：`artifacts/b1_seen20_metrics_20260727-072347.json`。

---

## 5. 训练与 loss

- 首轮 1000 步：日志 `b1-1000-step.log`；曲线 `artifacts/loss_curve_b1_20260727-072347.png`。
- 续训到 5000：从 step_1000 权重继续；日志 `b1-5000-step.log`；最后记录约 **step=2350，loss≈1.60**。
- 合并 loss 点：`artifacts/b1_loss_history_through_2400.json`（47 个点，latest=2350）。

配方（`orch_b1_train.sh`）：

```
task=aerial_joint_b0_ft_dagger
resume=<latest step_*.pt>
max_steps=5000
save_every=250
learning_rate=1e-5
lambda_video=0.0
lambda_action=1.0
eval_every=0
2×GPU DeepSpeed ZeRO-2 bf16
```

---

## 6. 代码 / 脚本变更清单（本地 worktree，多数未 commit）

### 已改（关键）

| 文件 | 变更 |
|---|---|
| `experiments/aerial/orchestration/b1_discover.py` | mirror ckpt + `dataset_stats.json`；`B1_STEPS=250..5000` |
| `experiments/aerial/scripts/orch_b1_train.sh` | 5000 步；权重 resume；路径默认 `/home/a25689` |
| `experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh` | steps 250..5000；shared-weights |
| `experiments/aerial/scripts/orch_b1_progress.sh` | **新增**进度汇总 |
| `experiments/aerial/scripts/orch_s1_report.sh` | candidate 1000..5000 |
| `experiments/aerial/eval/run_closed_loop.py` | yaw wrap |
| `experiments/aerial/openfly_actions.py` | `wrap_angle` 公开 |
| `src/fastwam/trainer.py` | `.pt` resume 从文件名恢复 `global_step` |
| `experiments/aerial/scripts/migrate_aerial_off_tmp_{31103,30682}.sh` | **新增**迁移脚本 |
| `experiments/aerial/scripts/sync_b0_ft_to_h100.sh` 等 | 默认路径离开 `/tmp` |
| 对应 tests | discover / train scripts / yaw wrap |

### 远端已热补但需与 worktree 对齐

- `:31103` / `:30682` runtime 上的 `run_closed_loop.py`、`openfly_actions.py`、`trainer.py`、`b1_discover.py`、`orch_b1_train.sh` 等曾直接 scp/stdin 部署。

**未 commit**：FastWAM worktree 仍有大量 `M`/`??`；本交接文档与 artifacts 在 `robomaster-tt-control`。

---

## 7. 运维速查

### 看进度

```bash
# :31103
export PATH=/home/a25689/aerial_wam_runtime/env/bin:$PATH
export AERIAL_FT_CACHE=/home/a25689/aerial_ft_cache
cd /home/a25689/aerial_ft_cache/repo
bash experiments/aerial/scripts/orch_b1_progress.sh
```

### 从 step_002250 续训到 5000（建议先处理 correction gate）

```bash
# :31103
export PATH=/home/a25689/aerial_wam_runtime/env/bin:/usr/bin:/bin
export AERIAL_FT_CACHE=/home/a25689/aerial_ft_cache
export STATUS_PATH=/home/a25689/aerial_cache/orchestration/status.json
export LOCK_PATH=/home/a25689/aerial_cache/orchestration/baseline_lock.manifest.json
export REPO_DIR=/home/a25689/aerial_ft_cache/repo
export SKIP_MANIFEST_VERIFY=1
# 确保 status.json: phase=RUN_B1_TRAIN, gates_passed=true
cd $REPO_DIR
# resolve_resume 会自动选最新 step_*.pt（当前应为 2250）
setsid -f bash experiments/aerial/scripts/orch_b1_train.sh \
  >$AERIAL_FT_CACHE/logs/ft/b1-5000-step.launcher.log 2>&1 < /dev/null
```

续训前必须三选一：

1. 放宽 `ft_source_monitor` 窗口阈值；或  
2. 关闭 / 跳过该 monitor；或  
3. 修好加权采样，使 correction 稳定落在 [0.20, 0.30]。

### 评测

- watcher：`orch_ckpt_watch_enqueue.sh`（已含 shared mirror + stats）。
- worker：`:30682` 上 `orch_eval_worker.sh`（注意迁移后日志勿再写到已删 `/tmp` inode；队列空闲后应重启 worker 并改日志路径）。

---

## 8. 建议的下一步（按优先级）

1. **修/放宽 correction-rate 门禁**，从 `step_002250` 续训到 5000。  
2. **抽查 step_250 vs step_1250** 评测是否异常重复。  
3. **决定是否重采 DAgger**（yaw wrap 后）并重算 stats / 重训 —— 否则航向信息仍受损。  
4. **把 worktree 变更 commit / 同步**到 runtime + ft_cache repo，消除热补分叉。  
5. 队列空闲后 **重启 eval worker**，修复日志落到持久路径。  
6. 5000 步完成后跑 `orch_s1_report.sh`，与 B0 NE=134 / S1=107.19 对比选点。

---

## 9. 本地落盘索引

| 路径 | 内容 |
|---|---|
| `docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md` | 本文档 |
| `artifacts/b1_seen20_metrics_20260727-072347.json` | 全量 B1 seen-20 汇总 + B0 baseline |
| `artifacts/b1_loss_history_through_2400.json` | loss 点到 step 2350 |
| `artifacts/loss_curve_b1_20260727-072347.png` | 首轮 1000 步曲线 |
| `artifacts/loss_history_b1_20260727-072347.json` | 首轮 1000 步 history |
| FastWAM worktree（未 commit） | 编排 / yaw wrap / trainer resume / 迁移脚本 |

---

## 10. 一句话结论

工程链路（持久化路径、ckpt mirror+stats、yaw wrap、5000 步续训机制、进度脚本）已基本打通并落盘；**模型侧 B1 尚未超过 B0**，且训练被 **correction-rate 门禁**截在 2250。下一步是放宽门禁续训到 5000，并单独评估是否重采矫正集以修复 yaw 污染。
