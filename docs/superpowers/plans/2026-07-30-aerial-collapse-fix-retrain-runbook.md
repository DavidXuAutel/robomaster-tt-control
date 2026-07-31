# Aerial B0 v2 动作塌缩修复 — 重训操作手册（实施方案）

> **致执行者（人或 Agent）：** 这是承接
> `plans/2026-07-30-aerial-b0-v2-collapse-fix.md`（设计蓝本）的**可执行重训方案**。
> 塌缩修复的*代码*（方案 B 的 `head_cls`、CE 损失、转换阶段 stop relabel、
> Stage-1 数据配置）**均已实现并通过单测**；本文是把这些代码变成一个「能训练、
> 会终止」策略的**主机侧执行序列**。各步用勾选框（`- [ ]`）跟踪。

**目标：** 用方案 B 的分类头 + stop 监督重训 b0_v2,使策略真正会**终止**
（根因 #1),并摆脱**目标盲**（根因 #2),随后以「学习到的 stop」做评测。

**蓝本：** `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md`
**规格：** `docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-v3.2.md`
**启动器：** `experiments/aerial/scripts/run_collapse_fix_retrain.sh`
**主机：** 训练 `:31126`（2×H100）,评测 `:30905`（OpenFly+AirSim）。

---

## 前置条件与不变量（三点 —— 先读这里）

以下三点是承重的。任一处错误都会复现 b0_v2 的 SR=0 失败。

1. **opt-in / 默认 no-op。** 塌缩修复的改动**不改变当前训练行为**。
   `configs/data/aerial_openfly.yaml`、`configs/task/aerial_joint_1cam_1e-4.yaml`
   以及原始的 `./data/openfly_lerobot/train_subset` 均**未被触碰**。
   `convert_openfly_to_lerobot.py` 的 `stop_relabel_radius` 默认 `None`（原始
   delta,输出逐比特一致 —— 由 `test_stop_relabel_none_is_noop` 锁定）,
   `loss_lambda_ce` 默认 0。**不运行本启动器,下面任何东西都不会触发。**

2. **训练分布/窗口的改变来自两个相互独立的开关：**
   - **(a) 数据：** 带 `--stop-relabel-radius 20` 重新转换 → 以零 body-delta
     注入 `stop`（原语 0）标签。
   - **(b) 配方/窗口：** `task=aerial_joint_collapse_fix` → `num_frames=9`、
     `action_video_freq_ratio=2`、`skip_padding_as_possible=true`,以及
     `λ_video=1.0 / λ_fm=0.1 / λ_ce=1.0` 配合 `enable_action_cls=true`。

   两者正交。**只翻 (b)** 会改变窗口,但 stop 标签仍停在 ~2/2709（~0.07%）的
   地板值 → CE 头永远学不会 stop → 策略仍然永不终止。**两者缺一不可。**

3. **重训前提：** `reconvert` **必须先于** `train`。没有 relabel 后的数据集,
   `labels.relabel_stop_on_trajectory` 的在线路径仍未接线（训练 `sample` 里
   缺逐帧位姿/goal),stop 监督就停在 ≈2/2709,策略会复现 b0_v2 的永不终止。

**归一化器说明（不变量 #2a 的后果）：** 重新转换会在新的「零占多数」的动作分布
上**重新计算** `FastWAMProcessor` 的 min/max 归一化(大量 `[0,0,0,0]` 行会压缩
动作 min/max 跨度)。因此 relabel 子集的归一化器与旧 b0_v2 checkpoint **不通用**。
本次是从头重训,故此为预期且无害 —— 但**切勿**拿旧 ckpt 去跑 relabel 数据的
闭环评测,否则反归一化出来的动作是错的。

---

## 文件映射

| 路径 | 在重训中的角色 |
|------|-----------------|
| `experiments/aerial/scripts/run_collapse_fix_retrain.sh` | reconvert → preflight → smoke → train 启动器（本方案的驱动） |
| `experiments/aerial/convert_openfly_to_lerobot.py` | `--stop-relabel-radius` stop relabel（不变量 #2a / #3） |
| `configs/task/aerial_joint_collapse_fix.yaml` | λ 配方 + `override /data: aerial_openfly_collapse_fix` |
| `configs/data/aerial_openfly_collapse_fix.yaml` | `num_frames=9`、`ratio=2`、`skip_padding`（不变量 #2b） |
| `src/fastwam/models/wan22/action_dit.py` | `head_cls` + `classify_from_tokens`（经 override 启用） |
| `src/fastwam/models/wan22/fastwam.py` | CE 分支（`loss_lambda_ce>0` + `head_cls`） |
| `experiments/aerial/eval/policy_fastwam.py` | 闭环在有 `primitive` 时用 `argmax(cls)` |
| `experiments/aerial/collapse_fix/compute_dmax.py` | 仅供参考的 d_max（见 §d_max） |

---

### 任务 1：重生 relabel 数据集（主机 `:31126`）

非破坏性：写入 `train_subset_stop20`,原始子集保持不变。

```bash
OPENFLY_ANN=<path/to/annotation.json> \
OPENFLY_IMAGE_ROOT=<path/to/images> \
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh reconvert
```

- [x] **第 1 步：** 运行 `reconvert`;确认 `./data/openfly_lerobot/train_subset_stop20` 已生成。
- [x] **第 2 步：** 校验:原始 `train_subset` 逐字节未变（不变量 #1）。
- [x] **第 3 步：**（可选）抽查几条 episode —— 末帧 + 近 goal 帧应带 `action==[0,0,0,0]`。

> `R=20.0` 对齐 `OPENFLY_SUCCESS_DIST_M`（`experiments/aerial/eval/metrics.py`）。
> 若评测阈值变动,用 `STOP_RADIUS=<m>` 覆盖。目标目录已存在时,用 `FORCE=1`
> 覆盖重生。

---

### 任务 2：preflight + smoke（主机 `:31126`）

```bash
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh preflight
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh smoke
```

- [x] **第 1 步：** preflight 通过：relabel 数据在位、2 块 GPU、`resume=null`、检测到 `head_cls` 接线、`RESUME`/`AERIAL_ALLOW_LEGACY_RESUME` 为空。
- [x] **第 2 步：** smoke（`SMOKE_STEPS=10`）打印**有限**的 `loss_video`、`loss_action`（fm）以及 **`loss_ce`** —— 出现非平凡的 `loss_ce` 就是「CE 头确实接上并在接收 stop 标签」的信号。
- [x] **第 3 步：** 无 NaN;峰值显存 < 90%。记录 steps/s → 据此设 `MAX_STEPS`。
  （`:31126` 实测 ~0.09 step/s；`loss_ce` 5.97→0.02；峰值显存 ~70.5/81.6 GiB≈86.5%；本机 Wan，`redirect_common_files=false` + `+enable_action_cls`。建议 Task3：`MAX_STEPS=1000` 或 `1500`，`SAVE_EVERY=500`。）

> 启动器在 task 配置之上**显式**传入配方（`enable_action_cls=true`、
> `λ_video/λ_fm/λ_ce`,以及 `data.train.dataset_dirs=[…train_subset_stop20]`）,
> 因此整条 override 链在日志里自解释。

---

### 任务 3：完整重训（主机 `:31126`）

```bash
MAX_STEPS=<来自 smoke> SAVE_EVERY=500 \
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh train
```

- [ ] **第 1 步：** 启动完整训练;checkpoint 落在 `checkpoints/weights/step_*.pt`。
- [ ] **第 2 步：** 观察 `loss_ce` 下降、预测原语直方图从「只有前进」的塌缩里铺开（用 `plot_loss.py` 记录/绘图）。
- [ ] **第 3 步：** 目标 ≤5 个 epoch 等效量（当前子集上）;数据扩充留到后续阶段。

---

### 任务 4：以「学习到的 stop」评测（主机 `:30905`）

- [ ] **第 1 步：** 用 `ORACLE_STOP=0`（经 `argmax(cls)` 的学习 stop）+ `closest_approach` 诊断入队各 ckpt。
- [ ] **第 2 步：** 将 SR / NE / SPL 与 b0_v2（SR=0）及 Stage-0 oracle-stop 天花板（~10%）对比。
- [ ] **第 3 步：** 锁定基线前,过 v3.2 §1 的硬成功标准。

---

## 并行评测（可选 —— 与任务 3 重叠以省整体墙钟）

**任务 3 与任务 4 默认是串行的**（先在 `:31126` 训完，再逐 ckpt 到 `:30905`
评测）。但两者跑在不同主机上，且评测消费的是 checkpoint 文件，所以可以让评测
在训练还在跑时就「边出边评」——用现有 B1 编排工具即可。

### 可行性约束（先读 —— 先前草案的漏洞）

1. **`:30905` 必须看得见 weights。** 训练默认写 `./runs/train/<hydra-stamp>/`，评测机
   看不到。必须用 `OUTPUT_DIR` 指到双方都能访问的路径（本机 `/home/a25689` 整盘是
   Ceph，用 `aerial_cache_shared/runs/aerial_collapse_fix/...` 即可）。
2. **`STEPS` 必须对齐本次 `MAX_STEPS`/`SAVE_EVERY`。** 例：`MAX_STEPS=1500`、
   `SAVE_EVERY=500` → `STEPS=500,1000,1500`。SOP 里若写到 `5000` 会空等永不出现的
   ckpt（watcher 本身不退出，但浪费且误导）。
3. **专用 `EVAL_QUEUE_DIR`。** 默认队列常残留旧 B0/B1 job；worker 会先认领它们，
   污染 AirSim / 结果。collapse-fix 用独立队列，例如
   `.../orchestration/eval_queue_collapse_fix`。
4. **评测必须启用 `head_cls`。** 仅 `task=aerial_joint_collapse_fix` 不够时，
   旧版会静默退回 continuous→nearest-primitive，**测不到「学习到的 stop」**。
   现已：task yaml 含 `enable_action_cls: true`，且 `run_closed_loop` 对
   `collapse_fix` task 追加 `+enable_action_cls`。
5. **`dataset_stats.json` 必须在 ckpt 旁**（trainer 写在 `output_dir/`；discover
   会 mirror 到 shared weights）。缺了反归一化错，动作塌成 yaw-only。
6. **`ORACLE_STOP=0`（默认）** 才是任务 4 口径；显式写出防 shell 残留 `=1`。

### 推荐并行命令

```bash
# ---- :31126 训练（Task 3）----
STAMP=20260731-collapse-fix-1500
OUTPUT_DIR=/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}
MAX_STEPS=1500 SAVE_EVERY=500 OUTPUT_DIR="$OUTPUT_DIR" \
  bash experiments/aerial/scripts/run_collapse_fix_retrain.sh train

# ---- :30905 watcher（Task 4 入队）----
STAMP=20260731-collapse-fix-1500
WEIGHTS_DIR=/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}/checkpoints/weights
STAMP=$STAMP \
TASK=aerial_joint_collapse_fix \
WEIGHTS_DIR=$WEIGHTS_DIR \
SHARED_WEIGHTS_DIR=$WEIGHTS_DIR \
STEPS="$(seq -s, 500 500 1500)" \
EVAL_QUEUE_DIR=/home/a25689/aerial_cache_shared/orchestration/eval_queue_collapse_fix \
ANN=/home/a25689/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json \
OPENFLY_ROOT=/home/a25689/aerial_eval_cache/OpenFly-Platform \
bash experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh --dry-run   # 先核对
# 去掉 --dry-run 后常驻

# ---- :30905 worker（认领 + 学习 stop 评测）----
ORACLE_STOP=0 \
EVAL_QUEUE_DIR=/home/a25689/aerial_cache_shared/orchestration/eval_queue_collapse_fix \
bash experiments/aerial/scripts/orch_eval_worker.sh
```

- [ ] **第 1 步：** `OUTPUT_DIR`/`WEIGHTS_DIR` 指向本次训练的 shared checkpoint 树
  （**不是** B1 默认 `b1-${STAMP}`）。
- [ ] **第 2 步：** `STEPS` 与 `SAVE_EVERY`/`MAX_STEPS` 对齐（本 run：500/1000/1500）。
- [ ] **第 3 步：** 独立 `EVAL_QUEUE_DIR`；worker `ORACLE_STOP=0`。
- [ ] **第 4 步：** 先 `--dry-run` 核对 steps / weights / queue，再正式起 watcher+worker。

### 本节涉及的脚本与代码改动

| 脚本 / 代码 | 是否需改 | 说明 |
|-------------|----------|------|
| `run_collapse_fix_retrain.sh` | **已改** | `OUTPUT_DIR` + 本机 Wan（`redirect_common_files=false`）+ `+enable_action_cls`。 |
| `orch_ckpt_watch_enqueue.sh` | **已改** | `STEPS` 可覆盖；collapse-fix 对齐 `SAVE_EVERY`。 |
| `orch_eval_worker.sh` | **无需改** | 认领任意队列；靠独立 `EVAL_QUEUE_DIR` 隔离。 |
| `aerial_joint_collapse_fix.yaml` | **已改** | `model.action_dit_config.enable_action_cls: true`（训/评同源）。 |
| `run_closed_loop.py` | **已改** | `collapse_fix` task 自动 `+enable_action_cls`。 |
| `b1_discover.py` / `eval_queue.py` | **无需改** | 全参数驱动；结果目录仍叫 `b1_<stamp>`（纯命名）。 |

> **命名提示：** 结果落在 `results/b1_<stamp>/...`、job kind=`b1` 只是历史命名。

---

## d_max（已知限制 —— 不阻塞 retrain-1）

`compute_dmax.py` 写出 `artifacts/collapse_fix_dmax.json`（`d_max_p90_forward`）,
`fastwam.py:593` 经 `getattr(self, "action_cls_d_max", 1e9)` 读取该阈值。
**但 `action_cls_d_max` 并未接进 `create_fastwam_joint`**,所以
`model.action_cls_d_max=<val>` 这个 override 目前**不起作用**。对 retrain-1 而言,
这意味着前向类的 d_max 过滤是**关闭**的（阈值 1e9）;少数类（stop + 转向 + 升降）
无论如何都豁免,因此 stop 监督不受影响。`preflight` 里跑 `compute_dmax` **仅供参考**。

- [ ] **可选后续（需主机验证）：** 给 `create_fastwam_joint` 加一个
  `action_cls_d_max` 参数 → 设为 `FastWAMJoint` 的属性,然后在 `launch()` 里传
  `model.action_cls_d_max=$(jq .d_max_p90_forward artifacts/collapse_fix_dmax.json)`。
  仅当前向原语的标签噪声被证明拖累 CE 时才值得做;否则保持失效即可。

---

## 完成判据

1. `train_subset_stop20` 已带 stop 标签重生;原始子集保持完好。
2. smoke 显示有限的 `loss_ce`（CE 头在接收 stop 监督）。
3. 完整重训 checkpoint 已保存;预测原语分布不再前进塌缩。
4. `:30905` 用 `ORACLE_STOP=0` 评测,报出 SR > 0（超过 b0_v2）且为学习到的 stop。

## 备注 / 风险

- 沙箱无法重转数据、无法访问 GPU、无法 SSH 主机 —— 任务 1–4 由用户在
  `:31126` / `:30905` 执行。本方案 + 启动器即 SOP。
- 所有纯 Python 的转换/标签逻辑已在本地单测
  （`test_convert_openfly_fixture.py`、`test_collapse_fix_labels.py`）;torch
  路径（head_cls、CE、denoise）由主机验证。
- 不要改动基线 `aerial_openfly.yaml` / `aerial_joint_1cam_1e-4.yaml` 或原始
  `train_subset` —— 塌缩修复路径与之完全并行。
