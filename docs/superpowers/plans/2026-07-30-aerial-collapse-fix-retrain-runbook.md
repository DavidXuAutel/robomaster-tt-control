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
在训练还在跑时就「边出边评」——用现有 B1 编排工具即可，无需新代码。

**思路：** `:31126` 训练照常写 `checkpoints/weights/step_*.pt`；在 `:30905`
起一个 watcher 轮询 weights 目录，每出现一个满足大小阈值的 ckpt 就把它作为一个
评测 job 入队；再起一个 worker 认领 job、用 `ORACLE_STOP=0`（学习到的 stop）跑
闭环。二者通过共享的 `eval_queue` 目录解耦。

```bash
# 在 :30905（评测机），watcher —— 监视本次 collapse-fix 训练的 weights 目录
STAMP=<collapse-fix-run-stamp> \
TASK=aerial_joint_collapse_fix \
WEIGHTS_DIR=<训练 checkpoint_root>/weights \
STEPS="$(seq -s, 500 500 5000)" \
ANN=<path/to/eval_annotation.json> \
OPENFLY_ROOT=<path/to/OpenFly-Platform> \
bash experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh

# 在 :30905，worker（另开一个 shell）—— 认领 job，用学习到的 stop 评测
ORACLE_STOP=0 \
bash experiments/aerial/scripts/orch_eval_worker.sh
```

- [ ] **第 1 步：** 确认 `WEIGHTS_DIR` 指向本次训练 trainer 实际的
  `checkpoint_root/weights`（**不是** B1 默认的 `b1-${STAMP}/...` 路径）。
- [ ] **第 2 步：** `STEPS` 用 `500,1000,…` 对齐 `SAVE_EVERY=500`（默认是 B1 的
  250 间距，若不覆盖会去等永不出现的 250-step ckpt）。
- [ ] **第 3 步：** worker 用 `ORACLE_STOP=0`（学习 stop）——这正是任务 4 的评测
  口径。脚本默认就是 `0`（见 `orch_eval_worker.sh:22`），显式写出是为了防止 shell
  里残留 `ORACLE_STOP=1`；只有 `=1` 才会走 oracle stop，那评的就不是「策略自己会
  不会终止」。
- [ ] **第 4 步：** 先 `orch_ckpt_watch_enqueue.sh --dry-run` 核对入队的 steps /
  weights 路径 / queue 目录再正式起。

### 本节涉及的脚本与代码改动

| 脚本 / 代码 | 是否需改 | 说明 |
|-------------|----------|------|
| `experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh` | **已改（本方案）** | 新增 `STEPS` 环境变量覆盖（`B1_STEPS="${STEPS:-$(seq -s, 250 250 5000)}"`）。不设 `STEPS` 时逐比特保持 B1 原有 250-间距行为；collapse-fix 用 `STEPS="$(seq -s, 500 500 5000)"` 对齐 `SAVE_EVERY=500`。`--help` 同步更新。`STAMP`/`WEIGHTS_DIR`/`TASK`/`ANN`/`OPENFLY_ROOT` 本就是 env 覆盖，按上面命令设即可。 |
| `experiments/aerial/scripts/orch_eval_worker.sh` | **无需改** | task 从 job JSON 读取，脚本本身与任务无关；`ORACLE_STOP` 已是 env 开关，跑时带 `ORACLE_STOP=0` 即可。 |
| `experiments/aerial/orchestration/b1_discover.py` | **无需改** | 全参数驱动（`--stamp/--weights-dir/--queue-dir/--results-root/--ann/--openfly-root/--task/--steps/--poll-s/--min-bytes`）。仅结果目录名带 `b1_` 前缀、job kind 记为 `b1`（纯命名，功能不受影响）。 |
| `experiments/aerial/orchestration/eval_queue.py` | **无需改** | 队列原语（enqueue/claim/complete）与任务无关，原样复用。 |
| `experiments/aerial/scripts/run_collapse_fix_retrain.sh` | **无需改** | 训练侧不感知评测；只需把 watcher 的 `WEIGHTS_DIR` 指到本次训练 trainer 的 `checkpoint_root/weights`。 |

> **命名提示：** 复用 B1 编排后，结果会落在 `results/b1_<stamp>/...`、job 标为
> `b1`。这只是历史命名，评的是 collapse-fix ckpt。若在意可后续给
> `b1_discover.py` 加 `--kind`/结果前缀参数，但对本次评测非必需，不阻塞。

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
