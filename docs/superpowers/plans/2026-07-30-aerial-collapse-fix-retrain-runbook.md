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
| `experiments/aerial/scripts/run_collapse_fix_retrain.sh` | reconvert → preflight → smoke → train；支持 `OUTPUT_DIR` / 本机 Wan |
| `experiments/aerial/convert_openfly_to_lerobot.py` | `--stop-relabel-radius` stop relabel（不变量 #2a / #3） |
| `configs/task/aerial_joint_collapse_fix.yaml` | λ 配方 + `enable_action_cls: true` + `override /data: aerial_openfly_collapse_fix` |
| `configs/data/aerial_openfly_collapse_fix.yaml` | `num_frames=9`、`ratio=2`、`skip_padding`（不变量 #2b） |
| `src/fastwam/models/wan22/action_dit.py` | `head_cls` + `classify_from_tokens` |
| `src/fastwam/models/wan22/fastwam.py` | CE 分支（`loss_lambda_ce>0` + `head_cls`） |
| `experiments/aerial/eval/run_closed_loop.py` | `collapse_fix` task 自动 `+enable_action_cls`；`redirect_common_files=false` |
| `experiments/aerial/eval/policy_fastwam.py` | 闭环在有 `primitive` 时用 `argmax(cls)` |
| `experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh` | 并行评测：轮询 weights → 入队（`STEPS`/`WEIGHTS_DIR`/`EVAL_QUEUE_DIR`） |
| `experiments/aerial/scripts/orch_eval_worker.sh` | 并行评测：认领 job，`ORACLE_STOP=0` 学习 stop |
| `experiments/aerial/scripts/collapse_fix_status.sh` | **进度查询**：训练（step/loss_ce/ETA）+ 评测（SR/NE/stop%/hit@20）一屏汇总 |
| `experiments/aerial/scripts/collapse_fix_status_remote.sh` | 一键 SSH 到远端跑上面的 status（本机拉远端进度；`--dry-run` 先核对 host/port/runtime） |
| `experiments/aerial/scripts/plot_loss.py` | 从 train log 画 loss 曲线 PNG（含 `loss_ce`，副轴）+ CSV；status 里 `PLOT=1` 会自动调它 |
| `experiments/aerial/collapse_fix/compute_dmax.py` | 仅供参考的 d_max（见 §d_max） |

**本机路径约定（`:31126` / `:30905` 当前 run）：**

| 变量 | 值 |
|------|-----|
| `STAMP` | `20260731-collapse-fix-1500` |
| `OUTPUT_DIR` / run root | `/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}` |
| `WEIGHTS_DIR` | `${OUTPUT_DIR}/checkpoints/weights` |
| `EVAL_QUEUE_DIR` | `/home/a25689/aerial_cache_shared/orchestration/eval_queue_collapse_fix` |
| `MAX_STEPS` / `SAVE_EVERY` / `STEPS` | `1500` / `500` / `500,1000,1500` |
| reconvert ann | `.../FastWAM/data/openfly_raw/Annotation/subset_train.json` |
| reconvert images | `.../FastWAM/data/openfly_raw` |
| eval ann | `/home/a25689/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json` |
| `OPENFLY_ROOT` | `/home/a25689/aerial_eval_cache/OpenFly-Platform` |

> **进度查询（训练 + 评测一屏）：** 任一主机上跑
> `bash experiments/aerial/scripts/collapse_fix_status.sh`（`WATCH=1` 循环）。
> 默认读上表路径:训练段给 step/loss_ce/ckpt/ETA(ETA 由 ckpt mtime 推算，
> `:30905` 上看不到 train log 时自动退化为「盘上 ckpt」)；评测段给
> `eval_queue_collapse_fix` 计数 + 逐 ckpt 的 SR/NE/SPL/stop%/close_m/hit@20
> （`stop%`=`n_stop_primitive/n`,验根因 #1;`close_m`/`hit@20` 验根因 #2）。
>
> **loss 曲线已整合进 status：** 默认每次跑 status 都会调 `plot_loss.py` 出图
> (`$LOG_DIR/loss_curve.png` + `.csv`),`PLOT=0` 关闭,`PLOT_OUT`/`PLOT_SMOOTH`
> 可调;`--dry-run` 那台没 train log 时自动跳过。也可单独跑
> `python experiments/aerial/scripts/plot_loss.py <log或logdir>`。图:左轴
> total/action/video(flow-matching),右副轴 `loss_ce`(停止分类器,CE nats)。
> **重点看 `loss_ce` 是否早期(~step 10)就饱和到很低**——那是根因 #1(学会终止)
> 的训练侧信号。
>
> **从本机一键拉远端进度：**
> `bash experiments/aerial/scripts/collapse_fix_status_remote.sh --dry-run` 先打印
> 将执行的 `ssh` 命令核对 `SSH_HOST`/`SSH_PORT`/`RUNTIME`,确认无误后去掉 `--dry-run`
> 即在远端跑一次 status。`TARGET` 一键切主机(host+port 一起翻转,不会只改一个):
> `TARGET=train`(缺省)= 推理/训练机 `a25689@10.239.121.21:31126`(本地有 train log
> + 共享 ckpt/queue,一次即出两段);`TARGET=eval` = 测试/评测机
> `a25689@10.239.121.23:30905`。`WATCH=1` 在远端循环;`STAMP`/`OUTPUT_DIR`/`STEPS`/…
> env 自动透传;`SSH_HOST`/`SSH_PORT`/`RUNTIME` 可显式覆盖。**注意**:`RUNTIME`
> 默认 `/home/a25689/aerial_wam_runtime/robomaster-tt-control`；先 `--dry-run`
> 核对远端该目录下已有 `collapse_fix_status.sh`。

> **Wan 权重：** 禁止再下；用 `checkpoints/Wan-AI/` + `redirect_common_files=false` +
> `DIFFSYNTH_SKIP_DOWNLOAD=true`。
> **注意：** 规则文件 `.cursor/rules/aerial-wan-weights-local.mdc` **仅存在于主仓
> `robomaster-tt-control`**,并**未** vendor 进 `aerial-wam` 分支。在训练机上
> checkout 该分支的执行者看不到它 —— 本条(禁下 Wan/DiffSynth 权重、用本机
> `checkpoints/Wan-AI/`)即其内容摘要,以之为准;需完整规则请回主仓查阅。

---

### 任务 1：重生 relabel 数据集（主机 `:31126`）

非破坏性：写入 `train_subset_stop20`,原始子集保持不变。

```bash
OPENFLY_ANN=/home/a25689/aerial_wam_runtime/FastWAM/data/openfly_raw/Annotation/subset_train.json \
OPENFLY_IMAGE_ROOT=/home/a25689/aerial_wam_runtime/FastWAM/data/openfly_raw \
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh reconvert
```

- [x] **第 1 步：** 运行 `reconvert`;确认 `./data/openfly_lerobot/train_subset_stop20` 已生成。
- [x] **第 2 步：** 校验:原始 `train_subset` 逐字节未变（不变量 #1）。
- [x] **第 3 步：**（可选）抽查几条 episode —— 末帧 + 近 goal 帧应带 `action==[0,0,0,0]`。
  （实测：末帧零动作 200/200；零动作行 740/2709 ≈ 27%。）

> `R=20.0` 对齐 `OPENFLY_SUCCESS_DIST_M`（`experiments/aerial/eval/metrics.py`）。
> 若评测阈值变动,用 `STOP_RADIUS=<m>` 覆盖。目标目录已存在时,用 `FORCE=1`
> 覆盖重生。

---

### 任务 2：preflight + smoke（主机 `:31126`）

```bash
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh preflight
# smoke 必须本机 Wan（启动器已默认 redirect_common_files=false + DIFFSYNTH_SKIP_DOWNLOAD）
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh smoke
```

- [x] **第 1 步：** preflight 通过：relabel 数据在位、2 块 GPU、`resume=null`、检测到 `head_cls` 接线、`RESUME`/`AERIAL_ALLOW_LEGACY_RESUME` 为空。
- [x] **第 2 步：** smoke（`SMOKE_STEPS=10`）打印**有限**的 `loss_video`、`loss_action`（fm）以及 **`loss_ce`** —— 出现非平凡的 `loss_ce` 就是「CE 头确实接上并在接收 stop 标签」的信号。
- [x] **第 3 步：** 无 NaN;峰值显存 < 90%。记录 steps/s → 据此设 `MAX_STEPS`。
  （`:31126` 实测 ~0.09 step/s；`loss_ce` 5.97→0.02；峰值显存 ~70.5/81.6 GiB≈86.5%。
  **本 run 锁定：** `MAX_STEPS=1500`，`SAVE_EVERY=500`。）

> 启动器在 task 配置之上**显式**传入配方（`+enable_action_cls`、
> `λ_video/λ_fm/λ_ce`、`data.train.dataset_dirs=[…train_subset_stop20]`、
> `redirect_common_files=false`）,因此整条 override 链在日志里自解释。

---

### 任务 3：完整重训（主机 `:31126`）

**必须**设 `OUTPUT_DIR` 到 shared（否则任务 4 / 并行 watcher 看不见 ckpt）。
勿再依赖 Hydra 默认的 `./runs/train/<stamp>/`。

```bash
STAMP=20260731-collapse-fix-1500
OUTPUT_DIR=/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}
MAX_STEPS=1500 SAVE_EVERY=500 OUTPUT_DIR="$OUTPUT_DIR" \
  bash experiments/aerial/scripts/run_collapse_fix_retrain.sh train
```

- [x] **第 1 步：** 启动完整训练;checkpoint 落在
  `$OUTPUT_DIR/checkpoints/weights/step_*.pt`（run 已启动；等 500/1000/1500）。
- [ ] **第 2 步：** 观察 `loss_ce` 下降、预测原语直方图从「只有前进」的塌缩里铺开（用 `plot_loss.py` 记录/绘图）。
- [ ] **第 3 步：** 本 run 锁定 `MAX_STEPS=1500`。注意步数口径:`batch_size=8`
  × 2 proc = 有效 batch 16,当前子集(~200 episode / 2709 帧,窗口量级 ~1.1k–2.7k)
  下 1500 step ≈ **9–21 个 epoch 等效量**,已**超过**设计蓝本原定的「≤5 epoch」
  软目标。之所以仍跑 1500:smoke 显示 `loss_ce` 在 ~10 step 即饱和到 0.02(离散
  stop/原语头学得极快),额外步数主要留给 **video/fm 头**继续收敛。
  **代价:小子集上跑 ~20 epoch 有记忆化/过拟合风险** —— 根因 #2(目标盲)本就要靠
  后续**数据扩充**解决,不是靠在小子集上多跑。若 Task 4 SR 在 step_000500→001500
  之间不升反降,优先取更早的 ckpt 并把数据扩充提前,而非再加步数。
  （1500 step @ ~0.09 step/s ≈ 4.5h wall；与并行任务 4 重叠。）

---

### 任务 4：以「学习到的 stop」评测（主机 `:30905`）

**推荐与任务 3 并行**（见下一节）。串行亦可：训完后再对
`step_000500/1000/1500` 入队。口径一律 `ORACLE_STOP=0`（学习 stop）。

- [ ] **第 1 步：** 用独立队列 + `ORACLE_STOP=0`（`argmax(cls)`）+ `closest_approach`
  诊断评各 ckpt（**不要**用默认 `eval_queue`，内有旧 b0v2 job）。
- [ ] **第 2 步：** 将 SR / NE / SPL 与 b0_v2（SR=0）及 Stage-0 oracle-stop 天花板（~10%）对比。
- [ ] **第 3 步：** 锁定基线前,过 v3.2 §1 的硬成功标准。

---

## 并行评测（与任务 3 重叠 —— 本 run 采用）

训练在 `:31126` 写 shared `OUTPUT_DIR`；`:30905` watcher 轮询 `WEIGHTS_DIR`、
入队到**专用** `EVAL_QUEUE_DIR`；worker 以 `ORACLE_STOP=0` 认领。默认 B1
`eval_queue` **禁止**混用。

### 可行性约束（硬性）

1. **`:30905` 必须看得见 weights** → 训练用 `OUTPUT_DIR` 指向
   `aerial_cache_shared/runs/aerial_collapse_fix/...`（勿用 Hydra 默认本地 stamp）。
2. **`STEPS` = 实际会落盘的 step** → 本 run：`500,1000,1500`。
3. **专用 `EVAL_QUEUE_DIR`** → `.../eval_queue_collapse_fix`。
4. **评测必须建 `head_cls`** → task yaml + `run_closed_loop` 已对 `collapse_fix`
   自动 `+enable_action_cls`。
5. **`dataset_stats.json` 在 run root / weights 旁**。
6. **`ORACLE_STOP=0`**；显式写出防 shell 残留 `=1`。

### 命令（本 run）

```bash
# ---- :31126 训练（Task 3）—— 已启动则跳过 ----
STAMP=20260731-collapse-fix-1500
OUTPUT_DIR=/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}
MAX_STEPS=1500 SAVE_EVERY=500 OUTPUT_DIR="$OUTPUT_DIR" \
  bash experiments/aerial/scripts/run_collapse_fix_retrain.sh train

# ---- :30905 watcher ----
STAMP=20260731-collapse-fix-1500
WEIGHTS_DIR=/home/a25689/aerial_cache_shared/runs/aerial_collapse_fix/m1b-${STAMP}/checkpoints/weights
EVAL_QUEUE_DIR=/home/a25689/aerial_cache_shared/orchestration/eval_queue_collapse_fix
mkdir -p "$EVAL_QUEUE_DIR"/{pending,running,done,failed}
STAMP=$STAMP TASK=aerial_joint_collapse_fix \
  WEIGHTS_DIR=$WEIGHTS_DIR SHARED_WEIGHTS_DIR=$WEIGHTS_DIR \
  STEPS="$(seq -s, 500 500 1500)" \
  EVAL_QUEUE_DIR=$EVAL_QUEUE_DIR \
  ANN=/home/a25689/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json \
  OPENFLY_ROOT=/home/a25689/aerial_eval_cache/OpenFly-Platform \
  bash experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh --dry-run
# 核对无误后去掉 --dry-run，nohup 常驻

# ---- :30905 worker ----
ORACLE_STOP=0 \
EVAL_QUEUE_DIR=/home/a25689/aerial_cache_shared/orchestration/eval_queue_collapse_fix \
bash experiments/aerial/scripts/orch_eval_worker.sh
```

- [x] **第 1 步：** `OUTPUT_DIR`/`WEIGHTS_DIR` 指向本次 shared checkpoint 树。
- [x] **第 2 步：** `STEPS=500,1000,1500` 对齐 `SAVE_EVERY`/`MAX_STEPS`。
- [x] **第 3 步：** 独立 `EVAL_QUEUE_DIR`；worker 将用 `ORACLE_STOP=0`。
- [x] **第 4 步：** `--dry-run` 已通过。
- [x] **第 5 步：** 正式拉起 watcher + worker（无 `--dry-run`）。
  （`:30905` watcher+worker 已常驻；队列 `eval_queue_collapse_fix`；等 step_000500。）

### 脚本状态

| 脚本 / 代码 | 说明 |
|-------------|------|
| `run_collapse_fix_retrain.sh` | `OUTPUT_DIR`、本机 Wan、`+enable_action_cls` |
| `orch_ckpt_watch_enqueue.sh` | `STEPS` 可覆盖 |
| `orch_eval_worker.sh` | 靠独立 `EVAL_QUEUE_DIR` 隔离 |
| `aerial_joint_collapse_fix.yaml` | `enable_action_cls: true` |
| `run_closed_loop.py` | `collapse_fix` → 自动 `+enable_action_cls` |
| `b1_discover.py` / `eval_queue.py` | 无需改；结果目录名仍为 `b1_<stamp>`（纯命名） |

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

- 任务在 `:31126` / `:30905` 执行；Agent 可通过 SSH 操作。本手册 + 启动器即 SOP。
- 所有纯 Python 的转换/标签逻辑已在本地单测
  （`test_convert_openfly_fixture.py`、`test_collapse_fix_labels.py`）;torch
  路径（head_cls、CE、denoise）由主机验证。
- 不要改动基线 `aerial_openfly.yaml` / `aerial_joint_1cam_1e-4.yaml` 或原始
  `train_subset` —— 塌缩修复路径与之完全并行。
- **禁止**再下载 Wan2.2 / DiffSynth 权重；用本机 `checkpoints/Wan-AI/`。
