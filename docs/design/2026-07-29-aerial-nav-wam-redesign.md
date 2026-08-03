---
title: Aerial WAM 导航训练整体方案重设计（v2）
type: design
date: 2026-07-29
status: proposal
supersedes:
  - docs/design/2026-07-24-aerial-b0-failure-replay-finetune-design.md
  - docs/superpowers/specs/2026-07-27-aerial-b0-to-b1-orchestration-design.md
inputs:
  - docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md
  - artifacts/b1_seen20_metrics_20260727-072347.json
hosts:
  train: "10.239.121.21:31126 (1 机 2×H100) — B0 从头训 + B1 FT 同机"
  eval: "10.239.121.22:30682 (1 机 1×H100 + AirSim，串行单消费)"
  renderer: "10.229.20.125:41451"
  forbidden: "10.229.66.70 (robot net)"
topology:
  b0_train: "from scratch，1×node 2×H100，ZeRO-2 no-offload bf16，num_processes=2"
  eval: "1×H100 串行，AirSim 单客户端"
---

# Aerial WAM 导航训练整体方案重设计（v2）

## 0. 为什么要重设计

v1（B0 joint → B1 failure-replay DAgger FT）在工程链路上跑通了，但**模型侧没拿到结果**：

- Baseline B0（step_004000）：**SR=0**，NE=134，SPL=0 —— 基线在 seen-20 上从不成功。
- B1 各 checkpoint **整体差于 B0**；仅 step_1000 出现 SR=0.05；续训后回退，step_250 与 step_1250 出现 **bit 级相同 NE 327.41277290843607**。

根因（详见 handover §3 与失败分析）可归纳为 **6 类**，v2 的每个设计决策都针对其中之一：

| # | 教训 | 证据 | v2 对策 |
|---|------|------|---------|
| L1 | correction-rate 硬门禁被采样噪声打断 | gate 在 0.17 触发；±1.6σ 硬边界 vs 200 样本随机量 | **确定性配额采样 + 软失败门禁**（已实现） |
| L2 | weights-only resume → 新优化器 + LR 重新 warmup 重锤 | `trainer.py` 调度器不快进；1000→1250 骤退且双 327 | **resume 纪律**：满状态恢复，或续训用常数 LR、调度器快进 |
| L3 | yaw 不 wrap → min/max normalizer 污染 | yaw 累积到 43rad；`norm=min/max`；数据未重生 | **角度用 (sin,cos) 表示 + 稳健归一化**，重生数据/stats |
| L4 | λ_video=0 去掉世界模型正则 → 表征漂移 | B0 joint 训练，B1 纯 action | **保留小 λ_video + L2-SP 锚定 + EMA** |
| L5 | 评测链路脆弱、结果不可信 | 缺 stats→path_length=0；重复 metrics | **评测可信化**：fail-closed stats、per-episode、防陈旧 |
| L6 | 基线太弱（SR=0）+ 矫正集过窄（40ep） | 无成功可保留；40 条难 bootstrap | **先修基线与评测口径，再谈增训；扩/宽矫正集** |

一句话方针：**先让"基线"和"评测"可信，再谈微调；任何自动门禁必须对噪声鲁棒；任何微调必须不能悄悄损伤基座。**

---

## 1. 设计原则（贯穿全流程）

1. **数据完整性优先**：角度 wrap 且以 (sin,cos) 表征；稳健归一化（分位数而非 min/max）；leak-free split；每份 stats 与数据同源同 sha。
2. **评测先可信，再用于选点**：无 `dataset_stats.json` 直接 fail-closed；落 per-episode 记录；metrics 以 `(ckpt_sha, ann_sha, seed, protocol)` 为键，杜绝陈旧/重复复用。
3. **门禁对噪声鲁棒**：确定性配比采样；软失败（连续 N 窗才 fail）；阈值来自可信基线。
4. **微调不自伤**：resume 满状态或调度器快进；保留正则（λ_video>0 / L2-SP-to-base）；EMA + 按可信评测选最优；可回滚。
5. **单一事实源**：runtime = 本仓 `aerial-wam` 分支的 checkout（整个 FastWAM 已 vendoring 进来），禁止 scp 热补；部署即 `git checkout aerial-wam`。
6. **一切可复现/幂等/可续跑**：保留 v1 编排里好的状态机 + 幂等队列 + checkpoint watcher。

---

## 2. 分阶段方案

### Stage 0 — 地基：数据与评测可信化（阻塞性前置，不做完不许进入后续）

**S0.1 角度表征与稳健归一化（针对 L3）**
- 训练/评测/专家标注一律 `wrap_angle`；状态与动作里的航向改用 **(sin θ, cos θ)** 两通道，而非裸弧度 → 从根上消除"累积到 43rad 撑爆 min/max"这一类问题（对 wrap 天然鲁棒）。
- 归一化从 `min/max` 改为**分位数**（如 1%/99%）或标准化；对角度通道用单位圆约束。
- 重算 `dataset_stats.json`；与数据同 sha 绑定。

**S0.2 评测口径与可信化（针对 L5，且是 L6 的必要条件）**
- **fail-closed**：eval 前强制校验 `dataset_stats.json` 存在且 sha 匹配 ckpt 来源；缺失/不符直接报错，不产出 metrics。
- **per-episode 记录**（v1 Task 2 已做）：`episode_id, success, NE, path_length, shortest_length, steps`。
- **防陈旧/重复**：metrics 以 `key=(ckpt_sha256, ann_sha256, seed, protocol_version)` 命名/校验；worker 只在 key 不存在时计算；发现两 ckpt 产出相同聚合值时自动抽查 per-episode 并告警（直接堵住"双 327"这类）。
- **连续 vs primitive 动作**：记录两者 L2 gap；**优先用连续动作闭环**执行，量化仅作对照（见 S1.2）。

**S0.3 采样与门禁鲁棒化（针对 L1，已实现）**
- `WeightedSourceDataset` 确定性配额（largest-remainder），任意窗口精确配比。
- `FTSourceMonitor` 软失败：单窗告警、连续 `max_consecutive_violations` 窗才 fail。

**S0.4 单一事实源（针对代码分叉）**
- runtime/ft_cache repo 一律 `git checkout <worktree-sha>`；CI 前禁止手工 scp 覆盖；部署脚本打印并校验 sha。

**Stage 0 验收**：held-out 上用带正确 stats 的 B0 复评，得到与 artifact 一致或更正后的 NE；per-episode 齐全；重复 ckpt 不再产生相同聚合；门禁在合成噪声下不误杀。

---

### Stage 1 — B0 基线：从头训练的完整设计（针对 L6，真正的瓶颈）

B0 SR=0 意味着"20% NE 下降"的 S1 目标建立在流沙上。v2 **B0 从头训**（不 bootstrap 任何既有 aerial checkpoint），先把口径查清，再按下述完整配方训练与锁定。

> **"从头训"的界定（已定）**：取 **D1 义**——视频世界模型 backbone 用**预训练 Wan2.2 权重初始化**，动作通路 + joint 目标在 OpenFly aerial 上**从零训**，**不加载任何 v1 aerial checkpoint**。
> - D1（采用）：站在通用 Wan2.2 肩膀上，只有 aerial 部分从零。2×H100 唯一现实选项。
> - D2（否决）：连 Wan2.2 也随机初始化，训视频世界模型需海量算力/数据，2 卡不现实。
>
> **清除残存 checkpoint（已定）**：为杜绝"不小心又从旧权重 bootstrap"，v2 启动前**清除/隔离**所有 pre-v2 aerial 权重——尤其残存的 B0 seed `m1b-20260722-012926/step_000500.pt`。并在训练/FT 脚本加 **guard**：`resume` 只接受 v2 自己产出的 checkpoint，遇到任何 pre-v2 aerial `.pt` 直接拒绝启动。清除操作见 §8 运维（建议先归档再删，保留 metrics 与最佳点 step_1000 供存档）。

**S1.0 归因 SR=0（先做，最高信息量）**
- 逐项排查评测协议：`max_steps=100` 是否过短、success 距离阈值、动作量化、renderer 一致性、seed。
- **连续动作执行（强怀疑项）**：v1 eval 走 nearest-primitive，可能把连续策略量化成几乎纯 yaw（早先 path_length=0 旁证）。改为连续 `step_delta` 执行、primitive 仅回落。若连续执行就让旧 B0 SR>0，则"基线不行"其实是"执行口径不行"，B0 重训规模可大幅缩减。
- oracle/PathExpert 纯专家跑同一 seen-20：专家高而策略 0 → 执行/策略问题；专家也低 → 任务/协议问题。

**S1.1 数据（针对 L3，一次性重生，勿增量热补）**
- 源：OpenFly aerial，路径过滤子集（沿用 v1 downloader 思路），LeRobot v2.1，FPS=10。
- **动作/状态 4D**：位置 delta(x,y,z) + 航向；航向用 **(sinθ, cosθ)** 而非裸弧度，全程 `wrap_angle`。
- **归一化**：分位数（1%/99%）或标准化，角度通道单位圆约束；产出 `dataset_stats.json` 与数据同 sha 绑定。
- split：train / held-out **seen-20** 用 `route_ids` 保证 leak-free，held-out 只评不训。
- 文本条件：201-shard text-embed cache，启动前校验齐全（避免走 Ceph 热路径）。

**S1.2 模型与目标（针对 L4）**
- FastWAM **joint**：`lambda_action=1.0`、`lambda_video=1.0`（B0 必须带 video，作为世界模型正则的根基）。
- 从 Wan2.2 预训练 backbone 初始化；`num_frames`/`context_len` 沿用 joint 配置；动作头按 4D + (sin,cos) 输出维度调整。
- 可训范围：从头训阶段**放开 DiT 训练**（不像 FT 那样 DiT-only 冻结其余），backbone 视显存决定是否部分冻结。

**S1.3 训练调度与拓扑（1 机 2×H100）**
- accelerate DeepSpeed **ZeRO-2 no-offload，bf16，num_processes=2**；nanfix optimizer param-group（≤~1B/group，防 NaN）。
- AdamW(betas=0.9/0.95, wd=1e-2)；`lr_scheduler=cosine` + warmup（约 5% steps）。
- **max_steps 需按 2 卡算力现实设定**：v1 的 joint baseline 用 5×H100 才跑 5000 步；2 卡下同预算 wall-clock 约 2.5×，须据吞吐重定 `max_steps` 或接受更长时钟（见 §5 风险）。`save_every` 取能覆盖多个评测点的粒度（如 500）。
- 首步校验：`loss_action`/`loss_video` 有限、无 NaN；记录吞吐与 ETA。

**S1.4 评测协议定稿（成为"可信基准"，针对 L5）**
- held-out seen-20，固定 seed，success 距离阈值、`max_steps`（重新标定，勿沿用可能过短的 100）、renderer 单客户端、动作量化口径固定。
- **连续动作闭环**为准，primitive 仅对照；记录连续 vs primitive L2 gap。
- fail-closed：无 `dataset_stats.json` 或 sha 不符即报错；落 per-episode 记录；metrics 以 `(ckpt_sha, ann_sha, seed, protocol)` 为键防陈旧/重复。

**S1.5 基线锁定**
- 对 B0 各 checkpoint 跑 S1.4 协议，选 **finite mean NE 最低**且 **SR>0** 者。
- **只有 SR>0 且可复现才允许 lock**；写 `baseline_lock.manifest.json`（ckpt sha、protocol version、全 metrics、`baseline_mean_ne`、`s1_ne = 0.8×baseline_mean_ne`）。SR 仍为 0 → 停在此阶段排障，不得进入 B1。

**Stage 1 验收**：B0 在 seen-20 上 **SR>0**、NE 可信、per-episode 可解释；评测协议与 stats 定稿并 sha 化；baseline manifest 写好。

---

### Stage 2 — 能真正帮忙的矫正数据（针对 L6 的数据侧）

- **扩量、拓宽**：40 → 数百 episode，覆盖多类失败模式；leak-free（`route_ids.assert_disjoint`）。
- **专家质量**：PathExpert 用 wrap 后角度；oracle gate 要求 SR≥0.8、median_NE<阈值、projection_failures=0。
- **迭代式 DAgger**（而非一次性）：用当前策略 rollout → 专家在偏离态相对标注 → 入矫正集 → 再训，形成闭环，比单批更贴合策略分布。
- correction 动作以连续 delta 存储（与 S1.2 一致）。

**Stage 2 验收**：矫正集 sha 化、与 held-out 不相交、oracle gate 通过、动作有限且 wrap 正确。

---

### Stage 3 — 不自伤的微调（针对 L2、L4）

**S3.1 resume 纪律（针对 L2）**
- 首选**满状态恢复**（修 ZeRO-2 param-group 不匹配：保证 FT 的 optimizer param groups 与保存时一致，或用 accelerator 规范 save/load）。
- 若只能 weights-only：**调度器快进到真实 `global_step`**（不重新 warmup），或续训**直接用常数低 LR**。杜绝"新 Adam + 重新 warmup"的重锤。
- 加断言：resume 后首 50 步 loss/grad 不得超过基线区间 X 倍，否则 abort（防再现 1000→1250 坍塌）。

**S3.2 保留正则、锚定基座（针对 L4）**
- **保留小 λ_video（>0）**，或对基座权重加 **L2-SP / EWC** 正则，或 KL-to-base 蒸馏项，抑制表征漂移与灾难性遗忘。
- **权重 EMA**；按可信评测选最优 checkpoint，保留可回滚。
- 冻结策略明确（当前 DiT-only）；评估"多冻结 vs 带 video 正则"哪个更稳。

**S3.3 混采与课程**
- 确定性 75/25（S0.3）；可加课程：先高 original 占比稳住，再抬 correction。

**Stage 3 验收**：resume 后无骤退；每 250 步 checkpoint 经可信评测；EMA 最优点被记录。

---

### Stage 4 — 选点与晋级（针对 L5 的用途侧）

- S1_NE 来自 **Stage 1 的可信基线**（动态阈值，不硬编码历史常数）。
- **成对逐 episode 比较**（improve/flat/regress 计数）+ **SR 不回退**约束，而非只看聚合 NE。
- 失败不自动扩数据/不跑 unseen；产出诊断 scaffold（失败分箱）。

---

## 3. 编排（保留 v1 的好部分，补三处闸门）

保留：幂等 FIFO 评测队列、checkpoint watcher（mirror + stats + sha）、状态机 supervisor、`/home` 持久化。

新增闸门：
1. **eval 有效性闸门**（S0.2）：stats 缺失/sha 不符/key 陈旧 → 拒绝入库。
2. **resume 安全闸门**（S3.1）：首 N 步异常 → abort 并置 `BLOCKED(reason)`。
3. **baseline 可信闸门**（S1）：SR=0 或协议未定 → 不允许 lock、不允许进入 FT。

状态机建议相位：`VERIFY_EVAL → REBUILD_OR_LOCK_BASELINE → COLLECT_DAGGER → FT(RESUME_SAFE) → EVAL_B1 → S1_REPORT`。

---

## 4. 与 v1 的差异一览

| 维度 | v1 | v2 |
|------|----|----|
| 角度 | 裸弧度、不 wrap | (sin,cos) + 全程 wrap |
| 归一化 | min/max | 分位数/标准化 |
| 采样 | 逐样本 multinomial | 确定性配额（已实现） |
| 门禁 | 单窗硬 raise | 连续 N 窗软失败（已实现） |
| resume | weights-only + 重新 warmup | 满状态 / 快进调度器 / 常数 LR + 安全闸门 |
| 正则 | λ_video=0 | λ_video>0 或 L2-SP/EWC + EMA |
| 评测 | 易缺 stats、可复现陈旧 | fail-closed + per-episode + key 防陈旧 |
| 基线来源 | bootstrap 残存 step_000500 | **从头训**（Wan2.2 backbone 初始化，无 aerial ckpt） |
| 基线锁定 | SR=0 也锁定 | SR>0 且可信才锁定 |
| 训练拓扑 | 3 机 5×H100 | **1 机 2×H100**（评测独立 1×H100） |
| 矫正集 | 40ep 一次性 | 数百 ep + 迭代 DAgger |
| 代码源 | worktree/runtime/热补分叉 | 单一 commit checkout |

---

## 5. 风险与取舍

- **最大不确定性**：SR=0 的归因（S1.0）。若"连续执行"就能让基线 SR>0，则重点从"从头重训"转为"修评测/执行口径"，B0 训练规模与风险大幅下降 —— 这一步必须最先做。
- **算力现实（2×H100 从头训）**：v1 的 joint baseline 用 5×H100 才跑 5000 步；2 卡下同预算 wall-clock 约 2.5×。从头训（即便从 Wan2.2 backbone 起）比 bootstrap 更贵，须据实测吞吐重定 `max_steps`，或接受更长时钟、或缩小规模/分辨率。**先跑 1-step/10-step smoke 估显存与吞吐再定预算。**
- (sin,cos) 表征改动会波及数据 schema、normalizer、动作头，需一次性重生数据与 stats，别做增量热补（正是 v1 踩的坑）。
- L2-SP/EWC 引入超参；先用"小 λ_video + EMA"这一最简正则验证是否已足够抑制回退。
- 迭代 DAgger 成本高（多轮 AirSim 采集），可先单轮扩量验证收益再决定是否迭代。

---

## 6. 建议执行顺序（最小风险路径）

1. **S1.0 连续动作执行 + S0.2 评测可信化** → 复评旧 B0，先看 SR 是否真的是 0。（最高信息量、最低成本，可能直接改变 B0 规模）
2. **S0.1 / S1.1 角度 (sin,cos) + 稳健归一化** → 一次性重生数据与 stats。
3. **S1.2–S1.5 从头训 B0**（1×2H100，joint，先 smoke 估预算再定 max_steps）→ 拿到 **SR>0** 的可信 baseline 并 lock。
4. **S3 resume 纪律 + 正则** → 用已实现的 S0.3 采样/门禁跑一次干净 FT。
5. **Stage 2 扩/宽矫正集**（若单轮收益不足再上迭代 DAgger）。
6. **Stage 4 选点**，与可信基线比 SR/NE。

> 已落地（本 `aerial-wam` 分支，vendoring 自 FastWAM `feat/aerial-b0-b1-orchestration@46a1138`，共 19 tests passed）：
> - **L1** 确定性配额采样 + 软失败门禁：`weighted_source_dataset.py`、`ft_source_monitor.py`、`ft_mix_dataset.py`、FT task config。
> - **L2/清除 ckpt guard**：`trainer.py::_assert_resume_allowed` 拒绝从 pre-v2 aerial checkpoint（默认拦 `m1b-20260722-012926`、`step_000500.pt`）resume，逃生舱 `AERIAL_ALLOW_LEGACY_RESUME=1`；测试 `test_resume_guard.py`。
> 其余为本设计提案，待确认后按上表顺序实施。

---

## 7. v2 实现规格（模型 / 数据 / 评测 / 训练）

> 以下取自本分支现有实现（`configs/model/fastwam_joint.yaml`、`configs/data/aerial_openfly.yaml`、`configs/task/aerial_joint_1cam_1e-4.yaml`、`experiments/aerial/eval/*`）。标 **[v2 改]** 处为本设计相对 v1 的改动点。

### 7.1 模型结构（FastWAM joint，MoT）

- **底座**：`Wan-AI/Wan2.2-TI2V-5B`；tokenizer `Wan-AI/Wan2.1-T2V-1.3B`，`tokenizer_max_len=128`；`load_text_encoder=false`（用预算好的 text-embed cache）。MoT mixed-attention。
- **Video DiT**：30 层，`hidden_dim=3072`，`ffn_dim=14336`，`num_heads=24 × head_dim=128`，`in_dim=out_dim=48`，`patch_size=[1,2,2]`，`video_attention_mask_mode=first_frame_causal`，`action_conditioned=false`。
- **Action DiT**：30 层，`hidden_dim=1024`，`ffn_dim=4096`，`24×128`，`action_dim=4`；预训练权重 `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`。
- **调度器**：video/action 均 flow-matching，`train_shift=infer_shift=5.0`，`num_train_timesteps=1000`。
- **损失**：`lambda_action=1.0`、`lambda_video=1.0`（joint）。
- **[v2 改]** 动作头输出维度随 (sinθ,cosθ) 表征调整（航向 1→2 通道，见 7.2）；`from scratch` 指不加载任何 aerial 微调 ckpt（backbone 仍用 Wan2.2 + 预训练 Action DiT）。

### 7.2 数据集（OpenFly → LeRobot v2.1）

- `RobotVideoDataset`，`dataset_dirs=./data/openfly_lerobot/train_subset`。
- 观测：单目 ego RGB `3×224×224`；`num_frames=17`；`action_video_freq_ratio=4`；`global_sample_stride=1`；`concat_multi_camera=horizontal`。
- 动作 / 状态：各 4D（body-frame `dx,dy,dz,dyaw`）；`delta_action_dim_mask=[T,T,T,T]`。
- 文本条件：`text_embedding_cache_dir=./data/text_embeds_cache/openfly`（201 shard），`context_len=128`。
- **[v2 改] 航向 (sinθ,cosθ)**：`dyaw` 及 state yaw 全程 `wrap_angle` 并以 (sin,cos) 表征，消除累积撑爆问题（L3）。
- **[v2 改] 归一化**：`norm_default_mode` 由 `min/max` → **分位数(1%/99%)/标准化**，角度通道单位圆约束；重算 `dataset_stats.json` 并与数据同 sha 绑定。
- **动作原语（评测用）**：`0 stop,1 fwd3,2 left30,3 right30,4 up3,5 down3,6 left3,7 right3,8 fwd6,9 fwd9`。

### 7.3 评测方案（closed-loop, seen-20）

- Bridge：`openfly`（AirSim，renderer `10.229.20.125:41451`，单客户端串行）；held-out `seen_airsim16_m1a20.json`。
- 参数：`--max-episodes 20`、`--seed 42`、`--max-steps 100`（**[v2 改]** 需重新标定，勿沿用 100）。
- rollout：每步 `render→state→policy.predict→step`，出 `stop(0)` 结束。
- 指标（`eval/metrics.py`）：`OPENFLY_SUCCESS_DIST_M=20.0`；`NE=‖final−goal‖`；`success=NE<20`；`SR=Σsuccess/n`；`SPL=Σ(ok·shortest/max(path,shortest))/n`。
- **[v2 改] 连续动作执行**为准（`step_delta`），primitive 仅回落 + 记录 L2 gap；**fail-closed stats**（缺 `dataset_stats.json`/sha 不符即报错）；**per-episode 记录**；metrics 以 `(ckpt_sha,ann_sha,seed,protocol)` 为键防陈旧/重复。

### 7.4 训练方法

- **并行**：accelerate DeepSpeed **ZeRO-2 no-offload，bf16，num_machines=1，num_processes=2**（`accelerate_zero2_no_offload_2proc.yaml`）。
- **优化器**：AdamW(betas=0.9/0.95, wd=1e-2)；**nanfix** optimizer param-group（≤~1e9 elems/group，`optimizer_max_params_per_group`）。
- **调度**：`lr_scheduler=cosine` + warmup(≈5% steps)。
- **B0 超参**：`batch_size` 8（v1 uncond 基准值，按显存调）、`learning_rate` 1e-4、joint `λ_action=λ_video=1`、`gradient_accumulation_steps=1`、`resume=null`（从头）。
- **[v2 改] max_steps 按 2×H100 现实设定**：先 1-step/10-step smoke 估显存与吞吐，再定 `max_steps` 与 `save_every`（建议 500）。
- **[v2 改] resume guard**：训练脚本与 trainer 拒绝任何 pre-v2 aerial ckpt（见 §0 / L2）。
- **[v2 改] resume 纪律（FT 阶段）**：满状态恢复或调度器快进到真实 `global_step`（不重新 warmup），避免 v1 的"新 Adam + 重 warmup"重锤。

---

## 8. 运维：清除残存 ckpt / 训练主机 bring-up / 启动

> ⚠️ 以下需在**能访问实验室网络**的终端执行。Claude Code 当前沙箱只放行 Anthropic/autel API，**无法 SSH 到 `10.239.121.21` / `10.239.121.22`**，故本节为可直接执行的 SOP，不由本会话代跑。

**主机**：训练 `ssh a25689@10.239.121.21 -p 31126`（1 机 2×H100）；评测 `ssh a25689@10.239.121.22 -p 30682`（1×H100 + AirSim）。密码见私密渠道，勿写入仓库。禁止触碰 `10.229.66.70`。

### 8.1 清除残存 checkpoint（先归档再删，详见 §0 决策）

```bash
# 命中确认 → 归档 → 删除 残存 B0 seed
find /home/a25689 -path '*m1b-20260722-012926*step_000500.pt' -print
ARCH=/home/a25689/aerial_archive_v1_$(date +%Y%m%d); mkdir -p "$ARCH"
find /home/a25689 -path '*m1b-20260722-012926*step_000500.pt' -exec cp -av {} "$ARCH"/ \;
find /home/a25689 -path '*m1b-20260722-012926*step_000500.pt' -delete
```

### 8.2 训练主机环境准备（`:31126`）

```bash
# 代码单一事实源：runtime = robomaster-tt-control 的 aerial-wam 分支（已 vendoring 整个 FastWAM）。
# 禁止 scp 热补；部署即 checkout。
export RUNTIME=/home/a25689/aerial_wam_runtime/robomaster-tt-control
# 首次：git clone -b aerial-wam <robomaster-remote> "$RUNTIME"
git -C "$RUNTIME" fetch && git -C "$RUNTIME" checkout aerial-wam && git -C "$RUNTIME" pull
export PATH=/home/a25689/aerial_wam_runtime/env/bin:$PATH
pip install -e "$RUNTIME"                                                       # 装 fastwam 包
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # 期望 2

# 数据 + 文本 embed + Wan2.2/Action DiT 权重就位（本地 /home，勿走 Ceph 热路径）
ls ./data/openfly_lerobot/train_subset ./data/text_embeds_cache/openfly | head
ls checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
# Wan2.2 backbone 缓存命中（避免 ModelScope 慢下载，见 v1 教训）
```

### 8.3 Smoke → 估预算 → 定 max_steps

```bash
cd "$RUNTIME"
accelerate launch --config_file experiments/aerial/scripts/accelerate_zero2_no_offload_2proc.yaml \
  --num_processes 2 scripts/train.py task=aerial_joint_1cam_1e-4 max_steps=10 save_every=10
# 确认 loss_action/loss_video 有限、无 NaN、峰值显存 <90%；据吞吐推算 max_steps 预算
```

### 8.4 正式从头训 B0

```bash
cd "$RUNTIME"
# resume 必须为空（从头）；guard 会拒绝任何 pre-v2 ckpt
accelerate launch --config_file experiments/aerial/scripts/accelerate_zero2_no_offload_2proc.yaml \
  --num_processes 2 scripts/train.py task=aerial_joint_1cam_1e-4 \
  max_steps=<按 smoke 估算> save_every=500 resume=null \
  model.loss.lambda_video=1.0 model.loss.lambda_action=1.0
# 输出权重 → checkpoints/weights/step_XXXXXX.pt；每 500 步供评测
```

### 8.5 评测（`:30682`，串行单 AirSim 客户端）

```bash
cd "$RUNTIME"
python -m experiments.aerial.eval.run_closed_loop \
  --bridge openfly --policy fastwam \
  --checkpoint <shared>/step_XXXXXX.pt \
  --ann <held-out>/seen_airsim16_m1a20.json \
  --openfly-root <OpenFly-Platform> \
  --max-episodes 20 --seed 42 --max-steps <重标定> \
  --out <results>/step_XXXXXX_seen20/metrics.json
# 只在 SR>0 且可复现时才 lock baseline（写 baseline_lock.manifest.json）
```
