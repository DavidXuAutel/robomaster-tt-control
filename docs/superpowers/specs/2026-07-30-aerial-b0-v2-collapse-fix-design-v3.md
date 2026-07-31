---
title: Aerial B0 v2 动作塌缩修复方案（v3）
type: design
date: 2026-07-30
status: draft-v3
supersedes: []
revises:
  - docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design.md
  - docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-revised.md
related:
  - docs/design/2026-07-29-aerial-nav-wam-redesign.md
  - docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md
  - artifacts/b0_v2_20260729-b0v2-10k-2gpu/
hosts:
  train: "10.239.121.21:31126 (2×H100) — B0 v2 已停于 step_005000"
  eval: "10.239.121.23:30905"
  renderer: "10.229.20.125:41451"
run_stopped:
  id: m1b-20260729-b0v2-10k-2gpu
  last_ckpt: step_005000
  resume_state: checkpoints/state/step_005000/
---

# Aerial B0 v2 动作塌缩修复方案（v3）

> **状态**：draft-v3 / 暂存。**不覆盖**原 draft 与 revised；作为评审修订后的下一版并存。
> **范围**：仅 aerial-wam 导航训练与闭环评测；不改 TT orbit / 真机避障路径。
> **相对 revised 的三处收紧**（见 §修订说明）。

## 修订说明

### 相对原 draft（已由 revised 吸收）

1. **loss 单调下降**，不是「动作头被饿死」——L2 奖励塌到条件均值；8× loss 降换来 SR=0，排除「只是还没训够」。
2. **oracle-stop 上限 OSR@20m ≈ 10%**（非粗估 ~65%）——只修终止不够。
3. **双根因并列**：①从不终止（stop 仅 2/2709）；②朝向错（最近接近均值 62.7 m）。单修任一个仍 SR≈0。

### 相对 revised（本 v3 新增）

1. **Stage 0 强制跑 `wm_instruction_probe`**：在动 Stage 3（CFG / dropout / 弱化图像锚）之前，用现有 ckpt 验证「目标通路是否已活」。未验证不开刀。
2. **Stage 2 验收不绑死 OSR=10%**：10% 是**当前塌缩策略**的几何上限；换头+学会转向后轨迹会变，OSR 本身会动。改用原语分布、stop 非零、最近接近距离改善。
3. **终止监督默认走 stop 重标**，独立 done 头作后备；**Stage 2+3 默认合并一次重训**，用 probe + 原语直方图做单因子诊断，避免小数据上连开两轮满训。

---

## 0. 背景与失败证据

B0 v2 从头训（`max_steps=10000`，`save_every=500`，`λ_action=λ_video=1`），在 **step 5000** 由主人手动停训（SIGTERM）。闭环 seen-20（n=20，`SUCCESS_DIST=20m`，`max_steps=100`）：

| step | NE (m) | SR | SPL | 说明 |
|------|--------|----|----|------|
| 500  | 517.2 | 0 | 0 | |
| 1000 | 472.0 | 0 | 0 | |
| 1500 | 460.0 | 0 | 0 | |
| 2000 | 534.3 | 0 | 0 | |
| 2500 | 562.2 | 0 | 0 | |
| 3000 | 604.4 | 0 | 0 | fwd9-episode 占比升到最高 |
| **3500** | **394.6** | 0 | 0 | NE 最低点，仍 ≈3× 静止基线 |

- **每个 checkpoint 都 SR=0**；NE 高方差、无收敛趋势，**始终 3–5× 于「静止不动」基线（NE ≈ 124 m）**。
- `steps ≡ 100`（**从不终止**）；`path_mean − NE_mean ≡ 117`（常数）。
- 越训 forward 步长越大：fwd9-episode 计数 500→3000 为 `2→0→0→5→8→8`，平均每步位移与 NE 相关系数 ≈ 1.0 —— 训练在「把前进步子放大」，不是「朝目标走」。
- 两路 loss 单调下降（`loss_action` 更低、降更快）。**loss 降 = 动作头把 (dx,dy,dz,dyaw) 越回归越贴条件均值**，恰好塌到 fwd6/fwd9。loss 降与 NE 差**不矛盾**：L2 目标奖励的正是塌缩。

### 0.1 几何复算：为什么终止修好也只有 ~10% 上限（对当前策略）

对 `step_001500` 逐 episode 用余弦定理算轨迹到目标的最近接近距离：

| 阈值 | 曾进入该半径的 episode（oracle-stop 上限 OSR） |
|------|------|
| ≤20m | **2/20 = 10%** |
| ≤30m | 5/20 = 25% |
| ≤40m | 9/20 = 45% |

- 最近接近距离：均值 **62.7m**、中位 50.6m、min 8.4、max 188.5。目标直线距离（shortest）均值 143.6m。
- 折算朝向误差 ~**26°**，且跨 checkpoint 不改善（训练只放大步长、不纠正朝向）。

**结论**：即便加上完美的「进 20m 就停」，**当前塌缩策略**的 SR 上限也只有 10%，因为 90% 的轨迹压根没路过目标 20m 内。**终止（根因①）与朝向/目标条件（根因②）是并列的，必须同修。**

> **注意**：换分类头并学会转向后，轨迹形状会变，最近接近分布与 OSR 会随之改变。因此后文 **不以 OSR=10% 作为换头后的 SR 天花板**。

### 0.2 训练集实测（`train_subset`）

- 200 episode / **2709** frames；平均长度 13.5、中位 13；**66% episode 短于 `num_frames=17`**。
- `skip_padding_as_possible: false` → 大量 padding 动作进 loss（数据形状 bug）。
- 专家动作 → 最近邻原语分布：

| 原语 | 占比 | 原语 | 占比 |
|------|------|------|------|
| fwd9 | 50.8% | left30 | 2.6% |
| fwd3 | 26.0% | right30 | 2.1% |
| fwd6 | 15.4% | down3 / up3 | 1.8% / 1.0% |
| **stop** | **0.1%（2 样本）** | left3 / right3 | 各 0.1% |

- dx 均值 6.25；dy/dz/dyaw 零值占比 88% / 96% / 86%。连续 L2 回归的条件均值落在 fwd6/fwd9 → **塌缩是目标函数与标签分布的数学结果**；同分布裸扩数据只会把条件均值估得更准（塌得更死）。

### 0.3 模型结构：主干 + 多头 + 损失合成（改动定位）

修复只动「动作支」与目标条件通路；video 主干默认不动。下图标出各改动的落点。

```
输入                     编码器                    主干(MoT 双专家联合注意力)               头                 输出
──────────────────────────────────────────────────────────────────────────────────────────────────────────

观测图像(first frame) ─VAE.encode─► first_frame_latents ─┐ 硬 pin latents_video[:,:,0]=first  ← 超强锚
                                                          │
latents_video(噪声) ───────────────► video_expert ─pre_dit┐│
                                                          ││   ┌──────────────────┐
文本指令 ─T5─► context(+mask) ─送进两个 expert 的 pre_dit ─┼┼──►│ MoT 混合注意力    │
              ▲ goal 唯一入口(弱,要跟图像锚抢,无 dropout)  ││   │ = 共享"主干"      │
                                                          ││   └───┬──────────┬────┘
本体状态 ─proprio_encoder(Linear→text_dim)─► token ─append┘│   video│      action│
                                                           │    tokens│      tokens│
latents_action(噪声) ──► action_expert ─pre_dit────────────┘        ▼          ▼
                                                              ┌──────────┐  ┌───────────────────────┐
                                                              │video head│  │ action_expert.post_dit│
                                                              │post_dit  │  │  └► self.head          │◄ 【动作头】
                                                              └────┬─────┘  │     Linear(hidden,4)   │  action_dit.py:98
                                                                   │VAE.decode└──────────┬────────────┘
                                                                   ▼                     ▼
                                                             imagined 帧          连续(dx,dy,dz,dyaw)
                                                            (WM 诊断片段)        └► 最近邻 → 原语 0~9(执行)

损失:  total = λ_video·‖pred_video−tgt‖²  +  λ_action·‖pred_action−tgt‖²      (λ_video=λ_action=1)
        └─ loss_video(视频支)                  └─ loss_action(动作支) ← L2 奖励塌到均值 = 塌缩源头
```

**动作头 vs 分类头**：现状「动作头」= `action_dit.py:98` 的 `self.head = nn.Linear(hidden, 4)`，输出连续 4D delta，`L_action` 是 flow-matching MSE，数学上奖励塌到条件均值。「分类头」= 在其旁**新增** `Linear(hidden, 10)`（见 §0.4，保留流匹配主干），标签取专家 delta 的最近邻原语 ID，主损失换成 CE（+class weight/focal），多峰分布下不塌到单一均值。

**关键点**：换分类头把 `L_action` 的主损失从「回归均值」换成「分类 argmax」，`λ_video·L_video`、主干、两个 `pre_dit` 的 context 注入都不动。**它治「塌到均值」，治不了「目标盲」** —— 目标盲是否成立，须先用 Stage 0 的 instruction-probe 验证，再决定 Stage 3 力度。

### 0.4 换头的实现分叉：保留流匹配 + 辅助分类头（锁方案 B）

现状 action 支是**流匹配去噪**：`infer_action` 对 `latents_action` 跑迭代去噪循环，`self.head` 输出的是**速度**而非动作本身。10 类分类头**不去噪**（一次前向读一个 class）。因此「换头」是一个**架构分叉，不是替换一行 Linear**：

- **方案 A（否决）**：动作支彻底改单步分类，丢掉 action 噪声 latent 与去噪循环。问题：视频支仍逐步去噪，MoT 每步耦合 video/action tokens；动作支退化为单步后与该耦合机制冲突，改动面大、风险高、难归因。
- **方案 B（锁定）**：**保留 action 流匹配主干与去噪循环不变**，在 `action_expert` 输出上**新增一个分类头**（读 pooled / 末步特征 → `Linear(hidden, 10)`）。训练时 **CE 为主损失**；原流匹配的连续 delta 回归降为**小权重辅助**（正则，并保住与 MoT 耦合的梯度通路）。推理读分类头 argmax → 原语，忽略连续 delta。

**理由**：B 只在动作支加一个 head + 调主损失权重，不动去噪循环、不动 MoT 耦合、不动 video 支 —— 风险最低、可归因。**§2 Stage 2 的所有描述均按方案 B 理解。**

---

## 1. 目标与成功标准

- **主目标**：seen-20 上 **SR > 0**，且 mean NE **严格好于**静止基线（≈124 m）与当前最差点（604 m）。
- **选点指标**：SR、NE、SPL、预测原语直方图、平均每步米数、非零 stop 计数、最近接近距离分布（均值/中位/`oracle_hit@20/30/40`）。
- **不做选点指标**：世界模型生成帧质量（PSNR/FID/主观好看）。
- **训练纪律**：以评测选点，不以「训满 N step」为目标；有效 epoch 量级控制在个位数。

---

## 2. 分阶段设计

推荐执行顺序：**Stage 0 → 1 → (2+3 合并重训) → 4**。上一阶段验收不过，不进入下一阶段。
（相对原 draft：padding 提前；目标条件化升为一等阶段；相对 revised：probe 前置、验收口径与终止监督默认策略收紧。）

### Stage 0 — 评测止血 + 几何诊断 + 指令敏感性探针（不重训）

1. **终止条件**：除 `primitive == 0` 外，增加 `‖pos − goal‖ < SUCCESS_DIST` 即成功终止（`--oracle-stop`）；超时仍记失败。
2. **最近接近距离日志**：逐 step 记录到目标的距离，落 metrics（`closest_approach_m`、`oracle_hit@20/30/40`）。量化 OSR、区分终止问题与朝向问题。
3. **执行口径对照（轻量）**：同一 ckpt 抽跑 nearest-primitive vs 连续 `step_delta`（若 bridge 支持）。预期多半救不了 SR；目的是排除「量化把好策略弄坏」。不要在这上面耗大量机时。
4. **复评 ckpt**：`step_001500`、`step_003500`（NE 最低）、对照 `step_000500`。
5. **强制：`wm_instruction_probe`**（固定 obs+state+seed，扫指令含空指令基线）：
   - 对 `step_001500` / `step_003500` 各跑一遍；
   - 读 `summary.json` + 原语/clip 是否随指令变化；
   - **分支决策**（决定 Stage 3 力度，见下表）。
6. **WM 诊断旁路（可选）**：1–2 episode 开 `--dump-wm-frames`（仅 step 0）；**不入库为选点依据**。

**Stage 0 → Stage 3 分支（强制）**

| instruction-probe 结果 | Stage 3 做法 |
|------------------------|--------------|
| primitive/clip **几乎不随指令变**（含空指令不可分） | 目标通路未激活 → 做 conditioning dropout + CFG；评估是否弱化首帧硬 pin（需 A/B，默认先只加 dropout/CFG） |
| primitive/clip **随指令显著变化** | 目标通路已活 → **不做**弱化图像锚；Stage 3 降级为「保持文本条件 + 依赖分类头/stop 重标/数据配额」；仅当闭环朝向仍差时再加轻量 CFG |

**验收**：协议版本号写入 metrics key；`closest_approach_*` 与 probe 结果落盘；`--oracle-stop` 后 SR 预期抬到 ~10%（对当前策略），确认终止必要非充分后进入 Stage 1。

### Stage 1 — 数据 / 窗口 / padding 卫生（**重训前提**）

- `num_frames` 降到与中位 episode 匹配（建议 **9**；中位长度 13）。
- `skip_padding_as_possible: true`；padding 动作不进 action loss。
- 修复 / 重算 `dataset_stats`（杜绝 NaN std；若本轮一并改角度表征，按 v2 规格走 wrap + (sin, cos)）。

**验收**：有效动作占比显著上升；首步 `loss_action` / `loss_video` 有限、无 NaN。**不在污染标签上训新头。**

### Stage 2 — 动作头改分类 + 终止监督（消除 L2 塌缩 + 学会停）

- **动作头**：连续 4D delta → 最近邻 10 类原语 ID；主损失为 **分类**（CE + class weight 或 focal）。连续 delta 仅作可选小权重辅助回归。
- **标签噪声**：最近邻打标在 fwd6↔fwd9 等边界有噪声 → 实现时加 label smoothing，或仅对「最近邻距离低于阈值」的样本进 CE。
- **终止监督（默认）**：把「接近目标帧」**重标为 stop（class 0）** + class weight，保证 stop 有效样本远多于 2/2709。评测侧距离终止已在 Stage 0 落地。
- **终止监督（后备）**：仅当 stop 重标后仍学不会停时，再加独立 done 头。注意 `oracle-stop` 用**真值目标位**、**仅评测诊断**，不是学到的能力；**部署终止靠 done 头 / stop 类**。评测协议内优先级：`oracle-stop`（诊断上限）> done 头 > stop 类，避免双信号打架。
- **保留** `λ_video > 0`（joint 世界模型正则）。

**验收（不绑死 OSR=10%）**：

1. held-out 预测原语分布不全塌到 fwd9（fwd9 占比显著低于训练先验 50.8%，且转向/升降类非零）；
2. 非零 stop（或 done）计数出现；
3. 闭环 **最近接近距离均值**相对同协议下的塌缩 baseline（如 step_1500/3500）显著下降（**≥20% 为占位阈值，待 Stage 0 给出 baseline 分布后校准**；或中位进入更小桶）；
4. SR > 0 为加分项，不是本阶段硬门禁（朝向未充分修好时 SR 仍可能很低）。

### Stage 3 — 目标条件化（治「目标盲」，与 Stage 2 配套；力度由 Stage 0 probe 决定）

> **仅当 Stage 0 probe 显示指令不敏感时，本阶段做满；否则按分支表降级。**

- **conditioning dropout**：训练时按概率丢弃文本/目标条件，逼模型真正利用 goal。
- **classifier-free guidance**：训好 uncond 分支后，推理用 CFG 放大目标条件（`text_cfg_scale` 已存在于 `policy_fastwam`）。
- **弱化图像锚（可选、需 A/B）**：仅在 dropout+CFG 仍不够、且 probe 仍失败时评估；默认**不**作为第一刀，以免伤 video 正则。
- **诊断**：复用 `wm_instruction_probe.py` 验证「动作/生成帧随指令变化」。

**验收**：instruction-probe 显示动作随指令显著变化（含空指令基线可分）；闭环最近接近距离均值进入 **<30m** 量级（或相对 Stage 2-only 基线再明显下降）；SR 突破「旧策略 OSR 10%」并持续 >0。

### Stage 2+3 默认合并为一次重训

- **默认**：Stage 1 卫生完成后，**一次**重训同时上分类头 + stop 重标 +（按 Stage 0 分支决定的）目标条件化改动。
- **归因**：用原语直方图区分「是否脱离 fwd9 塌缩」；用 instruction-probe 区分「目标通路是否激活」；用最近接近距离区分「朝向是否改善」。
- **例外**：仅当合并训后 probe 已活、原语已发散，但最近接近仍差 → 再开一轮轻量强化（CFG 权重 / goal token），不做第二次从零满训。

### Stage 4 — 扩数据（禁止同分布裸扩）

- 放宽 `download_openfly_subset.py` 的 `env_prefixes`；目标量级 **≥2k episodes 或 ≥2e4 frames**（以 OpenFly 可下池为准）。
- **配额采样**：抬高 stop / 转向 / 近目标帧占比，禁止只堆 fwd9。
- leak-free：held-out seen-20 与 train 的 `route_ids` 不相交。

**验收**：新集 stop 占比达数百分点级；训 ≤5 epoch 量级即做闭环选点。

---

## 3. 世界模型生成帧评测支路 — 结论

| 角色 | 是否做 |
|------|--------|
| 选点 / 门禁 / early-stop | **否** |
| 诊断旁路（`--dump-wm-frames` / `wm_instruction_probe.py`） | **是**；probe 在 Stage 0 **强制**，dump-wm 可选 |
| 训练目标 | 保持 `λ_video > 0` 即可 |

**理由**：当前失败是动作决策塌缩 +（待证）目标盲；video loss 已在降，视频支路不是主瓶颈。无需新建以生成帧质量为主的评测支路。

---

## 4. 明确不做

- 不把「训满 10000 step」当成功标准（8× loss 下降已证明多训无用）。
- 不把 WM 生成帧质量当主指标。
- 不在修动作头 / 终止 / 目标条件之前大规模扩同分布数据。
- 不单修终止就期待可用 SR（当前策略几何上限 10%）。
- **不在 Stage 0 probe 之前**弱化 first-frame pin 或大改 goal 注入。
- 不以 OSR=10% 作为换头后的 SR 硬天花板。
- 不改 orbit / TT 真机避障模块。
- 不从本草案直接开长训，须 Stage 0 验收后再定预算。

---

## 5. 与 v2 重设计文档的关系

本草案是对 `docs/design/2026-07-29-aerial-nav-wam-redesign.md` 的 **B0 失败后补丁**，不推翻 v2 的数据可信化与评测 fail-closed 原则：

- v2 **S1.0**（连续动作执行归因）→ 本草案 Stage 0（轻量对照）。
- v2 **S1.2** joint `λ_video` → 保留。
- v2 角度 (sin, cos) / 分位数归一化 → 并入 Stage 1，或单开数据重生任务。

---

## 6. 已锁定的取舍（相对 revised 文末「待确认」）

| # | 问题 | v3 锁定 |
|---|------|---------|
| 1 | 阶段顺序 | **Stage 0 → 1 → (2+3 合并重训) → 4** |
| 2 | 动作头 | **10 类分类 + class weight/focal**；连续回归仅可选辅助；label smoothing / 近邻阈值降噪 |
| 3 | 终止监督 | **默认 stop 重标**；独立 done 头仅后备 |
| 4 | Stage 2+3 | **默认合并一次重训**；probe 决定 Stage 3 力度 |
| 5 | Stage 0 probe | **强制**；未跑不开 Stage 3 满配 |
| 6 | 换头实现 | **方案 B**：保留 action 流匹配 + 新增辅助分类头，CE 为主损失、连续回归小权重辅助（见 §0.4） |

确认开工后下一步：拆 `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md`，先落 Stage 0 零重训代码（`run_closed_loop.py`：`--oracle-stop` + 最近接近距离日志 + 测试；跑通 `wm_instruction_probe`），再动 Stage 1 数据卫生与模型。
