---
title: Aerial B0 v2 动作塌缩修复方案（修订稿）
type: design
date: 2026-07-30
status: draft-revised
supersedes: []
revises: docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design.md
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

# Aerial B0 v2 动作塌缩修复方案（修订稿）

> **状态**：draft-revised / 暂存。**不覆盖**原 draft（`...collapse-fix-design.md`），作为其修订版并存。
> **范围**：仅 aerial-wam 导航训练与闭环评测；不改 TT orbit / 真机避障路径。

## 修订说明（相对原 draft 的三处更正）

本轮把原始训练日志拉回本地重解析（`train_20260729_083642.log`，1083 个 logged step），并对 7 个 checkpoint 的 per-episode metrics 做了几何复算，纠正三处结论：

1. **loss 是单调下降的，不是「动作损失上升 / 动作头被饿死」。** 原判断基于失败的 CSV 抽取，误判 `loss_action` 上升。真实数据：`loss_action` 与 `loss_video` 都单调下降，`loss_action` 降得更快、全程更低。**7 个 checkpoint、约 8× 的 loss 下降，换来 0 的 SR** —— 排除「只是还没训够」，坐实是目标函数与标签分布的结构问题。
2. **只修终止（stop）最多把 SR 抬到 ~10%，不是「可能发现基线可用」。** 用余弦定理复算最近接近距离后，真实 oracle-stop 上限 OSR@20m = **2/20 = 10%**（此前用 `resid<20m` 粗估的 ~65% 已作废）。轨迹本身极少路过目标 20m 内 —— 终止只是必要条件，不是充分条件。
3. **SR≡0 有两个并列根因，必须一起修。** ①**从不终止**（stop 仅 2/2709）→ 飞满 100 步冲过目标；②**朝向错**（最近接近距离均值 62.7m，跨 checkpoint 不改善）→ 轨迹压根不路过目标。单修任一个都仍然 SR≈0。

---

## 0. 背景与失败证据

B0 v2 从头训（`max_steps=10000`，`save_every=500`，`λ_action=λ_video=1`），在 **step 5000** 由主人手动停训（SIGTERM）。闭环 seen-20（n=20，`SUCCESS_DIST=20m`，`max_steps=100`）：

| step | NE (m) | SR | SPL | 说明 |
|------|--------|----|----|------|
| 500  | 517.2 | 0 | 0 | |
| 1000 | 472.0 | 0 | 0 | |
| **1500** | **460.0** | 0 | 0 | |
| 2000 | 534.3 | 0 | 0 | |
| 2500 | 562.2 | 0 | 0 | |
| 3000 | 604.4 | 0 | 0 | fwd9-episode 占比升到最高 |
| 3500 | 394.6 | 0 | 0 | NE 最低点，仍 ≈3× 静止基线 |

- **每个 checkpoint 都 SR=0**；NE 高方差、无收敛趋势，**始终 3–5× 于「静止不动」基线（NE ≈ 124 m）**。
- `steps ≡ 100`（**从不终止**）；`path_mean − NE_mean ≡ 117`（常数）。
- 越训 forward 步长越大：fwd9-episode 计数 500→3000 为 `2→0→0→5→8→8`，平均每步位移与 NE 相关系数 ≈ 1.0 —— 训练在「把前进步子放大」，不是「朝目标走」。
- 两路 loss 单调下降（`loss_action` 更低、降更快）。**loss 降 = 动作头把 (dx,dy,dz,dyaw) 越回归越贴条件均值**，恰好塌到 fwd6/fwd9。loss 降与 NE 差**不矛盾**：L2 目标奖励的正是塌缩。

### 0.1 几何复算：为什么终止修好也只有 ~10% 上限

对 `step_001500` 逐 episode 用余弦定理算轨迹到目标的最近接近距离：

| 阈值 | 曾进入该半径的 episode（oracle-stop 上限 OSR） |
|------|------|
| ≤20m | **2/20 = 10%** |
| ≤30m | 5/20 = 25% |
| ≤40m | 9/20 = 45% |

- 最近接近距离：均值 **62.7m**、中位 50.6m、min 8.4、max 188.5。目标直线距离（shortest）均值 143.6m。
- 折算朝向误差 ~**26°**，且跨 checkpoint 不改善（训练只放大步长、不纠正朝向）。

**结论**：即便加上完美的「进 20m 就停」，SR 上限也只有 10%，因为 90% 的轨迹压根没路过目标 20m 内。**终止（根因①）与朝向/目标条件（根因②）是并列的，必须同修。**

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

修复只动「动作支」，其余不动。下图标出各改动的落点。

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

**动作头 vs 分类头**：现状「动作头」= `action_dit.py:98` 的 `self.head = nn.Linear(hidden, 4)`，输出连续 4D delta，`L_action` 是 flow-matching MSE，数学上奖励塌到条件均值。「分类头」= 把它换成 `Linear(hidden, 10)`，标签取专家 delta 的最近邻原语 ID，`L_action` 换成 CE（+class weight/focal），多峰分布下不塌到单一均值。

**关键点**：换分类头只改 `L_action` 的形状（回归均值 → 分类 argmax），`λ_video·L_video`、主干、两个 `pre_dit` 的 context 注入都不动。**它治「塌到均值」，治不了「目标盲」** —— 目标盲的病根在文本那条线太弱（硬图像锚压制 + 无 conditioning dropout，CFG 训不出来）。所以换头必须与目标条件化（下述 Stage 3）配套，缺一闭环仍 SR≈0。

---

## 1. 目标与成功标准

- **主目标**：seen-20 上 **SR > 0**，且 mean NE **严格好于**静止基线（≈124 m）与当前最差点（604 m）。
- **选点指标**：SR、NE、SPL、预测原语直方图、平均每步米数、非零 stop/done 计数、最近接近距离分布。
- **不做选点指标**：世界模型生成帧质量（PSNR/FID/主观好看）。
- **训练纪律**：以评测选点，不以「训满 N step」为目标；有效 epoch 量级控制在个位数。

---

## 2. 分阶段设计（已按「padding 是重训前提」重排顺序）

推荐执行顺序：**Stage 0 → 1 → 2 → 3 → 4**。上一阶段验收不过，不进入下一阶段。
（相对原 draft：把 padding/窗口修复提到重训之前，并把「目标条件化」升为一等阶段，与换头配套。）

### Stage 0 — 评测止血 + 几何诊断（不重训，最高信息量）

1. **终止条件**：除 `primitive == 0` 外，增加 `‖pos − goal‖ < SUCCESS_DIST` 即成功终止（`--oracle-stop`）；超时仍记失败。
2. **最近接近距离日志**：逐 step 记录到目标的距离，落 metrics（`closest_approach_m`、`oracle_hit@20/30/40`）。这是量化 OSR 上限、区分「终止问题」与「朝向问题」的关键。
3. **执行口径对照**：同一 ckpt 跑 nearest-primitive（现状）vs 连续 `step_delta`（若 bridge 支持）。
4. **复评 ckpt**：`step_001500`（NE 次优）、`step_003500`（NE 最低）、对照 `step_000500`。
5. **WM 诊断旁路（可选）**：1–2 episode 开 `--dump-wm-frames`（仅 step 0），人工看表征；**不入库为选点依据**。

**验收（已知预期）**：协议版本号写入 metrics key。预期 `--oracle-stop` 后 SR 抬到 ~10%（OSR@20m 上限），仍远低于目标 —— 即确认「终止是必要非充分」，直接进入 Stage 1+2+3 的重训闭环；若意外 SR 明显 >10%，再重新评估 baseline 可用性。

### Stage 1 — 数据 / 窗口 / padding 卫生（**重训前提**）

- `num_frames` 降到与中位 episode 匹配（建议 **9**；中位长度 13）。
- `skip_padding_as_possible: true`；padding 动作不进 action loss。
- 修复 / 重算 `dataset_stats`（杜绝 NaN std；若本轮一并改角度表征，按 v2 规格走 wrap + (sin, cos)）。

**验收**：有效动作占比显著上升；首步 `loss_action` / `loss_video` 有限、无 NaN。**不在污染标签上训新头。**

### Stage 2 — 动作头改分类 + 终止监督（消除 L2 塌缩 + 学会停）

- **动作头**：连续 4D delta → 最近邻 10 类原语 ID；主损失为 **分类**（CE + class weight 或 focal），不再用纯 L2 回归当主损失；连续 delta 仅作可选小权重辅助回归。
- **终止监督**：独立 done 头，或把「接近目标帧」标为 stop/done；保证 stop/done 有效样本远多于 2/2709。（评测侧距离终止已在 Stage 0 落地，此处补训练侧信号。）
- **保留** `λ_video > 0`（joint 世界模型正则）。

**验收**：held-out 预测原语分布不全塌到 fwd9；出现非零 stop/done；闭环 SR 达到 ~OSR 上限（终止已修、朝向未修时的天花板）。

### Stage 3 — 目标条件化（治「目标盲」，与 Stage 2 配套）

> **这是修复朝向误差（根因②）的核心，缺它闭环仍 SR≈10% 封顶。**

- **conditioning dropout**：训练时按概率丢弃文本/目标条件，逼模型真正利用 goal，而非只吃图像锚。
- **classifier-free guidance**：训好 uncond 分支后，推理用 CFG 放大目标条件影响。
- **弱化图像锚的独占**：评估首帧硬 pin 与目标条件的竞争关系；必要时调整锚的注入强度或加目标 token 的显式通路。
- **诊断**：复用 `wm_instruction_probe.py`（固定 obs+state、扫指令）验证「动作/生成帧随指令变化」，确认目标通路已激活。

**验收**：instruction-probe 显示动作随指令显著变化（含空指令基线可分）；闭环最近接近距离均值显著下降（目标 <30m 量级）；SR 突破 OSR 10% 天花板。

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
| 诊断旁路（`--dump-wm-frames` / `wm_instruction_probe.py`，低频） | **是，可选** |
| 训练目标 | 保持 `λ_video > 0` 即可 |

**理由**：当前失败是动作决策塌缩 + 目标盲；video loss 已在降，视频支路不是主瓶颈。代码已具备 dump 能力，无需新建主评测支路。

---

## 4. 明确不做

- 不把「训满 10000 step」当成功标准（8× loss 下降已证明多训无用）。
- 不把 WM 生成帧质量当主指标。
- 不在修动作头 / 终止 / 目标条件之前大规模扩同分布数据。
- 不单修终止就期待可用 SR（几何上限 10%）。
- 不改 orbit / TT 真机避障模块。
- 不从本 draft 直接开长训，须 Stage 0 验收后再定预算。

---

## 5. 与 v2 重设计文档的关系

本草案是对 `docs/design/2026-07-29-aerial-nav-wam-redesign.md` 的 **B0 失败后补丁**，不推翻 v2 的数据可信化与评测 fail-closed 原则：

- v2 **S1.0**（连续动作执行归因）→ 本草案 Stage 0。
- v2 **S1.2** joint `λ_video` → 保留。
- v2 角度 (sin, cos) / 分位数归一化 → 并入 Stage 1，或单开数据重生任务。

---

## 6. 待确认（落实施工前）

1. 整体是否按 **Stage 0（诊断）→ 1（padding）→ 2（分类头+终止）→ 3（目标条件化）→ 4（扩数据）** 采纳？（相对原 draft 把 padding 提前、并新增 Stage 3 目标条件化。）
2. Stage 2 是否锁定为 **10 类分类 + class weight**（连续回归仅可选辅助）？
3. Stage 3 的目标条件化是否与 Stage 2 **合并成一次重训**（省一轮训练，但牺牲单因子归因），还是分两次训以便分别验收？

确认后下一步：拆 `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md` 实现计划，先落 Stage 0 的零重训诊断代码（`run_closed_loop.py` 加 `--oracle-stop` + 最近接近距离日志 + 测试），再动模型。
