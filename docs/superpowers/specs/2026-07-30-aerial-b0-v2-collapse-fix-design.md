---
title: Aerial B0 v2 动作塌缩修复方案（暂存稿）
type: design
date: 2026-07-30
status: draft
supersedes: []
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

# Aerial B0 v2 动作塌缩修复方案（暂存稿）

> **状态**：draft / 暂存。按 Stage 0→4 执行前需主人确认两处取舍（见文末「待确认」）。
> **范围**：仅 aerial-wam 导航训练与闭环评测；不改 TT orbit / 真机避障路径。

## 0. 背景与失败证据

B0 v2 从头训（`max_steps=10000`，`save_every=500`，`λ_action=λ_video=1`）在 **step 5000** 停训。闭环 seen-20 上：

| step | NE | SR | SPL |
|------|-----|----|----|
| 500 | 517.2 | 0 | 0 |
| 1000 | 472.0 | 0 | 0 |
| **1500** | **460.0** | 0 | 0 |
| 2000 | 534.3 | 0 | 0 |
| 2500 | 562.2 | 0 | 0 |
| 3000 | 604.4 | 0 | 0 |

NE 在 step 1500 后单调变差；平均每步位移与 NE 相关系数 ≈ 1.0。模型表现差于「静止不动」基线（NE ≈ 124 m）。

### 训练集实测（`train_subset`）

- 200 episode / **2709** frames；平均长度 13.5，中位数 13；**66% episode 短于 `num_frames=17`**。
- `skip_padding_as_possible: false` → 大量 padding 动作进 loss。
- 专家动作 → 最近邻原语：

| 原语 | 占比 | 原语 | 占比 |
|------|------|------|------|
| fwd9 | 50.8% | left30 | 2.6% |
| fwd3 | 26.0% | right30 | 2.1% |
| fwd6 | 15.4% | down3 / up3 | 1.8% / 1.0% |
| **stop** | **0.1%（2 样本）** | left3 / right3 | 各 0.1% |

dx 均值 6.25；dy/dz/dyaw 零值占比 88% / 96% / 86%。连续 L2 回归的条件均值落在 fwd6/fwd9 附近 → **mode collapse 是目标函数与标签分布的数学结果**，不是「还没训够」。

### 根因排序（不是「只差数据量」）

1. **终止依赖 stop 原语**，而 stop 仅 2/2709 → 几乎从不 `break`，跑满 `max_steps=100` 冲过目标 → SR≡0，NE≈飞出距离。
2. **L2 回归 + nearest-primitive** 在 92% 前进主导的多峰分布上必然塌缩；训越久越贴 fwd9。
3. **小集 + 长训**（~30 epoch @5k）过拟合放大塌缩；padding 污染监督。
4. 数据量本身偏小，但是**第三位**；同分布裸扩只会让条件均值估得更准（塌得更死）。

---

## 1. 路径对比与推荐

| | A. 评测侧止血 → 再改模型 | B. 直接换分类动作头重训 | C. 先扩数据再重训 |
|---|---|---|---|
| 第一步 | 距离终止 + 连续动作复评 step_1500 | 10 类分类 + class weight | 扩到数千 ep |
| 成本 | 低 | 中 | 高且无效 |
| 风险 | 可能发现基线可用 | 不修终止仍 SR≈0 | 同分布放大塌缩 |
| 结论 | **采用（主路径）** | Stage 2 | **否决作首步** |

推荐执行顺序：**Stage 0 → 1 → 2 → 3 → 4**。上一阶段验收不过，不进入下一阶段。

---

## 2. 目标与成功标准

- **主目标**：seen-20 上 **SR > 0**，且 mean NE **严格好于**静止基线（≈124 m）与当前最差点（604 m）。
- **选点指标**：SR、NE、SPL、预测原语直方图、平均每步米数。
- **不做选点指标**：世界模型生成帧质量（PSNR/FID/主观好看）。
- **训练纪律**：以评测选点，不以「训满 N step」为目标；有效 epoch 量级控制在个位数。

---

## 3. 分阶段设计

### Stage 0 — 评测止血（不重训，最高信息量）

1. **终止条件**：除 `primitive == 0` 外，增加 `‖pos − goal‖ < SUCCESS_DIST` 即成功终止；超时仍记失败。
2. **执行口径对照**：同一 ckpt 跑两路 —— nearest-primitive（现状）vs 连续 `step_delta`（若 bridge 支持）。
3. **复评 ckpt**：优先 `step_001500`（历史最佳 NE）；对照 `step_000500`、`step_003000`。
4. **WM 诊断旁路（可选）**：1–2 个 episode 开 `--dump-wm-frames`（仅 step 0），人工看表征是否崩；**不入库为选点依据**。

**验收**：协议版本号固定并写入 metrics key；若连续执行或距离终止后 SR>0 → 可收缩重训规模；若仍 SR=0 且动作几乎全是 fwd9 → 进入 Stage 1/2。

### Stage 1 — 终止 / stop 监督

- **1a（评测侧，与 Stage 0 合并）**：距离终止，不依赖 stop 标签。**优先落地。**
- **1b（训练侧，按需）**：独立 done 头，或把「接近目标帧」标为 stop/done；保证 stop/done 有效样本远多于 2/2709。

**验收**：专家回放 / oracle 在同一协议下 SR≥0.8；策略至少出现非零 stop/done **或** 距离终止被触发。

### Stage 2 — 动作头：消除 L2 塌缩

- **监督**：连续 4D delta → 最近邻 10 类原语 ID；主损失为 **分类**（CE + class weight 或 focal），不再用纯 L2 回归当主损失。
- **推理**：直接输出 class → primitive；连续 delta 仅作可选小权重辅助回归。
- **保留**：`λ_video > 0`（joint 世界模型正则）。

**验收**：held-out 预测原语分布不全塌到 fwd9；闭环出现转向/升降；NE 低于静止基线。

### Stage 3 — 窗口与 padding

- `num_frames` 降到与中位 episode 匹配（建议 **9**；中位长度 13）。
- `skip_padding_as_possible: true`；padding 动作不进 action loss。
- 修复 / 重算 `dataset_stats`（杜绝 NaN std；若本轮一并改角度表征，按 v2 规格走 wrap + (sin, cos)）。

**验收**：有效动作占比显著上升；首步 `loss_action` / `loss_video` 有限、无 NaN。

### Stage 4 — 扩数据（禁止同分布裸扩）

- 放宽 `download_openfly_subset.py` 的 `env_prefixes`；目标量级建议 **≥2k episodes 或 ≥2e4 frames**（以 OpenFly 可下池为准）。
- **配额采样**：抬高 stop / 转向 / 近目标帧占比，禁止只堆 fwd9。
- leak-free：held-out seen-20 与 train 的 `route_ids` 不相交。

**验收**：新集 stop 占比达数百分点级；训 ≤5 epoch 量级即做闭环选点。

---

## 4. 世界模型生成帧评测支路 — 结论

| 角色 | 是否做 |
|------|--------|
| 选点 / 门禁 / early-stop | **否** |
| 诊断旁路（`--dump-wm-frames`，低频） | **是，可选** |
| 训练目标 | 保持 `λ_video > 0` 即可 |

**理由**：当前失败是动作决策塌缩；video loss 已在降，视频支路不是主瓶颈。代码已具备 dump 能力（`policy_fastwam.dump_video` / `run_closed_loop --dump-wm-frames`），无需新建主评测支路。

---

## 5. 明确不做

- 不把「训满 10000 step」当成功标准。
- 不把 WM 生成帧质量当主指标。
- 不在修动作头 / 终止条件之前大规模扩同分布数据。
- 不改 orbit / TT 真机避障模块。
- 不从本 draft 直接开长训，须 Stage 0 验收后再定预算。

---

## 6. 与 v2 重设计文档的关系

本草案是对 `docs/design/2026-07-29-aerial-nav-wam-redesign.md` 的 **B0 失败后补丁**，不推翻 v2 的 Stage 0 数据可信化与评测 fail-closed 原则。优先对齐点：

- v2 **S1.0**（连续动作执行归因）→ 本草案 Stage 0。
- v2 **S1.2** joint `λ_video` → 保留。
- v2 角度 (sin, cos) / 分位数归一化 → 可与 Stage 3 合并，或单开数据重生任务；本草案不强制首轮就做完。

---

## 7. 待确认（落实施工前）

1. 整体是否按 **Stage 0→1→2→3→4** 采纳？
2. Stage 2 是否锁定为 **10 类分类 + class weight**（连续回归仅可选辅助），还是先只做「连续执行 + 距离终止、暂不改头」再视 Stage 0 结果决定？

确认后下一步：拆 `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md` 实现计划，再改代码。
