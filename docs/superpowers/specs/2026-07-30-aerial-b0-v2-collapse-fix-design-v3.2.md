---
title: Aerial B0 v2 动作塌缩修复方案（v3.2）
type: design
date: 2026-07-30
status: draft-v3.2
supersedes:
  - docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design.md
  - docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-revised.md
  - docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-v3.md
  - docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-v3.1.md
related:
  - docs/design/2026-07-29-aerial-nav-wam-redesign.md
  - docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md
  - docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md
  - docs/superpowers/plans/2026-07-30-aerial-collapse-fix-retrain-runbook.md
  - artifacts/b0_v2_20260729-b0v2-10k-2gpu/
hosts:
  train: "10.239.121.21:31126 (2×H100) — B0 v2 已停于 step_005000"
  eval: "10.239.121.23:30905"
  renderer: "10.229.20.125:41451"
run_stopped:
  id: m1b-20260729-b0v2-10k-2gpu
  last_ckpt: step_005000
  resume_state: checkpoints/state/step_005000/
code_status:
  stage0_oracle_stop: "已在 aerial-wam worktree 落地（run_closed_loop.py + tests，protocol=stage0-oracle-v1）；待真机复评与 probe 跑数"
---

# Aerial B0 v2 动作塌缩修复方案（v3.2）

> **状态**：draft-v3.2。相对 v3.1 的评审修订版；**取代**原 draft / revised / v3 / v3.1 作为当前首选规格（旧稿保留只读）。
> **范围**：仅 aerial-wam 导航训练与闭环评测；不改 TT orbit / 真机避障路径。
> **落地执行**：本规格的可执行重训 SOP 见
> [`plans/2026-07-30-aerial-collapse-fix-retrain-runbook.md`](../plans/2026-07-30-aerial-collapse-fix-retrain-runbook.md)（reconvert→preflight→smoke→train→eval，含三点不变量、normalizer 说明、`run_collapse_fix_retrain.sh` 启动器与 `d_max` 已知限制）；实现进度追踪见
> [`plans/2026-07-30-aerial-b0-v2-collapse-fix.md`](../plans/2026-07-30-aerial-b0-v2-collapse-fix.md)。

## 修订说明（相对 v3.1）

1. **修 `head_cls` 训推噪声错配（§0.4.1，核心）**：`head_cls` **感知时间步**（把流匹配噪声水平 `t` 的 embedding 并入输入）。训练在随机 `t~U`、推理在终端 `t_final` 都作为**同一个条件输入的不同取值**，不再是分布错配。删除 v3.1 里「必须在随机噪声分布上学会读特征」与推理读近干净 token 的自相矛盾表述。
2. **pooled 默认改 first-token（§0.4.1）**：标签只标「将执行的第一步」，故默认 `pooled = action_tokens[:, 0, :]`；mean / last-token 仅作消融。避免用整个 horizon 的特征稀释 step-0 信号。
3. **`d_max` 过滤豁免少数类（§0.4.1）**：转向 / 升降 / stop 标签样本**豁免** `d_max` 过滤（或按类施加）；`d_max` 只用于压 fwd6↔fwd9 边界噪声，不再进一步饿死稀有类。
4. **closest-approach 目标数值口径（§1 / Stage 3）**：§1 最终硬标准 = closest 中位 **< 40 m**；Stage 3 收敛目标 **< 30 m** 是**更严的中间目标**（非放宽、非笔误）。

> 承接 v3.1 的所有修订（统一方案 B 叙事、辅助流匹配默认保留、Stage 0 状态校准、可操作定义），此处不重复。

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
| **3500** | **394.6** | 0 | 0 | NE 最低点，仍远差于静止基线 |

- **每个 checkpoint 都 SR=0**；NE 高方差；本地复算：`steps ≡ 100`，`path_mean − NE_mean ≈ 116.8`（跨 ckpt 近似常数）。
- **静止基线**：若策略完全不动，NE = 起点到目标的**直线距离**均值（记为 `NE_stationary`）。该值与专家路径长 `shortest_mean≈143.6` **不是同一量**。历史口头量级 ≈124 m；Stage 0 复评时用 annotation 精确算并写入 metrics key。当前所有 ckpt 的 mean NE（395–604）均 **≥3×** 该量级。
- 越训 forward 步长越大；平均每步位移与 NE 相关系数 ≈ 1.0 —— 训练在「把前进步子放大」，不是「朝目标走」。
- 两路 loss 单调下降（`loss_action` 更低、降更快）。**loss 降 = 流匹配把连续动作越推越贴条件均值**，塌到 fwd6/fwd9。loss 降与 NE 差**不矛盾**。（loss 曲线 / fwd9-episode 计数来自训练日志离线解析；实现计划需挂脚本路径以便审计。）

### 0.1 几何复算：当前策略下终止修好也只有 ~10% 上限

对 `step_001500` 逐 episode 估计轨迹到目标的最近接近距离（离线几何复算；Stage 0 起以日志 `closest_approach_m` 为准）：

| 阈值 | 曾进入该半径的 episode（oracle-stop 上限 OSR） |
|------|------|
| ≤20m | **2/20 = 10%** |
| ≤30m | 5/20 = 25% |
| ≤40m | 9/20 = 45% |

- 最近接近：均值 **62.7m**、中位 50.6m。目标直线距离量级与 `shortest` 不同，见上。
- **结论**：对**当前塌缩策略**，完美「进 20m 就停」也只能到 SR≈10%。终止与朝向必须同修。
- **注意**：换分类头并学会转向后轨迹会变，OSR 会动；**不以 10% 作为换头后的 SR 天花板**。

### 0.2 训练集实测（`train_subset`）

- 200 episode / **2709** frames；平均长度 13.5、中位 13；**66% episode 短于 `num_frames=17`**。
- `skip_padding_as_possible: false` → 大量 padding 动作进 loss。
- 专家动作 → 最近邻原语：fwd9 50.8%、fwd3 26.0%、fwd6 15.4%、**stop 0.1%（2 样本）**；转向/升降合计个位数百分比。
- dx 均值 6.25；dy/dz/dyaw 零值占比 88% / 96% / 86%。连续 L2/流匹配的条件均值落在 fwd6/fwd9 → **塌缩是目标与标签分布的数学结果**；同分布裸扩只会塌得更死。

### 0.3 模型结构与改动落点（方案 B）

```
输入                     编码器                    主干(MoT)                         头                      输出
──────────────────────────────────────────────────────────────────────────────────────────────────────────────

观测图像 ─VAE─► first_frame_latents ─┐ 硬 pin latents_video[:,:,0]=first     （图像锚；是否过强待 probe）
                                      │
latents_video ────────► video_expert ─┤
                                      ├─► MoT 混合注意力 ─┬─► video post_dit ─► VAE.decode ─► imagined 帧
文本指令 ─T5─► context ───────────────┤                  │
（目标位置主要在文本里；是否生效待 probe） │                  │
proprio ─encoder─► append context ────┤                  │
                                      │                  └─► action post_dit tokens
latents_action ───────► action_expert ┘                       ├─► head_fm: Linear(H,4)          ─► 速度/delta（流匹配）
                                                              └─► head_cls: (tokens, t)→10 logits【新增，感知时间步】
                                                                     └─► 原语 logits

损失:  λ_video·L_video  +  λ_ce·CE(cls, prim_id)  +  λ_fm·L_fm(action)
        └─ 保留 joint 正则      └─ 主动作损失（分类）     └─ 默认保留、小权重（MoT 耦合）
```

- **不做方案 A**：不把动作支改成单步分类、不拆掉 action 噪声 latent / 去噪循环（与 MoT 逐步耦合冲突，改动面大）。
- **目标位置**主要经文本进入；图注不再断言「弱」，改由 Stage 0 probe 验证。

### 0.4 方案 B：保留流匹配 + 旁路分类头（锁定）

- 保留 `action_dit.py` 的 `self.head`（下称 `head_fm`）与 `infer_action` 去噪循环。
- **新增** `head_cls`（**感知时间步**）：读 action 支 post_dit 特征 **+ 流匹配噪声水平 `t`**，输出 10 类原语 logits。
- 训练：`λ_ce` 为主（建议初始 `λ_ce=1.0`），`λ_fm` 默认保留且 **λ_fm ≪ λ_ce**（建议初始 `0.1`，可扫）；`λ_video` 保持 `>0`（建议 `1.0`）。
- 推理：**执行原语 = `argmax(head_cls)`**；忽略连续 delta / 不再做 nearest-primitive。

#### 0.4.1 训推前向（写死，实现必须遵守）

**特征定义**

- `action_tokens`: MoT/`action_expert` 在当前 action 时间步上的 post_dit 隐状态，形状 `[B, T_action, H]`（`T_action = action_horizon`）。
- `pooled = action_tokens[:, 0, :]` → `[B, H]`。**默认取第一个 token**，对齐「只执行第一步」的标签；mean / last-token 仅作消融，不作默认。
- **`head_cls` 感知时间步**：`logits = head_cls(pooled, t)` → `[B, 10]`，其中 `t` 是该次 action 支前向的流匹配噪声水平。实现上把 `t` 的 embedding（复用 DiT 同款 timestep embedding）并入 `head_cls` 输入（拼接或 FiLM 皆可，实现计划定）。

**标签**

- 对专家连续动作 chunk，取**将执行的第一步**（与现评测 `replan_steps=1` 对齐）做 `delta_to_nearest_primitive` → `prim_id ∈ {0..9}`。
- 近邻距离（scaled L2）高于阈值 `d_max` 的样本：**不进 CE**（仍可进 `L_fm`），降低 fwd6↔fwd9 边界噪声。默认 `d_max` 取训练集近邻距离的 p90，Stage 1 统计后写入 config。
- **少数类豁免**：标签为**转向 / 升降 / stop** 的样本**不受 `d_max` 过滤**（或对少数类单独放宽 `d_max`）。`d_max` 只用于压前进原语（fwd3/6/9）间的边界噪声，不得进一步饿死本就稀有的转向 / 升降 / stop 类。
- CE 使用 class weight 或 focal；另加 label smoothing（建议 0.05）。

**训练前向（每个 train step）**

1. 按现有 joint 流程采样噪声、跑 MoT，计算 `L_video`、`L_fm`（`head_fm` 速度/目标，**不变**）；记录该次 action 支的噪声水平 `t`。
2. 在**同一次** action 支前向得到的 `action_tokens` 上取 `pooled`，算 `logits = head_cls(pooled, t)`。
3. `L_ce = CE(logits, prim_id)`（仅非 padding、且经少数类豁免后的 `d_max` 过滤、且经 stop 重标后的样本）。
4. `L = λ_video·L_video + λ_ce·L_ce + λ_fm·L_fm`。

> 说明：`head_cls` **以 `t` 为条件**读特征。训练时 `t~U` 遍历噪声全程（含推理所用的终端低噪 `t≈0`），推理时以终端 `t_final` 查询——两者是**同一条件输入的不同取值**，不构成训推分布错配。因此**不另开**「干净动作编码」旁路（那会分裂特征分布），也**不允许** `head_cls` 无视 `t` 而训随机噪声、推近干净。

**推理前向（闭环每步）**

1. 仍跑完整 `infer_action` 去噪循环（与现网关耦合一致；`return_video` 默认 false）。
2. 在**最后一次** `_predict_joint_noise` 得到的 `action_tokens`（对应终端噪声水平 `t_final`）上取 `pooled`，算 `logits = head_cls(pooled, t_final)`（实现上需让该次前向把 tokens 与 `t_final` 暴露给 `head_cls`；可缓存 last-step tokens）。
3. `primitive_id = argmax(logits)`；**不**再对 denoised delta 做 nearest-primitive。
4. bridge 执行 `step(primitive_id)`。

**禁止**

- 推理时跳过去噪、只跑单步分类（那是方案 A）。
- `head_cls` 无视时间步 `t`：训练用随机噪声特征、推理用近干净特征而不作 `t` 条件（v3.1 的隐患，v3.2 已修）。
- 训练用干净动作特征、推理用噪声特征（分布不一致）。

---

## 1. 目标与成功标准

- **最终成功（硬）**（seen-20，固定 seed + `protocol_version`）：
  1. **SR ≥ 0.15**（至少 3/20；避免「碰巧 1 个」）；
  2. **mean NE < NE_stationary**（严格好于静止）；
  3. **closest_approach 中位 < 40 m**（Stage 3 的收敛目标 **< 30 m** 更严，见下，非放宽）。
- **选点指标**：SR、NE、SPL、原语直方图、平均每步米数、非零 stop 计数、`closest_approach` 分布 / `oracle_hit@20/30/40`。
- **不做选点指标**：WM 生成帧 PSNR/FID/主观好看。
- **训练纪律**：以评测选点；有效 epoch 控制在个位数；不以训满 N step 为目标。

---

## 2. 分阶段设计

顺序：**Stage 0（跑数）→ 1（数据卫生）→ (2+3 合并重训) → 4（扩数据）**。上阶段验收不过不进入下阶段。

### Stage 0 — 评测止血 + 几何诊断 + 指令探针（不重训）

**代码状态**：`aerial-wam` worktree 已实现 `--oracle-stop`、`closest_approach`、`oracle_hit@*`、`terminated_by`、`protocol_version=stage0-oracle-v1` 及单测。本阶段交付是**跑数与结论**，不是再写一遍 oracle-stop。

1. **复评**（`oracle_stop=true`）：`step_001500`、`step_003500`、对照 `step_000500`；确认 SR 是否落在 ~OSR@20m（旧策略预期 ~10%）。
2. **落盘几何**：`closest_approach_m`、`oracle_hit@20/30/40`、`NE_stationary`（由 annotation 起点–终点直线距离均值）。
3. **强制 `wm_instruction_probe`**：对 `step_001500` / `step_003500`；指令集含空字符串 + ≥4 条语义不同导航指令；固定 obs/state/seed。
4. **`step_delta` 对照（可选，不进门禁）**：闭环 `Bridge` 协议目前只有 `step(primitive_id)`；连续执行仅在 dagger/oracle 路径以 `getattr(bridge,"step_delta")` 出现。**若要做对照，须先给 openfly bridge 加 `step_delta`**；否则跳过，不阻塞 Stage 0 验收。
5. **WM dump（可选）**：1–2 episode `--dump-wm-frames`（step 0）；不入库选点。

**Probe「显著变化」操作定义（锁定）**

- 设指令列表长度 N（含空指令），得到原语序列 `p[0..N)`。
- **不敏感**：`|set(p)| == 1`（全部相同，含全等于空指令结果）。
- **敏感**：`|set(p)| ≥ 3`，或非空指令与空指令原语不同且至少两个非空指令原语不同。
- 介于中间 → 记 `partial`，Stage 3 仅加轻量 CFG，不做弱化图像锚。

**Stage 0 → Stage 3 分支**

| probe | Stage 3 |
|-------|---------|
| 不敏感 | conditioning dropout + CFG；弱化首帧 pin 仅作后续 A/B，默认第一刀不加 |
| partial | 仅轻量 CFG；不加 dropout 满配、不弱化 pin |
| 敏感 | **不做**弱化 pin；依赖分类头 + stop 重标 + 数据；闭环朝向仍差再加轻量 CFG |

**验收**：复评 metrics 与 probe `summary.json` 落盘；oracle-stop SR 与几何诊断一致后进入 Stage 1。

### Stage 1 — 数据 / 窗口 / padding 卫生（重训前提）

- `num_frames` → **9**（中位 episode 长 13；且满足 VAE `T % 4 == 1`：9=4·2+1）。
- `skip_padding_as_possible: true`；padding 帧不进 `L_ce` / `L_fm`。
- 重算 `dataset_stats`（无 NaN std）；角度 (sin,cos) / 分位数归一化按 v2 **本轮可并行，不阻塞分类头**（分类标签是原语 ID）。
- 统计近邻距离分布，写入 `d_max`（默认 p90）；并单列转向 / 升降 / stop 的近邻距离分布，供少数类豁免阈值参考。

**验收**：有效动作占比显著上升；1-step/10-step smoke：`L_ce`/`L_fm`/`L_video` 有限、无 NaN。

### Stage 2 — 分类头 + 终止监督（按 §0.4 / §0.4.1）

- 落地 `head_cls`（**感知时间步**）、损失权重、训推前向。
- **stop 重标（默认，锁定规则）**：
  - 专家轨迹上，若某帧 `‖pos − goal‖ < R_stop`，将该帧动作标签改为 `stop (0)`；
  - `R_stop = OPENFLY_SUCCESS_DIST_M`（20 m）；
  - 每条 episode **最后一帧强制 stop**（即使距离 ≥ R_stop）；
  - 重标后再算 class weight。
- **done 头（后备）**：仅当 stop 重标后闭环仍几乎从不停时再加。终止优先级：**评测诊断** `oracle-stop`（真值目标，非部署能力）> done 头 > stop 类。
- `λ_fm` **默认保留**（小权重）；推理不用连续 delta。

**验收（不绑 OSR=10%）**

1. held-out / 闭环原语直方图：fwd9 显著低于 50.8% 先验，转向或升降非零；
2. 非零 stop 计数出现；
3. `closest_approach` 均值相对同协议塌缩 baseline（step_1500/3500）下降 —— **≥20% 为占位，Stage 0 给出分布后校准**；
4. SR>0 为加分，非本阶段硬门禁。

### Stage 3 — 目标条件化（力度由 Stage 0 probe 决定）

- 满配（仅「不敏感」）：conditioning dropout + CFG（`text_cfg_scale` 已存在）。
- 弱化 first-frame pin：**默认不做**；仅满配后 probe 仍失败再 A/B。
- 验收：probe 达「敏感」；`closest_approach` 中位进入更小桶（**目标 < 30 m**，比 §1 最终 40m 更严，作为 Stage 3 收敛目标；或相对仅 Stage2 再明显下降）；SR 持续 >0 并迈向 §1 硬标准。

### Stage 2+3 默认合并一次重训

- Stage 1 完成后一次重训：分类头 + stop 重标 +（按分支的）条件化改动。
- 归因：原语直方图 / probe / closest_approach 三分开看。
- 例外：合并后 probe 已敏感、原语已发散、接近仍差 → 只做轻量 CFG/goal 强化，不从零再满训。

### Stage 4 — 扩数据（禁止同分布裸扩）

- 放宽 `env_prefixes`；目标 **≥2k episodes 或 ≥2e4 frames**。
- 配额抬高 stop / 转向 / 近目标帧；`route_ids` 与 seen-20 leak-free。
- 训 ≤5 epoch 量级即闭环选点。

---

## 3. 世界模型生成帧评测支路

| 角色 | 是否做 |
|------|--------|
| 选点 / 门禁 | **否** |
| `wm_instruction_probe` | Stage 0 **强制** |
| `--dump-wm-frames` | 可选诊断 |
| `λ_video > 0` | 保持 |

无需新建以生成帧质量为主的评测支路。

---

## 4. 明确不做

- 不以训满 10000 step 为成功标准。
- 不以 WM 帧质量为选点指标。
- 不在修动作头/终止/条件化之前同分布裸扩数据。
- 不单修终止就期待可用 SR。
- 不在 Stage 0 probe 之前弱化 first-frame pin。
- 不以 OSR=10% 为换头后 SR 天花板。
- 不采用方案 A（拆掉 action 去噪循环）。
- 不让 `head_cls` 无视时间步 `t`（训随机噪声、推近干净）。
- 不改 orbit / TT 真机避障。
- 不跳过 Stage 0 跑数直接开长训。

---

## 5. 与 v2 重设计文档的关系

- v2 S1.0 连续执行归因 → Stage 0 可选 `step_delta`（须先补 bridge）；不进门禁。
- v2 S1.2 joint `λ_video` → 保留。
- v2 角度 (sin,cos) / 分位数 → Stage 1 可并行，不阻塞分类头。

---

## 6. 已锁定取舍

| # | 项 | 锁定 |
|---|----|------|
| 1 | 阶段顺序 | 0（跑数）→ 1 → (2+3 合并) → 4 |
| 2 | 换头 | **方案 B**：`head_fm` 保留 + `head_cls`（**感知时间步**）新增；训推见 §0.4.1 |
| 3 | 分类特征 | `pooled = action_tokens[:,0,:]`（first-token）；`head_cls(pooled, t)`，`t` 为流匹配噪声水平 |
| 4 | 损失 | `λ_ce` 主；`λ_fm` 默认小权重保留；`λ_video>0` |
| 5 | 推理 | `argmax(cls)`；不用连续 delta / nearest-primitive |
| 6 | 标签过滤 | `d_max`（p90）压 fwd6↔fwd9 边界；转向/升降/stop 少数类**豁免** |
| 7 | 终止监督 | stop 重标（R=20m + 末帧强制）；done 头后备 |
| 8 | Stage 3 | 由 probe 三档分支；默认不弱化 pin |
| 9 | 最终成功 | SR≥0.15 且 NE<NE_stationary 且 closest 中位<40m（Stage 3 收敛目标<30m）|

---

## 7. 下一步

1. 在 eval 机跑 Stage 0：oracle-stop 复评三 ckpt + 两遍 `wm_instruction_probe`，落盘并填 `NE_stationary`。
2. 拆实现计划 `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md`（Stage 1 数据卫生 → 方案 B 模型/损失，含 `head_cls` timestep embedding 接线 → 合并重训配方）。
3. 审计附件：挂上 loss 曲线与 OSR 几何复算脚本路径（若尚未进仓，补进 `artifacts/b0_v2_.../`）。
