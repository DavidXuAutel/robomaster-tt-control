# Aerial WAM RL 方案优化：DreamerV3 对齐与消融

**状态：** REVIEWED（引用已对原文核实，2026-08-04）
**日期：** 2026-08-04
**适用分支/worktree：** `aerial-rl-skeleton`
**上游依赖：**
- `docs/superpowers/specs/2026-08-03-aerial-wam-pure-vision-design-v2.md`（v2 纯视觉设计，本文的母方案）
- `configs/aerial_rl.yaml`（model-based RL 骨架 config）
- `experiments/aerial/rl/`（imagination / collector / corrector 骨架）

**参考论文：** Romero et al., *Dream to Fly: Model-Based RL for Vision-Based Drone Flight*, ICRA 2026 Vienna（RPG/UZH，DreamerV3 应用于纯像素无人机竞速）；配套引用 Geles et al. RSS'24 [10]、Xing et al. CoRL'24 [12]。

> **核实记录（2026-08-04）：** 本文引用的每个具体数字均已对 `~/Documents/ICRA26_Romero.pdf` 原文核对无误——T=16 想象 horizon、b2 从 0 起、总回报过 50.0 后拉起、HIL 9 m/s（habitat 渲染在环）、零视角奖励下注视涌现、PPO/SAC 在 20M 步内不收敛、λ-return AC + 熵正则、重建/decoder 为表征损失主干、收敛需 ~240 h。**唯一读原文才暴露的缺口是控制接口/任务域差（见 §1.5），它决定了"算法配方可整体抄、经验数字需重新验证"的边界。**

---

## 0. 一句话

现有 RL 方案本身已是 **Dreamer 式 model-based imagination RL**，且已与 Fast-WAM 主论点（"像素预测只用于预训练/蒸馏，在线不做逐步像素规划"）对齐。本文不改方向，只做两件事：**(1) 从 DreamerV3 抄一批成熟的世界模型/策略训练配方，省炼丹、防塌缩；(2) 定义两个能反哺 Fast-WAM 主论文的消融。** 所有内容以 §0.5 的 Fork A（AirSim 可用）为前提，优先级低于该 GO/NO-GO。

---

## 1. 为什么是 model-based，而不是 model-free PPO

结论先行：本方案选 model-based imagination 是对的，不要退回 model-free PPO/GRPO。

- AirSim 闭环跨网 **~12–14 Hz RGB-only**（`configs/aerial_rl.yaml: env.step_hz=12`），真环境采样极慢。
- 慢环境下**样本效率就是一切**——这正是 MBRL 相对 model-free 的核心优势，也是 Dream to Fly 全文的论据（PPO/SAC 在 2000 万步内学不出任何有意义飞行，DreamerV3 收敛）。
- 因此本方案的原生策略优化算法是 **Dreamer 的 imagination actor-critic**，不是 PPO。

> 说明：此前讨论里对操作类/离散 OpenFly 提过 model-free PPO/GRPO；在本方案（model-based + 12 Hz 慢环）语境下作废，以本文为准。

---

## 1.5 控制接口与任务域差（决定经验数字能抄多少）

**这是读原文才暴露、却决定全文可迁移边界的一节。** Dream to Fly 与本方案在两个维度上处于不同 regime，务必区分"抄算法配方（可整体迁移）"与"抄经验数字（需重新验证）"：

| 维度 | Dream to Fly | 本方案（aerial WAM v2） |
|------|--------------|------------------------|
| 动作空间 | **CTBR** `a=[c, ωx, ωy, ωz]`（集体推力 + 机体角速率），低层敏捷 | **4-D 运动学增量** `(dx,dy,dz,dyaw)`，经 `moveByVelocityAsync` 下发 |
| 任务 | 门定义航迹的竞速，目标（门）基本可见，9 m/s | 长时 SEARCH，目标不可见，慢速导航 |
| 控制率 | 高（实时机体率控制） | 闭环 ~12 Hz（跨网瓶颈） |

**这带来三条硬边界：**

1. **`b2` 惩罚的是机体角速率 `‖ω‖`——本方案动作里没有角速率这一维。** 你们的 `w_maneuver` 惩罚的是速度增量幅值。§2.4 的课程**机制**可整体抄，但"被惩罚的物理量"不同；且课程阈值 **`total reward > 50.0` 是论文的奖励尺度**，本方案尺度不同，**只能抄"从 0 起、达标后拉起"的机制，抄不了绝对阈值 50**。
2. **T=16 是在其控制率/敏捷动力学下的平衡。** 在本方案 12 Hz 运动学下，16 步 ≈ 1.3 s 前瞻，物理含义不同——可以试，但"抄 16"是 regime-dependent 的（另见 §2.6 关于实际 cap 的修正）。
3. **注视涌现（§3.2 消融 B）是"门定义航迹 + CTBR 敏捷"下的产物。** 在"目标不可见的 SEARCH + 运动学速度控制"下是否同样自发涌现，是真正 open 的问题——这正是把 B 定位成**消融**而非替换的根本原因。

**一句话：DreamerV3 的训练配方（§2.1）、λ-AC（§2.5）、reward 结构（§2.4）与动作/任务无关，可放心迁移；T=16、阈值 50、gaze 涌现这三个经验结论跨 regime，必须在本方案内重新验证。**

---

## 2. 直接采纳项（P0/P1）

### 2.1 [P0] 世界模型训练配方：抄 DreamerV3 原版

`[4]` 快 latent 世界模型 `(z_{t+1}, p_coll, progress, done)=f(z_t,a_t)` 本质就是 DreamerV3 的 RSSM + reward/continue 头。DreamerV3 "跨域免调参"靠的是一套具体 trick，直接搬：

| Trick | 作用 | 落到本方案 |
|-------|------|-----------|
| **symlog 变换** 预测 progress/reward | 压缩大尺度回归目标，稳 | progress、success 相关标量头 |
| **two-hot 离散编码**回归目标 | 降方差、防发散 | 同上 |
| **free-bits + KL balancing** | 表征损失 vs 动力学损失分开配比 | 现在 loss 权重全 1.0，需拆 |
| **回报百分位归一化** | actor 训练跨奖励尺度稳定探索 | 用于 actor 的 λ-return 归一化 |
| **固定 loss 权重** βpred=1, βdyn=1, βrep=0.1 | 免调 | 直接用 |

**验收关联：** 命中 v2 §7 V0 的"非塌缩"过关信号。

> **易混点（务必区分两个不同问题）：** DreamerV3 的 percentile return normalization 归一化的是 **actor 看到的总回报/优势**（原文："to ensure robust exploration across diverse reward **scales**"），解决的是跨任务/跨阶段的回报尺度稳定；它**不解决**你 `w_progress/w_collision/success_bonus` 这几个**奖励分量之间**的配比——那是 reward design 问题。证据：论文自己在 Eq.(1) 里仍用**固定** `b1=1.0, b2=0.01` + 独立的 collision/passed 项，并没有指望 return norm 去救分量权重。别把这两件事合并。

### 2.2 [P0] 把 `[4]` 做成真正的 RSSM（补确定性循环态 h）

- 任务是 **POMDP + 长时目标不可见（SEARCH）**。DreamerV3 的 RSSM 有**确定性循环态 `h` + 随机态 `z`** 两条路，`h` 专扛部分可观。
- 现方案写的是 `z_{t+1}=f(z_t,a_t)`（无循环态）→ 在目标遮挡段想象 rollout 会退化。
- **动作：** `[4]` 补 `h_t`。它在**短程控制层面**扛遮挡记忆，与 `[2]` 拓扑记忆（长程）分工，不冲突。

**验收关联：** v2 §7 V1（多步 rollout 达标）。

### 2.3 [P1] 复用 DreamerV3 标准 decoder 兼作"塌缩探测器"

- **先纠一处重复计费：** 原文 Eq.(7) 里 decoder/重建**是 DreamerV3 表征损失的主干**（"ensures the latent variables retain essential information"），始终在训练期开启，**不是可选附件**。所以只要抄了 §2.1 的 DreamerV3 配方，这个重建头**自动就有了**——本节不新增组件，只是明确"标准 decoder 顺便当塌缩早警用"。
- Dream to Fly Fig.4：重建质量随训练阶段（0.4M→1M→10M 步）变好，与策略能力相关。
- 本方案的真实取舍：`[4]` 从 Wan2.2 蒸馏、**在线不做像素**——所以 decoder 只在训练期在环，作为：
  - 表征信息充分性的锚（DreamerV3 原义）；
  - **免费的塌缩早警**：重建突然变糊/发散 = 世界模型崩了，比等 progress 指标崩更早发现。

**验收关联：** v2 §7 V0"非塌缩"，零额外成本（配方自带）命中。

### 2.4 [P1] 机动惩罚上课程

- **意外确认——你们的 reward 本就是 Dream to Fly 的 reward。** 原文 Eq.(1)：
  `rk = r_collision | r_passed | b1·(‖g−p_{k-1}‖−‖g−p_k‖) − b2·‖ω‖`，系数 `b1=1.0, b2=0.01`。
  对照 `aerial_rl.yaml`：`w_progress=1.0(=b1)`、`w_maneuver=0.01(=b2)`、`success_bonus=10.0(=r_passed)` 三项**系数逐字相同**；`w_collision=10.0` 与论文 `r_collision=−4.0` 量级/符号处理不同（本方案自定，非抄论文）。总体是同一 reward 结构，其中被 anneal 的 `b2` 尤其就是论文那一项。
- Dream to Fly：该系数 `b2` **从 0.0 开始，总回报过阈（原文 50.0）后渐增**，早期不压探索。
- 现 `configs/aerial_rl.yaml: reward.w_maneuver=0.01` 全程固定。
- **动作：** `w_maneuver` 做成 curriculum（progress/SR 过阈值后再拉起）。
- **两条跨 regime 注意（详见 §1.5）：**
  1. 论文 `b2` 惩罚**机体角速率 `‖ω‖`**（CTBR 动作）；本方案动作无角速率维，`w_maneuver` 惩罚的是**速度增量幅值**——机制可抄，物理量不同。
  2. 阈值 **50.0 是论文的奖励尺度，不可平移**；本方案要用自己的 progress/SR 曲线定阈，只抄"从 0 起、达标后线性拉起"的机制。

### 2.5 [P1] imagination actor-critic 用 Dreamer 原版

- V4（`corrector.enable_policy_update`）的策略更新用 **λ-return + REINFORCE + 回报归一化 + 熵正则 + value 目标 stop-grad**（DreamerV3 原样），不要自造 PPO。
- 它就是为"在学出来的世界模型里训策略"设计的，本设定下比 PPO 更稳、更省样本。

**验收关联：** v2 §7 V4（超越 BC 天花板）。

### 2.6 [P2] imagination horizon 10 → 15（→16 需抬 cap）

- Dream to Fly 用 **T=16**，明确是"算力与长时依赖的平衡"（注意其含义随控制率而变，见 §1.5：本方案 12 Hz 下 16 步 ≈ 1.3 s 前瞻）。
- 现 `imagination.horizon=10`。**实际硬上限是 `imagination.py:26 MAX_IMAGINATION_HORIZON=15`，不是 16**——代码在 `horizon>15` 时会直接报错并要求显式抬 cap。
- **动作：** V1 多步误差门禁通过后，先把 horizon 试到 **15**；若要到 16，**必须同时把 `MAX_IMAGINATION_HORIZON`/`horizon_max` 显式抬到 ≥16**（§7 draft 已含 `horizon_max: 16`）。**保持"多步误差不发散才提权"的门禁**（现方案已有，正确）。

---

## 3. 两个反哺主论文的消融（P2，★）

### 3.1 消融 A：在线想象规划 vs 蒸馏前馈策略 —— Fast-WAM 主问题的 aerial 实例

- v2 §4.5 中层用"想象打分"选动作序列（= test-time imagination）。Fast-WAM 核心问题正是"test-time 要不要想象"。
- Dream to Fly 部署时是**纯前馈 actor，不生成未来帧**——想象只是训练期装置。
- **消融设置：**
  - (a) 在线想象打分选轨（现设计）；
  - (b) 从想象 AC **蒸馏出的纯前馈策略**（部署零想象）。
- **价值：** 若 (b) 追平 (a) → 拿到跨域证据"**控制回路不需要 test-time imagination**"，直接强化 Fast-WAM 主论文；且砍掉部署想象开销（对 12 Hz 慢环尤其值钱）。

### 3.2 消融 B：主动搜索/注视行为是否免费涌现

- Dream to Fly 最大亮点：摄像头主动朝信息丰富区域的"感知感知"行为，**无任何视角奖励，纯端到端涌现**。
- 现方案在 `[2]` 手工做 frontier / 信息增益搜索。
- **消融设置：** 测端到端 imagination RL 是否**自发产生主动搜索/注视**，无需显式信息增益机器。
- **价值：** 哪怕部分成立都是强 finding；若成立可简化 `[2]`。
- **边界（见 §1.5）：** 论文的 gaze 涌现是"门定义航迹 + CTBR 敏捷 + 目标基本可见"下的产物；本方案是"目标不可见的 SEARCH + 运动学速度控制 + 12 Hz"，是否同样涌现完全 open。加之长时遮挡 POMDP 大概率仍需一定记忆——故定位为**消融**，不是替换。**这个跨 regime 差异本身就是该消融的科学看点：涌现是否依赖敏捷底层控制。**

---

## 4. HIL 中间验证档（P3，借 [10]/[12] 的 sim-to-real 桥）

- Dream to Fly 用 HIL（**habitat 渲染在环** + 真机动力学，命令直发真机）在 9 m/s 拿到小 sim-to-real gap。
- 本方案部署侧担心真机 IMU 接口 + sim RGB 域差。
- **动作：** 全真机部署前插一档 HIL（4090 渲染在环 + 真飞控动力学），先量 gap。
- **权衡：** 渲染器跨网 12 Hz，达不到 Dream to Fly 的实时性；HIL 只能做低速/离线校验，**不能当实时飞行验证**。且论文自陈"transfer 到真实相机像素仍有额外挑战"——HIL 用的是渲染像素，不是真机相机，域差没被 HIL 覆盖。

---

## 5. 纪律：数据/仿真底座优先于一切

沿用 v2 §0.5 的判断，且优先级压过本文其余全部：

- **真正瓶颈是数据/仿真底座（Fork A/B），不是模型架构。** 本文所有优化都在 **Fork A 成立**之后才有意义。
- **先过 §0.5.3 GO/NO-GO。** Fork B 下这些 RL 优化全无地基；此时唯一有意义的是"补带 IMU+稠密视频+碰撞的采集"或"降级回离散原语 VLN（SR≈10% 天花板）"。
- **成本预期要摆正："MBRL 省样本 ≠ 省时间"。** Dream to Fly 自陈 DreamerV3 收敛需 **~240 h 训练**（换样本效率的代价是 wall-clock）。本方案叠加"闭环 ~12 Hz 慢环 + 真机采样更慢"，训练 wall-clock 只会更长——排期时按"样本高效但耗时"来估，别把样本效率误读成快。

---

## 6. 落地优先级表

| 优先 | 优化 | 里程碑 | 类型 | 涉及文件 |
|------|------|--------|------|----------|
| P0 | DreamerV3 世界模型损失配方（symlog/two-hot/free-bits/回报归一化） | V0/V1 | 直接抄 | `experiments/aerial/rl/`（WM 训练）、loss 配置 |
| P0 | `[4]` 补循环态 h（RSSM） | V1 | 直接抄 | 世界模型模块、`imagination.py` |
| P1 | 复用 DreamerV3 decoder 兼塌缩探测（配方自带，不新增组件） | V0 | 零成本 | WM 训练、诊断日志 |
| P1 | `w_maneuver` 课程化 | V0/V4 | 改配方 | `configs/aerial_rl.yaml: reward` |
| P1 | imagination AC 用 Dreamer 原版 | V4 | 直接抄 | `rl/train_rl.py`、`corrector.py` |
| P2 | horizon 10→15（→16 需抬 `MAX_IMAGINATION_HORIZON`；门禁后） | V1 | 调参 | `configs/aerial_rl.yaml: imagination.horizon` |
| P2 | 消融 A（在线想象 vs 前馈）★论文 | V4 | 消融 | `rl/imagination.py`、eval |
| P2 | 消融 B（搜索行为涌现）★论文 | V4 | 消融 | `[2]` 记忆、eval |
| P3 | HIL 中间档 | 部署前 | 权衡 | eval bridge |

---

## 7. 建议的 `configs/aerial_rl.yaml` 改动方向（草案，非最终 diff）

```yaml
reward:
  w_progress: 1.0
  w_collision: 10.0
  w_maneuver: 0.0          # was 0.01 — 课程起点=0（终值=论文 b2=0.01）
  w_maneuver_final: 0.01   # 新增：课程终值
  maneuver_curriculum_threshold: <本方案自测的 progress/SR 阈值>  # 新增：过阈后线性拉起（勿抄论文的 50.0，尺度不同，见 §2.4）
  success_bonus: 10.0
  success_dist_m: 3.0

# 新增：世界模型训练配方（DreamerV3 对齐）
world_model:
  recurrent_state: true    # RSSM 确定性循环态 h（§2.2）
  pred_transform: symlog   # §2.1
  reward_head: two_hot     # §2.1
  free_bits: 1.0           # §2.1 KL free bits
  loss_scales: {pred: 1.0, dyn: 1.0, rep: 0.1}  # DreamerV3 固定权重
  decoder: train_only      # §2.3 DreamerV3 标准 decoder，仅训练期在环，兼作塌缩探测器

imagination:
  batch: 64
  horizon: 10              # V1 门禁通过后先试 15（§2.6）
  horizon_max: 16          # 抬 cap 才能到 16：代码默认 MAX_IMAGINATION_HORIZON=15

corrector:
  # ... 保持 V0/V1/V4 门禁 ...
  # V4 策略更新算法：DreamerV3 λ-return AC（§2.5），非 PPO
  policy_update_algo: dreamer_lambda_ac
  return_normalization: percentile   # DreamerV3 回报归一化——属 actor 回报层，非 reward design（§2.1）
```

> 上表字段名为**方向示意**，实现时以 `rl/train_rl.py` 与世界模型模块的真实 schema 为准，逐项对齐后再定最终 diff。

---

## 8. 与母方案的关系

- 本文**不取代** v2 spec，是其 V0–V4 的训练配方补充与消融清单。
- 建议在 v2 spec §7 训练分期表的每个阶段旁，引用本文对应小节（V0→§2.1/2.3、V1→§2.2/2.6、V4→§2.5/§3）。
- **§1.5（控制接口/任务域差）应作为全文的先决阅读**：它划出"DreamerV3 配方/λ-AC/reward 结构可整体抄" vs "T=16、课程阈值、gaze 涌现需在本方案 regime 内重新验证"的红线，避免把竞速 CTBR 的经验数字直接搬到运动学 SEARCH。
