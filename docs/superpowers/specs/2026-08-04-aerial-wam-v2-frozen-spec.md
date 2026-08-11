# Aerial WAM v2 — 冻结版建造契约（goal-first V-1 → V0，并定义 V1）

> **状态：FROZEN（2026-08-04；§4.1 数值门禁补丁同日）。** 本文是当前这一轮工作的**唯一权威建造契约**。它合并并**钉死**以下三份来源，之后不再发散：
> - 设计母本：[`docs/superpowers/specs/2026-08-03-aerial-wam-pure-vision-design-v2.md`](../specs/2026-08-03-aerial-wam-pure-vision-design-v2.md)（§1.2 纯视觉边界、§3 骨架、§4 模块、§7 分期、§8 接口、§11 非目标）
> - 配方补充：[`docs/design/2026-08-04-dreamerv3-alignment-optim.md`](../../design/2026-08-04-dreamerv3-alignment-optim.md)（§1.5 迁移边界）
> - 能力核查：[`docs/superpowers/specs/2026-08-03-aerial-sim-capability-verification-spec-v1.md`](../specs/2026-08-03-aerial-sim-capability-verification-spec-v1.md)（Fork A/A-/B）
>
> **本文不引入任何新设计。** 它只做一件事：把已确认的决策、门禁、接口、非目标固定成一份不可再议的清单，并与现有代码骨架一一对应。**凡本文未列入本期范围者，一律不做**（见 §8）。母本 §2–§6 的架构描述仍然有效，本文不复制、只索引。

### Glossary（命名债务，钉死）

| 用语 | 含义 | 不是 |
|---|---|---|
| **旧 H100「V1 validate」**（`_wm_train_validate` on `dataset_v1_rgb` / `wm_step_5000.pt`） | **非发散地板**：学习↓ + 开环 H≤15 有界 | 不是母本 §7 的 V1 档，也不是本期 V0 四信号 |
| **本期 V0** | RGB 编码 + 多帧深度头 + VIO；四信号全过 | 不是单柱 RGB RSSM |
| **本期 V1**（只定义） | τ + 短程想象规划 + τ/D̂ 双通道罩 | 不是「翻 `dynamics.kind=torch`」本身 |
| **`aerial_v1_wm_gate_invalidated`** | 作废上述地板 ckpt 的治理凭据 | 不否定配方代码可复用（须随机重训） |

---

## 0. 为什么需要冻结

旧 FastWAM 像素线在 B0 塌缩（SR=0、退化/逐位相同 checkpoint）。v2 已**goal-first 重设计**为模块化混合架构——从目标反推，而非改良塌缩模型。

一个 RGB-only RSSM checkpoint（`wm_step_5000.pt`，`dynamics_torch.py` 在 `dataset_v1_rgb` 上训练）曾被**错误宣告为「V1 gate 通过」**：它是**单柱 RGB-only 世界模型**，跳过了整个感知骨架（无深度头、无 VIO、无光流 τ），只越过了「非发散地板」这一必要非充分切片，②碰撞率↓ 与 ③τ/D̂ 双通道从未可测。

**冻结的动机**：把「下一步不是训一个更好的 RSSM，而是回到纯视觉 v2 规范沿 §7 阶梯 goal-first 重建」这一治理性结论，连同其全部边界，固定下来，杜绝再次滑向「单柱调参」。

---

## 1. 不可动摇的前提（已确认，禁止再议）

1. **`wm_step_5000.pt` 永久作废。** 不是 baseline，不是 warm-start，不做任何 pre-v2 权重 bootstrap。清洁重训一律从随机初始化开始。记忆凭据：`aerial_v1_wm_gate_invalidated`。
2. **纯视觉外感知边界（母本 §1.2）。** 策略与世界模型只消费 **RGB + proprio4 `(x,y,z,yaw)`**。深度 / IMU / 速度是**监督专用**，永不进入策略计算图——由 `env/obs.py::PolicyObservation` 在类型层强制。
3. **局部 VIO ≠ 全局 SLAM。** 只做滚动窗口内度量一致，**无回环、无全局 BA、不产出全局轨迹作规划坐标**。
4. **DreamerV3「配方可抄、调参不可抄」（补充 §1.5）。** 可整体迁移：symlog / two-hot / free-bits / KL-balance / RSSM(h+z) / λ-return AC。**禁止照搬**：`T=16`、reward 阈值 `50.0`、gaze 涌现结论——它们跨 regime，须在本方案内重新验证。硬上限 `MAX_IMAGINATION_HORIZON = 15`。
5. **未过关不叠加下一阶段。** 每一档有可独立验收的门禁；当前档全部信号通过前，不动下一档的旗标与代码路径。
6. **像素级 Wan2.2 仅作离线预训练/蒸馏源。** 绝不在线逐步像素规划（母本 §4.4/§11）。绝不重新下载 Wan2.2 权重，只用本地已有权重（记忆凭据：`no_wan22_download_in_scripts`）。

---

## 2. 本期锁定决策（session 决策，钉死）

| 决策项 | 锁定值 | 理由 |
|---|---|---|
| **深度/IMU 数据底座的 depth-rate 解法** | **4090 本地采集**（collector 跑在渲染主机，走 `127.0.0.1:41451` loopback） | 跨网 DepthPlanar ~1.3 s/帧（~0.7 Hz）是 float32 深度缓冲的跨网传输代价；loopback 消除它，使 `grab_depth=true` 每帧可行 → RGB + 稠密深度 + 时间同步 IMU/step |
| **本期范围** | **完整 V0（四信号全过）+ 定义 V1（仅判据/组件级）** | V1 只落判据与组件，不实现；V2/V3/V4 本期完全不碰 |

---

## 3. 骨架与接口契约（钉死，对应代码）

母本 §8 的接口在本仓的落点。**接口语义冻结**；实现可演进，语义不改。

| 接口 | 生产者 → 消费者 | 语义 | 代码落点 |
|---|---|---|---|
| `z_t` | `[1]` → `[2][4][5]` | 当前视觉状态 | `dynamics_torch.py`（RSSM `[h‖z]` 特征） |
| `D̂_t, σ_t` | `[1b]` → `[5]`/安全 | 局部深度 + 不确定度 | `_DepthHead`（from-scratch，canonical）**或** `DA3DepthHead`（DA3 骨干，①d 加固）经 `build_depth_head` 工厂分派（`backbone` 键：`scratch` 默认 / `da3`）；接口语义不变（见 2026-08-10 §3 修订） |
| `vio` | `[1c]` → `[2][3][5]` | 局部度量位移/速度/尺度/高度 | **`vio.py` 数学已落**（Step 3 Mac）；学习头仍待 4090 语料 |
| `τ` | `[1d]` → 安全罩 | 碰撞时间（独立于 `D̂`） | **V1 组件，本期只定义** |
| `mem` | `[2]` → 高层 | 局部占据 + 拓扑 | **V2，本期不做** |
| `goal_hyp` | `[3]` → 高层 | 命中假设 + 置信度 + 相对方位 | **V3，本期不做** |
| `mode` | `[5]` | `SEARCH \| NAV \| DONE` | **V3，本期不做** |
| `a_t` | `[5]` → 环境 | 执行动作 `(dx,dy,dz,dyaw)` | `collector.py` / `env/` |
| `p_coll` | `[4]` → 中层/安全 | 碰撞风险 | `dynamics_torch.py`（`wm_out.p_coll`） |

**复用、不重造的骨架**（保持原样）：
- `env/obs.py` — `Observation{rgb, state[7]=(x,y,z,vx,vy,vz,yaw), collided, depth, imu, t}`；`PolicyObservation` 强制 RGB+proprio4 边界。
- `safety.py` — `ThresholdSafetyShield` 已读 `obs.info["depth_min_pred"]`、`obs.info["tau_pred"]`、`wm_out.p_coll`；V0 只接 `depth_min_pred` 一路。
- `dynamics_torch.py` — RSSM + DreamerV3 配方 + 塌缩遥测（`post_entropy_frac`、`loss_recon`）：**复用代码，随机初始化重训**。
- `reward.py` — `NavigationReward` 已产出逐步 `progress`。
- `_wm_train_validate.py` — `_check_learning` 即 ① 非塌缩门。
- `dreamer_recipe.py` / `wm_data.py` / `wm_eval.py` / `imagination.py`（cap 15）/ `buffer.py` — 保持。

---

## 4. 阶段门禁（钉死）

### V-1（前置）— sim GO/NO-GO + depth-rate

在 4090 loopback 上跑 `sim_verify/run_all.sh`，取 **Fork A** 判决。Fork A 现在的合取（已实现，`verdict.py::decide`）：

```
connected ∧ real_rgb ∧ imu ∧ (baro∨gps) ∧ collision ∧ depth ∧ depth_rate ∧ physics ∧ continuous_frames
```

- `depth_rate`（新增，`sanity.depth_rate_ok` + `t2_capability._depth_continuous`）：背靠背抓 DepthPlanar，`fps ≥ L2F_DEPTH_MIN_FPS`（设为 ≥ 采集 `step_hz`）且时间戳单调。**在 4090 loopback 上跑**，测得即真实采集率——把「深度存在」升级为「深度快到足以采 V0 数据集」。
- 未过 → 停在 Fork B/A-，不进 V0。

### V0（本期主目标）— 四个**同权**信号，缺一不可

`_v0_gate.py`（H100 入口；numpy 度量在 `v0_metrics.py` / `vio.py`，Mac 单测）跑下表，**四者全过**才退出 0；否则非零。定性意图仍见母本 §7；**可执行数值钉死于 §4.1**（评估补丁 2026-08-04）。

| 信号 | 度量 | 复用 | 过关条件（摘要；细则 §4.1） |
|---|---|---|---|
| ① 非塌缩 | 训练曲线 + 深度重建见证 | `_wm_train_validate._check_learning`、`post_entropy_frac`、`loss_recon` | loss↓≥2%；recon 不劣化；min entropy-frac ≥ `collapse_entropy_frac`（默认 0.10）；深度 AbsRel 有界 |
| ② 接近量上升 | sim rollout progress-vs-random | `NavigationReward.progress`、同起点对照 | N=16；mean progress_sum 优于随机 +5.0 **或** mean 终点距优于随机 ≥3.0 m |
| ③ D̂ 尺度与 VIO 一致 | 相对尺度误差 | `vio.reproject_scale_error` | 运动窗上 median 相对误差 ≤ 0.25 |
| ④ 简单近障生效 | shield-on vs shield-off | `ThresholdSafetyShield`、`CollectStats.interventions` | 干预先于接触 ≥50%；近碰撞率 ≤ shield-off 的 80% |

**④ 的接线**：`collector.py` 须在调 `safety.should_override` **之前**由深度头产出 `depth_min_pred`（今天两者皆空）。**评测期**用 CLI/`_v0_gate` 临时 `safety.kind=threshold` 跑 shield-on/off 对照，**不**改默认 `configs/aerial_rl.yaml` 直到四信号全过。

**仅在四信号全过后翻转的旗标**：`world_model.depth_head.enable: true`、`safety.kind: threshold`。`dynamics.kind` / `enable_wm_update` 在 V0 bring-up 期间保持 V0 姿态（清洁 co-train 走离线 `_wm_train_validate`/`_v0_gate`，不走 live corrector）；只在进入 V1 时才翻到 `torch`/`true`。**严禁**顺带打开 `enable_policy_update`（那是 V4）。

### 4.1 V0 数值门禁表（钉死；实现以本表为准）

常量落点：`experiments/aerial/rl/v0_metrics.py::V0GateThresholds`（与下表同步；改阈值 = 修订本节）。

| ID | 参数 | 钉死值 | 协议 |
|---|---|---|---|
| ①a | `loss_drop_ratio` | last10% mean < first10% mean × **0.98** | 同 `_wm_train_validate._check_learning`；全 finite |
| ①b | `recon_non_worse` | last10% recon ≤ first10% recon | RGB recon；深度头接入后另计 ①d |
| ①c | `min_post_entropy_frac` | ≥ **`collapse_entropy_frac`**（默认 **0.10**） | 训练全程 min |
| ①d | `depth_absrel_max` | holdout median AbsRel ≤ **0.30** | 仅当 depth 头与 GT depth 语料存在；否则 ①d=SKIP（整门 FAIL——V0 需要深度柱） |
| ②a | `n_eval_episodes` | **16** | 与 annotation 起点对齐；seed=0 |
| ②b | `progress_margin` | mean(progress_sum_policy) ≥ mean(progress_sum_random) + **5.0** | 同起点；随机动作为 `U(-1,1)` clip 到 body_delta_limits |
| ②c | `dist_margin_m` | mean(final_dist_policy) ≤ mean(final_dist_random) − **3.0** | ②b ∨ ②c 任一即可过 ② |
| ③a | `min_motion_m` | 窗内 ‖Δp‖ ≥ **0.5** m 才计入 | 静止窗不参与尺度比 |
| ③b | `scale_rel_err_max` | median |Zpred − Zobs| / Zobs ≤ **0.25** | 2026-08-10：`vio.reproject_scale_error`（重投影，取代 band-median `scale_relative_error`）；阈值不变 |
| ③c | `scale_depth_band_m` | **[1.0, 40.0]** m | 2026-08-05 定；2026-08-10 band 敏感性证明为 domain-of-interest 掩码而非偏置源（GT-oracle 在 40/80/200/∞ 恒≈0.002），冻结 [1,40]（D̂ 最坏档仍 0.122 PASS） |
| ③d | `fwd_cos_min` | **0.7** | 2026-08-05：|cos∠(Δp, mean heading)|；取代 world-+x 前向代理 |
| ③e | `scale_support_ratio` | **0.6**（采样器仍用；估计器不再依赖，DEPRECATED-in-place） | 2026-08-10：重投影无死代理失效模式，保留只为 `_sample_scale_windows` 选前向接近窗，不参与 ③ 判定 |
| ③f | `min_scale_windows` | **≥ 8** | 2026-08-05：有效接近窗不足则 ③ FAIL（非放宽 0.25） |
| ④a | `near_collision_depth_m` | GT `depth_min` < **1.5** m | 度量钉死 1.5;shield 触发 `min_depth_m` = 度量 + 反应余量（默认 **3.0 m**），见下 ④ 反应余量注（re-freeze 2026-08-11） |
| ④b | `intervention_before_contact_min` | **≥ 0.50** | 在最终 `collided` 的 episode 中，首次 intervention 步号 < 首次 contact 步号 的比例 |
| ④c | `near_coll_rate_ratio_max` | shield-on / shield-off ≤ **0.80** | near_coll 帧占比；同 N、同起点；**仅评测 CLI 开罩** |

**④ 反应余量注（re-freeze 2026-08-11；不改 ④a 钉死度量 1.5 / ④b 0.50 / ④c 0.80）**：原 ④a 协议注要求 `ThresholdSafetyShield.min_depth_m` 与近碰撞度量 1.5 m **对齐**。实测（`dataset_v0_approach_20260805`，DA3 深度头 approach AbsRel≈0.167，accepted=12）证明该对齐使 ④ **结构性不可过**——预测器在 1.5 m 边界处偏乐观（d̂ 读得比 GT 远）：(i) "退到 `safe_depth` 再悬停"把机体停在 GT<1.5 带内 → `near_coll_rate_on`(0.205) ≫ `off`(0.034)、**ratio 反转 6.10**；(ii) 约半数过边界时 d̂ 仍 >1.5 → 触发晚一拍、前一步已前冲进碰撞 → 干预晚于接触，**before_frac=0.333**。**修订**：(a) shield latch 后**连续后退**（`override_action` 去掉 hover-at-safe 分支，每步 body −x），机体单调退带 → ratio 稳过；(b) shield **触发 `min_depth_m` 提到 3.0 m**（> 度量 1.5，作反应余量），提前于进带反应 → 不进带、零碰撞 → `before_frac` 空过（`check_shield_effectiveness` 无碰撞集 → 1.0）。**度量端 `near_collision_depth_m=1.5` 与 ④b/④c 钉死值不动**；改的只是 shield 触发/后退**行为**（shield 是被测系统，反应余量属其设计参数，非 gate 阈值）。起点前向深度最小 5.578 m > 3.0 m，余量不误触发起点。落点：`safety.py::ThresholdSafetyShield`（`min_depth_m=3.0` 默认、连续后退）· `v0_rollout_eval.run_shield_eval(shield_trigger_depth_m=3.0)`（与 metric 掩码解耦）· `train_rl._build_safety`（live 默认 3.0）。理由：④b「干预先于接触」在噪声预测器下，对"恰好边界反应"数学不可满足，反应余量为必需而非调参偏好。

**③ 适用性注记（不改钉死值 0.25；协议修订 2026-08-05）**：③ 用的深度侧长度 `ŝ_D = |median_band D̂_last − median_band D̂_first|`（`vio.scale_from_depth_change`，median 限在 **[1, 40] m** 导航带）是一个**代理**，只在**窗内含前向接近分量**（相机大致沿运动方向、正对场景使视深随位移单调变化，且 ŝ_D ≥ 0.6·‖Δp‖）时才与度量位移 `s_VIO=‖Δp‖` 同尺度。纯侧移/纯偏航/纯升降/贴墙平行巡航/开阔地平线窗上该代理失真——`③a min_motion_m≥0.5 m` 只滤静止窗；`③d/③e/③c` 滤掉无物理意义的窗。采数应偏**朝向表面接近**的轨迹；V1 τ 通道用 FOE 散度独立复核。**`--signal3-diagnose`（2026-08-05）**在改定协议下拆 GT-oracle vs D̂：GT 仍失败 → 再修订或重采，禁止为过门放松 0.25；GT 过而 D̂ 不过 → 加时序/Δ-depth 监督重训深度头。

**③ 估计器修订(2026-08-10；不改钉死值 0.25/0.30)**：③ 的深度侧长度由 band-median 差 `ŝ_D=|median_band D̂_last − median_band D̂_first|`(`vio.scale_from_depth_change`)**改为重投影** `vio.reproject_scale_error`：以内参 K(224²/hfov90→fx=fy=112, cx=cy=111.5)反投影首帧导航带 [1,40] m 像素→世界(GT proprio Δpos+Δyaw；AirSim NED body-fwd cam, DepthPlanar=Zc)→重投影至末帧，robust-median `|Zpred−Zobs|/Zobs`。**动因**：band-median 是代理，其自身几何偏置使 GT-oracle 前向窗=0.229–0.263(≈或超 0.25 线)，完美深度亦不可过门→测的是 band 像素构成 churn 而非尺度。重投影是几何恒等式，**GT-oracle→0.002** (Mac 0.005；H100 5 语料 0.002–0.004)，0.25 预算回归真实 D̂ 误差。**验证**(canonical `depth_step_5000`，H100 `mot-wam`)：同机同语料 OLD band-median GT-fwd 0.263 FAIL / D̂ 0.619 FAIL；NEW 重投影 GT-fwd 0.002 PASS / D̂ 0.122 PASS；5 语料 D̂ 0.112–0.148 全 PASS。band 扫描 [1,40..∞]：GT 恒≈0(非偏置旋钮)，D̂ 单调降(0.122→0.043)，[1,40] 是最坏档仍过关→冻结 [1,40]。`scale_support_ratio` 降级为采样器专用；`fwd_cos_min=0.7` 与 `min_scale_windows≥8` 不变。`--signal3-diagnose` 保留 band-median 腿作 A/B。依据 `docs/handover/2026-08-10-signal3-reprojection-estimator.md`。

Shield 对照协议：默认 yaml 保持 `safety.kind: null`；`_v0_gate --shield-eval` 在进程内构造 on/off 两套 collector，不写回配置文件。

### V1（本期只定义，不实现）

- **组件**：[1d] 光流 time-to-contact `τ`（FOE + 散度，**独立于深度网**，产 `obs.info["tau_pred"]`）；短程想象规划器（`imagination.py`，cap 15）在线为候选动作序列打分；τ/D̂ 双通道安全罩 `DepthTauShield = D̂ ∪ τ ∪ p_coll`（τ 路独立于 D̂，缓解共因失效）。
- **三个同权信号**：① 碰撞率相对 V0 baseline 下降；② 多步 rollout 达标（开环想象保真，`wm_eval.fidelity_verdict`，在**诚实留出**的清洁重训上评，不在 in-sample 失效 ckpt 上）；③ τ/D̂ 双通道各自独立验证（两路都验，both-fail 集合列举公开）。
- Horizon 恒 ≤ 15。

---

## 5. 数据契约 — schema v2（已实现，冻结）

`dataset.py` 落盘（`episode_arrays`/`write_episode`），`load_episode` 逐键 `key in raw.files` 向后兼容旧 npz：

| 键 | 形状 | 来源 | 用途 |
|---|---|---|---|
| `rgb` | `[N,H,W,3]` u8 | `obs.rgb` | 策略/WM 可见 |
| `proprio4` / 动作 / reward / done / collided | — | 原有 | 原有 |
| `vel` | `[N,3]` f32 | `state[3:6]` | VIO 回归目标 |
| `imu_ang_vel` / `imu_lin_acc` | `[N,3]` f32（缺帧 NaN） | `obs.imu` | VIO 输入 |
| `imu_present` | `[N]` bool | `bool(obs.imu)` | 缺 IMU 掩码 |
| `timestamps` | `[N]` f32 | `obs.t` | VIO 真实 dt |
| `depth` | `[N,H,W]` f32 | `obs.depth` | [1b] 目标；**全有或全无**（每帧都有才存） |

**边界护栏**：IMU/vel/depth GT **绝不**喂进 `wm_data.windows_to_arrays`。监督专用通道走**独立** `perception_data.py`（torch-free，`wm_data` 的刻意 sibling）+ `dt_from_timestamps`（glitch-robust 逐步 dt）。这是「监督信号不泄漏进 RGB-only 策略图」的结构性保证。

---

## 6. 构建顺序（未过关不叠加）与运行位置

| # | 步骤 | 运行位置 | 门禁 |
|---|---|---|---|
| 1 | Step 0 复原 V1 gate（config → V0 姿态） | Mac 改 config → H100 用 | ✅ 完成 |
| 2 | Step 2 dataset schema v2 | Mac（numpy 全可测） | ✅ 完成 |
| 3 | Step 1 depth-rate V-1 gate（sanity/verdict） | Mac 写+测；probe 跑 4090 | ✅ 完成（Mac 部分） |
| 4 | Step 1 4090 本地采集 + 重测 `step_hz` | 4090 采 → H100 落 | ⛔ 待 4090（gated by 3） |
| 5 | Step 3 [1b] 深度头 + [1c] VIO | 度量/adapter 数学 Mac 测；训练 H100 | 待（gated by 4） |
| 6 | Step 4 清洁 WM co-train + `_v0_gate` 四信号 | H100 | 待（四全过 → 翻 V0 旗标） |
| 7 | V1 | **本期不建** | — |

4090 采集：git checkout `aerial-rl-skeleton`（**禁 scp 热补丁**），`env_bridge.py` 常驻，`collect_v1.sh --airsim --host 127.0.0.1 grab_depth=true` 写 4090 本地目录，`rsync` npz（仅数据）到 H100 `experiments/aerial/rl/artifacts/`。重测 loopback + `grab_depth=true` 闭环 Hz，置 `env.step_hz` 于实测地板下，并同步更新 `_wm_train_validate._refuse_v0` 阈值（当前 8 Hz / `>8.5` 是**跨网** RGB-only 地板，对 4090-local+depth 已过时）——**该数值须实测得到，不得臆测**。

**①d 深度骨干（Step 5 子选项；2026-08-10 修订）**：`[1b]` 深度头有两个后端，经 `build_depth_head`（`backbone` 键）分派：`scratch`（默认，from-scratch `_DepthHead`，canonical `depth_step_5000.pt`）与 `da3`（`DA3DepthHead`：冻结 DINOv2-ViT-L + 可训练 DPT，从 Apache-2.0 `depth-anything/DA3METRIC-LARGE` warm-start，在我方 GT 上微调学 metric 尺度）。动因：from-scratch 头 ①d 封顶（代表性 0.281 薄余量、approach 0.31 FAIL），DA3 加固至 0.1x。DA3 仅动 ①d，**不动 ③**（③ 已由重投影估计器解决）。`da3` 路径 `delta_weight=0`/`nll_weight=0`（DPT 只出 depth）、默认冻结 encoder；纯 torch 子集 vendor 在 `experiments/aerial/rl/third_party/depth_anything_3/`（锁 commit `3d835ec`），权重仅训练机需要、ckpt 自包含。跑权威 `_v0_gate --signals 1,3` 于代表性+approach 两语料，①d 有余量 + ③ 仍过。护栏不变：canonical 不覆盖（DA3 ckpt stem `_da3`）；yaml `depth_head.enable`/`safety.kind` 四信号全过前保持关闭；四全过前不启动 4090 ②/④。依据 `docs/handover/2026-08-10-da3-depth-backbone.md`。

---

## 7. 当前实现状态（截至 2026-08-04，含评估落地补丁）

- **已完成并 checkpoint**：Step 0（config 复原 V0 姿态、`wm_step_5000.pt` 作废、清理陈旧 docstring）；Step 2（schema v2 + `perception_data.py` + `dt_from_timestamps`）；Step 1 Mac 部分（`depth_rate_ok`、`_depth_continuous` probe、Fork A 合取加 `depth_rate`、`verdict.decide` 抽为纯函数）。
- **评估落地（本补丁）**：§4.1 数值门禁表；glossary；`vio.py`（积分/尺度误差，Mac 可测）；`v0_metrics.py` + `_v0_gate.py`（阈值与判定骨架，完整 sim 评测仍待 4090 语料 + 深度头）；4090 本机采数手册。
- **测试**：rl numpy 套件 + sim_verify；见 CI/本地 pytest。
- **阻塞在 4090（本沙箱不可为）**：Step 1 剩余的 4090 本地采集、`step_hz` 重测、`_refuse_v0` 阈值更新。
- **下一步**：4090 loopback Fork A + depth 采数；其后 DepthHead 训练接入 ①d/③/④。

---

## 8. 非目标（明确不做）

**母本 §11（永久非目标）：** 不依赖深度传感器/预建全局度量地图；不以经典/全局 SLAM 作导航主链路（局部窗口 VIO 除外）；不在线用像素级视频模型逐步闭环；不把单帧模仿学习当搜索/规划的充分方案；不承诺对抗障碍场（细线/玻璃/高速动态）的避障保证。

**本期额外非目标（防发散）：**
- 不做 V2（混合记忆/探索）、V3（语义 grounding/SEARCH-NAV-DONE）、V4（想象中 RL）——本期只到 V0 + 定义 V1。
- 不实现 V1 组件（τ 网络、在线想象规划器、双通道罩），只落判据与组件清单。
- 不翻 `enable_wm_update`（V1）/`enable_policy_update`（V4）旗标作为副作用。
- 不 warm-start / bootstrap 任何 pre-v2 权重。
- 不照搬 DreamerV3 的 `T=16` / reward-50.0 / gaze 涌现。
- 不重新下载 Wan2.2。
- 不在 v2 spec 之外新增架构分支或「顺手优化」。

**任何超出本文 §4 门禁与 §6 顺序的工作，需先修订本文并重新冻结，方可进行。**

---

## 9. 文件清单

**改**：`configs/aerial_rl.yaml`（复原 gate；加 `world_model.depth_head`；`safety.kind: threshold`；重导 `step_hz`）· `dataset.py`（schema v2 + loader）· `dynamics_torch.py`（深度头；清洁重训）· `collector.py`（产 `depth_min_pred`，把 `wm_out` 传给罩）· `sim_verify/{probes/t2_capability.py, lib/sanity.py, verdict.py}`（depth-rate gate）· `_wm_train_validate.py`（`_refuse_v0` 阈值）。
**新**：`perception_data.py`（✅）· `vio.py`（✅ 数学）· `v0_metrics.py`（✅）· `_v0_gate.py`（✅ 骨架；完整评测待语料）· `docs/handover/2026-08-04-v0-4090-local-collect-runbook.md`（✅）· `dynamics_torch.py::DA3DepthHead` + `build_depth_head` 工厂（✅ 代码；①d DA3 骨干）· `experiments/aerial/rl/third_party/depth_anything_3/`（✅ vendored 纯 torch 子集，锁 commit `3d835ec`，含 `VENDOR.md`）· `train_depth_head.py --backbone da3`（✅ 训练路径）· `docs/handover/2026-08-10-da3-depth-backbone.md`（✅）。

---

*本文冻结。修改须显式修订本节以上任一「钉死」条目并注明日期，否则以本文为准。*
*修订 2026-08-04（评估落地）：§ glossary、§4.1 数值门禁、`vio`/`v0_metrics`/`_v0_gate` 落点。*
*修订 2026-08-05（③ 协议）：钉死导航带 [1,40] m、heading-forward `fwd_cos_min=0.7`、approach-support `scale_support_ratio=0.6`、`min_scale_windows=8`；**不改** `scale_rel_err_max=0.25`。依据 `--signal3-diagnose`：旧 full-frame median + world-+x 采样在开阔地平线/贴墙平行上使 GT oracle 亦不可达；0.6 使 GT oracle 在 resize 语料上可达（med≈0.21），从而把失败归因到 D̂。*
*修订 2026-08-10（③ 估计器）：band-median → 重投影 `vio.reproject_scale_error`（GT-oracle 0.26→0.002，canonical D̂ 0.122 PASS，5 语料佐证 + band 扫描）；band 冻结 [1,40]；`scale_support_ratio` 降级采样器专用；**不改** `scale_rel_err_max=0.25` / `depth_absrel_max=0.30`。*
*修订 2026-08-10（①d 深度骨干）：§3 `D̂_t,σ_t` 落点加 `DA3DepthHead`（DA3METRIC-LARGE 冻结 DINOv2-ViT-L + 可训 DPT）作 `_DepthHead` 之外的 `build_depth_head` 分派后端，仅加固 ①d 薄/失败余量（代表性 0.281 / approach 0.31 → 目标 0.1x）；§6 加 Step 5 深度骨干子选项，§9 加 vendored `third_party/depth_anything_3/` + `DA3DepthHead`/`build_depth_head` + `--backbone da3`。**接口语义、gate 数学、③ 估计器均不变；不改** `depth_absrel_max=0.30` / `scale_rel_err_max=0.25`。DA3 仅动 ①d，不动 ③。依据 `docs/handover/2026-08-10-da3-depth-backbone.md`。*
