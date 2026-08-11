# Aerial WAM v2 —— V0 项目总 RUNBOOK（活文档）

> **这是 aerial v2 pure-vision V0 的顶层入口 + 活文档。**
> - 想知道"项目在哪一步 / 每块去查哪份文档 / 怎么端到端跑" → 看这份。
> - **此后任何修改和调整,都在本文档底部 [§8 变更记录](#8-变更记录) 记一笔**(日期 + 改了什么 + 为什么)。
> - 阈值以冻结 spec §4.1 为**唯一权威**;本文只摘录并标注,**不在此新建第二处真相源**(那会触发 re-freeze)。

---

## 1. 一句话 & 当前阶段

**目标**:重建 goal-first 世界模型(干净重训现有 RSSM),让 V0 的**四个同权信号全过**,然后才翻 flags
(`depth_head.enable` / `safety.kind` / `corrector.enable_wm_update`)。当前在 **§6 Step 6**
(权威干净重训 + 四信号)。计划文件:`~/.claude/plans/humble-imagining-forest.md`。

**为什么**:旧 `wm_step_5000.pt` 被判定为单柱 RGB-only RSSM shortcut(已失效);必须从随机初始化干净重训,
结构性反 shortcut。

## 2. 四信号现状

| 信号 | 内容 | 评估位置 | 现状(2026-08-11) |
|---|---|---|---|
| **①a–c** | WM 训练健康(loss↓≥2% / recon 不劣 / min entropy-frac ≥0.10) | H100 离线(重训日志) | 🟡 干净重训 dry-run 已产出(`wm_ckpt_v2clean_20260810`,非权威);权威 a–c 待 Step-6 语料重跑 |
| **①d** | 深度 AbsRel ≤0.30 | H100 离线(DA3 ckpt) | ✅ 0.132 代表 / 0.167 approach OOD |
| **②** | 接近量↑(N=16 rollout vs random) | **4090 sim rollout** | ✅ 决定性通过:progress 24.13 vs random −5.11;final_dist 5.01m vs 34.99m |
| **③** | D̂ 尺度一致(reprojection,GT-proprio 位移) | H100 离线 | ✅ 0.05–0.12(重投影估计器,GT-oracle 0.002) |
| **④** | 近障避让(shield 开/关对比) | **4090 sim rollout** | 🟡 2026-08-11(晚³)harness 修已落地(后退罩+修 step bug+正前 probe+近障优先候选),**待 H100 pull + 重跑**;上一版不可测(`near_coll_rate_off=0`→ratio NaN) |

> ④ `near_coll_rate_off=0`(2026-08-11 rollout)根因:`HeuristicPolicy` 是纯 proprio 直线奔 goal、
> **不看 depth 不避障**;宽锥深度代理(`center_frac=0.5`)只证明"视野里有障碍",直线策略从旁 >1.5m 擦过。
> **已修(待重跑)**:`make_obstacle_facing_episodes` 加 **probe 验证** —— 代理通过后用直线策略空跑 24 步
> (shield 关),只留 GT 深度真进 <1.5m 的起点;因 ④ shield-off 臂跑同策略同起点 → `near_coll_off>0`
> 构造保证。同时收紧代理(`obstacle_max_m=15`、`center_frac=0.3`)。属 harness 几何修,不动 §4.1。
> 重跑看 scan 的 `probe.hits`;若 probe 找不到 hit(场景太开)→ ④ 诚实 fail-closed,提示换候选点。

## 3. 文档地图

**权威规格 / 设计**(定义"做什么"):
- **[frozen spec](docs/superpowers/specs/2026-08-04-aerial-wam-v2-frozen-spec.md) —— §4.1 四信号阈值,最权威,改阈值需 re-freeze。**
- [pure-vision design v2](docs/superpowers/specs/2026-08-03-aerial-wam-pure-vision-design-v2.md) —— 架构。
- [sim capability spec](docs/superpowers/specs/2026-08-03-aerial-sim-capability-verification-spec-v1.md) —— Fork A 判定。
- [signal3 OLS/axis proxy design](docs/superpowers/specs/2026-08-05-signal3-ols-axis-proxy-design.md)。

**分主题 handover**(定义"怎么跑某一块"):
- [4090 本地采集 runbook](docs/handover/2026-08-04-v0-4090-local-collect-runbook.md)
- [V1 WM H100 验证 runbook](docs/handover/2026-08-04-v1-wm-h100-validation-runbook.md)
- [signal3 reprojection estimator](docs/handover/2026-08-10-signal3-reprojection-estimator.md)
- [DA3 深度骨干](docs/handover/2026-08-10-da3-depth-backbone.md)

**基础设施**:[同步代码 & 建环境手册](experiments/aerial/scripts/RUNBOOK_sync_and_env.md)(三机 / 四脚本 / 六坑)。

**计划**:`~/.claude/plans/humble-imagining-forest.md`(V0 §6 Step 6 阶梯)。

## 4. 端到端跑法(§6 阶梯)

> 基础设施(推/拉代码、建 H100 环境、起 4090 渲染器)一律照
> [RUNBOOK_sync_and_env.md](experiments/aerial/scripts/RUNBOOK_sync_and_env.md),这里只列各阶段命令。

- **Step V-1（前置 gate）**:4090 loopback 跑 `experiments/aerial/sim_verify/run_all.sh` → 取 Fork A 判定
  (含 `depth_rate`:DepthPlanar fps ≥ 采集 step_hz)。Fork A 通过才进 Step 4。 → ✅ 已过
- **Step 4（4090 采集）**:`collect_v1.sh --airsim --host 127.0.0.1 grab_depth=true` → rsync npz 到 H100
  `experiments/aerial/rl/artifacts/`。实测闭环 Hz 设 `env.step_hz`。 → ✅ 产出 `dataset_v1_rgb`
- **Step 5（感知支柱）**:深度头 DA3(①d,已过——新语料上重验)。VIO 学习头是并行交付,不阻塞 gate。
- **Step 6（权威重训 + 四信号）**:
  - 干净重训(随机初始化,`--checkpoint-dir` 带日期):`_wm_train_validate --dataset <Step4语料> --steps N --save-ckpt --checkpoint-dir artifacts/wm_ckpt_v2clean_<date>`
  - ① 出:`_v0_gate --signals 1 --learning-log <新日志> --depth-ckpt <DA3>`
  - ③ 重跑:reprojection ≤0.25
  - **②④ rollout(4090 渲染器需先起)**——当前命令(artifact 走旧 checkout 绝对路径,见 [§5](#5-基础设施要点)):
    ```bash
    "$AERIAL_PY" -m experiments.aerial.rl._v0_gate --signals 2,4 --rollout-eval \
      --config configs/aerial_rl_rollout.yaml \
      --depth-ckpt /home/a25689/aerial-rl-skeleton/experiments/aerial/rl/artifacts/depth_ckpt_da3_20260810/depth_step_2000_da3_head.pt \
      --rollout-dataset /home/a25689/aerial-rl-skeleton/experiments/aerial/rl/artifacts/dataset_v1_rgb \
      --device cuda --emit experiments/aerial/rl/artifacts/v0_partial_24.json
    ```
    先盯 `[v0-gate] obstacle-facing scan: {...}` 看 `accepted`/16;跑完 `cat .../v0_partial_24.json`。
  - **合并**:`_v0_gate --merge <各信号 json>` → exit 0 才算 V0 过关 → 才翻 flags。

## 5. 基础设施要点

- **三机**:Mac(写代码)/ 8×H100 `a25689@10.239.121.22 -p 31126`(训练+gate+rollout 客户端)/ 4090 `10.229.20.125`(渲染器)。
- **H100 是临时容器**:重建后 torch/venv/artifacts 全丢。环境用 `INSTALL=1 source experiments/aerial/scripts/env_h100.sh` 一键重建。
- **两个 checkout**:`~/robomaster-tt-control`(新 clone,代码新、artifacts 空)vs `~/aerial-rl-skeleton`(旧的,数据/权重全在)。
  artifacts 不在 git → 从新 clone 跑、`--depth-ckpt`/`--rollout-dataset` 用 `~/aerial-rl-skeleton/...` 绝对路径,别拷贝。
- **共享盘** `/home/a25689/aerial_cache_shared/` 存 runs/orchestration,重建通常不清 → 找丢失产物先搜这里。
- 详见 [RUNBOOK_sync_and_env.md](experiments/aerial/scripts/RUNBOOK_sync_and_env.md)。

## 6. 治理红线（永不放宽）

- 四信号**全过前不翻 flags**;`enable_policy_update`(V4)绝不顺带打开。
- 阈值 = §4.1 冻结,改阈值 / 越出 §4 gate / §6 order → 先改并 re-freeze 冻结 spec(§8)。
- 干净重训禁 warm-start 失效 ckpt;canonical `depth_step_5000.pt` 不动;失效 ckpt 归档保留。
- 代码走 git,禁 scp 热补丁;`step_hz` 实测不猜。
- goal-input 属 V3,本周期不给 RSSM 加 goal 张量输入。

## 7. §4.1 阈值摘录（**以冻结 spec 为准**,此处仅速查）

| 信号 | 键 | 阈值 |
|---|---|---|
| ①d | `depth_absrel_max` | holdout median AbsRel ≤ **0.30**(缺深度语料则 ①d=SKIP → 整门 FAIL) |
| ①a–c | `_check_learning` / `post_entropy_frac` / `loss_recon` | loss↓≥2%;recon 不劣;min entropy-frac ≥ **0.10** |
| ② | `n_eval_episodes` / `progress_margin` / `dist_margin_m` | N=**16**;progress policy ≥ random + **5.0** ∨ final_dist policy ≤ random − **3.0**(任一即过) |
| ③ | `scale_rel_err_max` / `min_scale_windows` | reprojection median 相对误差 ≤ **0.25**;有效接近窗 ≥ **8** |
| ④ | `intervention_before_contact_min` / `near_coll_rate_ratio_max` | 接触前干预比例 ≥ **0.50**;shield-on/off near_coll 比 ≤ **0.80** |

---

## 8. 变更记录

> 格式:`YYYY-MM-DD —— 改了什么(为什么 / 依据)`。最新在上。

- **2026-08-11(晚³) —— ④ 后退罩 + 修 step 双夹 bug + 正前 probe + 近障优先候选(修 `near_coll_off=0`/ratio NaN)。**
  用户批准方向"后退罩+细step"。诊断链:上一版 `near_coll_off=0` 的真根因有三层,全修:
  1. **`HeuristicPolicy` 双重夹取 bug**(`train_rl.py`):`act` 里 `clip_body_delta` 用默认 30Hz
     上限 `[0.167,…]`,把 `step_m` 彻底废掉 → 5Hz rollout 实际每步只走 ~0.167m(而非物理上限
     1.0m=5m/s÷5Hz),probe/eval 永远够不到障碍。改:`act` 返回 `step_m`-缩放的**原始** delta,
     由 collector `act_delta` 用 `body_delta_limits(1/step_hz)` 正确按速率夹取(删孤儿 import)。
  2. **probe 命中判据 全图最小 → 正前中心裁剪**(`v0_rollout_eval.make_obstacle_facing_episodes`):
     旧判据 `_full_min_depth` 会把侧向擦碰/下沉见地当命中(docstring 自陈"巡航全图最小往往是
     地面")→ 跨网 RPC 抖动下 eval 复跑不复现(`fwd=13.4m` 接受、probe"命中"1.08m 侧向、
     `near_coll_off=0`)。改用 `_forward_min_depth(center_frac)`:要求直线策略**正面撞**墙 →
     eval 全图 near mask(<1.5,仍是冻结 §4.1)必然复现,ratio 可测。
  3. **shield 后退罩 latch(`safety.ThresholdSafetyShield`)**:旧 `override_action` 返回 zeros=悬停,
     会把机体**停在** 1.5m 近障带里(goal-seeker 一直命令前进、罩一直抵消)→ `near_on>near_off`
     ratio 反转/NaN。改:首次触发即**整集 latch**,后退(body −x)到预测间距恢复 >`safe_depth_m`(2.5m)
     再悬停 → `near_on≈0` 构造保证。加 `reset()`,`run_shield_eval` per-episode 清 latch(shield 实例跨集复用)。
  4. **近障优先候选 + 扫描参数**(`_v0_gate`):`_obstacle_candidate_positions` 按采集帧深度全图最小
     **近障优先排序**(RGB-only 无存深度则安全回退原序);`make_obstacle_facing_episodes` 加
     `preserve_order` 按此序扫(不打乱);scan 参数 `obstacle_max_m` 15→25、`probe_steps` 12→40
     (配 ~1m/步够到 25m 正前障)、`max_scans` 400→1000。诊断依据:上一版 `open_ahead=392/400`
     (98% 候选点前向 >15m 空)+ 只扫了 8.7%(400/4584 对)→ 巡航走廊本就多为开阔,把少数近障点排前是关键。
  harness 几何/罩行为修 + 一个真 step bug,**env / §4.1 阈值 / 模型 / flags 均未动**。①③② 的判读不受影响
  (② 决定性通过的 banked 结果仍成立;重跑会在障碍起点上重测 ②,仍应过)。**待 H100 pull + 重跑 ②④。**
- **2026-08-11(晚²) —— ④ probe 验证起点(修 near_coll_off=0)。**
  - `make_obstacle_facing_episodes` 加 `probe_policy/probe_steps/probe_near_m`:代理判据通过后,用同一
    `HeuristicPolicy` 空跑 24 步(shield 关),只保留 GT 深度真进 `<near_collision_depth_m` 的起点 →
    ④ shield-off 臂 `near_coll_off>0` 构造保证。`_v0_gate` 传入该策略,并收紧代理(`obstacle_max_m=15`、
    `center_frac=0.3`)。scan diag 新增 `probe.{hits,hit_depth_m}` + `rej.{proxy_ok,probe_no_hit}`。
    harness 几何修,env/阈值/模型/flags 均未动。**待 H100 pull + 重跑 ②④ 验证。**
- **2026-08-11(晚) —— ②④ rollout 判读 + 修 DA3 依赖漏装。**
  - ②④ rollout 出结果:**② 决定性过**(progress 24.13 vs −5.11;final_dist 5.01 vs 34.99m);
    **④ 不可测**(④b 干预 1.0 ✅,但 `near_coll_rate_off=0`/`n_contact=0`→ratio NaN)。判定:过 ② 的
    好策略天然不进 <1.5m 近障区,shield 无分母。**下一阶段 = 收紧 ④ 几何**(障碍上 start→goal 连线、
    `obstacle_max_m`↓、goal 置障碍远端),非改 §4.1。
  - **修 `env_h100.sh`**:DA3 深度头 hard-import `einops`+`addict`(vendored depth_anything_3 的
    DinoV2+DPT),之前最小依赖漏装 → 新 clone 跑 ②④/①③ 报 `ModuleNotFoundError: einops`。已加进安装
    列表 + 自检。`xformers` 有 try/except 回退,**不需要**。(纠正"DA3 是纯 torch"的错误判断。)
- **2026-08-11 —— 建本活文档 + 基础设施脚本化 + ④ 障碍生成器上线。**
  - 新增本 RUNBOOK(总入口 + 活文档约定)。
  - 新增 `experiments/aerial/scripts/{sync_push,sync_pull,start_renderer_4090,env_h100}.sh`
    + `RUNBOOK_sync_and_env.md`:把三机同步 / H100 临时容器建环境(torch cu128 + airsim,
    含 ensurepip / headless-cv2 / 两 checkout 坑)一键化。动因:每次代码提交/环境重建都极易出错。
  - `v0_rollout_eval.make_obstacle_facing_episodes` + `_v0_gate --rollout-dataset`:扫真实轨迹点找前向障碍,
    修 ④ 在开阔空域 `near_coll_rate_off=0` 的假失败(commit 540eb98/1c4178c)。属 ②④ **harness 几何修**,非 §4.1 改动。
  - 2026-08-11 首次带障碍生成器的 ②④ scan:`accepted 16/16`(前向障碍 ~21m);rollout 进行中,结果待记。
