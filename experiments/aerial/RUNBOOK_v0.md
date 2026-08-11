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
| **②** | 接近量↑(N=16 rollout vs random) | **4090 sim rollout** | ✅ progress margin 通过(progress_sum ≈24.13) |
| **③** | D̂ 尺度一致(reprojection,GT-proprio 位移) | H100 离线 | ✅ 0.05–0.12(重投影估计器,GT-oracle 0.002) |
| **④** | 近障避让(shield 开/关对比) | **4090 sim rollout** | 🟡 障碍生成器已修好开阔空域问题,scan 16/16;rollout 结果**待出** |

> ④ 之前卡在 `near_coll_rate_off=0`(巡航高度朝原点飞是开阔空域,无障碍可撞)。已加
> `make_obstacle_facing_episodes`(扫真实轨迹点找前向障碍),2026-08-11 scan `accepted 16/16`
> (障碍在前方 ~21m)。等 rollout 出 `near_coll_rate_off>0` / `intervention_before_contact≥0.5` / `ratio≤0.8`。

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

- **2026-08-11 —— 建本活文档 + 基础设施脚本化 + ④ 障碍生成器上线。**
  - 新增本 RUNBOOK(总入口 + 活文档约定)。
  - 新增 `experiments/aerial/scripts/{sync_push,sync_pull,start_renderer_4090,env_h100}.sh`
    + `RUNBOOK_sync_and_env.md`:把三机同步 / H100 临时容器建环境(torch cu128 + airsim,
    含 ensurepip / headless-cv2 / 两 checkout 坑)一键化。动因:每次代码提交/环境重建都极易出错。
  - `v0_rollout_eval.make_obstacle_facing_episodes` + `_v0_gate --rollout-dataset`:扫真实轨迹点找前向障碍,
    修 ④ 在开阔空域 `near_coll_rate_off=0` 的假失败(commit 540eb98/1c4178c)。属 ②④ **harness 几何修**,非 §4.1 改动。
  - 2026-08-11 首次带障碍生成器的 ②④ scan:`accepted 16/16`(前向障碍 ~21m);rollout 进行中,结果待记。
