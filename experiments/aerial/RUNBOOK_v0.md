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
| **④** | 近障避让(shield 开/关对比) | **4090 sim rollout** | 🟡 scan 已修(accepted=10/11);晚¹⁰ `_shield_diag` 定位盲目后退撞后墙(coll_after_latch=9/9);晚¹¹ 保持(悬停)反致惯性滑进带并停留(near_count_on 200/200、ratio 12.96);晚¹² **shield 有界状态反馈后退**(D̂<standoff 后退刹动量、D̂≥standoff 保持不撞后墙;re-freeze);晚¹³ 遥测:④c ratio 0.172 ✓ 但 ④b=0,盲退 REFUTED;晚¹⁴ 真因=eval 关了 collector 出生碰撞守卫误计出生嵌入为碰撞 → 恢复 `skip_reset_collision=True`+重采样,**待 4090 重跑预期 spawn_collision_drops>0、④ PASS** |

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
      --depth-ckpt /home/a25689/aerial-rl-skeleton/experiments/aerial/rl/artifacts/depth_ckpt_da3_near_20260811/depth_step_2000_da3_head.pt \
      --rollout-dataset /home/a25689/aerial-rl-skeleton/experiments/aerial/rl/artifacts/dataset_v0_headon_20260811 \
      --device cuda --emit experiments/aerial/rl/artifacts/v0_partial_24.json
    ```
    先盯 `[v0-gate] obstacle-facing scan: {...}` 看 `accepted`/16;跑完 `cat .../v0_partial_24.json`。
    (晚⁸:scan 现对每个位置先试采集记录航向再走 8 网格,头对头语料 `dataset_v0_headon_20260811` 才能被 probe 命中。)
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

- **2026-08-11(晚¹⁵) —— 晚¹⁴ H100 权威 rollout 验证生效(② PASS、④c 漂亮过 ratio 0.042、spawn_collision_drops=2、start_collided_on 全 False),但 ④b 仍 =0;加 opt-in 只读接触集取证 dump 定性唯一接触集。**
  晚¹⁴ 在 H100 跨网权威重跑(commit b7823b9,4090 渲染器 PID 79217):**② PASS**(progress 11.81 vs random −3.99、final_dist 18.26 vs 33.34,双余量过,n=12);**晚¹⁴ 修复实锤生效** —— `spawn_collision_drops=2`/`health_drops=0`、所有存活集 `start_collided_on=false`、`coll_after_latch=0`、9/10 集 shield-on 存活满 200 步;**④c 漂亮过** `near_coll_rate_ratio=0.042 ≤0.80`(on 0.00185 / off 0.0439)。
  **但 ④ 仍 FAIL 于 ④b** `intervention_before_contact_frac=0.0`,`n_contact_episodes=1`。唯一接触集(scored idx 0):`len_on=1, coll_first_on=0, interv_first=-1, start_fwd_min=6.18, start_full_min=3.029, along_heading_on=0.025, lateral_on=19.554, start_collided_on=false`。画像:**前向 GT 净空 6.2m(shield 前向 D̂ 未到 3.0 → 全程未干预)→ 第一步(len=1)即撞**,全场最近 3.03m 在**非前向**;且 `lateral_on=19.554m` —— 单步(5Hz,dt=0.2s)位移 19.5m 物理上不可能(≈100 m/s),其他集 lateral 都是多步累积 3~10m。**高度疑似 (a) teleport-jitter 退化集 或 (b) 前向盲区侧向碰撞**,两者都非 shield 控制律失败,但处置不同。
  **不能凭疑似排除(=gaming gate)**。加 **opt-in 只读取证 dump**(`--dump-contact-frames DIR`,默认关 ⇒ 权威 gate 逐字节不变):对每个接触集导出逐步表(pos/next_pos/`dpos_norm` 单步|Δpos|/vel/collided/GT full+fwd min/`min_depth_px` 最近像素归一化 row,col/预测 D̂/intervention/action)+ 实际渲染 RGB&depth 帧栈(npz)+ 出生帧/碰撞帧 PNG。`min_depth_px` 直接区分正前(中心)/侧向(左右缘)/地面(底部);`dpos_norm` 直接判 teleport-jitter。`run_shield_eval` 返回 `contact_dumps`,`_v0_gate` 挂到 `shield_diag["contact_dumps"]`。纯只读、绝不影响判决(dump 异常也只 warning)。
  **治理**:未动 §4.1(1.5/0.50/0.80)、shield 控制律、env/模型/flags;dump 是 opt-in 诊断。Mac 合成接触集 smoke 验证:`dpos_norm=19.5`、`min_depth_px={row:0,col:0}`、npz+4 PNG 均生成。**待**:H100 同命令加 `--dump-contact-frames` 重跑 → 据 dump 定性接触集 → 若确证退化/盲区则治理安全排除(无效试次 or boxed-in 起点筛选,仿晚⁸/spawn-drop),否则另定。flags 仍全关。commit 本次待 push。
- **2026-08-11(晚¹⁴) —— 晚¹³ 遥测实锤:盲退假设 REFUTED;真因是 ④ eval 关掉了 collector 的出生碰撞守卫,把"出生嵌入"误计为 shield 碰撞 → 恢复 `skip_reset_collision=True` + 重采样。**
  晚¹³ 4090 rollout 带回按集几何:**盲目后退撞后墙 REFUTED** —— 5 个接触集(#2/4/6/7/9)`along_heading_on` = −0.004/−0.145/−0.311/−0.009/−0.301(全≈0)、`lateral_on` ≤0.39,**几乎没动就撞了**,不是退进后墙。真相:5 集全 `len_on=1`、`start_full_min` 低到 0.663/0.879/0.907m(3 个在 1.5m 带内),而**同起点 off 臂飞了 8–13 步**(`len_off` 8/13/11/13/11)才撞 → on/off 臂**出生净空显著不同**。位移≈0 却撞,物理上只可能是**出生就贴/嵌在几何体里**(撞前向 FOV 之外的侧/后/地面几何)。
  **代码钉死根因**:`_run_one`(v0_rollout_eval.py)给 collector 传 `skip_reset_collision=False`,**单独关掉了** collector 默认(`True`)的"出生即碰撞→跳过"守卫;`_run_one_resilient` 只在抛 `RuntimeError` 时重试,出生嵌入只返回 length-1 集不抛异常 → 不重试不跳过 → 被当成一次 shield 碰撞 → `first_i<first_c`=`0<0`=False → **④b 结构性归零**。这与 scan 自己会 reject `spawn_collision`、以及 collector 到处用的默认**自相矛盾**。
  **修复(治理安全,episode 有效性,同 晚⁸/⁹/¹³ 类)**:(1) `_run_one` 恢复 `skip_reset_collision=True`;(2) `_run_one_resilient` 把"出生碰撞→空 episode"当**可重采样 transient**(4090 reset 非确定,on/off 同起点净空都不同已证)→ 重试(=collector 文档说的 "start pose may need resampling"),持续嵌入才 drop(成对丢保持配对);(3) 新增 `drop_stats`,`run_shield_eval`/`_v0_gate` 输出 `spawn_collision_drops`/`health_drops`(**可审计,非静默截断**);(4) `_episode_geom_diag` 加只读 `start_collided_on/off`(存活集须为 False)。
  **诚实警示**:好 shield 会让"净空充足起点"零碰撞 → ④b 依冻结 metric(v0_metrics.py:286,无接触→before_frac=1.0)**空过**;此时 ④ 靠 ④c(ratio 0.172,近带占用 5.7× 更低)+ off 臂每集都撞(证明每个场景都是真碰撞风险)成立。**未动** §4.1(1.5/0.50/0.80)、shield 控制律、env/模型/flags。冻结 spec ④ 方法学追加注记(晚¹⁴)。新单测 3 个(persistent-drop / resample-recover / drops-surfaced);rollout+followups+collector 44 全过。
  **待**:4090 同命令重跑 ④ → 看 `spawn_collision_drops`>0 且接触集消失/变真接近 → 期望 ④ PASS → `_v0_gate --merge` 全四 → exit 0 才翻 flags。commit 本次待 push。
- **2026-08-11(晚¹³) —— 晚¹² 修好 ④c(ratio 12.96→0.192 ✓),但 ④b 仍 FAIL(before_frac=0);加只读按集几何遥测定位 step-1 碰撞方向。**
  晚¹² 有界后退在 4090 权威 rollout 上**大幅改善 ④c**:`near_coll_rate_ratio` 12.96→**0.192 ≤0.80** ✓、`near_coll_rate_on` 0.385→0.0066、`coll_after_latch` 4→3、6 集里 3 集存活满 200 步(shield 成功)。但 **④ 仍 FAIL 于 ④b**:`intervention_before_contact_frac=0.0`。
  **只读定位**:`collided` 是 post-step、只在 done 的终止步(v0_rollout_eval.py:571)→ `first_coll_step=0` ⟹ 该集**长度=1**(第一个动作后即撞)。3 个接触集全长度≈1 → `first_i<first_c` 恒 `0<0`=False → ④b 恒 0。**根因收窄(非 shield 控制律、非出生嵌入)**:起点 `start_clearance_m=3.0` + `spawn_collision` 拒 → 起点前向 FOV 净空 ≥3.0m(不在带内);off 臂 ~9 步才撞 → 每步 ~0.5–2.8m → **一步跨不过 5–25m 前障** → step-1 碰撞对象**不是前障**。shield 退 body −x + **前视相机看不到后方** → 强指向**盲目后退撞未感知的后/侧墙**(3 个 boxed-in 起点);design-4 有界后退在后墙就在第一退步内时仍无能为力。
  **但碰撞方向尚未实证,不据此定 fix**。加只读 `_episode_geom_diag`(v0_rollout_eval.py:600;proprio 位置 + GT 起点净空,均不入策略图)→ 按集出 `along_heading_on`(沿起点朝向净位移,**<0=后退**)、`start_full/fwd_min`(起点净空)、`len_on/off`、`coll_first_on/off`、`interv_first`,经 `run_shield_eval` 返回 `episode_diag`、`_v0_gate` 挂到 `shield_diag["episodes"]` 打印。新单测 `test_episode_geom_diag_flags_backward_retreat`;51 全过。
  **治理**:纯只读诊断,§4.1(1.5/0.50/0.80)、shield 控制律、env/模型/flags **全未动**。**待**:4090 同命令重跑 ④ → 看 `episodes[*].along_heading_on` 符号:若接触集 <0 → 实锤盲退撞后墙 → fix = 起点选择排除 boxed-in(仿晚⁸ 记录航向,治理安全的 episode 过滤);若 ≥0 或起点净空<standoff → 另定。flags 仍全关。commit 本次待 push。
- **2026-08-11(晚¹²) —— ④ shield:保持(悬停)→有界状态反馈后退;修惯性滑进带停留(near_count_on 200/200、ratio 12.96)。re-freeze。**
  晚¹¹ 4090 权威 rollout 推翻晚¹⁰"保持"假设:零 body-delta **不刹前向动量** → latch 关掉策略后机体**惯性滑进 1.5m 带并停在里面** —— `near_count_on` 达 200/200、`near_coll_rate_on=0.385`、`ratio=12.96`(比后退设计 1.24 更差)、`coll_after_latch=4`、`first_coll_step max=8`、`first_near_on max=15`。**分析错误定位**:晚¹⁰ 误以为零 delta 能定住位置;实际设计 (2) 的后退**身兼两职** —— 抵消乐观预测器 + 刹前向动量;晚⁷ 消除了乐观偏差,但**动量刹车仍需要**,悬停把它一起丢了。
  **修法**:`override_action` 恢复 `retreat_step_m=3.0`,`D̂ < min_depth_m`(反应 standoff)时后退 body −x(刹动量+退出带),`D̂ ≥ standoff` 时保持零 delta(**不再后退**→不盲目倒进后墙)。晚⁷ 使 D̂ 近带准确/欠读(前向 6.4→0.65m)故 `D̂≥standoff ⟺ 真 ≈3m 净空`(先前"退到 D̂ 安全再悬停"停带里正因乐观 D̂ 在 GT 仍<1.5 时就过 standoff,该前提已消失);latch 使 standoff 稳定后策略不再逼近。综合了后退(晚¹⁰前)刹动量 + 保持(晚¹¹)不撞后墙两者的正确部分。collector.py:158-163 在 `depth_min_pred` 填入后、`override_action` 前调用 → 状态反馈拿得到当前步 D̂(已核 wiring)。
  **治理**:改的只是 shield **保持→有界状态反馈后退**控制律(shield 是被测系统);**④a 1.5 / ④b 0.50 / ④c 0.80 钉死值不动**;env/模型/flags 未动。冻结 spec §④a 追加"保持→有界状态反馈后退 更正(re-freeze 晚¹²)"。单测 `test_threshold_shield_holds_not_retreats_after_latch`→`test_threshold_shield_bounded_state_feedback_retreat_after_latch`;shield 4 + collector/rollout/metric 34 = 38 全通过。
  **待**:4090 同命令重跑 ④,预期 `coll_after_latch→0`、`ratio≤0.80`、④ PASS → `--merge` 四信号权威判决。flags 仍全关。commit 本次待 push。
- **2026-08-11(晚¹¹) —— ④ shield:连续后退→闩锁保持(悬停);修盲目倒退撞后墙(coll_after_latch=9/9)。re-freeze。**
  晚¹⁰ 机制遥测决定性:`near_before_latch=0`(闩锁不晚)、`coll_after_latch=9/9`(闩锁后全撞)、`first_interv` p50≈0 vs `first_coll` p50≈33、`steps_on` 34 vs `steps_off` 10.5。
  根因:`override_action` 每步 body −x **无后向感知**,策略被 latch 关掉后盲目倒退,封闭场景退进后墙 —— 活久 3 倍但仍 9/10 撞 = 把碰撞**转移**到后方,非避障。
  **前提已消失**:逼出"连续后退"的是**乐观预测器**(approach AbsRel 0.167),而 **晚⁷** 已消除(前向 D̂ 6.4→0.65m、近带 P(trigger)=1.0、近带准/欠读)。
  **修法**:`override_action` latch 后**保持(返回 `np.zeros(4)` 悬停)**、删 `retreat_step_m`。触发 `min_depth_m=3.0`+欠读 → 真 ≈3m(带外)闩锁 → latch 使策略不顶入、零 delta 无前冲无盲退 → 前向稳 ≥standoff(near_rate_on≈0)、无后墙撞(coll_after_latch→0)、~3m 先于接触介入(④b)。
  保持优于纯悬停(有 latch,不再逼近)、优于后退(晚⁷ 移除了后退所补偿的乐观偏差)。冻结 spec §④a 追加"后退→保持 修订(re-freeze 晚¹⁰)"。新单测 `test_threshold_shield_holds_not_retreats_after_latch`;顺修 2 个 latch 前的顺序依赖老测(负例用新实例)。全 shield/rollout/metric 测通过。
  **待**:4090 同命令重跑 ④,预期 `coll_after_latch→0`、`ratio≤0.80`、④ PASS → `--merge` 四信号权威判决。flags 仍全关。commit 本次待 push。

- **2026-08-11(晚¹⁰) —— scan 修复生效(accepted=11);② PASS;④ 仍 FAIL(near_coll ratio=1.24>0.80),加只读机制遥测定位。**
  晚⁹ 全画面/碰撞判据把 `accepted` 0→**11**、`probe hits`=11、`obstacle_ok`=11 —— scan 阻塞彻底解除。权威 rollout:
  **② PASS**(progress 10.5 vs random −0.65;final_dist 19.5 vs 26.5,双余量过)。**④ FAIL**:④b `intervention_before_contact=0.636≥0.5` ✓,
  但 ④c `near_coll_rate_ratio = on/off = 0.0389/0.0312 = **1.24** > 0.80` ✗,且 `n_contact_episodes=**11**`(shield 开启臂 11 集**全部仍碰撞**)。
  与"闩锁+单调后退首次 breach 后 GT 间距严格单调增、on 臂不该碰撞"的设计**直接矛盾** → 要么闩锁太晚(predictor 在实测帧偏乐观),
  要么后退量压不住前向位移/动量。**不靠改 shield/阈值凑过**(那是 gaming gate)。加纯后处理只读遥测 `_shield_diag`(从已返回 `masks` 算,
  不改 rollout/不动阈值/模型/flags):每臂逐集 `steps`/`near_count`、on 臂 `first_interv`/`first_coll`/`first_near` 步、
  以及 `near_before_latch`(near 早于闩锁数)、`coll_after_latch`(闩锁后仍碰撞数)。一次重跑即可判"闩锁太晚"还是"后退失效"。
  **待**:4090 重跑 ④(同命令,遥测自动带出 `shield mechanism diag`)→ 据实定位后再动。flags 仍全关。commit 本次待 push。

- **2026-08-11(晚⁹) —— ④ probe 判据对齐评测臂(中心裁剪→全画面 + 碰撞即接受);修 probe/eval 不匹配。**
  晚⁸ 上记录航向优先只把 proxy_ok 19→22,`accepted` 仍 0。加只读遥测(每个 proxy-OK probe 记
  `reached_fwd_m`/`reached_full_m`/`travel_m`/`collided`)后一轮定论:`travel_m` p50=**24.8m**(飞得动、到位),
  `collided`=**10/22**(真撞墙),但 `reached_fwd_m`(中心裁剪 0.3)最低只 **1.63m** 从没 <1.5。**根因**:probe 接受用
  `_forward_min_depth(中心裁剪)<1.5`,而 ④ 评测臂 `_episode_masks` 的 `near_coll_off` 用 `_full_min_depth(**全画面**)<1.5`
  —— probe 严过评测,头对头碰撞几何落在中心裁剪外(中心停 1.63m 但全画面必 <1.5),22 个真起点(含 10 碰撞)全被误杀。
  **修法(对齐 harness、不碰 §4.1 的 1.5m)**:probe 接受改为 **全画面 `_full_min_depth<near_m` 或 `collided`**
  (二者都被评测臂 `near_coll_off`/`collided_on` 同款读取 → near_coll_off>0 可复现;碰撞是最抗 RPC 抖动的铁证)。
  补 `reached_full_m` 遥测;新增 probe 单测(中心裁剪触底 1.6m 但角落 <1.5 → 全画面接受、旧中心判据会拒),19 测全过。
  **待**:4090 重跑 ④,预期 `accepted>0`(全画面/碰撞判据)→ `--merge` 四信号权威判决。flags 仍全关。commits 4eab52f/cf6de28/本次待 push。
- **2026-08-11(晚⁸) —— ④ scan 用采集记录航向(修 probe_no_hit=19/accepted=0);待 4090 用新头对头语料重跑。**
  晚⁷ 深度头修好后,4090 ④ rollout 仍找不到近障起点。专采头对头语料 `dataset_v0_headon_20260811`(34/34 可用)
  后扫 656 对:`candidates=82 / proxy_ok=19 / probe_no_hit=19 / accepted=0` —— **19 个朝障候选全被 probe 判否**。
  根因:`make_obstacle_facing_episodes` 丢弃采集记录的接近航向,改用 8 网格(0/45/…/315°,最多差 22.5°);
  中距正前障碍只擦到 0.3 中心裁剪边缘 → proxy 过但直线 probe 从旁擦过。**修法(harness,非 §4.1)**:
  `_obstacle_candidate_positions` 现返回 `(positions, 记录yaw)`;`make_obstacle_facing_episodes` 新增可选
  `candidate_yaws`,每个位置**先试记录航向**再走 8 网格兜底(头对头语料里记录 yaw 正对障碍 → probe 正撞)。
  向后兼容(不传则纯网格,单测不变);新增 off-grid 单测(0.3rad≈17° 网格打不中、给记录航向即命中),3 测全过。
  **治理**:选点=episode 过滤器非 gate 阈值(docstring 明载"②/④ harness 几何修正,非 §4.1");env/阈值/模型/flags 不动。
  **待**:4090 起渲染器 → `_v0_gate --signals 2,4 --rollout-eval --rollout-dataset dataset_v0_headon_20260811
  --depth-ckpt <晚⁷ 新头>` 重跑 ④ → `--merge` 四信号权威判决。flags 仍全关。
- **2026-08-11(晚⁷) —— near-band 重训验证双绿(①d 不退 + 近带感知实锤修复);待 4090 重跑 ④。**
  合并 `dataset_v0_local_depth + dataset_v0_approach_merged`(`_merge_datasets`)→ DA3 头 fresh 重训
  (near_weight=3.0,本地 HF cache 权重,`pip install safetensors` 解依赖)→ `depth_ckpt_da3_near_20260811`。
  **① 权威复验**(`--signals 1 --dataset dataset_v0_local_depth`):①d AbsRel=**0.0483 ≤0.30**,不退反降(旧代表 0.132)。
  **⑤ 诊断复跑**(`_diag_depth_vs_gt` on approach_merged):FORWARD 正前 `GT[0,1.5)` D̂p50 **6.415→0.645m**、
  `P(trig)` **0.000→1.000**、`P(over)` 1.000→0.377、AbsRel 6.757→0.198;full-field HEADLINE 近带 `GT<1.5` 与
  反应窗 `[1.5,3)` **P(trig) 双双=1.0**(旧 0.697/0.802)。1.5m 正前墙从被读 ~6.4m 修到 ~0.65m,shield 每帧必刹 →
  ④ 感知层根因消除。§4 ②④ 命令 `--depth-ckpt` 已切至新头。**待**:4090 起渲染器 → `_v0_gate --signals 2,4
  --rollout-eval` 重跑 ④(用新头)→ `--merge` 四信号权威判决。flags 仍全关。
- **2026-08-11(晚⁶) —— 定位 ④ 真根因=深度头近障乐观 → 深度头 near-band 重训(离线诊断先证,再修 loss)。**
  晚⁵ 修完 flaky 后 ④ 仍未过。用只读离线诊断 `_diag_depth_vs_gt`(不碰 gate/spec/config/flags,仅读 ckpt+语料,
  按 shield 同款 `DepthMinPredictor` 逐帧配 D̂ vs GT、按 GT 深度分箱)在 `dataset_v0_approach_merged`(115 集/14487 帧)
  上**决定性证实**:FORWARD 正前裁剪 `GT[0,1.5)` → D̂ p50=**6.415m**、`P(over)=1.000`、`P(trig)=0.000`
  —— 1.5m 正前墙被预测成 ~6.4m,shield 永不刹。这是**安全攸关的近前向深度质量问题**,被聚合 ①d(远/地板像素主导)
  掩盖;调 shield 余量/阈值都救不了(各 GT 箱的 D̂ 分布重叠)。修:`dynamics_torch.depth_head_loss` 加
  **near-band 强调项** `near_weight*mean(AbsRel | GT≤near_focus_m)`(默认 `near_weight=0.0` 保持旧行为/单测惰性;
  `train_depth_head._load_depth_cfg` setdefault `near_weight=3.0/near_focus_m=5.0`)。加可复用 `_merge_datasets.py`
  (npz 顺序重编号复制+provenance manifest)。**治理**:改训练 loss 属 §6 Step 5/6 范围内;①d gate 度量
  (`v0_metrics.depth_absrel`,全掩码,阈值 0.30)与所有 §4.1 阈值不动;`near_focus_m` 是训练超参非 gate 阈值。
  commits `df08bfa`(诊断)/`dc4a8b5`(near-band loss)/`39db5ea`(merge)。**待**:合并
  `dataset_v0_local_depth + dataset_v0_approach_merged` → 重训 DA3 头(fresh,`--eval-every 200`)→ 权威 ①d≤0.30 复验
  + 诊断复跑(FORWARD 近带 D̂p50 下压、`P(over)`↓)→ 双绿才上 4090 重跑 ④。env/模型/flags 未动。
- **2026-08-11(晚⁵) —— rollout 对 reset/健康失败重试+跳过(修 flaky 深度帧崩全局 gate)。**
  晚⁴ 重跑时崩在 shield-off 臂的 `env.reset`:`depth sanity failed: depth nearly constant (span=0.239, std=0.048)`
  —— 一个抖动/近似恒定的深度帧让 `_assert_healthy` 抛 `RuntimeError` 冒到顶,**整个 40min gate 挂掉**。
  且 ④ 专挑"正对障碍"起点,墙面填满 FOV 本就近似恒定深度 → 最易触发该守卫。修:`v0_rollout_eval`
  加 `_run_one_resilient`,对**瞬态 reset/健康失败**(白名单 marker:sanity/‌no depth/no imu/renderer)**重试 2 次
  (间隔 0.5s,reset 顶部 `_connect` 会重连),仍失败则**跳过该起点**(②/④ 都按起点整体跳,保持两臂配对、同 N)。
  **守卫不削弱**:坏帧永不被**评分**,只重试/跳过;每次跳过打 WARNING(不静默丢)。非瞬态错误(真 bug)照常抛。
  全跳空 → 现有 `depth_steps==0`/空数组守卫仍 fail-closed。env/§4.1/模型/flags 未动。
- **2026-08-11(晚⁴) —— ④ 连续后退 + 触发余量(re-freeze ④a 注;修 ratio 反转 6.10 + before_frac 0.333)。**
  晚³ 修好后 ④ 首次可测,但仍 FAIL:`near_on=0.205 ≫ off=0.034`(ratio 6.10)、`before_frac=0.333`、
  `n_contact=3`。诊断:**深度预测器在 1.5m 边界偏乐观**(approach AbsRel≈0.167),"恰好 1.5m 反应"必然太晚:
  1. **连续后退(`safety.override_action`)**:晚³ 的"退到 `safe_depth`(2.5)再悬停"仍把机体停在 GT<1.5 带内
     (d̂ 到 2.5 就停、GT 还 ~1.2)→ latch 整集刷分子。改:latch 后**每步都后退,永不悬停**,机体单调退带 →
     `near_on≈0`、总帧变大 → ratio 稳过。删 `safe_depth_m` 字段。
  2. **触发余量(`min_depth_m` 1.5→3.0)**:`before_frac≥0.5` 在噪声预测器下对"边界反应"数学不可满足
     (在碰撞边界才反应=已在边界)。shield 触发提到 3.0m(> 度量 1.5),提前于进带反应 → 不进带、零碰撞 →
     `before_frac` 空过(`check_shield_effectiveness` 无碰撞集→1.0)。落点:`run_shield_eval(shield_trigger_depth_m=3.0)`
     与 metric 掩码**解耦**、`_v0_gate` 显式传、`train_rl._build_safety` live 默认 3.0。
  **治理**:改的是 §4.1 ④a 协议注「与 1.5 对齐」→ 用户批准 **re-freeze**(冻结 spec 已加"④ 反应余量注")。
  **度量端 `near_collision_depth_m=1.5`、④b 0.50、④c 0.80 钉死值不动**;env/模型/flags 未动。起点前向最小 5.578m>3.0m
  不误触发。**待 H100 pull + 重跑 ②④。**
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
