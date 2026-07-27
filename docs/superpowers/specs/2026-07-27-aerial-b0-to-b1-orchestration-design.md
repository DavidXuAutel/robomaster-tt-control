# Aerial B0→B1 自动编排设计

日期：2026-07-27  
状态：已确认  
范围：B0 joint-video 收尾评测、baseline 锁定，以及 B1 failure-replay 训练与每 checkpoint 同步仿真评测  
依赖：`docs/superpowers/specs/2026-07-24-aerial-b0-failure-replay-finetune-design.md`（r2）

## 1. 目标

在当前 B0→joint-video 训练结束后，按训练结果自动锁定 B1 起始 checkpoint，启动 B1 failure-replay 训练，并在每个 checkpoint 持久化完成后同步完成 AirSim seen-20 仿真评测。

成功标准：

1. B0 的 `step_001000`–`step_005000`（凡存在且 SHA256 完整）均完成 held-out seen-20 评测；
2. 以 mean NE 最低者写入 `baseline_lock.manifest.json`，并计算 `S1_NE = 0.8 × baseline_mean_NE`；
3. B1 在前置门禁通过后自动启动；
4. B1 的 `step_000250` / `step_000500` / `step_001000` 在持久化完成后自动进入评测队列并完成 seen-20；
5. 全部结束后输出 S1 比较报告，且编排过程可断点恢复、不重复已完成步骤。

## 2. 机器分工

| 角色 | 主机 |
|------|------|
| B0 / B1 训练 | `a25689@10.239.121.22:31660`（2×H100） |
| 评测与回采客户端 | `a25689@10.239.121.22:30682`（1×H100） |
| AirSim 渲染 | `yao@10.229.20.125`，RPC `10.229.20.125:41451` |

规则：

- 训练与评测分卡，不抢训练 GPU；
- 评测串行占用 AirSim，避免多评测并发争用 renderer；
- 热路径使用各 pod 本地 `/tmp`；Ceph 仅作持久镜像；
- 禁止访问或修改 `10.229.66.70` / 机器人网络。

## 3. 编排形态：可恢复状态机

采用幂等状态机，而不是一次性 shell waiter 串联。每个阶段写明确 status / manifest；进程重启后只继续未完成工作。

状态文件根目录：

- 训练侧：`/tmp/aerial_cache/orchestration/`
- 评测侧：`/tmp/aerial_eval_cache/orchestration/`
- 持久镜像：`/home/a25689/aerial_cache_shared/orchestration/`

主状态机阶段：

1. `WAIT_B0_COMPLETE`
2. `EVAL_B0_CHECKPOINTS`
3. `LOCK_BASELINE`
4. `B1_GATES`
5. `RUN_B1_TRAIN`
6. `EVAL_B1_CHECKPOINTS`
7. `S1_REPORT`
8. `DONE` / `BLOCKED` / `FAILED`

任何 `BLOCKED` 必须写明阻塞原因，不得伪造数据或跳过门禁。

## 4. B0 收尾与 baseline 锁定

### 4.1 等待训练完成

当前 stamp：`20260727-072347-5k-2gpu-b0-to-joint-video`。

完成条件：

- 训练进程正常退出或日志达到 `step=5000/5000`；
- `step_005000.pt` 存在且伴随 `.sha256`；
- persist watcher 已将该权重同步到 shared / FastWAM runs。

仅接受完整 checkpoint（文件大小稳定 + sha256 存在）。`.partial` 或不完整拷贝不得进入评测。

### 4.2 评测所有可用 B0 checkpoint

对 `step_001000`–`step_005000` 中存在且完整的权重，按升序串行评测：

- annotation：held-out `seen_airsim16_m1a20.json`
- seed：42
- max_steps：100
- bridge：openfly → `10.229.20.125:41451`
- policy：fastwam
- task：与训练一致的 `aerial_joint_b1_joint`

已有有效 `metrics.json`（含有限 `NE`、`n=20`）则跳过。  
`step_001000` 已有结果（NE≈151.07，SR=0）可直接复用。

### 4.3 锁定规则

用户确认：自动选择 mean NE 最低者。

锁定产物 `baseline_lock.manifest.json` 必须包含：

- checkpoint 路径与 SHA256
- train stamp
- 全部候选 checkpoint 的 metrics 路径与 mean NE
- `baseline_mean_ne`
- `s1_ne = 0.8 * baseline_mean_ne`
- 选择时间与编排版本

若多个 checkpoint NE 相同，优先更晚的 step。

## 5. B1 前置门禁

在启动 B1 训练前必须全部通过：

1. `baseline_lock.manifest.json` 存在且 SHA256 校验通过；
2. `seen_airsim16_collection_source.json` 已部署到固定路径，且与 held-out route ID 交集为空；
3. collection-40 manifest 生成完成；
4. 路径专家通过单测 / mock / AirSim oracle-only 门禁（SR≥80%，median NE<20m，投影失败=0）；
5. DAgger 回采 40 episodes 完成，correction LeRobot dataset 验收通过；
6. 75/25 weighted mix 配置就绪；
7. 训练侧 `/tmp/aerial_ft_cache`（或等价本地缓存）完成 SHA256 同步 smoke；
8. 双 H100 1-step / 10-step smoke 通过（finite loss、峰值显存 <90%）。

任一门禁失败：进入 `BLOCKED`，报告阻塞项，**不**自动伪造 collection、不跳过 oracle、不擅自改 λ 或采样比例。

## 6. B1 训练配方

与 r2 failure-replay 设计一致：

- 主机：`:31660` 2×H100，ZeRO-2 no-offload，bf16
- resume：仅加载锁定 baseline 的**模型权重**，不恢复 optimizer/scheduler
- `lambda_action=1`，`lambda_video=0`
- 数据源：OpenFly original 75% / correction 25%（按 sample，非 episode）
- LR：`1e-5`，warmup 50，cosine
- max_steps：1000
- save_every：250 → 产出 `step_000250` / `step_000500` / `step_001000`
- 每 50 step 记录 source counts；任意完整 200-step 窗口 correction 命中率必须 ∈[20%, 30%]，否则该 run 无效

训练状态写 `ft.status`：`RUNNING` / `COMPLETED` / `FAILED`。

## 7. 每 checkpoint 同步仿真评测

### 7.1 触发

B1 训练侧 persist 一旦确认某 `step_*.pt` 完整（size 稳定 + sha256）：

1. 将该 checkpoint 追加到评测队列；
2. 评测侧串行消费队列；
3. 训练继续，不等待当前评测完成。

### 7.2 评测协议

与 B0 baseline 评测完全相同：held-out seen-20、seed 42、max_steps 100、同一 AirSim renderer、同一 action 执行路径。

产物：

- `/tmp/aerial_eval_cache/results/b1_<stamp>/step_XXXXXX_seen20/metrics.json`
- 持久镜像到 shared orchestration / runs 目录

### 7.3 并发约束

- 同时只允许一个 `run_closed_loop` 占用 AirSim；
- B0 剩余评测与 B1 评测共用同一队列，FIFO；
- 若 AirSim RPC 中断：当前 episode 重试一次；仍失败则标记该 checkpoint 评测 `FAILED` 并告警，不杀死训练。

## 8. S1 报告

B1 三个 checkpoint 全部评测完成后：

1. 选择 mean NE 最低者；
2. 与 `baseline_lock.manifest.json` 比较；
3. 通过条件：`best_ft_mean_ne ≤ s1_ne`；
4. 输出 `ft_selection_report.json`：mean/median NE、SR/SPL、逐 episode 差值、improve/flat/regress 计数、量化 gap（若可得）；
5. 若未通过：写诊断脚手架，**不**自动扩大数据、追加训练或启动 unseen。

## 9. 幂等与恢复

- 已有有效 metrics / lock / correction manifest 的步骤默认 skip；
- 仅接受完整 checkpoint + sha256；
- 训练意外退出：从最近完整 B1 checkpoint 恢复训练进度；若尚无 B1 checkpoint，则从锁定 baseline 权重重新启动 B1（不复用损坏产物）；
- 编排器自身崩溃：重读状态文件，从最近未完成阶段继续；
- 禁止把 `/tmp` 热产物当作唯一真相；关键 lock / metrics / correction 必须镜像到 shared。

## 10. 排程摘要

```text
WAIT_B0_COMPLETE
  -> EVAL_B0_CHECKPOINTS (1000..5000, reuse existing)
  -> LOCK_BASELINE (min mean NE)
  -> B1_GATES (collection / oracle / dagger / sync / smoke)
  -> RUN_B1_TRAIN (1000 steps, λ_a=1, λ_v=0, 75/25)
       └─ on each ckpt ready → enqueue EVAL
  -> EVAL_B1_CHECKPOINTS (250/500/1000 via shared queue)
  -> S1_REPORT
  -> DONE | BLOCKED | FAILED
```

## 11. 非目标

- 不在首轮自动把 B1 扩到 >1000 steps；
- 不在 S1 失败后自动启动第二轮 DAgger；
- 不在 seen 锁定前跑 unseen；
- 不把 step1000 的差 NE 当作最终 B0 基线（必须以全部可用 checkpoint 的最优者为准）。
