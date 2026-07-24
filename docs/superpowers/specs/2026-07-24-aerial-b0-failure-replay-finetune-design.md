# Aerial B0 失败回采与纠偏微调设计

日期：2026-07-24  
状态：待用户审阅  
范围：AirSim seen 路线上的 B0 纠偏式 DAgger 回采，以及双 RTX 4090 微调

## 1. 目标与成功标准

从 B0 `step_5000` 出发，使用 AirSim 闭环失败状态和 OpenFly annotation 路径专家构造纠偏数据，再进行一次受控微调。

第一阶段成功标准（S1）：

- 在固定 held-out seen-20 上，最佳微调 checkpoint 的平均 NE 相对 B0 `step_5000` 降低至少 20%。
- 同时报告 SR、SPL、median NE 和逐 episode NE 差值。
- held-out seen-20 不参与回采、训练或 checkpoint 选择之外的参数调节。
- 在 seen 结果锁定前不运行 unseen 评测。

S1 不要求 SR 大于零；出现正 SR 作为额外结果报告。

## 2. 机器分工

| 角色 | 主机 |
|------|------|
| B0 推理与失败回采客户端 | eval H100 `a25689@10.239.121.25:31126` |
| AirSim 渲染 | RTX 4090 `yao@10.229.20.125:22`，RPC `10.229.20.125:41451` |
| 纠偏微调 | 双 RTX 4090 `a25689@10.239.121.14:30879` |
| 现有 B1 联合训练 | train H100 `a25689@10.239.121.25:31893` |

渲染、推理和微调保持分离。微调不占用正在进行 B1 训练的 H100。

禁止通过 Desk API 修改机器人侧网络；本工作不访问或修改 `10.229.66.70`。

## 3. 数据划分与防泄漏

从 seen annotation 中建立两个不相交集合：

- `heldout-seen-20`：当前固定评测集合，只用于基线和微调后评测。
- `collection-seen-40`：同类 seen 场景中的 40 条不同路线，用于路径专家预检与纠偏回采。

以稳定 route ID 去重，并将两个集合的 route ID、源 annotation 哈希和随机种子写入 manifest。启动回采前必须断言交集为空。

## 4. 路径专家

### 4.1 路径进度

1. 将 AirSim 当前位置投影到 annotation 参考折线。
2. 保存单调路径游标；后续投影不得退回已通过的旧路段。
3. 从投影位置沿折线前视 6m；接近终点时将前视距离缩短为剩余距离。

### 4.2 动作生成

从当前位姿到前视目标计算连续 body-frame 动作：

`(dx, dy, dz, dyaw)`

动作按现有 aerial OpenFly 数据统计范围裁剪。训练数据保留连续动作；只有 AirSim 执行时才映射为最近 OpenFly primitive，避免将量化误差写入监督标签。

路径专家必须先通过：

- 投影、单调游标、坐标变换和动作裁剪单元测试；
- mock bridge 专家回放；
- collection-40 上 AirSim oracle-only SR ≥95%。

oracle-only 未过门禁时，不得进入 DAgger 回采。

## 5. 阈值接管 DAgger

正常情况下执行 B0 动作，同时由路径专家为每个有效状态生成监督标签。

满足任一条件时由专家接管：

- 当前位置距参考路径大于 9m；
- 偏离距离或 NE 连续 3 步恶化；
- 路径进度连续 8 步没有增长。

距参考路径小于 6m 且连续稳定 3 步后，将控制权交还 B0。

若偏离超过 30m，或连续 20 步无进展，则结束当前 episode。丢弃无法可靠投影的失真尾段，但保留终止前的有效纠偏样本。

每个训练样本包含：

- AirSim RGB；
- state；
- instruction；
- 连续专家动作（训练标签）；
- policy action、expert action、executed action；
- 路径进度、偏离距离、NE；
- intervention 标记和 episode/route 元数据。

专家为所有有效状态打标签，而不只记录接管帧。

## 6. 回采数据集

第一轮预算为 40 episodes，最多约 4000 个策略状态。结果写入独立 LeRobot correction dataset，不修改原 OpenFly 数据。

每条 episode 完成后原子更新 manifest，以支持断点续采。数据验收要求：

- collection 与 held-out route 无重叠；
- RGB、state、action 数量一致；
- 所有数值 finite；
- 专家动作位于训练 action 范围内；
- 每条异常终止均有原因；
- 汇总策略/专家接管率、路径偏离分布与有效样本数。

RPC 中断只重试当前 episode 一次。AirSim renderer 异常时，仅在 `10.229.20.125` 运行既有恢复脚本。

## 7. 双 4090 微调

### 7.1 已验证资源

`10.239.121.14:30879` 提供：

- 2× RTX 4090，每卡约 24GB；
- torch `2.7.1+cu128`；
- accelerate `1.12.0`；
- DeepSpeed `0.18.5`；
- 约 503GiB RAM；
- 已存在 FastWAM 代码和 Python 环境。

H100 上当前全参 ZeRO-2 训练约使用 63GB/卡，不能直接照搬到 4090。

### 7.2 配方

首选 DeepSpeed ZeRO-2 + optimizer CPU offload：

- 2 GPU；
- bf16；
- micro-batch 1/GPU；
- gradient accumulation 1；
- 保持 gradient checkpointing；
- 从 B0 `step_5000` 加载模型权重；
- 不恢复原 optimizer/scheduler；
- learning rate `1e-5`；
- warmup 50 steps；
- cosine decay；
- weight decay 与 B0 保持一致；
- 原始 OpenFly / correction weighted sampling = 75% / 25%；
- 最多 1000 steps；
- 保存 step 250、500、1000。

### 7.3 显存门禁

1. 先执行 1-step smoke。
2. 记录两卡峰值显存、loss 和参数 finite 状态。
3. 继续运行到 10 steps。
4. 通过条件：峰值显存低于 23GB/卡，且无 OOM/NaN。
5. 若 OOM，仅允许缩小通信 bucket并清理缓存后重试一次。
6. 第二次仍失败则停止 ZeRO-2 路线，改为单独评审 ZeRO-3 + offload；不得反复撞显存或擅自改变训练目标。

## 8. 评测与模型选择

将 step 250、500、1000 checkpoint 复制回 eval H100，并使用与 B0 基线完全相同的：

- held-out seen-20 annotation；
- seed；
- max steps；
- success distance；
- AirSim renderer；
- policy action quantization。

以平均 NE 最低的 checkpoint 作为候选。S1 的唯一主通过线是相对 B0 `step_5000` 的平均 NE 降幅至少 20%。

为避免均值掩盖退化，同时输出：

- SR / SPL；
- mean / median NE；
- 每条 episode 的 baseline 与 finetune NE；
- 改善、持平、退化 episode 数；
- 接管训练数据的失败类型分布。

## 9. 排程与恢复

1. 完成 B0 `step_5000` 指标并锁定基线。
2. 等当前 B0 视频队列释放 eval H100 和 renderer。
3. 生成 collection-40 manifest 并完成防泄漏检查。
4. 路径专家通过单测、mock 和 AirSim oracle-only 门禁。
5. 回采 40 episodes 并验收 correction dataset。
6. 通过 SHA256 校验复制 checkpoint、数据与配置到双 4090。
7. 运行 1-step/10-step 显存 smoke。
8. 运行最多 1000-step 微调。
9. 回传三个 checkpoint，完成 held-out seen-20 评测并选择模型。

训练进程需保存明确状态文件。意外退出时只从最近完整 checkpoint 恢复；损坏或未完成产物不得参与同步和评测。

## 10. 未达标处理

若所有 checkpoint 均未达到 S1：

1. 按转向错误、停滞、过冲、高度错误和异常终止分类；
2. 检查专家动作与执行 primitive 的差异；
3. 检查 correction 采样比例是否实际达到 25%；
4. 检查改善是否仅集中在少数路线；
5. 提交诊断后再决定是否调整接管阈值、扩大 collection 或进行第二轮 DAgger。

首轮失败不会自动扩大数据、追加训练或开始 unseen 评测。
