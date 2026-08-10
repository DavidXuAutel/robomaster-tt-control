# Signal ③ OLS + 光轴投影代理设计

状态：待主人审阅；仅为实现设计，不修改 frozen spec。

## 1. 目标与边界

当前代理以端点深度中位数变化
`|median(D_last) - median(D_first)|` 对比三维净位移 `||Δp||`。现有数据表明：

- 75-episode 合并集的 GT forward-only 为约 0.229，距离 0.25 门限过近；
- window 从 8 增到 16 后 GT 反而恶化，说明端点聚合噪声较大；
- base64 模型已将 AbsRel 降到 0.205，但现代理下 D̂ 仍约 0.319。

本设计只替换 Signal ③ 的尺度代理与诊断路径，不改变 ①d、②、④，不翻任何
enable 旗标。frozen spec 的正式修改必须由主人/Claude 裁决。

## 2. 推荐代理

### 2.1 光轴 VIO 路径

对窗口内每一步速度积分得到位置 `p_t`，由窗口平均 heading 得到水平光轴单位向量
`u`。累计有符号光轴路径：

`s_t = Σ_i max(0, (p_i - p_{i-1}) · u)`

最终尺度基准为 `s_axis = s_{L-1}`。只保留现有 forward cosine、最小运动和支持率
条件；支持率相对 `s_axis` 计算，不再相对 `||Δp||` 计算。

### 2.2 多帧 OLS 深度尺度

每帧在导航波段 `[1, 40] m` 内计算深度中位数 `d_t`。在至少 4 个有效帧上拟合：

`d_t = a - α s_t`

预测的深度尺度为：

`ŝ_D = |α| s_axis`

窗口相对误差为：

`rel = |ŝ_D - s_axis| / max(s_axis, ε) = ||α| - 1|`

Signal ③ 仍取所有有效窗口 `rel` 的中位数，并保留 `n >= 8` 的最低样本要求。
门限暂不改，离线实验仍以 0.25 作比较。

## 3. 数据流与代码边界

- `experiments/aerial/rl/vio.py`
  - 新增光轴累计路径计算；
  - 新增 OLS 深度尺度计算；
  - 保留旧端点代理供 A/B 对照和兼容测试。
- `experiments/aerial/rl/v0_metrics.py`
  - 增加可选的 `proxy_kind="ols_axis"`；
  - 默认值暂不切换，直到 frozen spec 裁决。
- `experiments/aerial/rl/_v0_gate.py`
  - diagnose 支持同时输出 legacy 与 OLS-axis 的 GT / D̂ 行；
  - 正式 gate 暂不切换默认代理。
- `experiments/aerial/rl/tests/`
  - 匀速正向接近平面时 `α≈1`；
  - 横移和后退不计入正向累计路径；
  - 缺帧、常深度、低运动、无有效波段均 fail closed；
  - L=16 的合成噪声结果不应劣于 L=8。

第一阶段不修改 `depth_delta_scale_loss`。只有 GT 离线验证形成明确余量、且 D̂ 仍
失败时，才另行设计与 OLS-axis 对齐的训练损失。

## 4. 离线否证实验

在同一 75-episode round-robin 合并集、相同候选窗口上同时计算：

1. legacy GT / D̂，window 8；
2. OLS-axis GT / D̂，window 8；
3. OLS-axis GT / D̂，window 16。

进入下一阶段需同时满足：

- GT OLS-axis median rel ≤ 0.18，形成明显而非贴线的余量；
- GT `n >= 8`，且不低于 legacy 有效窗口数的 50%；
- window 16 不显著劣于 window 8；
- 不依赖改变 `scale_rel_err_max=0.25` 才成立。

若 GT 仍 ≥ 0.23，或 window 16 继续恶化，则否决 OLS-axis，不改 frozen spec，
转而验证中心裁剪的有符号逐像素 ΔZ 代理。

## 5. 验收与安全

- 离线代理实验不得覆盖 canonical checkpoint；
- `depth_head.enable`、`dynamics.kind`、`enable_wm_update`、`safety.kind` 保持不变；
- 任何正式代理切换都需要 frozen spec 的带日期修订；
- 代理切换后必须重新跑 Signal ①/③；四信号未全过前不得翻旗标；
- 不下载 DA-V2 或 Wan 权重。

## 6. 实施顺序

1. 仅实现可选 diagnose 路径和单测；
2. 在已有数据上做 GT-first 离线 A/B；
3. GT 达到验收条件后，再评 D̂；
4. 主人/Claude 裁决并修订 frozen spec；
5. 才允许将 OLS-axis 设为正式 gate，并视需要对齐训练损失。
