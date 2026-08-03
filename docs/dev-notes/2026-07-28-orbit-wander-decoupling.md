# 椅子绕飞（Orbit）与随机漫游（Wander）模块解耦

日期：2026-07-28
起因：主人 2026-07-28 铁丝笼采数期间提出——绕飞是独立模块，参数与规则
不得与随机漫游相互覆盖影响。本笔记记录耦合点核查结果与此后的边界约定。

## 1. 核查结论：绕飞基线未被破坏

逐条对照 `docs/design/2026-07-27-orbit-control-principles.md` §4 已验证基线，
`configs/default.json` 的 orbit 节 22 项与 fsm 节 orbit 相关 4 项**全部与基线一致**，
无一被漫游调参波及。安全不变量 #3 亦成立：
`target_nearness 0.69 ≤ danger_thresh 0.78 − 0.06 = 0.72`。

原因是漫游的历次调参（turn_thresh / clear_thresh / cruise_pitch / h_min / h_max /
depth_stale_s）全部落在独立档 `configs/wander-cage.json`，未回灌 default.json。
default.json 本轮唯一改动是 wander 节新增 `pano_step_deadband_deg`、`h_missing_s`，
与 orbit 无关。

## 2. 实际存在的耦合点

| # | 耦合点 | 性质 | 处置 |
|---|---|---|---|
| 1 | `default.json` 单文件同时承载 orbit 与 wander 两节 | 潜在：任何人在此调漫游参数都可能手滑碰到 orbit 节 | 新建 `configs/orbit-chair.json`，把 orbit 全部参数**显式写全**，不再依赖 default.json 回退。绕飞真机一律用该档启动 |
| 2 | `fsm` 节的 `max_height_cm` / `min_battery_pct` / `depth_stale_s` / `max_auto_engaged_s` 为两模块共用字段名 | 潜在：同名不同需求（如 wander-cage 因撞铁丝网把 depth_stale_s 收到 1.5，绕飞并不需要） | 两档各自显式赋值，互不回退。绕飞保持 3.0 基线：环绕是围椅子转，不朝铁丝网直冲 |
| 3 | `WanderPolicy` 借用 `AvoidanceController.zone_nearness()` 取三区近度，依赖 `avoid` 节的 `band_top` / `band_bottom` | **真耦合**：改这两项会同时改变漫游与绕飞/避障的近度读数 | 暂不拆分（拆分需改 wander.py 结构，属规格书管辖，只允许 Claude 改）。当前记为已知约束：**禁止为漫游调参而动 `avoid.band_*`** |
| 4 | `orbit_mode` / `wander_mode` 互斥开关 | 设计如此，非缺陷 | 两档各自钉死取值，避免忘设导致跑错模块 |

## 3. 此后的边界约定

1. 绕飞真机：`--config configs/orbit-chair.json`；漫游真机：`--config configs/wander-cage.json`。
   两档不会同时加载，互不覆盖。
2. 任一模块的调参只准写进自己的档，**不回灌 `configs/default.json`**。
   default.json 的 orbit/fsm 节视为 2026-07-27 已验证基线快照，改动仍受
   orbit 原则文件守则约束（非 Claude Agent 需逐条对照安全不变量与历史坑）。
3. `avoid` 节的 `band_top` / `band_bottom` 为两模块共享，任何改动须同时评估
   漫游与绕飞影响，并跑 `tests/test_orbit.py tests/test_avoidance.py tests/test_wander.py`。
4. 新增跨模块安全阈值时沿用坑 #9 的教训：全项目只允许一个来源，接配置，禁止硬编码。

## 4. 附：本次为何漫游做不了「椅子绕飞」

主人一度在漫游模式下期待绕椅子飞行，实际观察到「没识别到椅子就扭头走了」。
根因是需求与模块能力错配，非参数问题：

- 漫游遇障响应为 `turn_min_deg=50 ~ turn_max_deg=130` 的偏航转向，语义是
  「转开去别处」，本就不是「绕着目标转」。当次抽到 126°，接近掉头。
- 漫游安全不变量要求 `roll ≡ 0`（`tt_control/wander.py:841`），而
  `tt_control/avoidance.py:99` 记录的 2026-07-24 真机结论是
  「yaw 绕障几何上无法绕过椅子，改用 roll 横移」。二者直接冲突。
- 三区近度取中带**中位数**，两米外的椅子占不到 mid 区一半像素，抬不动 mid。
  为此把 `turn_thresh` 降到 0.30 只会让远墙与地面的整体近度触发误转向——
  即主人观察到的现象。该值已随本次调整回收，见 `wander-cage.json` 注释。

椅子绕飞的正确载体是 OrbitController（POI 视觉伺服环绕），2026-07-27 真机
已验证连续环绕 2min13s 无 abort。
