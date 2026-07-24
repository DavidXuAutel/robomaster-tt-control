# 文档索引

RoboMaster TT 视觉避障 / 真机-仿真链路项目文档。按**用途**分类，文件名统一采用 `YYYY-MM-DD-<主题>` 前缀，便于按时间检索。

## 目录说明

| 目录 | 放什么 | 面向 |
|---|---|---|
| [`design/`](./design) | 设计与规格：相对稳定的"应该怎么做"（架构、方案、接口设计） | 内部 |
| [`dev-notes/`](./dev-notes) | 开发过程笔记：决策记录、合并集成、踩坑与验证 | 内部 |
| [`handover/`](./handover) | 交接与同步：面向大众/服务器的对外说明与执行报告 | 对外 |
| [`references/`](./references) | 外部技术方案分析等参考资料，供开发引用作背景 | 内部 |

## 现有文档

### design/ — 设计与规格
- `2026-07-16-tt-control-design.md` — RoboMaster TT 统一控制界面设计说明
- `2026-07-17-tt-visual-avoidance-design.md` — 半自动视觉避障设计说明
- `2026-07-17-tt-simulation-plan.md` — 视觉避障仿真方案（规格）
- `2026-07-17-simulation-plan.html` — 仿真方案（可视化/汇报版）

### dev-notes/ — 开发过程笔记
- `2026-07-18-avoidance-dev-notes.md` — 视觉避障离线开发说明
- `2026-07-18-merge-notes.md` — 三会话代码合并集成说明
- `2026-07-20-auto-watchdog-notes.md` — AUTO 半自动看门狗开发记录

### handover/ — 交接与同步
- `2026-07-20-changes-and-sync-for-dazhong.md` — 代码变更与服务器同步说明（面向大众）
- `2026-07-20-gesture-control-handover.md` — 手势控制模块交接说明
- `2026-07-20-sync-execution-report.md` — 手势控制上云同步执行报告
- `2026-07-20-real-flight-test-checklist.md` — 视觉避障真机测试现场 Checklist（分层通用版）
- `2026-07-20-single-machine-flight-checklist.md` — 单机（服务器一肩挑）挡板首飞傻瓜 Checklist

### references/ — 外部参考
- `2026-07-20-scoutxwam-world-model-analysis.md` / `.html` — ScoutXWAM 世界模型技术方案分析

## 约定
- 新增文档请放入对应用途目录，沿用 `YYYY-MM-DD-<主题>` 命名。
- 同一文档若需 Markdown 与 HTML 两份，使用相同主文件名、不同扩展名。
- 新增后请在本索引补一行登记。
