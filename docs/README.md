# 文档索引

RoboMaster TT 视觉避障 / 真机-仿真链路项目文档。按**用途**分类，文件名统一采用 `YYYY-MM-DD-<主题>` 前缀，便于按时间检索。

## 目录说明

| 目录 | 放什么 | 面向 |
|---|---|---|
| [`design/`](./design) | 设计与规格：相对稳定的"应该怎么做"（架构、方案、接口设计） | 内部 |
| [`dev-notes/`](./dev-notes) | 开发过程笔记：决策记录、合并集成、踩坑与验证 | 内部 |
| [`handover/`](./handover) | 交接与同步：面向大众/服务器的对外说明与执行报告 | 对外 |
| [`references/`](./references) | 外部技术方案分析等参考资料，供开发引用作背景 | 内部 |

## 从哪开始

| 我要做什么 | 读这个 |
|---|---|
| 飞真机 / 采数据 / 选飞行模块 | `handover/2026-07-28-flight-modes-and-data-collection-runbook.md` |
| 人工示教 + 意图标注（动作头 SFT） | `handover/2026-07-29-teleop-intent-data-collection.md` |
| 改环绕（Orbit）代码或参数 | `design/2026-07-27-orbit-control-principles.md`（仅 Claude 可改） |
| 改漫游（Wander）代码或参数 | `design/2026-07-27-wander-explore-design.md`（仅 Claude 可改） |
| 了解项目全部能力 | `handover/2026-07-25-capability-inventory.md` |
| 空地协同（无人机×机器狗）演示 | `handover/2026-07-30-air-ground-phased-gates.md` + `design/2026-07-30-air-ground-mission-architecture.md` + `design/2026-07-31-air-ground-g1-g2-design.md` |

## 现有文档

### design/ — 设计与规格
- `2026-07-16-tt-control-design.md` — RoboMaster TT 统一控制界面设计说明
- `2026-07-17-tt-visual-avoidance-design.md` — 半自动视觉避障设计说明
- `2026-07-17-tt-simulation-plan.md` — 视觉避障仿真方案（规格）
- `2026-07-17-simulation-plan.html` — 仿真方案（可视化/汇报版）
- `2026-07-21-single-drone-solidify-and-data-delivery-design.md` — 单机避障「做扎实」+ 数据交付设计方案
- `2026-07-23-ds-depth-avoidance-plan.md` — Tello 接入 DS 路由器、远端深度推理与避障绕飞方案
- `2026-07-25-wam-training-scenarios.md` — WAM 世界模型训练场景规划：21 个飞行场景分类 + 数据采集管线设计
- `2026-07-27-orbit-control-principles.md` — **POI 环绕模块设计原则与参数守则**（安全不变量 + 历史坑 + 参数安全范围；**仅 Claude 可改**）
- `2026-07-27-wander-explore-design.md` — **随机漫游模块实现规格书**（含分工与验收阶梯；**仅 Claude 可改**）
- `2026-07-30-air-ground-mission-architecture.md` — 空地协同 MissionBrain 架构说明（语义地图 + 事件契约 + 适配器边界）
- `2026-07-31-air-ground-g1-g2-design.md` — G1–G2 真机适配设计（Codex 审过：串行 Scout / abort 广播 / 防假绿）
- `2026-07-31-air-ground-mission-overview.html` — 空地协同方案总览（术语悬停 + 设计理由，面向阅读）

### dev-notes/ — 开发过程笔记
- `2026-07-18-avoidance-dev-notes.md` — 视觉避障离线开发说明
- `2026-07-18-merge-notes.md` — 三会话代码合并集成说明
- `2026-07-19-realdrone-verify-gesture.md` — 手势起飞真机验证 Runbook
- `2026-07-20-auto-watchdog-notes.md` — AUTO 半自动看门狗开发记录
- `2026-07-26-orbit-preflight-bug-review.md` — POI 绕飞飞前缺陷审查与修复方案（P0/P1 + checklist）
- `2026-07-26-preflight-safety-bugs-plan.md` — 第二轮飞前安全缺陷：新鲜度/急停/禁公网回落/分模块测试
- `2026-07-27-wander-codex-cross-review.md` — Wander × Codex 交叉评审记录
- `2026-07-28-wander-cage-flight-plan.md` — Wander 铁丝笼真机验证方案与规格偏离申请
- `2026-07-28-wander-defect-ticket.md` — Wander 模块缺陷审查与修复工单
- `2026-07-28-wander-side-danger-abort.md` — 首飞 abort 复盘：侧区触发 danger 导致必然 abort
- `2026-07-28-wander-verify-graylock.md` — VERIFY 灰区死循环 + 铁丝笼二飞复盘
- `2026-07-28-orbit-wander-decoupling.md` — **绕飞与漫游模块解耦**：耦合点核查、边界约定、需求与模块错配的反例

### handover/ — 交接与同步
- `2026-07-28-flight-modes-and-data-collection-runbook.md` — **飞行/采数必读**：模块×配置档对照、启动命令、采数 SOP、飞后校验、打包交付、模块红线、故障速查
- `2026-07-29-teleop-intent-data-collection.md` — **人工示教（含意图标注）采数方案**：给 Kairos/WAM 动作头 SFT；任务清单、SOP、验收与打包
- `templates/teleop_manifest.csv` — 示教交付总表模板（episode ↔ intent）
- `2026-07-20-changes-and-sync-for-dazhong.md` — 代码变更与服务器同步说明（面向大众）
- `2026-07-20-gesture-control-handover.md` — 手势控制模块交接说明
- `2026-07-20-sync-execution-report.md` — 手势控制上云同步执行报告
- `2026-07-20-real-flight-test-checklist.md` — 视觉避障真机测试现场 Checklist（分层通用版）
- `2026-07-20-single-machine-flight-checklist.md` — 单机（服务器一肩挑）挡板首飞傻瓜 Checklist
- `2026-07-21-tello-avoidance-and-data-handover.md` — 避障能力边界与飞行数据交接（面向大众）
- `2026-07-22-flight-sop.md` — 现场飞行 SOP：校正飞行 + 避障采集飞行
- `2026-07-22-wam-brain-integration-plan.md` — 无人机接入 WAM 大脑对接方案（接口契约待双方确认）
- `2026-07-23-depth-server-api-handover.md` — 深度推理服务器远程访问与调用指南（面向大众）
- `2026-07-23-morning-flight-checklist.md` — 新服务器开飞清单
- `2026-07-25-orbit-avoidance-handover.md` — POI 环绕避障开发交接：视觉伺服居中 + 环绕飞行，含完整参数、调试记录、启动命令
- `2026-07-25-capability-inventory.md` — 能力清单总览：11 个模块、所有键盘映射、FSM 状态、安全保护、参数表
- `2026-07-30-air-ground-phased-gates.md` — 空地协同分阶段验收门 G0–G5（契约/狗/机/交接/气检/端到端）

### references/ — 外部参考
- `2026-07-20-scoutxwam-world-model-analysis.md` / `.html` — ScoutXWAM 世界模型技术方案分析
- `world_model_report_jiedu.md` — 世界模型 · ScoutWAM 汇报的通俗解读与独立评估
- `大众具身智能遥操作数据飞轮_研究分析报告.md` — 大众遥操作数据飞轮系统研究分析报告

## 约定
- 新增文档请放入对应用途目录，沿用 `YYYY-MM-DD-<主题>` 命名。
- 同一文档若需 Markdown 与 HTML 两份，使用相同主文件名、不同扩展名。
- 新增后请在本索引补一行登记。
