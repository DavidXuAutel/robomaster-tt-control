# 空地协同 · 模块设计索引

总览（给人读）：`../2026-07-31-air-ground-mission-overview.html`  
迭代规格：`../2026-07-31-air-ground-g1-g2-design.md`  
验收门：`../../handover/2026-07-30-air-ground-phased-gates.md`

## 是否每个模块都要详细设计？

| 类型 | 要不要单独详设 | 说明 |
|---|---|---|
| 控制面 / 安全 / 事件契约（M0） | **要** | abort、误派狗、串行 scout 会炸演示 |
| 感知契约（M1） | **要** | 锚点/物体解耦、AprilTag fail-fast |
| 测试基建 / 场景 runner（M2） | **要**（可偏测试设计） | 防假绿，双账本 |
| Dog 显式 mode + FakeNav（M3） | **要**（短） | 防 stub 假绿 |
| Autel spike 四态（M4） | **精简 checklist 即可** | 不扩 SDK 抽象 |
| 地图校验 + 文档（M5） | **精简** | 规则表 + SOP |

流程：详设 → Codex 批判审 → 改稿落盘 → 开发 → 单测/回归 → Codex 测审 → 下一模块。

## 模块清单（本迭代）

| ID | 目录 | 目标 | 状态 |
|---|---|---|---|
| M0 | `modules/M0-control-plane/` | Brain 防火墙、串行 scout、abort 广播、mission_id | **已实现**（pytest mission 套件绿） |
| M1 | `modules/M1-marker-detect/` | MarkerSpec、蓝锚/红物体、apriltag fail-fast | 待开始 |
| M2 | `modules/M2-g2-scripted/` | 合成 20/20 runner、evidence 校验 | 待开始 |
| M3 | `modules/M3-dog-backend/` | DogSdk 显式 mode + FakeNav | 待开始 |
| M4 | `modules/M4-autel-spike/` | 四态 spike、删假断言 | 待开始 |
| M5 | `modules/M5-map-docs/` | SharedMap 校验、狗探测清单 | 待开始 |

## 目录约定

每个模块目录固定三件套（可缺 review 直至 Codex 跑完）：

```text
modules/Mx-name/
  design.md           # 详细设计（接口、行为、测试矩阵）
  codex-review.md     # Codex 审稿原文摘要 + 采纳/拒绝表
  notes.md            # 可选：实现偏差、真机笔记
```

跨模块总览与 HTML 仍放在 `docs/design/` 根下，不塞进 modules。
