# 空地协同 · 模块设计索引

总览（给人读）：`../2026-07-31-air-ground-mission-overview.html`  
迭代规格：`../2026-07-31-air-ground-g1-g2-design.md`  
验收门：`../../handover/2026-07-30-air-ground-phased-gates.md`

## 闸门（主人 2026-07-31 批准）

- M1–M5 **纯软件**按序自动：详设 → 实现 → 单测/回归  
- 装依赖 / 真机 / 猜型号场地 → **暂停汇报**  
- 模拟只标 `synthetic` / `SKIP`，不冒充真机  

## 模块清单

| ID | 目录 | 状态 |
|---|---|---|
| M0 | `modules/M0-control-plane/` | **已实现** |
| M1 | `modules/M1-marker-detect/` | **已实现** |
| M2 | `modules/M2-g2-scripted/` | **已实现**（synthetic_contract） |
| M3 | `modules/M3-dog-backend/` | **已实现**（软件 FakeNav） |
| M4 | `modules/M4-autel-spike/` | **已实现**（四态；硬件 NOT_RUN） |
| M5 | `modules/M5-map-docs/` | **已实现** |

## 回归命令

```bash
.venv/bin/python -m pytest \
  tests/test_mission_*.py \
  tests/test_marker_detect.py \
  tests/test_g2_scripted_scenes.py \
  tests/test_g1_fake_nav.py \
  tests/test_dog_stub.py \
  tests/test_drone_scout_adapters.py \
  tests/test_shared_map.py \
  -q
```

## 暂停项（需主人）

- 安装 `pupil-apriltags`  
- Tello 实拍 `recorded_tello` 账本  
- 狗型号确认后挂真 `NavBackend`  
- Autel hardware spike（`--require-hardware`）  
