# 空地协同 · 模块设计索引

总览（给人读）：`../2026-07-31-air-ground-mission-overview.html`  
迭代规格：`../2026-07-31-air-ground-g1-g2-design.md`  
验收门：`../../handover/2026-07-30-air-ground-phased-gates.md`  
真机搁置：`../../handover/2026-07-31-air-ground-deferred-hardware.md`

## 闸门（主人已批准）

- M0–M5 **纯软件**已落地  
- 真机/人工项见搁置清单，不冒充验收  

## 模块清单

| ID | 目录 | 状态 |
|---|---|---|
| M0 | `modules/M0-control-plane/` | **已实现** |
| M1 | `modules/M1-marker-detect/` | **已实现**（含真库 AprilTag 测） |
| M2 | `modules/M2-g2-scripted/` | **已实现**（synthetic_contract） |
| M3 | `modules/M3-dog-backend/` | **已实现**（FakeNav） |
| M4 | `modules/M4-autel-spike/` | **已实现**（四态；硬件搁置） |
| M5 | `modules/M5-map-docs/` | **已实现** |

## 依赖

```bash
.venv/bin/pip install -r requirements-dev.txt   # 含 pupil-apriltags
```

## 回归命令

```bash
.venv/bin/python -m pytest \
  tests/test_mission_*.py \
  tests/test_marker_detect.py \
  tests/test_apriltag_real.py \
  tests/test_g2_scripted_scenes.py \
  tests/test_g1_fake_nav.py \
  tests/test_dog_stub.py \
  tests/test_drone_scout_adapters.py \
  tests/test_shared_map.py \
  -q
```

## 真机/人工待测（搁置）

见 `docs/handover/2026-07-31-air-ground-deferred-hardware.md`：

- Tello 实拍 `recorded_tello`
- 标记物物理尺寸冻结
- 狗型号 + 真 NavBackend / G1 到点
- Autel hardware spike
- 室外 G5

打印约定：`docs/handover/2026-07-31-air-ground-marker-print-guide.md`
