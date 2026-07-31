# 空地协同 · 真机/人工待测清单（已搁置）

日期：2026-07-31  
软件 M0–M5 已合入；下列项**明确不做完、等主人有设备/场地时再开**。

## 搁置项

| ID | 内容 | 阻塞原因 | 软件侧已备好 |
|---|---|---|---|
| HW-T1 | Tello 实拍账本 `recorded_tello`（暗光/模糊/反光等） | 需真机+场地+飞行 | `synthetic_contract` 20/20；录制目录约定见下 |
| HW-T2 | 标记物物理尺寸/巡航高度阈值冻结 | 需实拍标定 | 蓝锚+红物体 / AprilTag 软件契约已冻 |
| HW-D1 | 宇树狗型号确认 + 真 `NavBackend` | 待连设备填探测表 | `DogSdkAdapter(mode=backend)` + FakeNav；清单 `docs/dev-notes/2026-07-31-dog-model-probe.md` |
| HW-D2 | G1 真机 10 次到点 | 需狗+地图路点 | D01–D04 软件用例 |
| HW-A1 | Autel hardware spike（`--require-hardware`） | 需道通 SDK/机 | 四态 spike；dry_run=simulated |
| HW-G5 | 室外端到端 10 连跑 | 大场地联调 | Supervisor / 串行 scout / abort |

## 实拍账本目录约定（有素材再填）

```text
tests/fixtures/mission/recorded_tello/
  README.md          # 来源、光照、飞机、日期
  R01_happy/ ...
```

当前该目录仅占位，**不得**用合成图冒充。

## 软件已完成（可回归）

```bash
.venv/bin/pip install -r requirements-dev.txt   # 含 pupil-apriltags
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
