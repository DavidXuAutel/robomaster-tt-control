# 空地协同 · G1–G2 真机适配设计稿

日期：2026-07-31  
状态：**待主人确认后开工**  
前置：`docs/design/2026-07-30-air-ground-mission-architecture.md`  
验收门：`docs/handover/2026-07-30-air-ground-phased-gates.md`  
Codex 审稿：`APPROVE_WITH_CHANGES`（结论已吸收，见 §9）

---

## 0. 目标与非目标

### 目标（本迭代）

把已合入的 **G0 骨架** 推到 **室内可分测的真机可测层**：

1. **G2 优先（不依赖狗型号）**：标记物契约固化 + 合成契约 20/20 + 真机录制集分开记账 + Autel 四态 spike。
2. **G1 预埋**：显式 `mode=stub|backend` + 可观测 FakeNav；真狗 SDK 待型号确认后挂接。
3. **修控制面安全缺口**：串行 Scout、Brain 锚点防火墙、abort 广播到执行端。
4. **数据工程**：场景可回放、证据可解码、门禁可量化、禁止假绿。

### 非目标

- 共享稠密 SLAM / VPR / LLM / MQTT（本迭代继续进程内 bus）
- 改 orbit / 回灌 `default.json` / 多机逻辑进 `AvoidanceFSM`
- 开放词汇检测（物体 A = 标记物）
- 完整 FakePerception/FakeGas 框架（DogStub 已够；Fake 仅测 Nav + 最小 double）
- Autel RTK/KMZ 大抽象（只做四态 spike 报告）
- `RegionConfirmer.mode=auto`（禁止静默猜检测模式）

---

## 1. 现状缺口（对照代码 + Codex）

| 缺口 | 代码位置 | 危险 |
|---|---|---|
| 颜色锚点是自证循环：看到色块就贴当前 `color_anchor_id` | `detect.py` / `drone_tello.py` | 多区域时无法区分 AX-01/AX-02 |
| Brain 对 `anchor ∉ region.anchor_ids` 只 warning 仍派狗 | `brain.py:_on_drone_target_found` | 误派狗 |
| 启动时并行发所有 `drone.scout`，adapter 单槽覆盖 | `brain.py` / adapters | 多区域只搜最后一个 |
| abort 经 Brain 变 `mission.failed`，狗只听 `mission.abort` | `runner.py` / `dog_base.py` | SAFE_FAILED 但狗继续走 |
| `DogSdkAdapter` 缺任一 backend 隐式 stub | `dog_sdk.py` | FakeNav 测试假绿 |
| Autel dry_run 勾 True + 测试 `or True` | `drone_autel.py` / tests | 假绿 |
| 合成红圆同时当锚点与物体 | fixtures / runner | 20/20 无真机证明力 |

---

## 2. 标记物与区域锚点契约

### 2.1 物理规格（初值，真机标定后冻结）

| 用途 | ID | 形态 | 备注 |
|---|---|---|---|
| 区域锚点 | `TAG-{n}` 优先；单点可用 `AX-01` | **AprilTag36h11**；单点室内可用**蓝色板** | 多 region **禁止**同色蓝区分 |
| 物体 A | label=`object_a` | **红色板**（与锚点不同通道） | 禁止红锚点+红物体双用途 |

15cm / 3m / 0.5% 仅为起步建议；**以 Tello 实拍标定后再写入阈值**，不在本迭代假冻结。

### 2.2 硬规则

1. 区域确认与物体检测解耦；禁止单色双用途。
2. **单点室内**：允许 `蓝色 AX-01 + region_x`（仅证明「锚点存在」）。
3. **两区域及以上**：必须唯一 AprilTag ID（或等价唯一编码）；颜色模式不得声称能辨 region。
4. `mode=apriltag` 缺 `pupil-apriltags` → **启动失败**，禁止静默回退颜色。
5. 配置显式 `mode=color|apriltag`，无 `auto`。

### 2.3 检测接口

```text
MarkerSpec:
  label: str
  kind: color | apriltag
  hsv_ranges?: [...]
  tag_ids?: [int]
  min_area_ratio: float
  min_confidence: float   # Scout 上报门槛，默认 0.55

detect_marker(frame, spec) -> Detection | None
```

事件字段集不改。

### 2.4 Brain 防火墙（接单前全部拒绝则零 `dog.inspect`）

拒绝条件：

- `confidence < min_confidence`
- `region_id ∉ mission.region_ids`
- `anchor_id ∉ region.anchor_ids`（**硬拒绝，不再 warning 放行**）
- `target_label != mission.target_label`
- `anchor_age_ms` 超阈值（可配置，默认宽松如 5000ms）
- `observed_at` 晚于 `deadline` 或明显早于任务 start

每项对应「拒绝且零 inspect」单测。

---

## 3. Scout 串行调度（单机）

本迭代只有一架 Scout：

```text
mission.start
  → 只发 region_ids[0] 的 drone.scout
  → scout_failed / 本区超时 → 发下一个
  → target_found → 停侦察，进 DOG_NAV
```

放在 **MissionBrain**（或极薄 `ScoutScheduler` 同包），不进 `AvoidanceFSM`。

负例终点：帧预算/视频结束/场景 deadline → 明确 `drone.scout_failed` 或 runner 确定性结束态；禁止「一直挂着算通过」。

---

## 4. Abort 与进程内 Supervisor

### 4.1 广播语义

```text
operator / UI → mission.abort{mission_id, reason}
        │
        ├─► MissionBrain.handle   （校验 mission_id 匹配，否则丢弃）
        ├─► DogAdapter.abort      （→ NavBackend.cancel）
        └─► ScoutAdapter.abort    （停侦察 / 悬停策略由飞控侧已有逻辑）
```

**同一原始 abort 事件**必须同时送 Brain 与所有执行 adapter；不得只依赖 Brain 转发出的 `mission.failed`。

### 4.2 测试 D03

完整 EventBus 路径：`FakeNav.cancel_calls == 1`；之后任意 tick **不得**再发 `dog.arrived`。

---

## 5. G2 — 无人机单独

### 5.1 双账本（防假绿）

| 账本 | 内容 | 可否冒充真机 G2 |
|---|---|---|
| `synthetic_contract` | 参数化/合成场景 ≥20 | **否**；只证明契约与检测解耦 |
| `recorded_tello` | 室内实拍小集（正例/仅蓝/仅红/暗光/模糊/反光/小红噪声） | **是**；N/N 单独报告 |

### 5.2 合成场景矩阵（期望序列必须确定）

| ID | 类型 | 期望终点 |
|---|---|---|
| S01–S05 | Tag/蓝锚 + 红物体，正确 region | 一次 `drone.target_found`；一次 `dog.inspect` |
| S06–S08 | 仅锚点 | 结束态无 inspect；最终 `scout_failed` 或 runner FAIL_EXPECT_NO_DISPATCH |
| S09–S10 | 仅红物体 | 同上 |
| S11–S12 | streak < need_frames | 同上 |
| S13–S14 | 低置信 | 同上 |
| S15 | **错误 Tag ID**（图像内容可识别，非 meta 灌 region） | Brain 拒绝；零 inspect |
| S16 | 重复检出 | 仍一次 found / 一次 inspect |
| S17 | 中途 abort | SAFE_FAILED；Scout+Dog abort；零后续 inspect |
| S18 | 两区域串行：先错 Tag 区失败，再对区成功 | 仅对区一次 inspect |
| S19 | 暗光合成（期望**预先写死**：found 或 not） | 按 meta 断言，禁止「按阈值随意」 |
| S20 | 小面积红噪声 | 零 inspect |

场景可用 **参数化 manifest** 生成，不必 20 套大 PNG。  
S15/S18 **必须**从图像 Tag ID 识别，禁止 `meta.json` 直接喂正确/错误 region 给检测器。

Runner 校验：`evidence_uri` 文件存在且可解码（`cv2.imwrite` 返回值必查）。

### 5.3 Tello / Autel 交付

- `scripts/mission/g2_scripted_runner.py`：读场景 → 喂帧 → 断言事件序列  
- `need_anchor_frames` 构造参数必须生效（现状 bug：硬编码 3）  
- Autel spike 状态机改为 **`PASS | FAIL | SKIP | NOT_RUN`** + `mode=simulated|hardware`  
  - hardware PASS 须附 device 身份、时间戳、证据  
  - 真机必需项 FAIL/NOT_RUN → 脚本非零退出  
  - 删除测试中所有 `or True` 假断言  

---

## 6. G1 — 狗单独（预埋）

### 6.1 显式模式

```text
DogSdkAdapter(mode="stub" | "backend", ...)
```

- `stub`：明确用 DogStub  
- `backend`：缺任一必需 backend → **立即报错**，禁止隐式回退  
- 报告输出实际 mode；自动回退不得算通过  

### 6.2 FakeNav（本迭代核心）

可观测：`goto_calls`, `cancel_calls`, `last_goal_id`；可配置 N tick arrived / 拒绝 goto。

Perception/Gas：测试内最小 double 即可，不新造完整 Fake 框架。

### 6.3 用例

| ID | 用例 | 期望 |
|---|---|---|
| D01 | 10× 同 goal | 10× arrived；FakeNav.goto 参数正确 |
| D02 | goto 拒绝 | `dog.inspect_failed` stage=nav |
| D03 | 导航中 abort（完整 bus） | cancel==1；无 arrived |
| D04 | SDK 调用抛异常 | 转失败事件 + cancel；不打断无保护循环 |
| D05–D06 | gas disconnect / cal stale | 沿用 DogStub 既有覆盖即可标明引用 |

Backend 调用约定：异常 → 失败事件；同步调用需有文档级最大时长/取消语义（实现可先 timeout wrapper）。

`is_arrived()` 允许电平语义；边沿由 adapter `_arrived_emitted` 锁存。

型号确认前类名用 `DogNavBackend` 示例，不提前固化 `Unitree*` 目录。  
探测清单：`docs/dev-notes/2026-07-31-dog-model-probe.md`（字段清单，无「明天」字样）。

---

## 7. SharedMap 启动校验

启动时程序校验（非仅人工 checklist）：

- `frame == "dog_map"`
- region key == 内部 `region_id`
- `anchor_id` **全局唯一**
- 每个活动 region 有非空 `anchor_ids` 与 `dog_goal_id`

---

## 8. 实现分期（确认后执行）

| 阶段 | 内容 | 验证 |
|---|---|---|
| P0 | Brain 防火墙 + 串行 scout + abort 广播 + mission_id 校验 | 单测零误派 / D03 |
| P1 | MarkerSpec + 锚点/物体解耦；apriltag 缺库 fail-fast；need_frames 生效 | detect 单测 |
| P2 | G2 scripted runner + 合成 20/20 + evidence 可解码 | `synthetic_contract` |
| P3 | DogSdk 显式 mode + FakeNav + D01–D04 | pytest |
| P4 | Autel 四态 spike + 删假断言 | 脚本机读 summary |
| P5 | SharedMap 校验 + 狗探测清单 + 真机录制 SOP | 文档可执行 |

CI：

```bash
.venv/bin/python -m pytest \
  tests/test_mission_*.py \
  tests/test_drone_scout_adapters.py \
  tests/test_dog_stub.py \
  tests/test_g2_scripted_scenes.py \
  tests/test_g1_fake_nav.py \
  -q
```

---

## 9. Codex 交叉审记录

**VERDICT: APPROVE_WITH_CHANGES**（已全部吸收为上文硬规则）

| ID | 意见 | 处理 |
|---|---|---|
| C1 | 蓝色锚点自证循环；多区须唯一 Tag；apriltag 禁静默回退 | **采纳** §2 |
| C2 | Brain 错误锚点仍派狗 | **采纳** §2.4 硬拒绝 |
| C3 | 多 scout 覆盖；须串行 | **采纳** §3 |
| C4 | abort 到不了狗 cancel | **采纳** §4 |
| C5 | DogSdk 隐式 stub 假绿 | **采纳** §6.1 |
| C6 | Autel dry_run / `or True` 假绿 | **采纳** §5.3 |
| C7 | 合成 20/20 冒充真机；S19 非确定 oracle | **采纳** §5.1–5.2 |
| I* | 负例终点、need_frames bug、map 校验、异常转失败 | **采纳** |
| O* | 砍 auto / 完整 Fake 框架 / Autel 大抽象 / 20 套大 PNG | **采纳** |
| 标记色 | 蓝/Tag + 红 A，多区必须 Tag | **同意（有硬条件）** |
| 无 MQTT | 本迭代同意；先做实进程内 abort/串行 | **同意** |

---

## 10. 待主人确认（开工闸门）

1. 本设计（含 Codex C1–C7）是否按 **P0→P5** 开工？  
2. 标记物：**单点蓝 AX-01 + 红物体 A**；多区强制 AprilTag —— 是否接受？  
3. 本迭代不做 MQTT，先修进程内 supervisor —— 是否接受？  
4. G2 双账本：合成契约 ≠ 真机 G2 —— 是否接受？  
