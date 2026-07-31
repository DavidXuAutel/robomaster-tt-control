# M0 · 控制面安全（Brain 防火墙 / 串行 Scout / Abort 广播）

状态：**已实现**（Codex 详设 REJECT→改稿；实现测审 FAIL→修 C1–C3；mission 套件绿）  
日期：2026-07-31  
审稿：`codex-review.md`  
依据：`docs/design/2026-07-31-air-ground-g1-g2-design.md` §2.4–§4  

范围：`mission_brain/brain.py`、`mission_brain/supervisor.py`（新）、`mission_brain/runner.py`、  
`adapters/dog_base.py`（可选 mission_id 校验）、`adapters/dog_stub.py`、`adapters/dog_sdk.py`、  
`adapters/drone_base.py` / tello（abort 锁存）、相关 tests  

红线：不改 orbit / AvoidanceFSM / default.json；事件信封字段集不动。

---

## 1. 目标

1. 错误锚点 / 错标签 / 不新鲜观察 **绝不** 派 `dog.inspect`
2. 单 Scout **串行** 侦察；只认当前区结果
3. 匹配 mission 的 abort / SAFE_FAILED：在同一 Supervisor 事务内对 Brain + Scout + Dog 做停机尝试
4. **其他 mission 的 abort：Brain 不变，且执行端 cancel_calls==0**

---

## 2. 非目标

MQTT；MarkerSpec（M1）；完整 FakeNav 矩阵（M3，本模块仅最小 Fake 测 abort）；Autel（M4）；多线程消息中间件。

约定：**单线程 tick 循环**；`MissionSupervisor` 为唯一操作入口（串行化）。

---

## 3. 设计

### 3.1 Brain 防火墙

接单前（进入 DOG_NAV 前）全部通过；否则 log + 保持 SCOUTING + 零 inspect。

用注入时钟 `now = self._now()`（非信任事件内的墙钟 alone）：

| 检查 | 说明 |
|---|---|
| `state == SCOUTING` | |
| `region_id == active_scout_region` | 非当前区 found → 拒绝 |
| `region_id ∈ mission.region_ids` | |
| `target_label == mission.target_label` | |
| `anchor_id ∈ region.anchor_ids` | 硬拒绝（无 warning 放行） |
| 数值有限：confidence / ages / timestamps | NaN/inf 拒绝 |
| `anchor_age_ms >= 0` 且 `<= max_anchor_age_ms` | 默认 5000 |
| `now <= deadline` | |
| `observed_at <= now + skew`（默认 1s） | 拒绝未来钟 |
| `now - observed_at <= freshness_s`（默认 5s） | Brain 侧新鲜度 |
| `confidence >= min_confidence` | |

过期 `mission.start`（`now > deadline`）→ 不发 scout，直接 fail 或拒绝 start。

### 3.2 串行 Scout

状态：`active_scout_index`、`active_scout_region`、可选 `active_scout_command_id`。

- start：只派 `region_ids[0]`；重置 stage 计时
- 仅接受 **当前区** 的 `drone.target_found` / `drone.scout_failed`
- 当前区 failed 或 **本区** stage timeout：index++，派下一区并 **重置** `_stage_entered_at`；无下一区 → fail `scout` reason code `all_regions_exhausted`
- 整任务 `deadline` 优先于本区超时 → fail `deadline`
- 迟到的上一区 failed/found → 忽略
- 活动任务中新 `mission.start` → **拒绝**（需先 abort）

稳定 reason code（字符串，不改 schema）：`abort` / `deadline` / `all_regions_exhausted` / `stage_timeout` / …

### 3.3 Abort（Brain）

- 无活动 mission → 忽略
- `event.mission_id != self.mission_id` → 忽略（**先校验再记 seen**，避免占坑）
- 匹配 → `_fail("abort", reason)` → `mission.failed`

### 3.4 Supervisor（取消所有者，唯一）

```text
MissionSupervisor(brain, scout, dog, bus, *, operator_sources={"operator"})
```

- `publish_operator(ev)`：仅允许 `mission.start` / `mission.abort`，且 `source ∈ operator_sources`
- 路由按 **EventType**，不靠 source 前缀猜角色
- **abort 路径**（同一线性化事务，顺序固定）：
  1. 校验：有活动 mission 且 mission_id 匹配；否则 **整段跳过**（不碰执行端）
  2. `brain.handle(abort)`
  3. best-effort：`dog.abort` / `scout.abort`（各自 try/except，互不影响）
- **`mission.failed`（本 mission）**：若尚未对本失败做过 stop，再 best-effort 停 Scout+Dog 一次（覆盖 deadline/gas 等非 operator abort）
- **不** 再让 adapter `on_brain_event` 听 abort（避免双 cancel）；取消只归 Supervisor
- `wire()` 幂等（`_wired` 锁存）

「同时」表述改为：**同一线性化事务内对全部参与方完成投递尝试**。

### 3.5 执行端 abort 锁存（纳入 M0）

`DogStub` / `DogSdk` / Scout adapter：

- `abort(reason)` 设 `_aborted=True`（或 generation++），清 pending inspect/nav
- 之后 `tick` / `process_frame` **强制静默**（即使 FakeNav `is_arrived()` 仍 True）
- `cancel()` 抛异常也不清除 aborted 锁存

### 3.6 文件清单

| 文件 | 变更 |
|---|---|
| `mission_brain/brain.py` | 防火墙、串行、abort id、拒 start、时间检查 |
| `mission_brain/supervisor.py` | 新建 |
| `mission_brain/runner.py` | 用 Supervisor |
| `adapters/dog_stub.py` / `dog_sdk.py` | abort 锁存 |
| `adapters/drone_tello.py`（及 base 若需） | abort 后静默（tello 已有部分） |
| `tests/test_mission_brain.py` | 防火墙 + 串行 |
| `tests/test_mission_supervisor.py` | 新建 |
| `tests/test_mission_*.py` | 回归适配 |

---

## 4. 测试矩阵（验收硬门槛）

### Brain

| ID | 期望 |
|---|---|
| B-FW-01..05 | 锚点/标签/age/未来时/合法 found |
| B-FW-06 | now>deadline 的 found → 零 inspect |
| B-FW-07 | NaN confidence → 零 inspect |
| B-SER-01 | start 只 scout 第一区 |
| B-SER-02 | 一区 failed → 二区 scout；stage 计时重置（二区享有完整 timeout） |
| B-SER-03 | 一区 found → 不发二区 |
| B-SER-04 | 非当前区 found → 零 inspect |
| B-SER-05 | 上一区迟到 failed → 不跳过当前区 |
| B-AB-01 | 他 mission abort → 状态不变 |
| B-AB-02 | 本 mission abort → SAFE_FAILED |
| B-ST-01 | 活动中新 start → 拒绝，旧任务继续 |

### Supervisor

| ID | 期望 |
|---|---|
| S-AB-01 | 本 mission abort：`cancel_calls==1`；其后 FakeNav arrived 仍 **零** `dog.arrived` |
| S-AB-02 | 他 mission abort：`cancel_calls==0`，Scout 仍可侦察 |
| S-AB-03 | dog.abort 抛错时 Scout/Brain 仍完成停机/失败 |
| S-FAIL-01 | deadline → mission.failed 后执行端被 stop 一次 |
| S-OK-01 | 单区 happy → COMPLETE |

回归：

```bash
.venv/bin/python -m pytest \
  tests/test_mission_events.py \
  tests/test_mission_brain.py \
  tests/test_mission_replay.py \
  tests/test_mission_e2e.py \
  tests/test_mission_supervisor.py \
  tests/test_dog_stub.py \
  tests/test_drone_scout_adapters.py \
  -q
```
