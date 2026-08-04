# Wander 模块缺陷审查与修复工单（2026-07-28）

- 审查对象：`tt_control/wander.py`（792 行）、`tests/test_wander.py`（13 用例）、
  `tt_control/avoidance_fsm.py` 的 wander 分支、`configs/default.json` 的 `wander` 节
- 依据规格：`docs/design/2026-07-27-wander-explore-design.md`（chmod 444，只读）
- 前置必读：`docs/design/2026-07-27-orbit-control-principles.md` §2 安全不变量 / §3 历史坑
- 已有评审：`docs/dev-notes/2026-07-27-wander-codex-cross-review.md`（本文是其后续，不重复其已处置项）
- 基线状态：`.venv/bin/python -m pytest tests/test_wander.py -q` → **13 passed**
- 结论：**规格符合度高，但存在 3 项真机安全缺陷 + 4 项规格落空 + 6 项质量问题；其中 5 项已附可复现证据**

---

## 0. 给实现方（Grok）的执行规则

**必须遵守（违反即打回）：**

1. **改动范围只允许**：`tt_control/wander.py`、`tests/test_wander.py`、
   `configs/default.json` 的 `wander` 节。
2. **禁止触碰**：`OrbitController` 及 orbit 相关 FSM 路径、`app.py` 的降落重试路径、
   `EpisodeRecorder` 的帧↔行同步核心、任何 Claude 所有权文档
   （`2026-07-27-orbit-control-principles.md`、`2026-07-27-wander-explore-design.md`）。
3. **禁止硬编码阈值**（orbit 坑 #9）。本工单要求新增的参数一律先进
   `configs/default.json` 的 `wander` 节 + `WanderParams` dataclass，代码里只许引用 `self.p.xxx`。
4. **禁止引入新第三方依赖**。
5. **保持纯函数式内核**：`decide()` 不碰 socket / 线程 / sleep；所有时间依赖通过 `now` 注入。
6. **每条修复必须带回归测试**，且测试断言的是「规格要求的行为」而非「实现的镜像」。
7. 提交前 `.venv/bin/python -m pytest tests/ -q` 全绿
   （已知无关预存失败：`test_main_depth_guard.py::test_depth_inference_without_service_returns_2`）。
8. **一条一提交**，commit message 注明本工单编号（如 `fix(wander): P0-1 ...`）。
   不要把三批混在一个提交里——真机回归时无法归因。

**分批要求：** 第一批（P0）改完先请主人真机短飞验证，通过后再动第二批。
第三批（P2 + 测试整改）可与第二批合并。

**遇到规格与本工单冲突：** 以 `2026-07-27-wander-explore-design.md` 字面为准，
并在本文件末尾「争议记录」追加一条，停下等裁决，**禁止自行决定**。

---

## 1. 复现脚本

以下脚本只读导入仓库，不写任何文件。放到临时目录运行，用于修复前确认现象、修复后确认消失。

```python
"""Probe suspected wander.py defects. Read-only: imports repo, writes nothing."""
import sys
import numpy as np

sys.path.insert(0, r"<仓库根目录>")

from tt_control.wander import (
    WanderParams, WanderPolicy,
    WANDER_CRUISE, WANDER_TURN, WANDER_VERIFY, WANDER_PANO,
)


def grid(v, shape=(96, 128)):
    return np.full(shape, v, dtype=np.float32)


def wall(mid=0.70, left=0.15, right=0.55, shape=(96, 128)):
    n = grid(0.10, shape)
    w = shape[1]
    t = w // 3
    n[:, :t] = left
    n[:, t:2 * t] = mid
    n[:, 2 * t:] = right
    return n


def tel(h=120, yaw=0, bat=80):
    d = {"bat": str(bat), "yaw": str(yaw)}
    if h is not None:
        d["h"] = str(h)
    return d


# PROBE 1: below-band altitude recovery
pol = WanderPolicy(WanderParams(seed=1, alt_change_prob=1.0, alt_throttle=25,
                                alt_segment_s_min=9, alt_segment_s_max=9,
                                segment_s_min=50, segment_s_max=50,
                                free_turn_prob=0.0, h_min_cm=80, h_max_cm=200), seed=1)
pol.decide(grid(0.15), tel(h=120), now=1.0, depth_ts=1.0)
pol._alt_throttle = 25          # force a CLIMB sub-segment
pol._alt_until = 100.0
for h in (120, 79, 60, 200, 201, 250):
    d = pol.decide(grid(0.15), tel(h=h), now=2.0 + h * 0.001, depth_ts=2.0 + h * 0.001)
    print(f"  h={h:>4}cm  cmd=+25(climb)  ->  throttle={d.axes.throttle}")

# PROBE 2: h-missing latch trigger rate (control frames, not depth frames)
pol = WanderPolicy(WanderParams(seed=2, h_missing_frames=5, alt_change_prob=1.0,
                                free_turn_prob=0.0, segment_s_min=50, segment_s_max=50), seed=2)
pol.decide(grid(0.15), tel(h=120), now=1.0, depth_ts=1.0)
pol._alt_throttle = 25
pol._alt_until = 1000.0
t = 1.0
for i in range(6):
    t += 1.0 / 24.0                       # 24 Hz control loop
    d = pol.decide(grid(0.15), tel(h=None), now=t, depth_ts=1.0)  # SAME depth_ts
print(f"  after {6/24.0:.2f}s of missing h at 24Hz -> latched={pol._h_latch_zero}")
d = pol.decide(grid(0.15), tel(h=120), now=t + 5.0, depth_ts=t + 5.0)
print(f"  h restored to 120cm, 5s later -> throttle={d.axes.throttle} (latch permanent)")

# PROBE 3: PANO travel accumulates |yaw noise| while hovering
pol = WanderPolicy(WanderParams(seed=3, pano_complete_deg=340.0), seed=3)
pol._state = WANDER_PANO
pol._pano_phase = "scan"
pol._pano_start_t = 0.0
pol._pano_start_yaw = None
rng = np.random.default_rng(0)
t = 0.0
for i in range(300):                       # 12.5s at 24Hz, drone NOT rotating
    t += 1.0 / 24.0
    noisy_yaw = float(rng.normal(0.0, 0.5))   # +-0.5 deg jitter, no real rotation
    pol.decide(grid(0.15), tel(yaw=noisy_yaw), now=t, depth_ts=t)
print(f"  _pano_travel_deg accumulated = {pol._pano_travel_deg:.1f} deg (threshold 340)")

# PROBE 4: corner window vs pano duration (DEFAULT params)
pol = WanderPolicy(WanderParams(seed=4, corner_max_turns=4, corner_window_s=20.0), seed=4)
pol._obstacle_turn_times = [100.0, 101.0, 102.0, 103.0]
pol._pano_done_in_window = True
d = pol._start_obstacle_turn((0.2, 0.7, 0.5), now=118.0)
print(f"  t=118s: abort={d.abort_reason!r}")
pol2 = WanderPolicy(WanderParams(seed=5, corner_max_turns=4, corner_window_s=20.0), seed=5)
pol2._obstacle_turn_times = [100.0, 101.0, 102.0, 103.0]
pol2._pano_done_in_window = True
d2 = pol2._start_obstacle_turn((0.2, 0.7, 0.5), now=125.0)
print(f"  t=125s: abort={d2.abort_reason!r}  <-- lost")

# PROBE 5: dead-reckon turn clock corrupted by DANGER_HOLD
pol = WanderPolicy(WanderParams(seed=6, turn_confirm_frames=1, turn_min_deg=120,
                                turn_max_deg=120, yaw_speed=40,
                                yaw_dead_reckon_dps_per_unit=1.0,
                                danger_hold_s=0.4, segment_s_min=50, segment_s_max=50,
                                free_turn_prob=0.0), seed=6)
t = 10.0
pol.decide(grid(0.2), tel(h=120, yaw=None), now=t, depth_ts=t)
t += 0.2
d = pol.decide(wall(0.70), {"bat": "80", "h": "120"}, now=t, depth_ts=t)   # no yaw telemetry
t += 0.2
d = pol.decide(grid(0.90), {"bat": "80", "h": "120"}, now=t, depth_ts=t)   # danger interrupts
t += 5.0                                   # 5s stuck in DANGER_HOLD
d = pol.decide(grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)   # danger clears
t += 0.04
d = pol.decide(grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
est = 40 * 1.0 * (t - pol._turn_start_t)
print(f"  state={d.state}  dead-reckon est={est:.0f}deg vs target 120deg")
```

**修复前的实测输出（基线）：**

```
PROBE 1  h=120 -> 25 | h=79 -> 0 | h=60 -> 0 | h=200 -> 0 | h=250 -> 0
PROBE 2  after 0.25s of missing h at 24Hz -> latched=True
         h restored to 120cm, 5s later -> throttle=0 (latch permanent)
PROBE 3  _pano_travel_deg accumulated = 169.6 deg (threshold 340)
PROBE 4  t=118s: abort='wander_cornered'
         t=125s: abort=''  <-- lost
PROBE 5  state=WANDER_VERIFY  dead-reckon est=210deg vs target 120deg
```

---

## 2. 第一批：P0 真机安全项

### P0-1　PANO 结束后跳过 VERIFY 直接前冲

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_step_pano()` seek 分支，约 `556-562` 行（`abs(err) <= turn_arrive_tol_deg` 命中处），以及无遥测降级分支约 `546-552` 行 |
| 现状 | 转到选定朝向后直接 `_enter(WANDER_CRUISE, now)` + `_begin_cruise_segment(now)`，下一帧即输出 `_seg_pitch` 前进 |
| 违反 | 规格 §2.1「转完不能立刻前冲」；orbit 坑 #2「转身瞬间的旧深度帧」 |
| 危害 | 全景选向是本模块转角最大的动作（最多 360°），转完瞬间的 nearness 完全是转身前的画面（端到端 ~0.5s 延迟）。普通遇障转向都做了 VERIFY，风险最高的这条反而漏了 |

**要求的行为：** PANO seek 完成后必须与普通转向走**同一条** VERIFY 路径，
只认 `depth_ts > turn_end_ts` 的新鲜深度帧，连续 `verify_frames` 帧开阔才允许前飞。

**改法建议：** seek 完成处不要进 CRUISE，改为：

```python
self._turn_end_ts = now      # 与 _step_turn 完成时同语义
self._verify_clear = 0
self._enter(WANDER_VERIFY, now)
```

无遥测降级分支（`pano_seek_timeout`）同样处理。注意 `_step_verify` 的超时用
`self._state_entered`，`_enter` 已负责刷新，无需额外处理。

**验收测试（新增）：** `test_pano_enters_verify_not_cruise`
- 构造进入 PANO → 完成 scan → seek 到位
- 断言状态为 `WANDER_VERIFY` 而非 `WANDER_CRUISE`
- 断言该帧及后续 `pitch == 0`，直到喂入 `depth_ts > turn_end_ts` 的连续 `verify_frames` 个开阔帧后才回 CRUISE
- 断言喂入 `depth_ts <= turn_end_ts` 的开阔旧帧**不**计数（与 `test_verify_rejects_stale_depth_before_turn_end_ts` 同手法）

---

### P0-2　DANGER 打断转向后，航位推算时钟被污染

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_step_turn()` 约 `356-360` 行；`decide()` 的 `_resume_after_danger` 恢复路径约 `279-291`、`428` 行 |
| 现状 | dead-reckon 用 `est = abs(yaw_cmd) * dps * (now - self._turn_start_t)`；DANGER_HOLD / RETREAT 期间墙钟照走，`_turn_start_t` 不做任何补偿 |
| 违反 | 规格 §2.2「转角测量」；实际造成未转到位即判定完成 |
| 危害 | 实测（PROBE 5）：目标 120°，转向刚开始被 danger 打断，5s 后危险解除，下一帧估算值 210° → 直接进 VERIFY，**实际一度未转**。朝向完全错误，且很可能仍朝着刚触发 danger 的障碍物 |
| 范围 | 仅影响 yaw 遥测读不到的降级路径，但规格 §2.2 明确要求支持该路径 |

**要求的行为：** dead-reckon 估算只能累加「实际在发 yaw 杆量」的时长，
不得把 DANGER_HOLD / RETREAT 等零杆期间计入。

**改法建议（二选一，倾向方案 A）：**

- **方案 A（推荐）**：把 `_turn_start_t` 的墙钟差改为累加式。新增 `self._turn_yaw_elapsed: float`，
  只在 `_step_turn` 真正输出 yaw 杆量的那一帧累加 `now - self._turn_last_tick`，
  并在每帧末更新 `_turn_last_tick = now`。进入 DANGER_HOLD 时不更新，恢复时把
  `_turn_last_tick` 重置为 `now`（丢弃暂停时长）。
- **方案 B**：进入 DANGER_HOLD 时记 `_turn_paused_at = now`，从 DANGER_HOLD 恢复到
  `WANDER_TURN` 时执行 `self._turn_start_t += (now - self._turn_paused_at)`。改动小，
  但对「DANGER→RETREAT→再回 TURN」的路径要一并覆盖。

无论哪种方案，**不得**引入新的墙钟来源，时间只能来自 `now` 参数。

**验收测试（新增）：** `test_dead_reckon_turn_not_credited_during_danger_hold`
- 复刻 PROBE 5 场景（无 yaw 遥测，转向中被 danger 打断 5s 后解除）
- 断言 danger 解除后的下一帧状态仍为 `WANDER_TURN`（不是 VERIFY）
- 断言继续喂帧直到真实累计发杆时长满足目标角度后，才转入 VERIFY

---

### P0-3　深度冻结期间仍输出满前进杆

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_step_cruise()` 约 `334` 行（`pitch = self._seg_pitch`）；`decide()` danger 检测约 `280` 行（`if new_depth and danger > ...`） |
| 现状 | 图传冻结或推理卡住时 `depth_ts` 不变 → `new_depth` 恒 False → danger 检测不触发，而 CRUISE 仍以 `_seg_pitch`（最高 25）持续前冲，直到 FSM 的 `depth_stale_s = 3.0` 兜底 |
| 违反 | 规格 §3「danger 处理三段式」的精神；orbit 安全不变量 #4「检测不可靠时不得盲飞」 |
| 危害 | 室内 3 秒全速前冲距离很长。Codex 评审第 1 条要求「hold 计时只在新深度帧累加」是对的，但连带把 danger 的**进入条件**也绑死在新帧上，属于过度修正 |

**要求的行为：** 感知失效时应逐步收杆，而不是维持原杆量等全局超时。

**改法建议：** `_step_cruise` 增加深度新鲜度 taper。记录最近一次 `new_depth` 为真的
`now`（`self._last_new_depth_t`），计算 `age = now - self._last_new_depth_t`：

- `age <= depth_taper_s` → pitch 不变
- `depth_taper_s < age` → pitch 线性递减，在 `age >= depth_taper_s * 2` 时降到 0

**新增参数（必须落 `configs/default.json` 的 `wander` 节 + `WanderParams`）：**

| 参数 | 建议默认 | 说明 |
|---|---|---|
| `depth_taper_s` | 0.6 | 深度停更超过此时长开始收杆（≈3 个深度帧周期，5Hz 下） |

**验收测试（新增）：** `test_frozen_depth_tapers_cruise_pitch`
- CRUISE 稳定前飞后，固定 `depth_ts` 不变、`now` 持续推进
- 断言 pitch 单调不增，且在 `2 * depth_taper_s` 后为 0
- 断言 `depth_ts` 恢复更新后 pitch 能回到段内取值

---

## 3. 第二批：P1 规格落空项

### P1-4　高度钳制写反，无法飞回巡航带

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_apply_h_clamp()` 约 `713-729` 行 |
| 现状 | 第一个 `if h < h_min or h > h_max: return 0` 双向归零；后面两个单向判断因此只在 `h` 恰好等于边界时才可能生效，**近似死代码** |
| 违反 | 规格 §2.3「高度用遥测 h 闭环钳在巡航带内」 |
| 危害 | 实测（PROBE 1）h=79cm 给爬升 +25 输出仍为 0。Tello 起飞悬停高度约 80-100cm，紧贴下界，气流一扰动就跌破，此后整段飞行高度控制全失效 |
| 佐证 | 后两行单向判断的存在本身说明作者本意就是单向钳制，被前一行覆盖了 |

**要求的行为：** 单向钳制——高于上界只禁止正 throttle，低于下界只禁止负 throttle，
反方向（回到带内）永远允许。

**改法建议：** 删除双向归零那一行，保留并修正两个单向判断：

```python
if h >= self.p.h_max_cm and throttle > 0:
    return 0
if h <= self.p.h_min_cm and throttle < 0:
    return 0
return throttle
```

**验收测试（改造 `test_height_band_and_missing_h_zeros_throttle`）：**
- 保留现有「h 在带内 → throttle 透传」断言
- **新增**：`h = h_min - 20`（跌破下界）且 throttle 为正 → 断言 throttle **透传**（可爬回）
- **新增**：`h = h_min - 20` 且 throttle 为负 → 断言 throttle == 0
- **新增**：`h = h_max + 50`（超上界）且 throttle 为正 → 断言 throttle == 0
- **新增**：`h = h_max + 50` 且 throttle 为负 → 断言 throttle **透传**（可降回）

---

### P1-5　高度闩锁按控制帧计数，0.25 秒丢遥测即永久失效

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_apply_h_clamp()` 约 `717-721` 行；由 `_pack()` 每个控制帧调用 |
| 现状 | `_apply_h_clamp` 在 ~24Hz 的控制环里被调用，`h_missing_frames = 5` 实际等于 **0.21 秒**；`_h_latch_zero` 一旦置位在整个 episode 内不可恢复 |
| 违反 | orbit 坑 #8 / 规格 §7 第 4 条：全模块 frames 参数均按 5Hz 深度帧计，此处按控制帧计属不一致 |
| 危害 | 实测（PROBE 2）0.25s 丢遥测即闩死，且 h 恢复后仍为 0。AUTO 挂载瞬间 `client.state` 可能仍是空字典，等于起飞就闩死。叠加 P1-4，规格 §2.3「高度微调也是数据」实际采不到数据 |

**要求的行为：** 计数口径改为时间（或深度帧），与全模块一致。

**改法建议：** 用时间判据替代控制帧计数。记录最近一次成功读到 `h` 的 `now`
（`self._h_last_ok_t`），`now - self._h_last_ok_t >= h_missing_s` 才置零。

**新增参数：**

| 参数 | 建议默认 | 说明 |
|---|---|---|
| `h_missing_s` | 1.0 | 连续读不到 `h` 超过此时长 → 停用高度控制（≈5 个深度帧周期） |

`h_missing_frames` 保留在 dataclass 中标记为 deprecated（旧配置不炸），但不再参与逻辑。

**⚠️ 待裁决（见 §6）：** 闩锁是否改为可恢复。规格原文写「throttle 永远 0，只做平面漫游」，
本工单倾向改为可恢复（连续读到 `h` 满 `h_missing_s` 即解除），但这属于改规格，
**Grok 不得自行决定**。在裁决下达前，先只改计数口径，闩锁行为保持现状。

**验收测试（改造）：**
- 断言以 24Hz 喂 `h` 缺失帧、总时长 < `h_missing_s` 时**不**闩锁
- 断言总时长 >= `h_missing_s` 时闩锁
- 裁决为「可恢复」后再补解除测试

---

### P1-6　全景扫描把 yaw 噪声当行程累加

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_step_pano()` 约 `503-508` 行 |
| 现状 | 注释写「累计有符号最短角差」，代码却是 `self._pano_travel_deg += abs(step)`——逐帧取绝对值再求和 |
| 违反 | 注释与实现不符；Codex 评审第 5 条处置意图未落实 |
| 危害 | 实测（PROBE 3）：飞机**完全不转**，仅 ±0.5° yaw 噪声，12.5s 累计出 169.6° 虚假行程（阈值 340 的一半）。实飞时会明显提前判定「扫完一圈」，此时 `_pano_samples` 覆盖方位不足一圈，「选最开阔方向」依据残缺——本用于脱困的动作反而可能选到没扫到的方向 |

**要求的行为：** 累计**有符号**角差，判定时取总和的绝对值；并对单帧增量加噪声死区。

**改法建议：**

```python
step = _yaw_delta(cur_yaw, self._pano_last_yaw)
if abs(step) >= self.p.pano_step_deadband_deg:
    self._pano_travel_deg += step          # 有符号累加
    self._pano_last_yaw = cur_yaw
```

判定处改为 `abs(self._pano_travel_deg) >= self.p.pano_complete_deg`。
注意 `_pano_last_yaw` 只在增量被采纳时更新，否则死区会把持续慢转也滤掉。
同时**修正注释**使其与实现一致。

**新增参数：**

| 参数 | 建议默认 | 说明 |
|---|---|---|
| `pano_step_deadband_deg` | 0.5 | 单帧 yaw 增量死区，低于此值视为传感器噪声不计入行程 |

**验收测试（新增）：** `test_pano_travel_ignores_yaw_noise`
- 复刻 PROBE 3：`yaw` 只有 ±0.5° 噪声、无真实旋转，喂 300 个控制帧
- 断言 `_pano_travel_deg` 绝对值 < 30（远低于阈值），且状态仍为 `WANDER_PANO`
- 补一条正向用例：真实同向慢转 340° → 断言判定完成

---

### P1-7　防打转 abort 实际不可达

| 项 | 内容 |
|---|---|
| 位置 | `wander.py` `_prune_corner_window()` 约 `693-697` 行；`_start_obstacle_turn()` 约 `571-581` 行 |
| 现状 | `_pano_done_in_window` 只在 `_obstacle_turn_times` 被 prune 空时清除。而 `corner_window_s = 20.0` 与 `pano_timeout_s = 20.0` 同量级，全景扫描在 30°/s 下转完 340° 就需 11.3s，PANO 走完后那 4 个转向时间戳早已滑出窗口 |
| 违反 | 规格 §2.2「全景选向后仍在 corner_window_s 内再触发 → abort "wander_cornered"」 |
| 危害 | 实测（PROBE 4）：t=118s 仍能 abort，t=125s（真实 PANO 结束 + 巡航一会儿）abort 消失。飞机会在角落反复「遇障 4 次 → 全景 → 遇障 4 次 → 全景」空转耗光电池，且该段数据价值极低 |
| 佐证 | `test_corner_pano_then_abort` 把 `corner_window_s` 从默认 20 调到 30 才能通过，见 §5 |

**要求的行为：** 「PANO 结束后 `corner_window_s` 内再遇障即 abort」这一判据
不得依赖 `_obstacle_turn_times` 是否为空。

**改法建议：** 把布尔标志 `_pano_done_in_window` 换成时间戳 `self._pano_done_at: Optional[float]`：

- PANO scan 完成时置 `self._pano_done_at = now`
- `_start_obstacle_turn` 开头判定：
  `if self._pano_done_at is not None and now - self._pano_done_at <= self.p.corner_window_s: abort`
- `_prune_corner_window` 中改为：`if self._pano_done_at is not None and now - self._pano_done_at > self.p.corner_window_s: self._pano_done_at = None`，
  不再与 `_obstacle_turn_times` 是否为空耦合

**验收测试（改造 `test_corner_pano_then_abort` + 新增）：**
- **必须改回默认 `corner_window_s`（20.0），并删除对 `pol._pano_start_yaw` / `pol._pano_start_t` 的私有状态注入**
- 断言 PANO 结束后 `corner_window_s` 内再遇障 → `abort_reason == "wander_cornered"`
- 新增反向用例：PANO 结束后**超过** `corner_window_s` 再遇障 → 正常转向、**不** abort

---

## 4. 第三批：P2 质量项

逐条改动都很小，可合并为一个提交，但每条需带断言。

| # | 问题 | 位置 | 要求 |
|---|---|---|---|
| P2-8 | 转向到位只看绝对角差，不校验方向；注释声称做了「方向一致性」检查，实际没有——转反了也判到位 | `_step_turn()` 约 `364-366` | 增加方向校验：`_yaw_delta(cur_yaw, _turn_start_yaw)` 的符号须与 `_turn_dir` 一致才计入 `turned`；或修正注释使其不谎称已实现（**倾向前者**） |
| P2-9 | 两处 abort 直接构造 `WanderDecision` 绕过 `_pack`，跳过 roll 归零与高度钳制（当前杆量全零无害，但破坏「所有输出统一过安全钳」不变量） | 约 `475-481`、`575-581` | 统一改为经 `_pack()` 返回，abort 信息通过 `_pack` 新增参数传递 |
| P2-10 | `_pano_start_t <= 0.0` 哨兵在 `now=0.0` 时会被多重置一帧（规格 §5 专门警告过 0.0 哨兵，orbit 为此炸过） | 约 `497` | 改为 `Optional[float]`，用 `None` 做哨兵 |
| P2-11 | `__init__` / `begin_episode()` 内部调 `time.time()` 生成 seed，违反 §5「所有时间依赖通过 now 注入」 | 约 `119`、`190` | seed 为 0 时由调用方（`app.py` / FSM 装配处）注入时间戳；内核不引用 `time` 模块 |
| P2-12 | 进入 PANO 时 `turns_total += 1`，但并未执行转向，污染 §9.3 验收指标「遇障转向 ≥ 8 次」 | 约 `586` | PANO 入口不计 `turns_total`；`panos_total` 已单独统计 |
| P2-13 | 首帧无深度时 SEG 事件被挂起延后发出，frames.csv 里事件时间戳与实际段起点不符 | `decide()` 约 `259-271` | 首帧无深度时不预开段，或在深度首帧到达时重开段并重发 SEG |

---

## 5. 测试整改清单

现状 13 个用例全绿，但有两处属于「迁就实现」而非「验证规格」，
违反规格 §8.2「防止测试写成实现的镜像——断言抄实现代码 = 无效测试」。

| # | 用例 | 问题 | 整改 |
|---|---|---|---|
| T-1 | `test_corner_pano_then_abort` | 用 `corner_window_s=30.0`（默认 20.0）绕开 P1-7；且直接写私有状态 `pol._pano_start_yaw` / `pol._pano_start_t` | 改回默认参数、去掉私有状态注入；见 P1-7 验收 |
| T-2 | `test_height_band_and_missing_h_zeros_throttle` | 只断言「超带→0」，未断言「跌破下界应允许爬回」，等于把 P1-4 的错误行为固化成期望 | 见 P1-4 验收 |
| T-3 | `test_invariant_roll_zero_pitch_neg_only_retreat` | 「全状态遍历」实际未覆盖 `WANDER_PANO` | 把 PANO 纳入遍历，断言其 `roll == 0`、`pitch == 0` |
| T-4 | — | 缺 PANO 后必须 VERIFY 的回归 | 见 P0-1 验收 |
| T-5 | — | 缺规格 §9.2 的仿真冒烟（固定 seed 跑 10 分钟 `sim_drone`，断言无 abort、转向次数 > 0、无「连续 30s 杆量全零且非 DANGER/VERIFY」） | 新增，作为 CI 可跑用例 |

---

## 6. 待裁决项（Grok 不得自行决定）

| # | 议题 | 背景 | 建议 |
|---|---|---|---|
| D-1 | 高度闩锁是否改为可恢复 | 规格 §2.3 原文「throttle 永远 0，只做平面漫游」。但遥测短暂抖动就永久禁用高度控制过于严苛，会让 §2.3「高度也是数据」的要求落空 | 倾向改为可恢复（连续读到 `h` 满 `h_missing_s` 即解除）。属改规格，需主人或 Claude 裁决 |
| D-2 | 高度子段门槛是否放宽 | `_cruise_throttle` 要求 `mid < clear_thresh`(0.40) 才做高度微调，而 `turn_thresh` 是 0.58，中间地带占比不小 → 高度样本天然稀疏 | 先修完 P1-4 / P1-5 采一次真机数据，用 `episode_check.py` 的 throttle 直方图判断是否还需放宽，**不要提前调参** |

---

## 7. 完成定义（DoD）

1. 三批修复全部合入，每条带回归测试，`pytest tests/ -q` 全绿（除已知预存失败）。
2. §1 复现脚本 5 个 PROBE 全部不再复现基线现象。
3. 规格 §9.2 仿真冒烟通过（T-5）。
4. 主人真机验收：按规格 §9.3 阶梯跑「首飞（保守参数，2 分钟）」→「验收飞行（默认参数，≥5 分钟）」，
   `episode_check.py` 指标全部达标。
5. 验收通过后由 Claude 在 `2026-07-27-wander-explore-design.md` 补「已验证基线」节
   （该文件 chmod 444，**Grok 与本工单作者均无权修改**）。

---

## 8. 争议记录

（实现方若认为本工单某条与规格冲突或不可实现，在此追加：编号 / 理由 / 建议，然后停下等裁决。）

- 暂无
