# Wander 模块缺陷审查与修复工单（2026-07-28）

> 本文由 Opus5 粘贴稿经 **本机 PROBE 复现 + Codex 交叉校对** 复原。  
> 表格已重整；与规格/代码冲突处在 §8 争议记录与各条 Disposition 中标明。  
> **后续实现以本文 Disposition 为准**（非未校对的粘贴原文）。

- 审查对象：`tt_control/wander.py`、`tests/test_wander.py`、`configs/default.json` 的 `wander` 节  
  （FSM/App 仅作证据引用；本工单默认改动范围见 §0）
- 依据规格：`docs/design/2026-07-27-wander-explore-design.md`（chmod 444，只读）
- 前置必读：`docs/design/2026-07-27-orbit-control-principles.md` §2 / §3
- 已有评审：`docs/dev-notes/2026-07-27-wander-codex-cross-review.md`（本文是其后续，不重复已处置项）
- 复原校对：Grok + Codex（2026-07-28）；PROBE 基线已在仓库根目录复现

## 0. 执行规则与采纳总表

### 0.1 必须遵守

1. 默认改动范围：`tt_control/wander.py`、`tests/test_wander.py`、`configs/default.json` 的 `wander` 节。  
   触碰 `app.py` / FSM 全局深度守卫仅在争议记录立项后由主人裁决。
2. 禁止改：Orbit 控制路径、降落重试、`EpisodeRecorder` 帧↔行同步、Claude 所有权文档。
3. 禁止硬编码安全阈值；新参数进 `wander` 节 + `WanderParams`，代码只引用 `self.p.xxx`。
4. 禁止新第三方依赖；`decide()` 保持纯函数式（时间只经 `now`）。
5. 每条修复带「规格行为」回归测试，禁止镜像实现。
6. 提交前：`.venv/bin/python -m pytest tests/ -q` 全绿  
   （已知无关预存失败见规格 §7.8，以当时仓库为准）。
7. 建议一条一提交；P0 优先。真机短飞验证后再大规模调参。

### 0.2 Disposition（经 Codex 校对）

| ID | 摘要 | Disposition | 理由（一句话） |
|---|---|---|---|
| P0-1 | PANO seek 跳过 VERIFY | **ADOPT** | 规格 §2.1 硬要求；全景转角最大 |
| P0-2 | dead-reckon 被 DANGER 暂停污染 | **ADOPT** | PROBE5 确认；未转到位进 VERIFY |
| P0-3 | 冻深度仍满 pitch + taper | **REJECT** | 深度新鲜度属全局层；taper 双轨超时；见 §8 |
| P1-4 | 高度双向归零无法回带 | **ADOPT** | PROBE1；规格 §2.3 闭环 |
| P1-5 | h 缺失按控制帧计数 | **ADOPT（口径+可恢复）** | `h_missing_s` 计时；主人 D-1=B 后连续读满可解除 |
| P1-6 | PANO `abs(step)` 吃噪声 | **ADOPT（修订）** | 有符号行程；死区走参数非裸字面量 |
| P1-7 | corner abort 实际不可达 | **ADOPT** | PROBE4；deadline 从 PANO 完成起算 |
| P2-8 | 到位不校验转向方向 | **ADOPT** | 注释撒谎；应做符号校验 |
| P2-9 | abort 绕过 `_pack` | **REJECT** | 当前已零杆；纯风格无安全增益 |
| P2-10 | `_pano_start_t<=0` 哨兵 | **ADOPT** | `now=0` 合法（orbit 同款坑） |
| P2-11 | 内核 `time.time()` 取 seed | **ADOPT（wander-only）** | 首次 `decide(now)` 定 seed，不改 app |
| P2-12 | PANO 计入 `turns_total` | **DEFER** | 计数语义未定契约 |
| P2-13 | 无深度首帧 SEG 挂起 | **REJECT** | 生产 FSM 无深度不调 decide |
| T-1…T-5 | 测试整改 + §9.2 冒烟 | **ADOPT** | 去掉迁就实现的断言 |

安全优先级：**不炸机 > 控制正确性/可复现 > 数据指标洁癖**。

## 1. 复现脚本与基线输出

脚本只读导入仓库。修复前应复现下列输出；修复后对应现象消失。

```python
"""Probe suspected wander.py defects. Read-only."""
import numpy as np
from tt_control.wander import WanderParams, WanderPolicy, WANDER_PANO

def grid(v, shape=(96, 128)):
    return np.full(shape, v, dtype=np.float32)

def wall(mid=0.70, left=0.15, right=0.55, shape=(96, 128)):
    n = grid(0.10, shape)
    w = shape[1]; t = w // 3
    n[:, :t] = left; n[:, t:2 * t] = mid; n[:, 2 * t:] = right
    return n

def tel(h=120, yaw=0, bat=80):
    d = {"bat": str(bat), "yaw": str(yaw)}
    if h is not None:
        d["h"] = str(h)
    return d

# PROBE 1 — below-band climb blocked
pol = WanderPolicy(WanderParams(
    seed=1, alt_change_prob=1.0, alt_throttle=25,
    alt_segment_s_min=9, alt_segment_s_max=9,
    segment_s_min=50, segment_s_max=50, free_turn_prob=0.0,
    h_min_cm=80, h_max_cm=200), seed=1)
pol.decide(grid(0.15), tel(h=120), now=1.0, depth_ts=1.0)
pol._alt_throttle = 25; pol._alt_until = 100.0
for h in (120, 79, 60, 200, 201, 250):
    d = pol.decide(grid(0.15), tel(h=h), now=2.0 + h * 0.001, depth_ts=2.0 + h * 0.001)
    print(f"PROBE1 h={h} -> throttle={d.axes.throttle}")

# PROBE 2 — missing-h latch at control rate
pol = WanderPolicy(WanderParams(
    seed=2, h_missing_frames=5, alt_change_prob=1.0,
    free_turn_prob=0.0, segment_s_min=50, segment_s_max=50), seed=2)
pol.decide(grid(0.15), tel(h=120), now=1.0, depth_ts=1.0)
pol._alt_throttle = 25; pol._alt_until = 1000.0
t = 1.0
for _ in range(6):
    t += 1.0 / 24.0
    pol.decide(grid(0.15), tel(h=None), now=t, depth_ts=1.0)
print(f"PROBE2 latched={pol._h_latch_zero}")
d = pol.decide(grid(0.15), tel(h=120), now=t + 5.0, depth_ts=t + 5.0)
print(f"PROBE2 restored throttle={d.axes.throttle}")

# PROBE 3 — pano noise travel
pol = WanderPolicy(WanderParams(seed=3, pano_complete_deg=340.0), seed=3)
pol._state = WANDER_PANO; pol._pano_phase = "scan"
pol._pano_start_t = 0.0; pol._pano_start_yaw = None
rng = np.random.default_rng(0); t = 0.0
for _ in range(300):
    t += 1.0 / 24.0
    pol.decide(grid(0.15), tel(yaw=float(rng.normal(0.0, 0.5))), now=t, depth_ts=t)
print(f"PROBE3 travel={pol._pano_travel_deg:.1f}")

# PROBE 4 — corner window
pol = WanderPolicy(WanderParams(seed=4, corner_max_turns=4, corner_window_s=20.0), seed=4)
pol._obstacle_turn_times = [100.0, 101.0, 102.0, 103.0]
pol._pano_done_in_window = True
print("PROBE4 t=118", pol._start_obstacle_turn((0.2, 0.7, 0.5), now=118.0).abort_reason)
pol2 = WanderPolicy(WanderParams(seed=5, corner_max_turns=4, corner_window_s=20.0), seed=5)
pol2._obstacle_turn_times = [100.0, 101.0, 102.0, 103.0]
pol2._pano_done_in_window = True
print("PROBE4 t=125", pol2._start_obstacle_turn((0.2, 0.7, 0.5), now=125.0).abort_reason)

# PROBE 5 — dead-reckon + danger
pol = WanderPolicy(WanderParams(
    seed=6, turn_confirm_frames=1, turn_min_deg=120, turn_max_deg=120,
    yaw_speed=40, yaw_dead_reckon_dps_per_unit=1.0, danger_hold_s=0.4,
    segment_s_min=50, segment_s_max=50, free_turn_prob=0.0), seed=6)
t = 10.0
pol.decide(grid(0.2), tel(h=120, yaw=None), now=t, depth_ts=t)
t += 0.2
pol.decide(wall(0.70), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
t += 0.2
pol.decide(grid(0.90), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
t += 5.0
pol.decide(grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
t += 0.04
d = pol.decide(grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
print(f"PROBE5 state={d.state} est={40*(t-pol._turn_start_t):.0f}")
```

### 1.1 本机复现基线（2026-07-28）

```
PROBE1  h=120 -> 25 | h=79 -> 0 | h=60 -> 0 | h=200 -> 0 | h=201 -> 0 | h=250 -> 0
PROBE2  latched=True after 0.25s @24Hz; restored throttle=0
PROBE3  travel=169.6 (thresh 340)
PROBE4  t=118 abort=wander_cornered; t=125 abort='' (lost)
PROBE5  state=WANDER_VERIFY est=210 vs target 120
```

## 2. 第一批 P0

### P0-1　PANO 结束后跳过 VERIFY 直接前冲 — ADOPT

- **位置**：`_step_pano()` seek 完成 / `pano_seek_timeout`（约 543–562 行）
- **现状**：到位后 `_enter(WANDER_CRUISE)` + `_begin_cruise_segment`，下一帧即 pitch 前进
- **违反**：规格 §2.1；orbit 坑 #2（转身旧深度）
- **改法**：seek 完成处与普通转向一致：
  ```python
  self._turn_end_ts = now
  self._verify_clear = 0
  self._enter(WANDER_VERIFY, now)
  ```
  无遥测超时分支同样处理。
- **测试**：`test_pano_enters_verify_not_cruise`  
  （含旧 `depth_ts` 不计数、新鲜帧连续 `verify_frames` 才 CRUISE）

### P0-2　DANGER 打断后 dead-reckon 时钟污染 — ADOPT

- **位置**：`_step_turn()` dead-reckon；`_resume_after_danger` 恢复
- **现状**：`est = |yaw_cmd| * dps * (now - _turn_start_t)`，暂停墙钟仍计入
- **证据**：PROBE5
- **改法（方案 A）**：新增 `_turn_yaw_elapsed`，仅在真正输出 yaw 杆的帧累加；  
  进入 DANGER/RETREAT 不累加；恢复时 `_turn_last_tick = now`
- **测试**：`test_dead_reckon_turn_not_credited_during_danger_hold`

### P0-3　冻深度仍满前进 + `depth_taper_s` — REJECT

- **Opus 主张**：Wander 内按深度年龄线性收杆
- **Codex / 复原结论**：REJECT  
  - 规格把深度过期交给全局安全（`depth_stale_s` / App watchdog）  
  - Wander 再 taper = 双轨阈值，动作标签被推理抖动污染  
  - 另：FSM `_last_depth_ts = now`（有 nearness 即刷新）是**全局**问题，不在本工单默认改动范围；见 §8
- **若主人坚持收杆**：优先下调已有 `fsm.depth_stale_s`，勿在 Wander 再造一套

## 3. 第二批 P1

### P1-4　高度钳制写反 — ADOPT

- **位置**：`_apply_h_clamp()`
- **现状**：`h < min or h > max → return 0` 覆盖单向回带逻辑（PROBE1）
- **改法**：删除双向归零；仅  
  `h >= max and throttle > 0 → 0`；`h <= min and throttle < 0 → 0`
- **测试**：改造 `test_height_band_and_missing_h_zeros_throttle`（含跌破下界允许爬升）

### P1-5　h 缺失计数口径 — ADOPT（口径 + 可恢复）

- **现状**：`h_missing_frames=5` 在 ~24Hz `_pack` 上计数 ≈0.21s 即闩（PROBE2）
- **改法**：`h_missing_s`（默认 1.0）按时间计缺失；`h_missing_frames` 保留但不参与逻辑
- **D-1（主人 2026-07-28 = B）**：连续读到 `h` 满 `h_missing_s` → 解除闩锁，高度控制恢复  
  （偏离规格 §2.3「永远 0」字面，待 Claude 改设计文档）

### P1-6　PANO 行程吃噪声 — ADOPT（修订）

- **现状**：`+= abs(step)`（PROBE3）
- **改法**：有符号累加，完成判据用 `abs(travel) >= pano_complete_deg`；  
  可选 `pano_step_deadband_deg`（默认 0.5）经参数表配置，禁止裸字面量
- **测试**：`test_pano_travel_ignores_yaw_noise` + 真实慢转 340° 正向用例

### P1-7　防打转 abort 不可达 — ADOPT

- **现状**：`_pano_done_in_window` 随 obstacle 窗口 prune 清空（PROBE4）
- **改法**：`_pano_done_at: Optional[float]`，在 **seek/VERIFY 入口完成全景流程时**计时；  
  `now - _pano_done_at <= corner_window_s` 内再遇障 → `wander_cornered`
- **测试**：默认 `corner_window_s=20`；禁止注入私有 PANO 字段；超时后不 abort

## 4. 第三批 P2

| ID | Disposition | 要求 |
|---|---|---|
| P2-8 | ADOPT | 到位：有符号进度 × `_turn_dir` ≥ 目标角 |
| P2-9 | REJECT | abort 已零杆；不做纯风格重构 |
| P2-10 | ADOPT | `_pano_start_t: Optional[float]`，`None` 哨兵 |
| P2-11 | ADOPT | seed=0 时推迟到首次 `decide(now)` 用 `now` 定种；内核去掉 `time.time()` |
| P2-12 | DEFER | `turns_total` 语义未定 |
| P2-13 | REJECT | 生产路径无「无深度调 decide」 |

## 5. 测试整改（T）

| ID | 动作 |
|---|---|
| T-1 | `test_corner_pano_then_abort` 改回默认 window，去私有注入 |
| T-2 | 高度测试覆盖「出带可回带」 |
| T-3 | 不变量遍历含 PANO；RETREAT 时长 ≤ `retreat_s` |
| T-4 | 新增 P0-1 / P0-2 / P1-6 / P2-8 回归 |
| T-5 | §9.2 加速时间 10 分钟仿真冒烟（合成 nearness + 可选 SimDrone；不 sleep 真 10 分钟） |

## 6. 待裁决（实现方不得擅自定）

| ID | 议题 | 本批默认 |
|---|---|---|
| D-1 | 高度闩锁是否可恢复 | **主人 2026-07-28 选 B：可恢复**（连续读到 h 满 `h_missing_s` 解除）。偏离规格 §2.3「永远 0」字面，待 Claude 改设计文档 |
| D-2 | 高度子段 `mid < clear_thresh` 是否放宽 | 先修 P1-4/5 再采数看直方图 |
| D-3 | P0-3 / FSM 冻深度守卫 | **主人 2026-07-28 选 A：维持全局 depth_stale**；不在 Wander 加 taper |
| D-4 | `turns_total` 是否含进 PANO 的那次遇障 | DEFER P2-12 |

## 7. DoD

1. ADOPT 项合入且带回归；`pytest tests/ -q` 全绿（除已知无关失败）。
2. §1 五条 PROBE 不再复现基线缺陷现象。
3. T-5 §9.2 冒烟通过。
4. 真机阶梯仍按规格 §9.3；通过后由 **Claude** 写「已验证基线」。

## 8. 争议记录

| 编号 | 议题 | 结论 |
|---|---|---|
| C-1 | Opus P0-3 taper vs 规格全局深度过期 | **REJECT taper**；FSM `nearness is not None` 刷新墙钟是全局缺陷，另案 |
| C-2 | P2-11 要改 app.py vs 工单禁止改 app | **wander-only**：首次 `decide(now)` 定 seed |
| C-3 | Opus「闩锁永久」 vs reset 可清 | 订正为「AUTO 会话内」；**后被主人改为可恢复（D-1=B）** |
| C-4 | P2-9 abort 绕 _pack | REJECT（无害） |
| C-5 | P2-13 SEG 首帧 | REJECT（非生产路径） |

---

## 9. 落地状态（2026-07-28）

- ADOPT 项已合入 `tt_control/wander.py` / `tests/test_wander.py` / `configs/default.json` wander 节
- PROBE 1–5 修复后本机复测：P1 可爬回、P2 0.25s 不闩、P3 travel≈0、P4 窗口有效、P5 保持 TURN
- `tests/test_wander.py`：**21 passed**（含 §9.2 加速冒烟）
- 全量：`107 passed`，仅已知预存失败 `test_main_depth_guard.py::test_depth_inference_without_service_returns_2`
- Codex 对修复 diff 二次审查：**GATE PASS**，无新增 [P1]/[P2]

*复原责任：Grok；对抗校对：Codex；粘贴源：主人转发的 Opus5 审查稿。*
