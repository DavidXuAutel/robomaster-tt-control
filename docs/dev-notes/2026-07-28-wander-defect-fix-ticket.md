# Wander 模块缺陷审查与修复工单（2026-07-28）

> 来源：Opus 5 审查粘贴稿，经 Grok + Codex 对照仓库现状复原/校对。  
> 规格（只读）：`docs/design/2026-07-27-wander-explore-design.md`  
> 前置：`docs/design/2026-07-27-orbit-control-principles.md` §2/§3  
> 前序评审：`docs/dev-notes/2026-07-27-wander-codex-cross-review.md`（本文不重复其已处置项）  
> 复原日实测：§1 五个 PROBE 现象与原稿基线一致。

---

## 0. 审查对象与基线

| 项 | 内容 |
|---|---|
| 代码 | `tt_control/wander.py`、`avoidance_fsm.py` wander 分支、`configs/default.json` wander 节 |
| 测试 | `tests/test_wander.py`（13 用例，复原时全绿） |
| 结论 | 规格符合度高；存在 **3 项真机安全缺陷 + 4 项规格落空 + 6 项质量问题**；其中 5 项有可复现证据 |

### 执行边界（相对原稿的会话修订）

原稿要求「每条带回归测试 / 一条一提交 / 先 P0 真机再 P1」。  
**主人 2026-07-28 当场指示**：与 Codex 协作落地改进；**不做回归测试、不做代码审查**。  
因此本工单落地阶段：只改代码与配置，测试整改（§5）与提交策略交主人另定。

仍遵守的红线：

- 优先改 `tt_control/wander.py` + `configs/default.json` 的 `wander` 节
- 不改 orbit 控制、降落路径、EpisodeRecorder 同步核心、Claude 所有权文档
- 新阈值进 `WanderParams` + `default.json`，代码只引用 `self.p.xxx`
- 无新第三方依赖；`decide()` 保持纯函数式（时间靠 `now`）
- 与规格冲突 → §8 争议记录，停等裁决

---

## 1. 复现脚本与基线输出

只读导入仓库的 probe（仓库根目录运行）：

```python
"""Probe suspected wander.py defects. Read-only."""
import numpy as np
from tt_control.wander import WanderParams, WanderPolicy, WANDER_PANO

def grid(v, shape=(96, 128)):
    return np.full(shape, v, dtype=np.float32)

def wall(mid=0.70, left=0.15, right=0.55, shape=(96, 128)):
    n = grid(0.10, shape)
    w = shape[1]; t = w // 3
    n[:, :t] = left; n[:, t:2*t] = mid; n[:, 2*t:] = right
    return n

def tel(h=120, yaw=0, bat=80):
    d = {"bat": str(bat), "yaw": str(yaw)}
    if h is not None:
        d["h"] = str(h)
    return d

# PROBE 1–5：见原稿；2026-07-28 复原实测输出如下
```

**修复前实测（2026-07-28，与 Opus 原稿一致）：**

| Probe | 现象 |
|---|---|
| 1 高度带 | `h=120→25`；`h=79/60/200/201/250` 给爬升 +25 均得 `throttle=0`（跌破下界无法爬回） |
| 2 h 缺失闩锁 | 24Hz 下约 **0.25s** 即 `latched=True`；h 恢复后仍 `throttle=0` |
| 3 PANO 噪声 | 无真实旋转、±0.5° 抖动 12.5s → `_pano_travel_deg≈169.6`（阈值 340 的一半） |
| 4 角落 abort | `t=118` 仍 `wander_cornered`；`t=125` abort 丢失（`abort=''`） |
| 5 dead-reckon | danger 打断 5s 后 `est=210°` vs 目标 120° → 直接 `WANDER_VERIFY` |

---

## 2. 第一批 P0（真机安全）

### P0-1　PANO 结束后跳过 VERIFY 直接前冲

- **位置**：`_step_pano()` seek 完成 / `pano_seek_timeout`
- **现状**：seek 到位 → 直接 `WANDER_CRUISE` + 开段前飞
- **违反**：规格 §2.1「转完不能立刻前冲」；orbit 坑 #2
- **改法**：与普通转向相同 → 设 `_turn_end_ts = now`，清 `_verify_clear`，`_enter(WANDER_VERIFY)`
- **判定**：✅ 真问题，必改

### P0-2　DANGER 打断后 dead-reckon 时钟被污染

- **位置**：`_step_turn()` dead-reckon；danger 恢复路径
- **现状**：`est = yaw_cmd * dps * (now - _turn_start_t)`，HOLD/RETREAT 墙钟照计
- **危害**：PROBE 5 已证实未转到位进 VERIFY
- **改法（方案 A）**：累加 `_turn_yaw_elapsed`，仅在真正发 yaw 杆量的帧累加；进 HOLD 时冻结，恢复 TURN 时重置 last tick
- **判定**：✅ 真问题，必改

### P0-3　深度冻结期间仍满杆前冲

- **位置**：`_step_cruise` + danger 进入绑 `new_depth`
- **现状**：冻帧时 danger 不进，CRUISE 仍满 pitch，靠 FSM `depth_stale_s=3s` 兜底
- **改法**：`depth_taper_s`（默认 0.6）：停更超过后线性收杆，`2*taper` 到 0
- **判定**：✅ 真问题，必改（与前序「hold 只认新深度」互补，非回滚）

---

## 3. 第二批 P1（规格落空）

### P1-4　高度钳制写反，无法飞回巡航带

- **现状**：`h < min or h > max → return 0` 双向归零，单向逻辑成死代码
- **改法**：删双向归零；只禁止「越上界仍爬 / 越下界仍降」
- **判定**：✅ 真问题，必改（PROBE 1）

### P1-5　高度闩锁按控制帧计数

- **现状**：`h_missing_frames=5` @~24Hz ≈ 0.21s 永久闩死
- **改法**：改为时间 `h_missing_s`（默认 1.0）；`h_missing_frames` 保留但不参与逻辑
- **闩锁可否恢复**：见 §6 D-1 — **裁决前保持永久闩锁，只改口径**
- **判定**：✅ 真问题；恢复性等待裁决

### P1-6　PANO 把 |yaw 噪声| 当行程

- **现状**：`+= abs(step)`，注释却写有符号累加
- **改法**：有符号累加 + `pano_step_deadband_deg`（默认 0.5）；完成判据用 `abs(travel)`
- **判定**：✅ 真问题，必改（PROBE 3）

### P1-7　防打转 abort 实际不可达

- **现状**：`_pano_done_in_window` 随 turn 窗口清空而清掉；PANO 耗时 ≈ window
- **改法**：改 `_pano_done_at: Optional[float]`，按「PANO 结束后 window 内再遇障」计时
- **判定**：✅ 真问题，必改（PROBE 4）

---

## 4. 第三批 P2（质量）

| # | 问题 | 判定 |
|---|---|---|
| P2-8 | 转向到位不校验方向 | ✅ 改：符号须与 `_turn_dir` 一致 |
| P2-9 | abort 绕过 `_pack` | ✅ 改：统一经 `_pack` |
| P2-10 | `_pano_start_t <= 0` 哨兵 vs `now=0` | ✅ 改：`Optional[float]` / `None` |
| P2-11 | `__init__/begin_episode` 调 `time.time()` | ⚠️ 完整修复需动 `app.py`（原稿禁区）；内核：`seed==0` 时要求调用方传入，缺省用固定派生避免藏墙钟——见落地说明 |
| P2-12 | 进 PANO 时 `turns_total += 1` | ✅ 改：入口不计 |
| P2-13 | 首帧无深度预开 SEG | ✅ 改：有深度再开段 |

---

## 5. 测试整改清单（本会话不执行）

主人指示本次不做回归测试。下列保留备查：

| # | 说明 |
|---|---|
| T-1 | `test_corner_pano_then_abort` 用 window=30 迁就 P1-7 |
| T-2 | 高度测试固化了 P1-4 错误期望 |
| T-3 | 「全状态」未覆盖 PANO |
| T-4 | 缺 PANO→VERIFY |
| T-5 | 缺 §9.2 仿真冒烟 |

---

## 6. 待裁决项（实现方不得自行决定）

| # | 议题 | 建议 | 本会话处置 |
|---|---|---|---|
| D-1 | 高度闩锁是否可恢复 | 倾向可恢复；属改规格 | **只改计时口径，闩锁仍永久** |
| D-2 | 高度子段 mid 门槛是否放宽 | 先修 P1-4/5 再看数据 | **不动** |

---

## 7. DoD（完整模块仍以此为准）

1. P0/P1/P2 采纳项合入（测试另议）  
2. §1 五个 PROBE 基线现象消失  
3. §9.2 仿真冒烟（T-5，另议）  
4. §9.3 真机阶梯 + Claude 补「已验证基线」

---

## 8. 争议 / 复原校对记录

| # | 内容 |
|---|---|
| R-1 | 粘贴稿表格错乱已按逻辑复原；行号以 2026-07-28 仓库为准（约数） |
| R-2 | PROBE 5 原稿写「clear 后下一帧 VERIFY」——实测为 clear 回到 TURN，再一帧才 VERIFY；缺陷本质（暂停时长计入 dead-reckon）不变 |
| R-3 | P2-11 与「禁止改 app.py」张力：完整「调用方注入时间戳」需 App/FSM；见落地 commit 说明 |
| R-4 | 主人指示本次不做回归测试/代码审查，覆盖原稿 §0「每条带测试」条款 |

### 落地分工（Grok + Codex，2026-07-28）

| 角色 | 职责 |
|---|---|
| Codex | 对照规格甄别真伪；对 P0-3 taper 曲线 / P1-7 时间戳语义做对抗检查 |
| Grok | 复原本文档；实现全部 ✅ 采纳项；跑 PROBE 验证消失 |
