# POI 环绕（Orbit）模块设计原则与参数守则

> **⚠️ 文件所有权：本文件只允许 Claude (Claude Code) 修改。**
> 其他 Agent（Cursor / Codex / 其他工具）：**只读**。你可以引用本文件的原则，
> 但不得编辑本文件，也不得在未逐条对照「安全不变量」与「历史坑」前修改
> `tt_control/avoidance.py` 的 OrbitController、`tt_control/avoidance_fsm.py`
> 的 orbit 相关路径、`configs/default.json` 的 orbit/fsm 节。
> 本文件已设为只读权限（chmod 444），这是有意为之，不要 chmod 回去。

本文沉淀 2026-07-23 ~ 07-27 五天真机调试的全部结论。每一条参数和结构
都对应一次真机炸机/失控/误停，**改之前先读「历史坑」**。

**状态：2026-07-27 真机验证通过。** 最终验证飞行：连续环绕 2min13s，
全程无任何 abort/watchdog 事件，pos 稳定 ±0.05，手动 L 降落成功。
当前 `configs/default.json` 即已验证基线，健康指标见 §6。

## 1. 模块架构

```
深度图 nearness (96x128, 值越大越近, ~5Hz)
   │
   ▼
OrbitController.decide()          ── tt_control/avoidance.py
   ├─ _lock_target: 获取(严格峰均比) → ROI 跟踪(宽松) 滞回
   ├─ 信号调理: 位置 EMA + 距离 tn EMA + 单帧跳变拒绝(max_pos_jump)
   ├─ roll: 随 |pos| 线性 taper（比例混合，非开关）
   ├─ yaw = 环绕前馈(-roll×ff_ratio) + 居中 P(带死区)
   └─ pitch: (target_nearness − tn) P 控制；偏离中央时禁止前冲
   │
   ▼
AvoidanceFSM._finish_orbit_decision()  ── tt_control/avoidance_fsm.py
   ├─ DANGER 去抖: 持续 ≥ orbit_danger_hold_s 才 abort，期间零杆
   ├─ LOST episode: 闪断不复位计时，超时 abort；重获需连续 N 帧
   └─ 全局: 深度过期 / 低电量 / AUTO 全局超时(max_auto_engaged_s)
```

控制频率注意：**控制环 ~24Hz，深度只有 ~5Hz**。所有滤波时间常数、
连续帧计数都要按这个比例理解——「连续 3 个控制帧」≈ 只有 0.6 个深度帧，
不构成独立证据。

## 2. 安全不变量（任何修改不得破坏）

1. **DANGER / LOST / ACQUIRE 状态下杆量必须全零**（OrbitController 层保证）。
   任何去抖/宽限逻辑只允许「推迟 abort」，绝不允许「危险期间继续给杆」。
2. **椅子明显偏离中央（phase=center）时禁止正 pitch 前冲**，负 pitch 后退保命
   永远允许。偏离时距离读数不可靠，前冲=向障碍物冲。
3. **`target_nearness ≤ danger_thresh − 0.06`**：目标距离必须给危险阈值留余量，
   否则正常环绕会周期性触发 DANGER。
4. **检测不可靠时不得盲飞**：chair_pos 为 None 时 pitch 必须为 0
   （见 test_search_no_blind_pitch）。
5. **每次真机暴露的 bug，修复必须带回归测试**进 `tests/test_orbit.py`。
   改动后 `tests/test_orbit.py` + `tests/test_avoidance.py` 必须全绿。
6. **不要在主线程给 Tello 发任何带回执的命令来「辅助」异步 land**
   （见坑 #7，UDP 回执竞态）。

## 3. 历史坑（按时间序，改代码前逐条自查）

| # | 现象（真机） | 根因 | 修复 | 教训 |
|---|---|---|---|---|
| 1 | engage 后立即横移，不进环绕 | ACQUIRE（收集确认帧）被当成检测失败，掉进 AVOID_TURN | FSM 三级处理：ACQUIRE→缓进；LOST+不够近→缓进；LOST+够近→AVOID_TURN | 获取中 ≠ 失锁 |
| 2 | 环绕左右猛摆，椅子摆出视野 | roll/yaw 二元开关（bang-bang）：横移推开→满yaw拉回→再推开 | roll 随 \|pos\| 线性 taper，yaw 始终参与 | 视觉伺服禁止开关式控制律 |
| 3 | 靠近椅子后悬停不环绕 | orbit_enter_nearness=0.40 高于椅子近度平台(0.35~0.39)，永远不触发 | 降到 0.35（=clear_thresh） | 进入阈值必须 ≤ 真机实测平台值 |
| 4 | 靠近后来回摇头（yaw 振荡） | gain=200/max_yaw=50 过猛 + 检测重心逐帧抖动无滤波 + 无死区，叠加 ~0.5s 延迟 | gain→80、max_yaw→25、位置 EMA、yaw 死区 0.05 | 高增益 P + 延迟 = 必振荡 |
| 5 | 绕到侧面突然悬停（AUTO 被踢） | 检测单帧跳变(±0.3) → tn 尖刺 0.89 → 单帧超 danger_thresh → 立即 abort | max_pos_jump 0.45→0.30、tn 加 EMA、DANGER 去抖 0.4s | 单帧深度证据不足以 abort；先零杆再确认 |
| 6 | 半径偏大，永远到不了目标距离 | pitch_distance_gain=2.5 太弱（误差 0.14 只给 pitch 7） | gain→4.0 | 看日志 tn 是否真收敛到 target，别只调 target |
| 7 | 键盘 L 无法降落（界面按钮可以） | 主线程先发 rc(0,0,0,0)，其 ok 回执与异步 land 的回执在 UDP 缓冲区竞态 | 删除主线程 rc 调用，land 走独立线程+0.25s 延迟+3 次重试 | 一问一答的 UDP 协议里不要并发发命令 |
| 8 | EMA 加了但没用 | α=0.4 在 24Hz 控制环下时间常数仅 ~80ms，滤不掉 5Hz 深度帧级跳变 | α→0.25，且跳变拒绝才是主力 | 滤波参数按「深度帧率」算，不是控制帧率 |
| 9 | 改了 fsm.max_auto_engaged_s 仍在 120s 被踢 | app.py 里另有一个 app 级 AutoWatchdog，max_engaged_s=120 硬编码，与 FSM 超时是两套 | app 级看门狗改为复用 FsmParams（max_auto_engaged_s / depth_stale_s） | 超时/安全阈值全项目只允许一个来源（configs/default.json）；新增看门狗必须接配置，禁止硬编码 |

## 4. 参数表（configs/default.json）

### orbit 节

| 参数 | 当前值 | 安全范围 | 说明 / 耦合关系 |
|---|---|---|---|
| direction | 1 | ±1 | 环绕方向。前馈符号自动跟随，无需另调 |
| target_nearness | 0.69 | ≤ danger_thresh−0.06 | 环绕半径（越大越近）。想缩半径先确认日志里 tn 已收敛到当前目标 |
| distance_deadband | 0.05 | 0.03~0.08 | 距离死区 |
| orbit_roll | 10 | 5~14 | 环绕线速度。调大时 yaw 前馈自动等比放大；>14 未验证 |
| yaw_centering_gain | 80 | 60~120 | 居中 P 增益。**调回 200 必复现摇头（坑#4）** |
| centering_deadband | 0.25 | 0.20~0.35 | roll taper 半宽。**调回 0.12 必复现 roll 开关振荡（坑#2/#5）** |
| pos_smooth_alpha | 0.25 | 0.15~0.4 | 位置/tn EMA 新帧权重。越小越稳但延迟越大 |
| yaw_deadband | 0.05 | 0.03~0.08 | P 修正死区（前馈不受影响） |
| yaw_ff_ratio | 0.6 | 0.4~1.0 | 前馈 = −roll×此值。横移必然使目标反向漂移，前馈让 yaw 不必等误差 |
| pitch_distance_gain | 4.0 | 2.5~5.0 | 距离 P。太弱→半径收不进（坑#6）；太强→前后冲 |
| max_yaw | 25 | 20~30 | **50 必复现甩头（坑#4）** |
| max_pos_jump | 0.30 | 0.25~0.35 | 单帧位置跳变拒绝。合法帧间移动 ~0.1；**0.45 放过尖刺（坑#5）** |
| danger_thresh | 0.78 | 0.75~0.82 | 环绕危险阈值，与 target_nearness 联动（不变量#3） |
| acquire_frames | 2 | 2~3 | 获取确认帧数。=1 会把噪声当目标 |

### fsm 节（orbit 相关）

| 参数 | 当前值 | 说明 |
|---|---|---|
| orbit_enter_nearness | 0.35 | 进入环绕的中区近度阈值。**必须 ≤ 真机椅子近度平台（坑#3）**；LOST 回退阈值 = 此值+0.15（写死在 FSM） |
| orbit_lost_timeout_s | 5.0 | 失锁 episode 超时。闪断不复位计时 |
| orbit_relock_frames | 5 | 重获需连续帧数 |
| orbit_danger_hold_s | 0.4 | DANGER 去抖。期间零杆悬停，安全等价；**设 0 = 退回坑#5** |
| max_auto_engaged_s | 86400 | **AUTO 全局超时看门狗**：AUTO 挂载起计时，到时无条件收杆悬停并解除 AUTO。2026-07-27 应主人要求实际取消（设 24h）。注意 app.py 的 app 级 AutoWatchdog 也读此值（坑#9），改这里即全局生效 |

## 5. 修改守则（后续 Agent 必读）

1. **先看日志再改参数**。`AUTO dbg` 行包含完整状态：
   `fsm=… sub='ORBIT pos±… phase L… M… R… tn… yaw… pit… rc(…)'`。
   判断依据：pos 波动幅度（>±0.2 = 检测/滤波问题）、tn 是否收敛到
   target（没收敛 = pitch 增益问题，不是 target 问题）、abort 原因
   （watchdog disengage 行）。凭现象猜参数 = 本文档所有坑的共同起点。
2. **一次只改一个参数**，真机验证后再改下一个。多参数同调无法归因。
3. 振荡类问题的排查顺序：**检测跳变(max_pos_jump/日志pos) → 滤波(α) →
   增益(gain/max) → 结构(taper/前馈)**。不要上来就调增益。
4. 控制律结构性修改（改公式而非参数）必须同时更新本文件（仅 Claude）
   和 `tests/test_orbit.py`，并在真机验证前跑全量测试。
5. 测试命令：`.venv/bin/python -m pytest tests/test_orbit.py tests/test_avoidance.py -q`
   （注意用 venv Python 3.11，系统 python3 是 3.9 跑不了）。
   已知无关预存失败：`test_main_depth_guard.py::test_depth_inference_without_service_returns_2`。
6. 真机启动/清理规则见 `.cursor/rules/tt-flight-test.mdc` 与 CLAUDE.md。
7. **降落路径不要动**：land 首次超时 + 重试成功是 Tello UDP 的正常现象，
   不是 bug，不要「优化」重试逻辑；键盘 L 与界面 Land 按钮必须保持
   同一代码路径（坑#7 的教训，两条路径曾产生回执竞态）。

## 6. 已验证基线（2026-07-27 真机，回归对照用）

改动后先真机短飞一次，把 `AUTO dbg` 日志与下面的健康特征对照，
任何一条明显劣化即视为回归，回滚改动：

| 指标 | 健康值 | 劣化含义 |
|---|---|---|
| pos 波动 | 常态 ±0.05，偶发 ≤ ±0.2 | 超出 → 检测跳变/滤波退化（先查坑#5/#8） |
| yaw | 常态个位数，前馈基线约 ∓6 | 频繁 ±15 以上 → 增益/滤波问题（坑#4） |
| tn | 0.5~0.75 随视角自然起伏，向 target 收敛 | 停在 0.5 以下不收敛 → pitch 增益（坑#6） |
| phase | 长期 orbit，center 只短暂出现 | 长期 center → taper/前馈失效（坑#2） |
| roll / pitch | roll 4~10（taper 生效）；pitch 常态 0 或小值 | pitch 频繁 ±10 → tn 尖刺（坑#5） |
| abort 事件 | 无 orbit_danger / orbit_lost / engaged>Ns | 出现即查对应坑（#5 / 跟踪 / #9） |
| 时长 | 可连续环绕 ≥2min（已验证） | 中途停 → 看 watchdog disengage 行找原因 |
