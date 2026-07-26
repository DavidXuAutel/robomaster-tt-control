# POI 绕飞飞前缺陷审查与修复方案

日期：2026-07-26  
状态：✅ 飞前最小集已实施（2026-07-26；单测 `tests/test_orbit.py` 通过）  
范围：`OrbitController` + `AvoidanceFSM` 环绕路径 + `app.py` AUTO 挂载

## 1. 背景

下次真机复现绕飞前，对绕飞栈做缺陷审查（本地复现 + Codex 对抗/短跑会诊）。  
目标：尽量避免复现时再踩安全/状态机类 bug。

相关代码：
- `tt_control/avoidance.py` — `OrbitController` / `OrbitParams`
- `tt_control/avoidance_fsm.py` — `_run_orbit` / `_step_orbit` / `FsmParams.orbit_*`
- `tt_control/app.py` — `orbit_mode=True`、AUTO 切换、`abort_reason` 处理

交接文档（行为声称）：`docs/handover/2026-07-25-orbit-avoidance-handover.md`

## 2. 缺陷清单（按飞行风险）

### P0 — 飞前必修

#### P0-1 失锁宽限期内仍前进

- **现象**：目标丢失（`LOST` / `search`）后，仍输出前进 `pitch≈+25~+27`，持续至 `orbit_lost_timeout_s`（默认 5s）才悬停。
- **位置**：`avoidance.py`（`OrbitController.decide` 距离项与状态解耦）；`avoidance_fsm.py` `_run_orbit` 宽限期内原样下发 `orbit_dec.axes`。
- **触发**：绕飞中遮挡 POI / 目标滑出视场 / 深度骤降。
- **风险**：失锁后仍前冲，撞障或飞向未知区域。
- **复现**：合成 nearness 全场 `0.05` → `state=LOST`，`axes.pitch≈25`。

#### P0-2 DANGER / orbit_lost 不写 abort_reason

- **现象**：FSM 进入 `HOVER` 且杆量清零，但 `abort_reason=''`。`app.py` 仅在 `abort_reason` 非空时调用 `_disengage_auto`。
- **结果**：AUTO 仍为 `ON`，飞机永久悬停，直到全局 AUTO 超时；HUD 无明确解除原因。
- **位置**：`avoidance_fsm.py` `_run_orbit`（`orbit_danger` / `orbit_lost`）；`app.py` ~730。
- **风险**：安全停住了，但挂载态错误 → 复现体验差，且与「保护应解除 AUTO」语义不一致。

### P1 — 复现易卡 / 误绕

#### P1-1 SPACE / V 暂停不复位 FSM

- **现象**：`_hover` / `_toggle_auto→ARMED` 只改 `_auto`，不调用 `_fsm.reset()`；暂停期间几乎不调用 `fsm.step(..., auto_on=False)`。
- **结果**：再挂 AUTO 可能续上 `_orbit_active`，直接 strafe，而非干净 APPROACH 重获 POI。
- **位置**：`app.py` hover/toggle/auto_decision；`avoidance_fsm.py` auto_on=False 分支（实际很少走到）。

#### P1-2 均匀近度场假 POI

- **现象**：`_chair_horizontal` 在权重均匀时重心居中 → `chair_pos=0.0` → `strafe`。
- **风险**：空场/墙面均匀近度时「空绕」。

#### P1-3 环绕无侧向刹停（Codex）

- **现象**：`DANGER` 仅看中区 mid；横移时侧障可能不触发保护。
- **建议**：orbit 路径增加侧向 nearness 刹停（`max(L,R)` 或侧区阈值）。

#### P1-4 冻结图传可伪装新鲜深度（Codex）

- **现象**：旧帧反复推理会刷新时间戳，深度看门狗可能不触发。
- **建议**：核对 `AsyncInferWorker` / depth 时间戳语义（帧捕获时 vs 推理完成时）。

### P2 — 技术债

- `AvoidState.ORBIT` / `_step_orbit` 为死代码；真路径是 `APPROACH` + `_run_orbit`（`_orbit_active` 滞回）。
- Orbit / FSM 环绕路径几乎无单测（现有 `tests/test_avoidance.py` 只覆盖 cruise 避障）。

## 3. 修复方案（建议实施顺序）

### 第一步（飞前最小集）

1. **LOST / search 强制零杆**  
   - `OrbitController.decide`：`chair_pos is None` 或 `state in {LOST}` 时 `pitch=roll=yaw=0`。  
   - FSM 宽限期内亦不得前进（可直接用控制器已清零的 axes）。

2. **orbit_danger / orbit_lost 填 abort_reason**  
   - 例：`abort_reason="orbit_danger"` / `"orbit_lost"`，走现有 `_disengage_auto` → AUTO 回 ARMED。

3. **暂停路径复位**  
   - SPACE / V→ARMED / `_disengage_auto` 调用 `_fsm.reset()`（或保证 `step(auto_on=False)` 被调用）。

### 第二步（飞后可收）

4. 假 POI：加最小权重峰度 / 总权重阈值，均匀场返回 `None`。  
5. 侧向危险刹停。  
6. 深度时间戳与冻结图传。  
7. 清理或真正启用 `AvoidState.ORBIT`；补 Orbit/FSM 单测。

## 4. 飞前必测 Checklist（≤5）

1. 绕飞中遮挡目标 → HUD=`LOST` 后 **pitch 必须为 0**，并退出 AUTO（或明确悬停保护）。  
2. 贴太近 `DANGER` → **AUTO 回 ARMED**（非 ON+悬停）。  
3. 绕飞中 **V 暂停再挂** → 从干净 APPROACH 起步，不得瞬切 strafe。  
4. 绕飞中 **SPACE** 再 V→ON → 同上，FSM 已 reset。  
5. 开阔均匀背景 → 不得长时间假 strafe（`roll≠0`）。

## 5. 验证方式

- 合成 nearness 单测：LOST 零杆、DANGER abort、暂停后再 ON 不续 `_orbit_active`。  
- 真机：按 §4 Checklist 走一遍，SPACE 随时可接管。

## 6. 审查来源

| 来源 | 结论 |
|------|------|
| 本地复现脚本 | 钉死 P0-1、P0-2、P1-1、P1-2、死代码 ORBIT |
| Codex adversarial（超时） | 同意 LOST 仍前进；指出 AUTO/FSM 不 reset |
| Codex 短跑 consult | 5 条候选全 AGREE；严重度略偏软（LOST 标 P1） |
| 综合定级 | LOST 前进、abort 缺口按 **P0**；暂停复位按 **P1** |

## 7. 决策与落地

- ~~下次复现前先修 §3 第一步，再飞。~~ → **已落地**（含 Claude 修正：侧向 DANGER 升 P0、`chair_pos is None` 钳 pitch、LOST/DANGER 零杆、abort_reason、SPACE/V `fsm.reset()`）。
- 假横移峰度阈值 / 冻结图传时间戳：仍可飞后迭代。

### 7.1 代码改动摘要

| 文件 | 改动 |
|------|------|
| `tt_control/avoidance.py` | LOST/DANGER 零杆；无目标即 LOST；均匀场拒假 POI；距离用椅子列近度；`max(L,M,R)` 侧向 DANGER |
| `tt_control/avoidance_fsm.py` | abort_reason；`None` 哨兵；进环绕前侧向危险 `approach_hold` 不前冲 |
| `tt_control/app.py` | `_hover` / V OFF→ARMED / ON→ARMED / ARMED→ON 均 `_fsm.reset()` |
| `tests/test_orbit.py` | 安全路径单测 + `now=0` 回归 |

## 8. Claude Code 审核意见（2026-07-26）

**总评**：五个缺陷定位全部准确，修复方向正确；P0-1 修复描述边界过宽，P1-3（侧向刹停）严重度偏软。

**全部确认存在**：P0-1、P0-2、P1-1、P1-2、P1-4、死代码 ORBIT。

**异议 / 修正**：
1. P0-1 勿用裸 `chair_pos is None` 零全部杆量（单帧遮挡会抖）；应按 `state==LOST` 零杆，**另**在 `chair_pos is None` 时单独钳 `pitch=0`（堵「search 盲飞前进」窗口）。
2. P1-3 侧向刹停应升 **P0**，纳入飞前第一步（orbit 横移时侧向才是主碰撞面）。

**建议补充**：
- Checklist 加：环绕横移中侧方放障 → 应 DANGER 刹停。
- `_fsm.reset()` 明确三处：`_hover`、V ON→ARMED、V OFF→ARMED。

**批准实施**：是，条件为采纳上述 1–2 条修正后再实施 §3 第一步。
