# VERIFY 灰区死循环 + 铁丝笼二飞复盘（2026-07-28）

场地：铁丝网笼，约 11–12 × 4.5–5 m。椅子 1 张 1.5 m、2 张 0.5–0.8 m。
配置：`configs/wander-cage.json`。二飞未录制。

日志时段 20:18:07–20:20:01（约 2 分钟），一飞的 `wander_danger` abort 已消失
（见 `2026-07-28-wander-side-danger-abort.md`），本轮 abort 计数 0。

---

## P0-1 铁丝网是反向信号（无法软件根治）

飞机在 20:19:57 撞上铁丝网。撞击前 6 秒 `pitch=13` 全速直行，`mid` 序列：

```
0.20 → 0.16 → 0.20 → 0.21 → 0.24 → 0.17 → 0.09
```

最后一帧 **0.09 是整段飞行最低值**——离网越近，模型越判定为"远"。

成因：DA-V2 单目深度依赖纹理/遮挡/透视线索。铁丝线径细，在 96×128 深度网格上
被抹平；透过网眼看到的是笼外远景，整片区域被判成远。越近网格越发散，
远景占比越大，`nearness` 反而越低。

危害等级高于"看不见"：看不见只是不减速；**反向信号会让策略主动把铁丝网方向
选成最开阔的去处**。

规格书 §「wander 新增坑」第 1 条已预判此形态并明确
「纯软件无法根治，靠场地布置 + turn_thresh 留反应距离 + danger 三段式兜底，
**禁止**为此给深度管线打补丁」。本次遵守，未碰深度管线。

### 处置

- 主人当前无挂布条件 → 采用**人工接管转向**（`app.py` 已有能力，AUTO ON 下
  WASD 覆盖自动决策，`ctrl_state="MANUAL"`，松手自动交回，全程不解除 AUTO）。
- `cruise_pitch` 10~14 → **5~8**。原速约 1 m/s，笼半宽 2.25 m，
  人工接管只有约 2 s 反应窗口，降速后约 5 s。这是人工接管方案成立的前提。
- 撞击后视频流中断，`depth_stale_s=3.0` 让飞机又执行了 3 s 转向指令才被
  看门狗断开 → 降到 **1.5**。

### 数据可用性（主人 2026-07-28 裁定）

小昭原判「人工接管数据不可用于训练」，前提错误，已纠正：大众训练的是
**视觉 → 操控**映射，输入为 RGB + 结构化动作，不含深度。铁丝网在 RGB 中
清晰可见（规则网格），人工转向有充分视觉依据，可学习。此类数据反而补上了
深度管线的盲区。

录制链路已验证对齐：`app.py:1063` 手动覆盖时 `_act_axes = self._last_rc`
（键盘实际杆量）、`_act_state="MANUAL"`；`_record_frame` 落盘
`act=self._act_axes, ctrl_state=self._act_state`。人工段可事后按 `MANUAL` 筛出。

行为说明：手动打杆期间提前 return，不调 `_auto_decision()`，FSM 冻结在原状态，
松手后接着走。

---

## P0-2 VERIFY 灰区死循环（本次逻辑修复）

### 现象

二飞出现 8 次 `TURN(...,retry)`，飞机原地反复打转不前进。
VERIFY 期 `mid` 实测 0.41 / 0.41 / 0.41 / 0.52 / 0.56。

### 根因

`_step_verify` 超时（`verify_timeout_s=3.0`）后**无条件** `_start_retry_turn`，
没有降级出口；而放行条件是 `mid < clear_thresh` 连续 `verify_frames` 帧。

于是 `mid` 落在 `[clear_thresh, turn_thresh]` 区间时：不够 clear 永不放行 →
超时 retry → 转完仍在区间 → 再 retry，无限循环。

本质是**判据不一致**：VERIFY 的放行标准（0.40）比 CRUISE 的巡航容忍标准
（0.45）更严。CRUISE 时 `mid=0.42` 照飞不误，VERIFY 时 `mid=0.42` 却死活不放行。

### 为什么参数无解

灰区宽度 = `turn_thresh − clear_thresh` = 迟滞宽度。二者是同一个量：

- 迟滞窄 → VERIFY 易放行，但刚进 CRUISE 又够到转向门槛 → 墙前抽搐（orbit 坑 #2）
- 迟滞宽 → 不抽搐，但灰区变宽 → 死循环概率上升

一飞后把 `turn_thresh` 0.58→0.45 时迟滞只剩 0.05；本次调到
`turn_thresh=0.40 / clear_thresh=0.30` 后迟滞回到 0.10，但灰区同步变宽。
**纯参数不可能同时消灭抽搐与死循环**，只能给超时开出口。

### 修复

`tt_control/wander.py::_step_verify`：超时后若 `mid < turn_thresh`
（前方已不构成障碍）则 `_enter(WANDER_CRUISE)` + `_begin_cruise_segment`，
`sub="verify_timeout_pass"`；否则维持原 `_start_retry_turn`。

判据统一为「不构成障碍即可走」，与 CRUISE 保持一致。

### 规格偏离声明

偏离 `docs/design/2026-07-27-wander-explore-design.md`：

- §5 状态图第 63 行「不通过（verify_timeout_s 内 mid 始终高）→ 回 WANDER_TURN 追加转角（同向）」
- §9.1 验收项 6「VERIFY 超时 → 同向追加转角」

主人 2026-07-28 裁定采纳。原语义在 `mid > turn_thresh` 时完整保留，
仅新增灰区出口。请 Claude 复核后决定是否回写规格书。

### 回归测试

- 新增 `test_verify_timeout_gray_zone_passes_to_cruise`：灰区 `mid=0.50`
  （`clear=0.40 / turn=0.58`）超时 → 进 CRUISE、不含 retry 事件、
  下一帧 `pitch > 0`（确认不是零杆卡住）。
- 修改 `test_verify_timeout_retries_same_direction`：超时帧 `_wall(0.55)` →
  `_wall(0.70)`。原值低于 `turn_thresh=0.58`，按新逻辑会走放行，
  抬高后才继续覆盖 retry 路径。原测试意图不变。

`pytest tests/ -q` → 109 passed, 1 failed
（`test_depth_inference_without_service_returns_2`，规格书 §7.8 已知预存失败，
与本次改动无关）。

---

## P1 椅子漏检

一飞、二飞累计 10+ 次转向，`TURN(...,obstacle)` **始终为 0**，全部是 free/retry。

`turn_thresh=0.45` + `turn_confirm_frames=2` 在笼内凑不齐：逐帧归一化使
`mid` 单帧可从 0.48 掉到 0.20。但 20:19:31–33 存在连续三帧 0.41/0.43/0.45，
说明门槛压到 **0.40** 可抓到。已调整，未动去抖帧数（去抖是 orbit 坑 #5 的安全设计）。

---

## P2 IP 漂移

飞机重启后 DHCP 从 `192.168.0.100` 变 `192.168.0.101`，而 `.100` 被网内其他
设备占用——**ping 通但 SDK 不响应**，误导排查方向。

流程约定：每次启动前先 `python3 station_mode.py find`，不要沿用记忆中的 IP。

---

## 本轮参数汇总（`configs/wander-cage.json`）

| 参数 | 原值 | 新值 | 依据 |
|---|---|---|---|
| `cruise_pitch_min/max` | 10 / 14 | 5 / 8 | 人工接管反应窗口 2s → 5s |
| `turn_thresh` | 0.45 | 0.40 | 撞前连续三帧 0.41/0.43/0.45 |
| `clear_thresh` | 0.40（默认） | 0.30 | 保住 0.10 迟滞 |
| `depth_stale_s` | 3.0 | 1.5 | 撞击后 3s 才断 AUTO |

---

## 待办

- [ ] 正式采集前给铁丝网挂不透明遮挡（主人当前无条件，暂以人工接管替代）
- [ ] 观察三飞是否出现 `TURN(...,obstacle)`；仍为 0 则需重新审视 `turn_thresh`
- [ ] 请 Claude 复核 P0-2 偏离，决定是否回写规格书
