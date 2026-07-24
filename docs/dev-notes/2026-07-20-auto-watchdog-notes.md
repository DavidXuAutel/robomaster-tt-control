# AUTO 半自动看门狗 — 开发记录（2026-07-20）

## 背景
真机前审查发现：视觉避障半自动（AUTO）此前**无任何自动解除机制**，超时与感知失联完全依赖人工接管。首飞风险较高，补一个安全看门狗。

## 改动
- 新增 `tt_control/auto_safety.py`：`AutoWatchdog`（纯逻辑，无 I/O，便于单测）。
  - `check(now, engaged_at, last_depth_ts) -> str`：返回应解除 AUTO 的原因，空串表示继续。
  - 规则：① 单次挂载超时 `max_engaged_s=30s`；② 感知失联 `depth_stale_s=1.5s`（`last_depth_ts<=0` 时从挂载时刻起算宽限期）。
- 新增 `tests/test_auto_safety.py`：5 项单测（未挂载不触发 / 新鲜继续 / 超时 / 失联 / 从未有深度的宽限）。
- 接入 `tt_control/app.py`（4 处，最小侵入）：
  1. `import AutoWatchdog`；
  2. `__init__` 增 `self._watchdog = AutoWatchdog()`、`self._auto_engaged_at = 0.0`；
  3. `_toggle_auto` 进入 `ON` 时记录 `self._auto_engaged_at = time.time()`；
  4. `_update_rc_stream` 的 AUTO 分支下发前调用看门狗，命中则 `_auto="OFF"`、`rc(0,0,0,0)` 悬停、HUD 显示原因、`return`。

## 验证
- `python -m py_compile tt_control/app.py tt_control/auto_safety.py` 通过。
- `python -m pytest tests/test_auto_safety.py tests/test_avoidance.py -q` → 10 passed。

## 备注
- 参数 `max_engaged_s / depth_stale_s` 可按现场调；如需 CLI 化再提。
- 本次接入期间 IDE 对 `app.py`（仓库最大文件）读取偶发重复/错乱，已通过“干净段交叉验证 + 唯一锚点 StrReplace + py_compile 复核”确保改动正确。
