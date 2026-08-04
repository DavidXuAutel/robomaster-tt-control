# Wander × Codex 交叉评审记录（2026-07-27）

对照规格：`docs/design/2026-07-27-wander-explore-design.md` §2 / §3 / §7 / §9.1。

## Codex [P1] 与处置

| # | 发现 | 处置 |
|---|---|---|
| 1 | danger hold 用控制环墙钟，冻帧可空转进 retreat/abort | ✅ hold 仅在新深度帧累加；retreat 结束后需 `depth_ts > retreat_end_ts` 才 abort |
| 2 | `turn_end_ts` 用 last depth ts，转向中帧可能通过 VERIFY | ✅ 改为转向结束墙钟 `now` |
| 3 | wander abort 后降落写成 `completed` | ✅ App latch `_episode_outcome=("aborted", reason)` |
| 4 | seed/计数非 episode 作用域 | ✅ `WanderPolicy.begin_episode()`；recorder 启动时调用 |
| 5 | PANO `%360>=340` 跨圈漏判；post-pano abort 条件过严 | ✅ 累计 yaw travel + timeout；pano 后窗口内再触发即 abort |
| 6 | wander 未强制 350cm / 15% bat | ✅ App 启动 warning（不硬拒，避免单测/调参被卡死）；配置注释标明 |

## Codex [P2] 与处置

| # | 发现 | 处置 |
|---|---|---|
| 7 | PANO 字面量阈值 | ✅ 收入 `pano_complete_deg/min_s/timeout_s` |
| 8 | ctrl_state HUD 串破坏 action_source | ✅ 录制用短状态名；`_AUTO_STATES` 含 wander 状态 |
| 9 | §9.1 覆盖缺口 | 部分加强（stale depth 延迟、retreat 后新深度）；仿真/真机阶梯仍待主人验收 |

## 自检

- `.venv/bin/python -m pytest tests/test_wander.py -q` 应全绿
- 启用方式：`fsm.wander_mode=true` 且 `fsm.orbit_mode=false`；建议 `max_height_cm=350`、`min_battery_pct=15`

## 结论

控制内核 + FSM 接线 + 录制字段 + `episode_check.py` 已落地；模块 DoD（§9 真机阶梯 + 已验证基线）未完成，需主人真机验收后由 Claude 补文档。
