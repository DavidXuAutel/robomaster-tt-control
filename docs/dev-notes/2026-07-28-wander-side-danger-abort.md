# Wander 首飞 abort 复盘：侧区触发 danger 导致必然 abort

- 日期：2026-07-28
- 提出方：Cursor（非 Claude Agent，按 `.cursor/rules/wander-module-guard.mdc` §1 只提异议、不擅自改规格）
- 证据：`logs/wander-cage-p1.log`（铁丝笼首飞，configs/wander-cage.json）
- 待裁决：规格 `docs/design/2026-07-27-wander-explore-design.md` §2.1 的 `max(zones) > danger_thresh`

## 1. 现象

主人反馈「看到椅子就卡住了，没法继续随机往前飞」。实际不是策略卡死，
是 **AUTO 被 abort 解除**，飞机悬停在原地：

```
20:02:23,281  WANDER_CRUISE | p=11 M0.28      rc=(0, 11, 0, 0)
20:02:23,847  rc=(0, 0, 0, 0)                 ← DANGER_HOLD 零杆
20:02:24,250  rc=(0, -15, 0, 0)               ← WANDER_RETREAT
20:02:25,218  AUTO watchdog disengage: wander_danger
```

## 2. 关键事实

1. **全程 mid 从未超过 turn_thresh(0.58)**：日志中 M 最大 0.46。
 8 次 WANDER_TURN 全部是 `free` turn，**遇障转向一次都没触发过**。
2. **abort 前一帧 M=0.28（前方开阔）**，0.6s 后即进 DANGER_HOLD。
 `danger = max(left, mid, right)`，mid 只有 0.28 → 顶穿 0.78 的只能是
 **左区或右区**（铁丝网侧墙 / 地面 / 侧后方的椅子）。
3. abort 判据在 `_step_retreat`：后退 `retreat_s=0.7s` 后拿到新深度帧，
 danger 仍 > 0.78 → `abort_reason="wander_danger"`。

## 3. 根因（结构性，不是调参能解）

漫游的安全不变量是 **roll ≡ 0、pitch ≥ 0（RETREAT 除外）**，
即**唯一的保命动作是沿机头方向直退**。

直退**不改变侧向近度**。所以只要侧区把 danger 顶过阈值：

```
DANGER_HOLD(0.4s) → RETREAT(0.7s) → 侧区仍高 → abort
```

是**必然路径**，与场地大小无关，与飞行技巧无关。窄笼（4.5~5m 短边）
只是让它必然发生得更早。

这条与 `tt_control/avoidance.py:86-88` 已固化的真机教训直接冲突：

```python
# 遇障急停兜底:正前方(中区)非常近 = 要正撞 → 直接悬停,不往里冲(优先级最高)
# (侧向很近应转开而非急停,故只看 mid)
if mid > p.estop_thresh:
```

即 2026-07-24 真机已经付过一次学费：**急停判据只看 mid，侧向近度应当转开**。
wander 规格 §2.1 用 `max(zones)` 做急停+abort，把这条教训丢了。

## 4. 裁决与落地（主人 2026-07-28 拍板）

| 项 | 裁决 | 落地 |
|---|---|---|
| R-1 danger 只看 mid | ✅ 采纳 | `tt_control/wander.py` `decide()` `danger = mid` |
| R-2 打印 L/M/R | ✅ 采纳 | `tt_control/app.py` AUTO dbg 增加 `zones=L../M../R..` |
| R-3 侧区超阈转向 | ⏸ 暂缓 | 等 R-2 实测 L/R 分布 |
| §5 turn_thresh | ✅ 下调 0.58 → 0.45 | `configs/wander-cage.json`（仅本场地档，不动 default.json） |

回归测试：`tests/test_wander.py::test_side_zone_alone_does_not_trigger_danger`
（侧区 0.95/0.92 + mid 0.28 → 全程 CRUISE 不 abort）。
`pytest tests/ -q`：107 passed，仅 §7.8 已登记的预存失败。

### R-1（已采纳）danger 判据只看 mid

```python
danger = mid          # 原：danger = max(zones)
```

- 三段式 hold → retreat → abort 保留不变，仅收窄触发源到「正前方要撞」。
- 与 avoidance.py 的 estop 语义对齐，无新参数，无新阈值字面量。
- 代价：侧向贴近不再有软件兜底。判断依据：漫游只向前平移，
 90° 侧向障碍不构成正撞；擦碰风险由场地 + 主人目视兜底（首飞本就在监护下）。

### R-2（同批做，纯观测无风险）AUTO dbg 打印 L/M/R + danger

当前 `AUTO dbg` 只打印 M，导致本次必须靠反证法推断是侧区触发。
补上 `L%.2f M%.2f R%.2f`，下次直接读数。

### R-3（暂不做，等 R-2 数据）侧区超阈 → 转向而非急停

结构上更自洽（侧向危险 = 转开，前向危险 = 保命三段式），但要新增
`side_turn_thresh` 参数，且窄笼里可能频繁转向撞上 `corner_max_turns=4/20s`
→ PANO → `wander_cornered`。先拿 R-2 的真实 L/R 分布再决定。

## 5. 遗留问题（本票不解，登记）

**遇障转向在本场地从未触发**（mid 峰值 0.46 < turn_thresh 0.58）。
可能原因：1.5m 椅子在 ~80cm 飞行高度下只占中区一小块，中位数拉不高；
或单目相对深度在笼内被侧墙拉平（规格 §7 坑 #1）。
处置：先按 R-2 收数，地面标定（正对椅子读 M）后再议是否下调 turn_thresh
或调整 `band_top/bottom`。注意 `WanderPolicy` 自建 `AvoidanceController()`
用的是 dataclass 默认值，不读 `configs.avoid` 节——若要调 band 需先修这条
（阈值单一来源，规格 §7 坑 #9）。
