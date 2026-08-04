# M0 Codex 审稿记录

日期：2026-07-31  
VERDICT（初审）：**REJECT** → 已改稿吸收 CRITICAL，现可开工  

## 采纳 / 拒绝

| ID | 意见 | 处理 |
|---|---|---|
| C1 | 错误 mission abort 仍杀执行端 | **采纳**：Supervisor 校验 mission_id 匹配且有活动任务后才 abort 执行端 |
| C2 | 时间/新鲜度可绕过 | **采纳**：注入时钟 `now`；有限数值；`now<=deadline`；observed 新鲜度；过期 start 拒发 scout |
| C3 | cancel 后仍可能 arrived | **采纳**：M0 含 dog_stub/dog_sdk abort 锁存；cancel 后 FakeNav 仍 arrived → 零事件 |
| C4 | 串行缺当前区校验与计时重置 | **采纳**：`active_scout_region`；只接受当前区；换区重置 stage 计时 |
| C5 | abort 广播无故障隔离 | **采纳**：逐目标 try/except best-effort；单线程 Supervisor 为唯一入口（不做完整消息队列） |
| I1 | 勿靠 source 前缀路由 | **采纳**：按 EventType 白名单 |
| I2 | 新 start 覆盖旧任务 | **采纳**：活动任务中拒绝新 start（需先 abort） |
| I3 | SAFE_FAILED 须停执行端 | **采纳**：Supervisor 对本 mission 的 `mission.failed` 做一次受控 cancel |
| I4–I5 | 测试假绿 / 旧测试不够 | **采纳**：收紧断言；新增负例为验收硬门槛 |
| O1 | 砍双路 abort | **采纳**：仅 Supervisor 拥有取消 |
| O2 | 砍多套并行状态 | **采纳**：单 active_index / active_region |
| O3 | 不扩 EventBus | **采纳** |
| — | 完整多线程队列总线 | **拒绝（过度）**：本迭代约定单线程 tick；Supervisor 串行入口即可 |

## 实现测审（第二轮）

VERDICT（初）：**FAIL** → 已修 C1–C3

| ID | 意见 | 处理 |
|---|---|---|
| C1 | `publish()` 绕过 operator 白名单 | **采纳**：route 内再校验 source |
| C2 | 他 mission 的 `mission.failed` 停机 | **采纳**：要求 mid 匹配且状态 SAFE_FAILED |
| C3 | 空 `anchor_ids` 放行任意锚点 | **采纳**：`anchor_id not in anchors` 硬拒绝 |
| I* | 二区完整 timeout 等加强测 | 记入后续；不挡 M0 收口 |
