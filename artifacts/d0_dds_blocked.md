# D0 真机 DDS 订阅验收 — BLOCKED（现场未排期）

权威：`docs/design/2026-08-05-dog-deployment-loop-plan.md` §3-D0 验收第 3 条：

> 真实 DDS subscriber 在狗侧网络实测 `/sportmodestate` 连续 10 分钟无断流  
> （这一步需要现场，若现场未排期则先交 Loopback 全绿 + 现场清单）。

## 软件侧已交

- `DdsTransport` 已实现 `/sportmodestate` + `/lowstate` 订阅回调、`t_mono`、`sample_age_s`、`is_dds_stale(>500ms)`
- Loopback + `_ingest` 单测全绿（见 `tests/test_dog_unitree.py`）
- `runtime/dog_runtime.py` `transport=dds` 时 `health_check.ok` 计入 `dds_stale`

## 现场清单（排期后执行）

1. 狗侧网络主机安装 `unitree_sdk2py`，确认网卡名写入 `configs/dog/topsee.json` → `dds_interface`
2. `transport` 改为 `"dds"`，跑：
   ```bash
   .venv/bin/python -m runtime.dog_runtime --config configs/dog/topsee.json --once-health
   ```
   期望：`dds_stale=false`、`ok=true`
3. 连续订阅 10 分钟：
   ```bash
   .venv/bin/python -m runtime.dog_runtime --config configs/dog/topsee.json --run --health-every 5
   ```
   期间 `dds_stale` 不得持续为 true；断流应触发 `safe_hold(dds_stale)`
4. 产物：把终端日志与 `health_check` 抽样 JSON 存为 `artifacts/d0_dds_10min_<date>.jsonl`
5. 通过后删除本 BLOCKED 标记，在交接文档写明实测日期与接口名

## 状态

**BLOCKED** — 等待主人安排狗侧网络现场；不得用假数据冒充 10 分钟无断流。
