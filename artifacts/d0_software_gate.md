# D0 软件门禁报告（2026-08-06）

权威方案：`docs/design/2026-08-05-dog-deployment-loop-plan.md` v3.1

## 交付对照

| # | 项 | 状态 |
|---|---|---|
| 1 | 真实 DDS subscriber + t_mono + dds_stale | 代码已交；10min 真机见 `d0_dds_blocked.md` |
| 2 | `runtime/dog_runtime.py` 装配/健康/日志轮转/磁盘水位 | 已交；含 `--run` 主循环 |
| 3 | abort → `Arbiter.force_release` | 已交（可选 arbiter/on_abort） |
| 4 | ack_confidence [0,1] + audit jsonl | 已交（先落盘再改内存） |
| 5 | 电量 fail-closed `battery_unknown` | 已交 |
| 6 | 探针头/gap 对齐 §2 | 已交（E7→G12，E9→alarm-schema，E10→battery-field） |
| 7 | `configs/dog/topsee.json` | 已交 |

## 测试

```text
tests/test_dog_runtime.py + arbiter/unitree/sdk/integration/topsee_tools
→ 相关套件全绿
全量：357 passed；1 failed = 预存 test_main_depth_guard（无关）
```

## Codex 评审

- R1 `artifacts/d0_codex_review.txt` → **FAIL**（4×P1）
- R2 `artifacts/d0_codex_review_r2.txt` → **PASS**（0×P1）
- 驳回：[P1]「D0 禁止 SportClient 命令槽」——v3.1 D0 未作此禁；Move 安全盒与租约闩属既有 M10，Gate 唯一出口是 D5。
- 已修：gap 标注、现场 BLOCKED 清单、runtime 主循环、`health_check.ok` 在 dds 模式计入 stale、confidence 先审计后提交、`on_abort` 不得旁路 `force_release`、磁盘检查 fail-closed。
