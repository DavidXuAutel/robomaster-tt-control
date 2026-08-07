# D0 DDS 验收 — L1 已过 / L2 BLOCKED（阻塞性质：集成商政策，非现场排期）

权威：`docs/design/2026-08-05-dog-deployment-loop-plan.md`（v3.4）§3-D0「验收（分三级）」。

更新：2026-08-07。**较上一版有实质变化，不要按旧结论行事。**

## 阻塞性质变更（最重要）

上一版记「BLOCKED — 等待主人安排狗侧网络现场」。**这个判断已过时。**

集成商书面答复（方案 §1.5-V1）：**不建议**第三方直连宇树 DDS，并发下发会导致运动异常；
改为提供其「状态转发」与「运控接口」。

所以 L2 的阻塞是**政策性**的，不是排期性的——**排到现场也不解决**。
路径只有两条：①取得集成商书面批准直连 DDS 只读；②改用其状态转发接口承载 E4-S。

## L0 软件 — 通过

见 `artifacts/d0_software_gate.md`。

## L1 真实 DDS · 合成源 — 通过（2026-08-07）

工具 `tools/dds_selfcheck.py`。**真实 CycloneDDS + 真实 `unitree_go` IDL**，
不再是 `LoopbackTransport` 桩。

| 运行 | 发布端 | 结果 | 产物 |
|---|---|---|---|
| 闭环自检 | 本工具自带（会填 `stamp`） | 500/500 帧；字段零缺失；设备钟单调；断流可检；只读模式下命令被拒 | `d0_dds_selfcheck_20260807.json` |
| observe | **宇树官方 MuJoCo B2 桥**（第三方发布端） | sportmodestate 1771 帧 / 147.6Hz、lowstate 1802 帧 / 150.2Hz；间隔 p99 ≈8ms、max ≈21ms；字段零缺失 | `d0_dds_mujoco_observe_20260807.json` |

第二行的意义：发布端是**宇树自己写的**，排掉了「我方自编自测、两端一起理解错字段」的盲区。

### L1 明确不证明什么

- **真机时序**：真狗的发布抖动、丢包、负载下退化，一概未测；
- **命令通道**：宇树 MuJoCo 桥只订阅 `rt/lowcmd`（底层关节），**不提供 SportClient 服务**，
  所以 `Move`/`StopMove` 在仿真里也验不了。E4-C 无法靠仿真解锁；
- **现场网络**：改装机的网段隔离与广播风暴约束（§1.5-V2）。

### 已知的假绿陷阱（工具已修，读产物时仍需注意）

宇树 MuJoCo 桥**不给 `stamp`/`tick` 赋值**，因此 observe 模式下 1771 个 `t_device` 全为 0.0，
`device_stamp_monotonic` 会**空洞成立**。必须同时看 `device_stamp_advancing`
（该次运行为 `false`，产物 `notes` 已如实标注）。设备钟提取的正确性由闭环自检那次覆盖。

## L2 真机 — BLOCKED

前置：集成商就以下任一给出书面答复（方案附录 B 第 5 项）——
①允许直连 DDS 只读；或②提供状态转发接口的字段/频率/时钟契约。

到位后执行：

```bash
# 载体为 DDS 时
.venv/bin/python tools/dds_selfcheck.py --observe --seconds 600 --poll-hz 300 \
  --domain-id 0 --family b2 --interface <狗侧网卡> \
  --out artifacts/d0_dds_10min_$(date +%Y%m%d).json
```

达标线：两个 topic 均 `frames > 0` 且持续 600s；`missing_fields` 为空；
`device_stamp_advancing = true`；间隔 max 不得出现 >500ms 的空洞（会触发 `dds_stale`）。

## 复现环境备注

- 本机（macOS/Apple Silicon）可跑 CycloneDDS 收发，但需两端用同一网卡配置。
  `ChannelFactoryInitialize` **不读 `CYCLONEDDS_URI`**，它自行拼配置传给 `Domain()`，
  两个进程若一个指定网卡、一个自动探测，会互相发现不到。
- `unitree_sdk2py.utils.thread` 依赖 Linux 专有的 `timerfd_create`，macOS 上不可用。
  我方 `DdsTransport` 不依赖该模块；仅官方示例与仿真桥受影响。
