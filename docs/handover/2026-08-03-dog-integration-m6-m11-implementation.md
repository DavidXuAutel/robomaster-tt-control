# 机器狗对接 M6–M11 实现交接（2026-08-03）

权威设计：`docs/design/2026-08-03-dog-integration-plan.md`（本文只讲「代码落成什么样、怎么自测、还卡在哪」）。

本轮把方案里**不依赖真机就能做**的部分全部落地：慢通道（拓普视平台 HTTP）、
快通道骨架（宇树 DDS）、两者的互斥仲裁、平台点位绑定、气体标定台账、只读探针。
真机相关的 E1/E3/E4 三个实验按方案要求**留空并显式报错**，没有用假数据糊过去。

## 一、落地清单

| 里程碑 | 文件 | 说明 |
|---|---|---|
| M7a | `adapters/topsee_rsa.py` | 纯标准库 RSA PKCS#1 v1.5 公钥加密（登录用） |
| M7b | `adapters/topsee_client.py` | 平台 HTTP 会话、`code:0` 语义、401 自动重登、`PollCache` 后台轮询 |
| M7c | `adapters/dog_topsee.py` | `TopseeNav`（到点三态）/ `TopseePerception` / `TopseeGas` |
| M9 | `adapters/gas_ledger.py`＋`configs/mission/gas_calibration.example.json` | 气体标定人工台账 |
| M10 | `adapters/dog_unitree.py` | DDS 运动通道 + 速度安全盒 + 无硬件 `LoopbackTransport` |
| M11 | `adapters/dog_arbiter.py` | 8 状态仲裁机、租约、preflight 门禁、命令看门狗 |
| M8 | `mission_brain/map_model.py`＋`tools/export_dog_bindings.py` | SharedMap v2 `platform_binding` 与导出/漂移检查 |
| M6 | `tools/topsee_probe.py` | 只读探针 + E0/E2/E5–E10 实验 |
| 契约 | `adapters/dog_sdk.py` | 两个可选钩子 + 空 readings 处理（见下） |

### 三个新增依赖为零的技术选择

1. **RSA 自己实现**。`requirements.txt` 里没有加密库，也不为一个登录接口引
   `cryptography`。公钥加密不涉及私钥运算，无需恒定时间实现。正确性由假平台
   用配套私钥真解一遍来验证（`tests/test_topsee_rsa.py`）。
2. **HTTP 用 `urllib`**。`requests` 没进 `requirements.txt`。
3. **标定台账用 JSON 而非方案写的 YAML**。没装 PyYAML，字段语义与方案一致。

### `dog_sdk.py` 的三处契约扩展（全部向后兼容）

三个 backend 协议签名一个字没改，`FakeNav` / `FakeGas` 行为完全不变。
新增的是两个用 `getattr` 探测的**可选**钩子，加一处空数据处理：

| 扩展 | 作用 | 不做会怎样 |
|---|---|---|
| `nav.poll_fault()` | 平台状态对不上白名单 / 查不到任务 / 导航超时 → 报明确 reason | 静默返回 False，被 `stage_timeout_s` 掩盖成「狗走得慢」，排障时完全看不到真因 |
| `gas.calibration_reason()` | 区分「无标定数据源」与「台账过期」 | 两种完全不同的问题共用一个 reason |
| 空 `readings` → `GAS_FAILED(no_gas_data)` | 平台只能按窗口回查历史，窗口内无数据是正常结果 | 空列表会在 `validate_event` 里抛异常，炸掉整条 mission |

## 二、自测结果

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
# 352 passed, 3 skipped
```

新增 129 个用例，其中值得单独点出来的：

- `tests/test_dog_arbiter.py` — 方案 §4.5 的 I1–I10 安全不变量。**任何一条红都意味着
  真机上可能双源控制（直接摔狗），不许 xfail、不许放宽阈值。**
- `tests/test_dog_integration.py` — 端到端装配（SharedMap v2 → Arbiter → TopseeNav/Gas →
  DogSdkAdapter → 事件契约校验）。单测各自绿不代表拼起来能跑。
- `tests/fixtures/topsee_fake.py` — 可编排假平台，刻意复刻已核验的平台怪癖：
  `code:0` 而非 HTTP 200、401 返回**纯文本**「未登录」、`sendNavigate` 不回任务标识。
  不复刻这些，测试只会给出虚假的安全感。

### 顺手修掉的一个既有 bug

`tt_control/flight_config.py:45` 用 `open(path)` 没指定编码。JSON 按 RFC 8259 恒为
UTF-8，不声明会在非 UTF-8 locale 的机器上（中文 Windows 的 cp936）读带中文注释的
`configs/default.json` 直接抛 `UnicodeDecodeError`。macOS 默认 UTF-8 所以一直没暴露。

## 三、真机联调前必须做的事

### 1. 跑只读探针，回填 6 个配置空位

```powershell
$env:TOPSEE_PASSWORD="<密码>"
.venv\Scripts\python.exe tools\topsee_probe.py `
    --base-url http://192.168.0.85:8001 `
    --account <账号> --robot-id B2000397 `
    --out artifacts\topsee_probe.json
```

探针**默认严格只读**，一条动作命令都不发。E1（会让狗真的走过去）必须显式加
`--allow-motion`，脚本还会要求在命令行再确认一次现场安全。

| 探针实验 | 回填到 |
|---|---|
| E2 状态枚举 | `TopseeNav(arrived_states=..., enroute_states=...)` |
| E10 状态字段 | `DogControlArbiter(battery_provider=...)` 的电量字段名 |
| E6 时间格式 | `dog_topsee._fmt_time()`（不符只改这一处） |
| E5 地图结构 | 确认 `tools/export_dog_bindings.py` 的递归取点判据够用 |
| E8 会话有效期 | `TopseeClient(token_header=...)` 与重登策略 |
| E9 告警结构 | `TopseePerception(mode="alarm_uri")` 的字段名 |

### 2. 导出点位绑定

```powershell
.venv\Scripts\python.exe tools\export_dog_bindings.py `
    --base-url http://192.168.0.85:8001 --account <账号> `
    --robot-id B2000397 --map configs\mission\shared_map.example.json --write
```

飞行前自检用 `--check`：绑定漂移就非零退出。**这一步不是可选的**——平台
pointsId 形如 `快速打点-1785465994716`，重建图或重打点后失效，手写进配置的话
失效后软件层完全看不见（派单「成功」，狗不动，或走到另一个点）。

标签与平台点位靠**点位名精确匹配**，对不上用 `--alias` 明确指定。
刻意不做模糊匹配：猜错点位比报错危险得多。

### 3. 三件代码解决不了的事

| 项 | 现状 | 影响 |
|---|---|---|
| E3 手动/自动巡检切换接口 | 已核验 275 个接口里**没有**（F14），需抓 Web 前端 | Arbiter 停在 `WAITING_HUMAN_MODE_SWITCH`，**必须人工在 Web/APP 切模式**后调 `ack_human_mode_switch(by=...)` |
| E4 DDS 命令是否被边缘侧独占 | 未验证（G11），需在狗侧网络实机测 | `probe_dds_authority()` 失败即转 `SAFE_HOLD`，绝不带着不确定的控制权进 `WAM_ACTIVE` |
| G7 `updateControllerUser` 的 `state`/`force` 取值 | OpenAPI 无枚举 | 未配置时**跳过抢权并记警告**，不假装成功。真机前必须抓包补上 |

### 4. 两个必须人工维护的输入

- **定位置信度**：平台全库 0 处 confidence 字段（G4），所以 `ack_confidence(value, by=...)`
  是人工录入且带审计留痕，默认 300 秒过期（长期趴下会丢定位，手册 §5.6）。
  既没 provider 又没人工确认 → preflight 直接拒绝，不放行。
- **气体标定台账**：平台无标定接口（F13）。照 `configs/mission/gas_calibration.example.json`
  填真实标定时间并进版本库。**禁止为了让气检通过而填假时间**——那等于把安全门禁拆了。

## 四、两个不许碰的红线

1. **DDS 位姿只准进 WAM 落盘通道，绝不进 `mission_brain.events`。**
   事件契约 v1 的 `FORBIDDEN_KEYS` 会在运行时拒绝 `pose_xyz` / `global_pose`；
   而且 odom 未与平台地图对齐时冒充全局坐标，会让无人机 Scout 与狗的 goal 错配。
   `tests/test_dog_integration.py::test_no_pose_leaks_into_events` 守这条。

2. **速度安全盒必须由我们实现。** 遥控器档位的软限位（低速档爬坡 <45°、台阶 <25 cm）
   在 DDS `Move()` 直控路径上**不生效**（F30）。`SpeedLimits` 默认
   `1.0 / 0.6 / 1.0`，远低于 B2 的 5 m/s 极限；放宽只能是显式决定。

## 五、还没做的

- **M12 WAM 采集落盘**（`EpisodeRecorder` 的狗侧扩展）：依赖 E4 结论与真机时钟对齐
  方案，等真机联调后再做。
- **`main.py` 接线**：目前这套是库，没接进 GUI 主程序。等 E2/E3 有结论、配置空位
  填完再接，否则接进去也只能跑在假枚举上。
