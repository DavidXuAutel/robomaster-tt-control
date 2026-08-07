# 机器狗部署闭环方案 v3.4（真机联调 + 边端部署改造）

日期：2026-08-05（阶段范围补丁：2026-08-06；公开脱敏：2026-08-06；**集成商接口政策修正：2026-08-07**）
状态：**可据此开发**（当前只授权 **D0**；已完成对抗评审与阶段范围评审，见附录 C）

> **v3.4 修正摘要（2026-08-07，依据集成商书面答复与 SDK 实测）**：
> ①集成商不建议第三方直连 DDS，DDS 快通道降为**非生产路径**，E4 拆为 E4-S（只读）/ E4-C（命令，需书面批准）；
> ②到点判定主证据改为平台 MQ 推送，HTTP 轮询降为兜底（F16 事实推翻）；
> ③E4 原第④步会复现集成商警告的双头同时下发，**已按零速度权属握手重设计**；
> ④§5 中「E4③已验证 deadman」是错误陈述，已改回未验证；
> ⑤新增欠账 Q7/Q8（机型客户端错配、状态字段残缺），已于当日修复。
关系：**扩展并接替** `2026-08-03-dog-integration-plan.md`（v2）的 §7 路线图。
v2 的 §2 事实基线（F1–F32 / G1–G16）与 §5 契约变更**仍是接口事实的权威**；实验编号以本文 §2 为准（v2 §8 编号作废，映射表见 §2）。
实现现状见 `docs/handover/2026-08-03-dog-integration-m6-m11-implementation.md`（M6–M11 软件已落地，352 pytest 绿）。

> **近期交付范围**：完整工程路线仍为 D0–D8；2026-09-14 前执行范围与停止线见 §0.3 / §3.0。当前只授权 D0，未经项目负责人书面批准不得提前启动后续阶段。  
> **公开仓库脱敏**：本仓库将上 GitHub 公开；见 §0.4 与 `CLAUDE.md`「公开仓库脱敏准则」。

---

## 0. 本次改造改了什么（一句话）

项目重心从「把狗接进 MissionBrain 编排」升级为「**部署闭环**」：

> **服务器当老师 → 边端跑学生模型 → 量化评测（时延/内存/功耗/成功率）→ 独立故障哨兵 → 影子部署 → 受限闭环**。

改造理由：
1. 远程 GPU（cloudflared 隧道 4090）在实时链路里＝延迟污染 + 单点故障，只应做教师与离线标注；
2. 「记录了数据」≠「数据可训练」——丢帧/时钟漂移/动作延迟目前无人自动检查；
3. 让生成式模型直接控 60kg 真狗是错误顺序——必须先有独立于模型的安全哨兵与影子期。

### 0.1 核心铁律

**模型只出提议（proposal），执行权永远在独立哨兵/仲裁层手里；模型进程崩溃，狗必须能停。**

### 0.2 Gate 的两种作用域（本节解决 v3.0 的架构矛盾）

现有 `DogControlArbiter` 的不变量 I3/I4 规定：`MISSION_NAV`（平台慢通道导航中）期间**禁止**发任何 DDS `Move`；平台侧又没有连续速度接口（G10）。因此「在原生导航上叠加减速」**不可实现**，明确放弃。ProposalGate 按 Arbiter 状态分两种作用域：

| Arbiter 状态 | Gate 作用域 | Gate 能做的动作 |
|---|---|---|
| `MISSION_NAV`（平台导航中） | **观察者**（shadow） | 只记录裁决 + 告警；风险持续超阈时**只能走任务级停止**（`TopseeNav.cancel()` → `stopTask`，这是 v2 已允许的路径），不发任何 Move |
| `WAM_ACTIVE`（快通道段） | **命令闸** | 对模型提议做 PASS/SLOW/STOP 裁决后经 Arbiter 发 `Move`；这是唯一的闭环作用域 |
| 其余状态 | 硬 dry-run | 零输出 |

```text
传感器 → episode 落盘/QA → 异步边端推理 → RiskProposal
                                       ↓
             ┌── MISSION_NAV：观察者（告警/任务级停止）
ProposalGate ┤
             └── WAM_ACTIVE：命令闸 → DogControlArbiter.move() → DDS Move
```

### 0.3 分阶段交付边界与优先级（开发 Agent 必读）

| 阶段 | 交付定位 | 当前执行边界 |
|---|---|---|
| **近期阶段：基础能力与离线验证** | 完成 D0，并按 §3.0 的时间窗口逐步进入 D1、D2 与 D5 离线骨架 | 2026-09-14 前只执行 §3.0 明确列出的切片 |
| **后续阶段：数据、训练、影子与闭环** | 按 D3–D8 完成数据采集、模型产线、影子验收和受限闭环 | 默认排在阶段评审之后，未获书面批准不得提前启动 |

**裁决规则**
1. 当前授权范围优先于完整路线图；完成当前阶段不自动授权下一阶段。
2. 阻塞时只允许补充当前阶段内的测试、报告和现场准备，不得借阻塞跳到后续阶段。
3. **本仓库资产合规**：代码、数据、配置、模型、checkpoint、日志及硬件采集结果只能用于本项目批准的研发与交付；引入外部资产必须具备明确来源、许可和审计记录，禁止与无关仓库或设备互拷。
4. 开发 Agent 只执行本文明确授权的本仓库任务；人员安排、采购和现场协调由项目负责人按公司流程处理。

### 0.4 公开仓库脱敏准则（强制）

本仓库内容会上传 GitHub、对公众与合作方可见。**事情可以做，文档与元数据不得暴露仓库外的私人规划意图。**

1. 文档只写本项目的工程目标、事实、约束、决策、风险与验收标准；不写个人生活策略、职业意图、私人优先级或外部规划代号。
2. 禁止出现私人目录、用户名、本机历史档案、私人知识库、战略台账及其文件名或链接；引用依据必须位于本仓库或经批准的公开来源。
3. 代码注释只解释机制、安全约束和必要取舍，不记录个人动机或与本项目无关的计划。
4. Commit / PR / Issue / 评审记录使用工程语言，不出现「练兵、作品集、保个人项目、离开准备」等私人目的表述。
5. 文件名、artifact 名与报告元数据用功能/阶段/日期/版本命名，不嵌入私人项目代号、姓名、主机用户名或规划体系编号。
6. 配置只提交脱敏示例；不得提交凭据、真实账号、私人绝对路径、未经批准的内部地址。
7. 本仓库资产不得流入无关仓库、个人设备或未经批准的外部系统；外部资产须有来源、许可与审批记录。
8. 发布前扫描私人代号、绝对路径、凭据与敏感元数据，并核对 Markdown / HTML 等派生版本口径一致。

---

## 1. 现状基线（开发 Agent 必读）

### 1.1 已落地（不要重写）

| 模块 | 文件 | 说明 |
|---|---|---|
| 平台 HTTP 会话 | `adapters/topsee_client.py` | `code:0` 语义、401 重登、`PollCache` 后台轮询 |
| 慢通道后端 | `adapters/dog_topsee.py` | `TopseeNav`（到点三态+`poll_fault`）/ `TopseePerception` / `TopseeGas` |
| RSA 登录 | `adapters/topsee_rsa.py` | 纯标准库 |
| 气体台账 | `adapters/gas_ledger.py` + `configs/mission/gas_calibration.example.json` | 人工标定台账，禁伪造 |
| 快通道骨架 | `adapters/dog_unitree.py` | `SpeedLimits` 速度安全盒（默认 vx 1.0 / vy 0.6 / vyaw 1.0）；`DdsTransport` 真实订阅（含 `connect_readonly()` 只读路径、机型能力表 `FAMILY_API_METHODS`）+ 无硬件 `LoopbackTransport`（测试桩） |
| DDS 订阅自检 | `tools/dds_selfcheck.py` | 真实 CycloneDDS + 真实 IDL 的订阅链路自检；`--observe` 模式对外部发布端计量，即真机 D0 的验收形态 |
| 仲裁层 | `adapters/dog_arbiter.py` | 8 状态机、租约、preflight 门禁、命令看门狗；I1–I10 不变量由 `tests/test_dog_arbiter.py` 固化 |
| 语义绑定 | `mission_brain/map_model.py` + `tools/export_dog_bindings.py` | SharedMap v2 `platform_binding`、漂移检查（`--check` 非零退出） |
| 只读探针 | `tools/topsee_probe.py` | E0/E2/E5–E10 只读；E1 需 `--allow-motion` 双确认 |

### 1.2 已知欠账（评审坐实，编入 D0 必修）

| # | 欠账 | 证据 | 后果 |
|---|---|---|---|
| Q1 | **真实 DDS 订阅未实现**：`DdsTransport` 从未创建 subscriber，`read()` 恒返回 `None`（`dog_unitree.py:202-255`） | Codex 评审 §一致性5 | 真机位姿/lowstate/E4 全部不可用 |
| Q2 | **生产接线缺失**：库未接进任何运行时；`DogSdkAdapter.abort()` 只调 `nav.cancel()`，不会自动 `Arbiter.force_release()` | handover §五；评审 §一致性9 | 现在的架构图是纸面链路 |
| Q3 | `ack_confidence` 无持久审计、不校验 `[0,1]`/NaN | 评审 §一致性8 | NaN 可绕过 ≥0.9 门禁 |
| Q4 | 电量门禁 fail-open：battery provider 缺失或返回 `None`/NaN 时直接放行（`dog_arbiter.py:255-263`） | 评审 §安全7 | 门禁形同虚设 |
| Q5 | `probe_dds_authority()` 只发 `StopMove`+读一帧状态，**不能证明 Move 权限/无抢权** | 评审 §一致性6 | E4 必须重设计（见 §2） |
| Q6 | 探针文件头与 gap 标注有错（写着 E1–E9；E7/E9/E10 的 gap 编号错） | 评审 §一致性2 | D0 顺手修正 |
| Q7 | **机型客户端错配**：导入 go2 的 `SportClient`，却调用 b2 独有的 `SwitchGait`/`BodyHeight`/`MoveToPos`；且 api_id 1036 在 go2 上是 `Heart` 不是 `MoveToPos` | 2026-08-07 实测 `unitree_sdk2py` 两家族方法表 | 真机上 `AttributeError` 而非干净报错；目标机 B2 用错客户端。既有测试全走 Loopback（照单全收），从未覆盖 `DdsTransport.call()`，属假绿 |
| Q8 | **状态字段残缺**：丢弃报文自带设备钟 `stamp`（只留接收时刻 `t_mono`）、IMU 只留 `rpy`（丢加速度/角速度/四元数）、电机只留 `q`（丢 `dq`/`tau_est`/温度/丢包）、足底力整体未取 | 同上 | 用接收时刻分不清「狗侧卡住」与「网络卡住」；v2 §数据契约点名的 12 电机+IMU+足底力不完整 → 打穿 D3 |

> **状态（2026-08-07）**：Q1–Q6 见 `artifacts/d0_software_gate.md`；Q7/Q8 已于本日修复并加回归测试
> （`tests/test_dog_unitree.py` 的机型能力表与状态契约两组）。Q7/Q8 是本轮新发现，不在原评审清单内。

### 1.3 悬而未决（disallow assumptions）

| 项 | 编号 | 处置 |
|---|---|---|
| 单点派单是否产生可查任务 | E1 | D7 前必须实测；否则 `is_arrived()` 退化为位姿距离判据（需 DDS 位姿，依赖 Q1） |
| 手动/自动巡检切换接口 | E3 | 无接口（F14）。人工切完调 `ack_human_mode_switch(by=...)`，流程演练 ≥3 次 |
| 运动权限（状态可读） | E4-S | 载体可为 DDS 只读或集成商状态转发；协议见 §2 |
| 运动权限（命令下发） | E4-C | D7 生死线。集成商不建议直连 DDS（§1.5-V1），须先取得书面批准或改用其运控接口；协议见 §2 |
| 集成商运控接口契约 | V1 | **五项必答**：最高下发频率、端到端时延、停发归零时间、实际速度回读、权属交接机制。未答齐不得写实现，也不得据此排 D7 |
| 集成商 MQ 契约 | V3 | broker/协议/认证/QoS、`robotNum`↔`robotId` 映射、消息与 `pointsId` 的唯一关联键；未明前 MQ 只写入口径不写实现 |
| `updateControllerUser` 取值 | G7 | 未配置时跳过抢权并记警告，真机抓包后回填 |
| 定位置信度 | G4 | 无接口。人工 `ack_confidence` + D0 补审计 |
| B2 现机算力配置 | — | 宇树官方标配＝i5+i7 双 x86，选配最多 3 块 Jetson Orin NX；本机被拓普视改装过，**必须 D1 现场盘点** |
| 深度流可用性 | E7 扩展 | E7 只验证接口形状；深度帧协议/内参/时间戳须真机确认，不可得则 recorder 降级 RGB 主时钟 |

### 1.4 算力分工（D1 盘点后按此就位）

| 计算单元 | 角色 | 禁令 |
|---|---|---|
| B2 i5 平台机 | 厂商导航栈 | **绝不动**：不装包、不升级、不跑我们的进程 |
| B2 i7 用户机 | ProposalGate、Arbiter、recorder、DDS 客户端、CPU 推理基线 | 不跑训练 |
| Jetson Orin NX（若有/增购） | 学生模型 TensorRT 推理、功耗实验 | 未经厂商镜像+恢复演练不升 JetPack |
| 4090 服务器 | 教师推理、伪标签、蒸馏训练、离线回归 | **退出实时控制链** |
| 5×H100 | 仅当 4090 训不动时启用 | 不为小模型长期占用 |

> 注意：无 Jetson 时模型与 Gate 同在 i7，只是**进程隔离**不是物理隔离；Gate、Arbiter、DDS 客户端共享 i7 故障域——这是残余风险，靠 §6 的 fail-safe 设计与实体急停兜底，不许在文档里宣称「物理隔离」。

### 1.5 集成商接口政策修正（2026-08-07，依据书面答复）

本机 B2 由系统集成商改装并封装。其技术支持书面答复三点，直接推翻本方案若干前提：

| # | 集成商答复 | 对本方案的后果 |
|---|---|---|
| V1 | **不建议**第三方直连宇树 DDS，并发下发会导致运动异常；改为提供他们的「状态转发」（遥测）与「运控接口」（运动命令） | DDS 快通道降为**非生产路径**。D0 原验收「真机 DDS 连续 10 分钟无断流」的阻塞原因从「现场未排期」变为**政策不允许**，性质不同，不能靠等排期解决 |
| V2 | 上位机（工控机）与狗**同网段**；其余设备刻意分在不同网段，避免广播风暴干扰运动控制 | 我方设备大概率不会被允许接入狗的运动网段。我方进程只能落在 B2 的 **i7 用户机**（§1.4），且 i5 平台机红线不变 |
| V3 | 任务结果有 **MQ 推送通道**（topic 形如 `service_robot_task_result.[robotNum]`） | 推翻 v2 事实 **F16**（原记「平台无对外 MQTT/WS」）。到点判定的主证据从 HTTP 轮询改为 MQ 订阅 |

**由此确定的三条口径**

1. **快通道的承载者未定。** 「连续速度 10–50Hz + deadman + 实际速度回读」这组能力，现在只可能由集成商的运控接口提供。该接口的**契约等级未知**——只拿到调用示例不足以判断，必须拿到：最高下发频率、端到端时延、停发后归零时间、能否读回实际速度、错误码、权属交接机制。这五项决定 D7 是否成立（见附录 B 的问询清单）。
2. **我们目前没有合法的运动级急停。** 原 abort 链依赖 DDS 的 `StopMove`/`Damp`，现已不可用。这与做不做闭环无关，是安全链的缺口，必须在 D7 前由运控接口补齐或由实体急停兜底并写入现场规程。
3. **到点判定改为双源交叉校验**（优于原设计，非简单替换）：
   - 主证据：MQ 推送（低时延）；
   - 兜底：`PollCache` 轮询 `getCurrentByRobotId`（补丢消息、乱序、MQ 断连）；
   - 两者均不可用 → `nav_status_unknown` **fail-closed**，不得默认「已到达」。
   - `TopseeNav` 已有「到点三态」抽象，MQ 是**给同一输出增加证据源**，不是重写。
   - ⚠️ `instruction.complete == true` **不等于成功到达**（失败/取消/超时同样可能置真）。没有结果码前禁止直接发 `DOG_ARRIVED`。

**未知项（禁止假设，未拿到书面契约前不得写实现）**：运控接口的五项契约；MQ 的 broker 地址/协议/认证/QoS；`robotNum` 与 `robotId` 的映射；MQ 消息与本次 `sendNavigate(pointsId)` 的唯一关联键。

---

## 2. 实验编号（唯一权威表，v2 §8 编号作废）

以 `tools/topsee_probe.py` 的代码编号为准。D0 需修正探针文件头注释与 gap 标注。

| 实验 | 类型 | 问题 | v2 旧编号 | 挂在阶段 |
|---|---|---|---|---|
| E0 | 只读 | 登录链路/授权有效 | （新增） | D1 |
| E1 | **动作** | 单点 `sendNavigate` 是否产生可查任务；**顺带制造到达/取消/无效点位，采齐 E2 枚举** | v2-E1 | D7 前（封闭场地） |
| E2 | 只读 | `currentState/totalState` 取值采样 | v2-E2 | D1 起持续采样；**完整枚举依赖 E1 制造状态，D1 只交「已见值集合」** |
| E3 | 人工抓包 | 手动/自动巡检切换 | v2-E3 | D7 前 |
| E4-S | 真机（只读） | **状态可读**：`/sportmodestate` + `/lowstate` 连续 10 分钟可读，字段齐全、设备钟推进、断流可检。载体可以是 DDS 只读，**也可以是集成商的状态转发接口**——换载体不改验收内容 | v2-E4① | D2b 前 |
| E4-C | 真机（命令） | **运动命令权属协议**，四步须全过：①可控——围栏内发极小幅速度，用**实际速度回读**确认执行；②deadman——停发命令后确认自动归零速并测出归零时间；③权属原子性——**先在零速度下**验证「获取/拒绝」是原子的（非持有者被确定性拒绝），而非混合；④仅在集成商书面确认仲裁安全后，才允许在持有权属下做微幅运动复核 | v2-E4②③④ | D7 前（生死线） |

> **E4 为何拆分（2026-08-07）**：集成商已书面表示不建议第三方直连 DDS（§1.5-V1），所以「只读遥测」与「下发命令」的批准条件不同，必须能分别申请、分别验收；`DdsTransport.connect_readonly()` 就是这条边界在代码里的落点（该路径下 `call()` 恒拒绝）。
>
> 🔴 **原 E4 第④步「平台侧同时活动时观察是否抢权」已废止**：它要求两个控制器同时下发运动命令，**正是集成商警告会导致运动异常的操作**。新版④先在零速度下验证权属原子性；若集成商无法提供原子的权属交接，结论是 **E4-C 判负**，不是「那就试试同时发」。
>
> ⚠️ **deadman 无官方依据**：2026-08-07 检索未找到宇树对 `SportClient.Move` 机器人侧超时行为的权威文档说明。该项**不得按业界惯例推定**，只能实测或取得书面保证（对比：Boston Dynamics Spot 的速度命令强制带 `end_time`，云深处 Lite3 底层 SDK 明确 >1s 无命令进阻尼保护——宇树侧没有等价文档）。
| E5 | 只读 | `getRobotMapAll` 真实结构 | v2-E9 | D1 |
| E6 | 只读 | 气体历史时间参数格式 | （v2-E6 的子集） | D1 |
| E7 | 只读 | 取流地址响应结构（**不等于深度可用**；深度协议/内参/时间戳另在 D2b 真机确认） | v2-E8 前半 | D1→D2b |
| E8 | 只读 | 会话/token 有效期 | （新增，G8） | D1 |
| E9 | 只读 | 告警列表结构 | （v2 未编号） | D1 |
| E10 | 只读 | 实时状态字段名（电量） | （新增） | D1 |
| — | 人工问询 | 部署形态与写权限（云端 :8001 vs 离线 :8888） | v2-E7 | D1（问厂商/现场，不是脚本） |
| — | 人工台账 | 气体标定记录来源 | v2-E6 | 已由 `gas_ledger` 落地 |
| — | 无接口 | 置信度/延迟/实时位姿 HTTP 接口 | v2-E5 | 已判死（G4/G5），人工 ack + DDS 位姿替代 |

---

## 3. 阶段计划 D0–D8

> **排序原则：先补地基、再只读、再落盘、再影子、最后闭环。**
> 工期按单工程师 + 每周 2–3 天现场折算，**总计约 12–14 周**；厂商答复、场地审批、增购 Jetson 属外部等待，单列不计入。
> 每阶段交付物 = 代码 + 测试 + 一份 `artifacts/` 报告；无报告不算完成。

### 3.0 六周执行切片与 09-14 阶段停止线（开发 Agent 硬约束）

完整 D0–D8 是工程路线图，**不是**当前授权任务列表。按当前阶段交付安排，**2026-09-14 前**只允许：

| 周次 | 日期（约） | 本仓库授权 | 明确禁止 |
|---|---|---|---|
| W1 | 08-06～08-10 | **仅 D0 的 W1 切片**（见下 D0-W1） | D1–D8；训练；Move；E1/E4 动作 |
| W2 | 08-11～08-17 | D0 收口 + **可开始 D1** | D3/D4/完整 D5/D6/D7 |
| W3 | 08-18～08-24 | D2a（离线 recorder/QA/回放） | 同上 |
| W4 | 08-25～08-31 | D2b **条件起步**（有现场才录）+ **D5 骨架**（定义见 D5） | 完整 D5 万次矩阵；D6 验收 |
| W5 | 09-01～09-07 | D5 骨架上的**离线影子/dry-run 报告**（≠ D6 验收） | D6 正式验收；D3 大规模采集 |
| W6 | 09-08～09-14 | **阶段交接报告**（已完成部分归档） | **禁止启动** D3、D4、完整 D5、D6 验收、D7 解锁/闭环——除非项目负责人按公司流程另行书面批准 |

**09-14 前禁止跨线清单（默认）**：D3 采集战役、D4 师生训练产线、完整 D5（万次注入/真实 Move 出口/解锁）、D6 影子验收、D7 闭环。阻塞时只在当前授权阶段内补测试/报告/现场清单，**不得**用等待跳到后阶段。

### D0 地基修复与生产接线（第 1–2 周）★新增，评审必修

#### D0.0 基线复现与改动边界（动手前必做）

1. 复现 handover §二：`python -m pytest tests/ -q`（预期约 352 passed；以现场实测为准）。  
2. 结果不一致 → **停止改代码**，先报告差异。  
3. **禁止重写** §1.1 已落地模块（M6–M11）；**禁止改** `main.py`；**禁止放宽** I1–I10。  
4. 只补 Q1–Q6 与本阶段交付物。

#### D0-W1 检查点（截至 2026-08-10）★近期阶段 W1

下列四项 **W1 必须有结论**；现场未排期导致 DDS 无法实测时，必须显式标 `BLOCKED`，**禁止**把 Loopback 通过写成「状态流稳定可读」：

| # | 交付 | 完成标准 |
|---|---|---|
| 1 | 真实 DDS 实读 | `/sportmodestate`（及 lowstate）在狗侧网络持续可读；或 `BLOCKED`+现场清单 |
| 2 | `dog_runtime` + abort | runtime 可装配；abort → `Arbiter.force_release()`；相关测试绿 |
| 3 | 两类 fail-closed 门禁 | 电量未知/NaN 拒绝；confidence 越界/NaN 拒绝且审计落盘 |
| 4 | `configs/dog/topsee.json` 骨架 | 六键存在；**不伪填**未实测的 E 枚举 |

#### D0 完整交付物（含 W1 之后收口）

1. `adapters/dog_unitree.py`：实现真实 DDS subscriber（`/sportmodestate`、`/lowstate`，基于 `unitree_sdk2py` 的 CycloneDDS 绑定）；`read()` 返回带单调钟时间戳的最新样本；断流检测（>500ms 无样本 → `dds_stale`）。
2. `runtime/dog_runtime.py`（新）：生产接线入口——装配 TopseeClient/Nav/Gas、Arbiter、（后续）Gate 与 recorder 的**唯一**进程编排；含健康检查、日志轮转、磁盘水位检查（<10% 停录制）。`main.py` 不动。
3. abort 链路闭合：`DogSdkAdapter.abort()` 在持有 Arbiter 时必须触发 `Arbiter.force_release()`（保持 backend Protocol 签名不变，用可选钩子注入）。
4. Q3 修复：`ack_confidence` 校验 `0.0 ≤ v ≤ 1.0` 且 finite，持久化审计到 `logs/audit/confidence_*.jsonl`（who/value/expiry）。
5. Q4 修复：电量门禁 fail-closed——provider 缺失/`None`/NaN 一律**拒绝** preflight，reason=`battery_unknown`。
6. Q6 修复：探针文件头与 gap 标注更正为本文 §2。
7. 配置落位：`configs/dog/topsee.json`（E 回填六项的唯一入口：`arrived_states`/`enroute_states`/`battery_field`/`time_format`/`token_header`/`alarm_fields`），`dog_runtime.py` 只从这里读。

**D0 期间禁令**：禁止发送 `Move`；禁止 E1/E4 动作实验；禁止接真实命令 sink；禁止解除任何 dry-run。

8. Q7/Q8 修复（2026-08-07 新增）：`DdsTransport` 按机型能力表 `FAMILY_API_METHODS` 派发（默认 `family="b2"`，配置键 `dds_family`），表外 api_id 干净报错；状态转换保留设备钟 `t_device`、IMU 全量、电机 `q/dq/ddq/tau_est/温度/丢包`、足底力；`LoopbackTransport` 与 `DdsTransport` 的 `read()` 键集必须逐字段一致（有测试守护）。
9. `tools/dds_selfcheck.py`（新，2026-08-07）：订阅链路自检。两种模式——自带发布器的闭环自检，与 `--observe` 只订阅外部发布端。

**验收（分三级，不得混淆）**

| 级别 | 内容 | 证明什么 | 产物 |
|---|---|---|---|
| L0 软件 | `tests/test_dog_runtime.py` 装配/健康/磁盘/abort 全绿；`tests/test_dog_arbiter.py` battery `None`/NaN 与 confidence NaN/越界拒绝且审计落盘 | 逻辑正确 | 测试报告 |
| L1 真实 DDS·合成源 | `tools/dds_selfcheck.py` 在真实 CycloneDDS + 真实 `unitree_go` IDL 上跑通：订阅、字段完整、设备钟推进、断流可检、只读模式下命令被拒 | **订阅链路软件正确性**——此前只有 Loopback 桩覆盖，等于没测 | `artifacts/d0_dds_selfcheck_<date>.json` |
| L2 真机 | `--observe` 对着真狗（或集成商状态转发）连续 10 分钟，帧率/间隔/字段完整性达标 | 真机时序 | `artifacts/d0_dds_10min_<date>.json` |

- **L1 绿 ≠ L2 完成**，Loopback 绿更不等于任何一级。L2 现被集成商政策阻塞（§1.5-V1），须显式标 `BLOCKED` 并写明阻塞性质是**政策**而非排期。
- L1 的采样年龄指标反映轮询节拍，**不是 DDS 端到端时延**，禁止当延迟指标引用。
- 若发布端不填 `stamp`，`device_stamp_monotonic` 会空洞成立；必须同时看 `device_stamp_advancing`。

**依赖**：无。

### D1 现机盘点 + 只读探针 + 边端评测 harness（第 2–3 周）

**交付物**
1. `artifacts/b2_inventory.json`：现机盘点——几台计算机、CPU/Jetson 型号、内存、OS/JetPack/TensorRT 版本、可访问视频流地址与编码、网络拓扑。
2. 跑探针只读集（E0/E2/E5–E10）→ `artifacts/topsee_probe_*.json`；六项回填写入 `configs/dog/topsee.json`；部署形态（:8001/:8888）问询结论记入 inventory。
3. `tools/edge_bench.py`：模型评测 harness。
   - Backend 协议：`class BenchBackend: def load(model_path, device) / def infer(frame: np.ndarray) -> Any`；内置 `onnxruntime`、`torch`、`tensorrt` 三个实现；
   - 输入源：离线帧目录 / 视频文件 / RTSP 流；
   - 计时边界：**帧解码完成 → 推理输出返回**（含前处理，不含解码；CUDA 显式 `synchronize` 后取点）；warm-up 100 帧不计入；
   - 输出 JSON：P50/P95/P99、FPS、进程 RSS、显存、SoC 温度、功耗（Jetson `tegrastats` 采样 1Hz；x86 RAPL + `power_meter_w` 人工字段）、丢帧率（分母＝输入源产出帧数）、降频事件（时钟频率低于标称 90% 的秒数）。

**验收**
- harness 在 i7（CPU）与 Jetson（若有）各连续 30 分钟跑固定基准模型（`depth-anything-v2-small` ONNX），零崩溃、报告字段齐全；`tests/test_edge_bench.py` 用合成帧覆盖计时/丢帧/报告逻辑；
- 算力决策记录在案：`现有 NX 够用 / 增购 NX 16GB / 暂用 i7 基线`。

**依赖**：现场网络 + 只读账号。

### D2a 狗版 Recorder + 统一 QA/回放（离线骨架，第 3–4 周）

**交付物**
1. `adapters/dog_recorder.py`：按 v2 §6 规格——深度帧主时钟、`frames.csv` 字段表、`/lowstate` 降采样 50Hz 单独落盘、`T_map_from_odom` 写 `meta.json`；深度不可得时降级 RGB 主时钟并在 meta 声明 `clock=rgb`（此时 `depth_*` 列留空、`has_depth=0`）。API：`DogRecorder(out_dir).start() / ingest_frame(...) / ingest_pose(...) / ingest_action(...) / stop()`，内部单写线程 + 有界队列（满则丢最旧并计数）。**schema 带 `schema_version: 2`，与无人机旧格式不兼容——不做兼容层，比较脚本各读各的**（v2 §6.7 明确禁止复用无人机检查器）。
2. `tools/episode_qa.py`：输出机器可读 QA manifest。指标定义：
   - **时钟单调性**：`t_mono_ms` 严格递增，违例数；
   - **跨流对齐误差**：每帧 `|rgb_ts - depth_ts|`（同一单调钟域），报 P99；
   - **动作标签延迟**：`frame.t_mono_ms - action.issued_t_mono_ms`，报 P99（真值＝命令下发时刻，不声称是执行时刻）；
   - **缺帧率**：`1 - 实收帧/期望帧`（期望帧＝主时钟源产出数）；
   - **地图变换跳变**：相邻帧 `map_*` 位置差 > 0.5m 且 odom 差 < 0.1m 的次数；
   - 继承 v2 §6.7 全部门槛：`frame_match`/`depth_p95_ok`/`pose_coverage≥95%`/`action_nonzero`/`single_owner_ok`/`conf_declared`/`cmd_watchdog_ok`/`no_platform_estop`/`duration≥120s`/`map_align_declared`。
3. `tools/episode_replay.py`：按原时序回放 episode 喂任意推理后端；支持 1x/加速/暂停；确定性（同输入同输出，`tests` 里校验哈希）。

**验收**
- 对 10 段样本（合成 + Loopback 生成）出 manifest；100 个注入损坏（乱序/缺帧/动作错位/租约重叠/地图跳变）全部检出，零漏报；
- 新数据硬门槛写死：非单调时间戳＝0、缺帧率 <1%、对齐误差 P99 ≤50ms；**QA 失败的数据禁止进训练集，无人工豁免通道**。

**依赖**：D1 的流地址（离线部分可先行）。

### D2b Recorder 真机验收（第 4–5 周，半周现场）

- 真实视频流 + 真实 DDS 位姿（D0 Q1 已修）录制 ≥5 段、每段 ≥120s；QA 全过；
- 深度通道结论落盘：可得（协议/内参/时间戳记入 meta）或不可得（正式启用 RGB 主时钟降级，报告说明）。

**依赖**：D0（DDS subscriber）、D1（流地址）、现场排期。

### D3 数据采集战役（第 5–6 周，现场为主）★新增

**交付物**
- 采集计划：≥6 条差异化路线（直行/转弯/坡道/窄道/动态行人区/弱纹理区），每条 ≥5 遍，操作方式＝平台 teach-and-repeat 导航 + 人工遥控段；
- **≥40 段 QA 通过的 episode**（D4 训练与 D6 标注的原料；训练/验证/测试按路线切分防泄漏：4/1/1 条路线）；
- 危险事件初步清单（供 D6 标注协议用的类别草案：迎面行人/障碍逼近/地面突变/急转/打滑等）。

**验收**：40 段全过 QA；切分清单落盘 `configs/dog/dataset_split.json`。

**依赖**：D2b。现场许可、路线安全评估先行。

### D4 教师→学生模型产线（第 6–8 周）

**交付物**
1. `server/teacher_labeler.py`：4090 上 DA-V2 批量产深度伪标签；**可通行性标签生成规则**（写死并版本化）：深度图分带（近/中/远）+ 地面平面拟合 → 每帧输出「前方 1.5s 走廊内最小障碍距离」与三档风险（`pass/slow/stop`），阈值初值 `slow<2.0m、stop<1.0m`（按 0.5 m/s 巡速换算，可配置）。
2. `experiments/dog/train_student.py`：学生模型（轻量 CNN/ViT-Tiny 级）——输入前向 RGB(+深度可选)、当前速度、名义速度；输出 `risk_score∈[0,1]`、三档 `action`、速度修正 `Δvx/Δvy/Δvyaw`（训练时即钳位在 `SpeedLimits` 内）。损失＝风险回归 + 档位分类 + 修正量回归；标签全部来自教师规则，**测试集另有 ≥500 帧人工复核标签**（D3 数据抽样，双人标注，分歧仲裁后为真值）。
3. 导出链：checkpoint → ONNX（固定 tensor 名：`rgb[1,3,H,W] float32 /255 归一`、`state[1,4]`；opset 17；静态 shape）→ **在目标机上**构建 TensorRT FP16/INT8（校准集＝训练集抽样 512 帧，清单落盘）。
4. 产物包 `artifacts/student_<ver>/`：checkpoint + ONNX + 校准清单 + 数据版本（episode manifest 哈希）+ git SHA + 依赖锁 + `edge_bench` 报告 + 精度报告。
5. 运行栈落位：`requirements-edge.txt`（onnxruntime / tensorrt 版本锁定，Jetson 侧按 JetPack 对应版本记录）。

**验收**
- 人工复核测试集上：风险二分类（stop vs 其余）AUC ≥0.90，档位准确率 ≥85%（基线＝教师规则自身在该测试集的表现，学生不得低于教师 3pp 以上）；
- 边端 60 分钟连跑：10Hz 输入 P95 ≤80ms、P99 ≤120ms；峰值系统内存 ≤12GB；计算模块平均功耗 ≤25W（Jetson；i7 基线只记录不设限）；
- INT8 相对 FP16：AUC 降幅 ≤0.02、档位准确率降幅 ≤2pp。

**依赖**：D3 数据。

### D5 ProposalGate 哨兵（第 6–9 周，与 D4 并行开发）

#### 六周窗口内的「D5 骨架」边界（09-14 前）

09-14 前的「D5 骨架」**仅指**下列集合；超出即属于未经批准的阶段扩项：

| 允许（骨架） | 禁止（完整 D5 / 越界） |
|---|---|
| `proposal_types.py` 契约 | 接真实 `Move` / 解除 dry-run |
| Gate 默认 `dry_run=True` + 假提议或回放输入 | 完整 ≥10,000 次故障矩阵验收 |
| 基础拒绝测试（NaN/越界/TTL/乱序等子集） | 解锁 PR / `gate-unlock` 流程 |
| 离线 dry-run 裁决日志报告 | 宣称「D6 影子已开始/已验收」 |

完整 D5 验收与 D6 仍按后文全文执行，但 **默认排在 09-14 之后**。

**交付物（完整 D5；默认在 09-14 后启动，提前启动须经项目负责人书面批准）**
1. `adapters/proposal_types.py`：见 §4 契约（含 `session_id`）。
2. `adapters/proposal_gate.py`：独立进程，50Hz 循环。
   - IPC：**UDS datagram**（`SOCK_DGRAM`，路径 `run/gate.sock`，权限 0600）；模型→Gate 单向；`session_id`（模型进程启动时间戳）+ `seq` 单调；乱序丢弃、`seq` 跳变计数、跨 session 重放拒绝；
   - 检查项与阈值（初值，全部进 `configs/dog/gate.json`）：租约有效；提议 TTL ≤200ms；观测年龄 ≤300ms；速度盒＝`SpeedLimits`；加速度 ≤1.0 m/s²、jerk ≤3.0 m/s³（对连续两条放行命令差分）；NaN/Inf 拒绝；推理超时（>500ms 无新提议→HOLD）；模型进程死亡（UDS 静默 >200ms 且 pid 消失）；DDS 断流 >500ms；SoC 温度 >85℃；电量 <25%（读 Arbiter 同源 provider，fail-closed）；
   - 裁决：`PASS / SLOW / STOP / DROP`（DROP＝单条丢弃不升级；STOP＝零速序列 + 通知 Arbiter `safe_hold()`；恢复滞环：STOP 后需连续 5s 全 PASS 才允许回 SLOW/PASS，且需 Arbiter 状态允许）；
   - **SLOW 融合算术（唯一定义）**：`v_cmd = clamp(v_nominal + Δv, SpeedLimits) * α`，`α = 1 - risk_score`，随后再过加速度/jerk 限幅；坐标系＝机体系（与 `Move` 一致）；多提议＝只认最新 `seq`；
   - 与 Arbiter 拓扑（唯一定义，取代 v3.0 两处矛盾表述）：**Gate 与 Arbiter 同进程**（`dog_runtime.py` 装配），Gate 是 Arbiter `move()` 的**唯一调用方**；模型是另一进程，经 UDS 进 Gate。
   - **唯一命令出口的机械保证**：`UnitreeSportClient.move()` 增加 `gate_token` 参数，token 由 Gate 构造时生成并私持；`tests/test_single_exit.py` 静态扫描仓库，`Arbiter.move(` 与 `SportClient.move(` 的调用点只允许出现在 `proposal_gate.py` 与测试目录。
   - **verdict 与实际命令双落盘**：`logs/gate/gate_*.jsonl` 每条含 `GateVerdict` + `applied_cmd (vx,vy,vyaw)`；异步有界队列写盘（满则丢日志不丢命令、计数告警），fsync 每 5s，磁盘 <10% 停写并告警。
3. 故障注入器 `tests/fixtures/fault_injector.py`：故障矩阵＝{乱序,重放,丢包,NaN,越界,旧观测,TTL 超,模型静默,模型假死(pid 在但不发),DDS 断流,温度,电量,租约失效,组合(任取二)}，每类含期望裁决。
4. 新增不变量测试 `tests/test_proposal_gate.py`：
   - I11 模型死亡 ≤100ms 检出（50Hz 环 5 tick），≤250ms 发出计划零速命令（dry-run 下为记录）；
   - I12 越过 `SpeedLimits` 的提议放行数恒 0；
   - I13 dry-run 下 sink 调用次数恒 0（spy 断言，不是字节比对）；
   - I14 STOP 后无「连续 5s PASS + Arbiter 允许」不得恢复；
   - I15 跨 session 的旧 `seq` 重放全部拒绝。

**dry-run 硬锁与解锁流程（唯一定义）**：`ProposalGate(dry_run=True)` 为构造默认且 `dog_runtime.py` 写死传 `True`；解锁＝一个显式 PR 同时改动 ①`dog_runtime.py` 传值 ②勾选 `docs/checklists/gate-unlock.md`（E1/E3/E4 结论文件路径 + D6 影子准入报告路径 + 双人签名行）③CI 校验清单文件存在且字段齐——三者缺一 CI 红。

**验收**
- 回放/仿真 ≥10,000 次故障注入：危险提议（定义＝任何应判 STOP/DROP 的注入样本）放行数＝0；检测时间（注入时刻→verdict 时刻）≤100ms；计划零速（verdict→零速命令入 sink/记录）≤250ms；正常回放（D3 干净数据）误 STOP <1%（分母＝50Hz 裁决 tick 数）。

**依赖**：D2a 回放器；`LoopbackTransport` 可全程离线开发；不依赖 E1/E3/E4。

### D6 影子部署（第 9–11 周）

**交付物**
1. 影子编排：`dog_runtime.py --shadow`——模型实时出提议、Gate 实时裁决（dry-run 硬锁），三路同录：①Gate verdict 流 ②平台导航状态（`PollCache` 快照）③人工遥控/干预标记（现场手持终端一键打点 `tools/mark_event.py`）。
2. 标注协议（写入 `docs/checklists/shadow-labeling.md`）：事件窗口＝10s 不重叠切片；危险类别＝D3 清单定稿（迎面行人/障碍 <1m/地面突变/急转/打滑/其他）；双人独立标注、分歧第三人仲裁；Gate 告警与真值窗口按时间交叠 ≥50% 记命中。
3. `tools/shadow_report.py`：召回/误停/时延/可用率 + 逐失败类别报告。

**验收**
- ≥2 小时、≥10 条路线段；流水线可用率 ≥99.5%（定义：Gate 每 tick 有裁决且输入年龄达标的 tick 占比；计划内暂停不计入分母）；
- 标注 ≥300 个不重叠窗口（≈50 分钟有效素材×多路线）：危险事件召回 ≥95%、误 STOP 率 ≤5%（分母＝标注为无危险的窗口数）；
- 时延保持 D4 门槛；
- 出具《闭环准入报告》：四项全绿才允许申请 D7。**本阶段不做命令级对照**（那需要 E1/G10，诚实地不承诺）。

**依赖**：D4 + D5 + D2b；平台侧只需只读。E1 仍可未决。

### D7 受限闭环 A/B（第 11–13 周 + 审批等待，硬门禁）

**准入条件（缺一不可）**
1. E1 有明确结论（可查任务 → 状态白名单；或不可查 → 位姿距离判据 + 点位坐标补齐）；
2. E3 人工切换流程演练 ≥3 次（含 `ack_human_mode_switch` 审计记录）；
3. E4 四步协议全过（§2：可读/可控/deadman/无抢权）；
4. D6《闭环准入报告》四项全绿；
5. Gate 解锁 PR 按 D5 流程合入；
6. 现场安全审批：封闭场地、实体急停在手、`SpeedLimits` 收紧至 `vx≤0.5, vy≤0.3, vyaw≤0.5`（分轴限制）、电量 >25%、双人在场。

**实验设计（对照臂重定义，符合 I3/I4）**
- **A 臂（基线）**：平台 teach-and-repeat 原生导航走完整路线；
- **B 臂**：同路线拆成「平台导航段 + 1 段 ≤20m 的 `WAM_ACTIVE` 走廊段」——走廊段内由学生模型提议、Gate 裁决、Arbiter 发 `Move`，走廊两侧虚拟墙 + 实体围栏；
- 配对设计：同一路线 A/B 交替、顺序随机；**≥60 次配对试验**（30 对）；
- 操作定义：成功＝到达走廊终点半径 0.5m 内且零越界零人工接管；人工干预＝任何一次实体急停/键盘 STOP/遥控夺权；安全包络越界＝DDS 位姿出走廊多边形或速度回读超限。

**验收**
- 零安全包络越界（一次即终止实验并回 D5 全量注入）；
- B 臂走廊段成功率 ≥90%，且相对 A 臂同段的人工干预率下降 ≥20%（相对比例）；给出 90% 置信区间（配对 bootstrap），区间下界 >0 才算证实；不足＝回 D4 迭代，不部署；
- 全程 episode 落盘且 QA 全过。

### D8 复盘与交接（第 13–14 周）

- 全链路报告：数据→训练→部署→影子→闭环各阶段指标汇总；欠账清单滚动（未过的 E、放宽的阈值、残余风险）；
- 下一期建议（扩走廊/多路线/气检联动）——只写清单，不自动立项。

---

## 4. 接口契约（开发 Agent 照此实现）

```python
# adapters/proposal_types.py（新文件，模型进程与 Gate 共享的唯一契约）
@dataclass(frozen=True)
class RiskProposal:
    session_id: int           # 模型进程启动时刻 epoch_ms（防跨 session 重放）
    seq: int                  # session 内单调递增
    obs_t_mono_ms: int        # 该提议依据的观测时间戳（单调钟）
    issued_t_mono_ms: int     # 提议发出时刻
    risk_score: float         # [0,1]
    action: str               # "pass" | "slow" | "stop"
    dvx: float; dvy: float; dvyaw: float   # 机体系速度修正建议，Gate 侧再次钳位
    model_ver: str            # 对应 artifacts/student_<ver>

@dataclass(frozen=True)
class GateVerdict:
    proposal_session: int
    proposal_seq: int
    decision: str             # "PASS" | "SLOW" | "STOP" | "DROP"
    reason: str               # ok/stale_obs/ttl_expired/nan/bounds/accel/jerk/
                              # model_dead/model_silent/dds_stale/thermal/battery/
                              # lease_invalid/out_of_scope(非WAM_ACTIVE)/dry_run/replay
    applied_vx: float; applied_vy: float; applied_vyaw: float  # 实发命令（dry-run 恒 0）
    t_mono_ms: int
```

- 传输：UDS datagram，单向 模型→Gate；Gate 不回包（模型不依赖 Gate 存活）；
- 单位/坐标系：全部 SI、机体系，与 `SportClient.Move(vx,vy,vyaw)` 一致；
- `FORBIDDEN_KEYS` 红线不变：位姿/点云/图像永不进 `mission_brain.events`；
- 配置唯一入口：`configs/dog/topsee.json`（平台回填）、`configs/dog/gate.json`（阈值）、`configs/dog/dataset_split.json`（数据切分）；
- 测试入口：`tests/test_dog_runtime.py` / `test_edge_bench.py` / `test_episode_qa.py` / `test_proposal_gate.py` / `test_single_exit.py`；全套 `python -m pytest tests/ -q` 必须全绿后才许提 PR。

---

## 5. 安全论证（fail-safe 摘要）

| 失效 | 行为 | 兜底 |
|---|---|---|
| 模型进程死/假死 | Gate ≤100ms 检出 → HOLD/零速 | Gate 独立于模型进程 |
| Gate 进程死 | Arbiter 命令看门狗超时 → 速度归零（I5）；`WAM_ACTIVE` 租约 TTL 过期 → `SAFE_HOLD` | 🔴 **狗侧 deadman 尚未验证**（E4-C②）。宇树无 `Move` 机器人侧超时的权威文档；在实测或书面保证到手前，本行只有软件侧兜底，**不得**据此声称硬件会自动停 |
| i7 整机死 | 🔴 软件侧全失效。当前**唯一**兜底＝**实体急停/遥控器**（现场双人必备，不可缺席） | 原文依赖的 deadman 未验证；且 DDS `StopMove`/`Damp` 已因集成商政策不可用（§1.5-V1），运动级急停缺口须在 D7 前补上 |
| STOP 不确认 | `safe_hold()` 后 1s 内 `/sportmodestate` 速度模 >0.1 m/s → 升级 `Damp` + 现场告警 | 新增：STOP 闭环确认（D5 实现） |
| 日志 I/O 阻塞 | 异步有界队列，丢日志不丢命令 | 磁盘水位 <10% 停录制 |
| 电量/温度未知 | fail-closed，拒绝放行 | D0 Q4 修复 |

---

## 6. 红线（违者回退，无豁免）

1. **模型永不直接发 `Move`**——唯一出口＝ProposalGate（`test_single_exit.py` 机械保证）；解锁走 D5 三件套 PR 流程。
2. **DDS 位姿只进 WAM 落盘通道**，绝不进 `mission_brain.events`（`test_no_pose_leaks_into_events` 守护）。
3. **速度安全盒由我们实现**（F30）；`SpeedLimits` 放宽必须显式 PR + 双人确认；D7 期间反向收紧。
4. **i5 平台机绝不动**；Jetson 不擅自升 JetPack。
5. **禁止伪造人工输入**：气体台账、`ack_confidence` 填假值＝拆安全门禁。
6. **QA 不过的数据禁止训练**，无豁免通道。
7. `MISSION_NAV` 期间 Gate 只做观察者与任务级停止，**永不发 Move**（I3/I4 延续）。
8. **本仓库资产合规**：本项目代码、数据、配置、模型、checkpoint、日志及公司硬件采集结果不得进入无关仓库、个人设备或未经批准的外部系统；外部资产也不得在来源、许可和审批不明确时混入本仓库。仅可复用许可证允许且来源可追溯的公开方法与指标定义（见 §0.3 / §0.4）。

## 7. Kill criteria

- 任一工作流两周无可运行工件+原始测量 → 停该工作流。
- 学生模型两轮优化仍超 P95 预算或 INT8 掉点超限 → 换模型/再蒸馏，**不靠买 AGX 解决**。
- E1/E3/E4 阻塞闭环超两周 → 狗停在影子模式（影子本身已是可交付成果），闭环验证转仿真/无人机床。
- Gate 出现一次危险提议放行或一次 STOP 未闭环确认 → 闭环资格清零，重跑 D5 全量注入。
- 第三方平台需长期逆向私有 API 才能拿基本状态 → 平台降级为 topo 命令源，不再深度耦合。

## 8. 与无人机/服务器的关系（一段话）

无人机栈冻结新功能，仅作「近端伴飞边端」故障注入床：`edge_bench` / `episode_qa`（各自 schema）/ Gate 降级逻辑必须在 Tello 链路复用验证；一旦需要 Tello 专属新框架即砍。服务器＝教师与离线评测环境：负责伪标签、蒸馏、批量回归，并产出可复现、版本化、可审计的离线工件；不再出现在任何实时控制路径。

---

## 附录 A. 阶段-依赖总览

```text
D0 地基(DDS订阅/接线/审计/fail-closed) ──┬─→ D1 盘点+探针+harness ─→ D2a QA/回放(离线)
                                        │                             ↓
                                        └────────────→ D2b 真机录制 ─→ D3 数据采集(≥40段)
                                                                       ↓
                          D5 ProposalGate(离线并行) ←─ 回放器        D4 教师→学生产线
                                        ↓                             ↓
                                        └──────→ D6 影子部署(dry-run) ←┘
                                                        ↓  (E1+E3+E4+准入报告+解锁PR+安全审批)
                                                 D7 受限闭环 A/B ─→ D8 复盘
```

## 附录 B. 外部等待项（不计入工期，提前触发）

1. 增购 Jetson Orin NX 16GB（D1 决策后立即走采购）；
2. 厂商问询：部署形态/写权限、G7 取值、深度流协议（v2 §10 清单继续有效）；
3. 封闭场地与安全审批（D7 前 4 周发起）；
4. 双人现场排期（D2b/D3/D6/D7 均需两人）；
5. **集成商接口契约（2026-08-07 新增，D7 的关键路径）**——问的是契约等级，不是调用示例：

| # | 必答项 | 为什么这条决定架构 |
|---|---|---|
| B5-1 | 运控接口最高下发频率、端到端时延 | ≥10Hz 连续速度 → 快通道可换载体，D7 存活；只有离散动作或 1–2Hz → D6 影子封顶 |
| B5-2 | 停发命令后的归零行为与时间（deadman） | §5 的失效兜底目前建在未验证假设上 |
| B5-3 | 能否读回**实际**速度 | 无回读则 STOP 无法闭环确认，`safe_hold` 形同虚设 |
| B5-4 | 权属获取/释放机制，是否原子、并发如何拒绝 | 决定 E4-C③ 能否通过；也直接暴露他们内部的仲裁设计 |
| B5-5 | 调运控接口时导航栈如何让路 | 这与我方 `DogControlArbiter` 是同一个问题，答案可直接对齐 |
| B5-6 | MQ：broker/协议/认证/QoS、`robotNum`↔`robotId`、与 `pointsId` 的关联键、结果码枚举 | 没有结果码就无法区分「到达/失败/取消」，`complete=true` 不可单独判成败 |
| B5-7 | i7 用户机是否允许运行我方 Gate/Arbiter/recorder/模型进程 | §1.4 算力分工的前提；需与 D1 现机盘点合并确认 |

6. **本机现状盘点前置**：B5-7 依赖 D1 盘点结论（本机被改装过，i7 是否在位/可访问未知，§1.3）。**盘点未做完，接口问询的第 7 条无法闭环**。

## 附录 C. 评审记录

- 对抗评审：2026-08-05，v3.0 判定需要修订，后续重写为 v3.1；
- 阶段范围评审：2026-08-06，补充 §0.3 分阶段交付边界、§3.0 停止线、D0.0/D0-W1、D5 骨架边界及资产合规红线；
- 公开脱敏评审：2026-08-06，移除仓库外私人规划指针与旧优先级表述，新增 §0.4；
- **接口政策修正评审：2026-08-07（v3.3 → v3.4）**。触发＝集成商书面答复。独立评审意见由 OpenAI Codex CLI 出具（含硬件替代方案检索），原始产物 `artifacts/d0_vendor_codex.txt`、`artifacts/d0_hw_candidate_codex.txt`。采纳：E4 拆分与第④步安全重设计、§5 deadman 错误陈述更正、MQ 双源交叉校验、D0 三级验收。**评审局限须记录**：该轮提示词中已包含「先别买硬件」的倾向与理由，故其采购结论不构成独立佐证；其市场价格与引用来源未经核实，不得用于采购决策。技术层结论（机型 API 差异、字段残缺）已由本仓库 SDK 实测独立坐实；
- SDK 实测：2026-08-07，`unitree_sdk2py` 的 go2/b2 `SportClient` 方法表、`SportModeState_`/`LowState_`/`IMUState_`/`MotorState_` 字段表，坐实 Q7/Q8。另记录一条平台限制：`unitree_sdk2py.utils.thread` 依赖 Linux 专有的 `timerfd_create`，**macOS 不可用**（我方 `DdsTransport` 不依赖它，仅影响需要该模块的官方示例/仿真桥）；
- 已吸收的主要修正：实验编号统一（§2）、D7 对照臂重设计、Q1–Q6 纳入 D0、fail-safe 论证、测量协议、总工期 12–14 周及 09-14 前阶段截断；
- 本文及仓库内变更记录构成公开工程决策依据，不引用仓库外私人规划或评审材料。

## 附录 D. 给开发 Agent 的开工约束（可贴粘）

```text
1. 当前只授权执行 D0；不得开始 D1–D8，也不得顺手搭训练、Gate 控制或闭环代码。
2. 修改前先复现 handover §二测试基线；若结果不同，停止修改并报告差异。
3. 复用方案 §1.1 已落地模块；禁止重写 M6–M11、修改 main.py 或放宽 I1–I10。
4. W1 截止（08-10）只交四项：真实 DDS 实读（或 BLOCKED）、dog_runtime+abort、两类 fail-closed 门禁、topsee 配置骨架。
5. D0 期间禁止发送 Move、运行 E1/E4 动作实验、接真实命令 sink 或解除任何 dry-run。
6. 现场未排导致 DDS 无法实测时，明确标记 BLOCKED；Loopback 通过不得写成「状态流稳定可读」。
7. 阻塞时只补 D0 范围内测试、报告和现场清单；不得用等待为理由跳到 D3/D4/D5/D6。
8. 本仓库代码、数据、配置、模型、checkpoint、日志和硬件采集结果不得进入无关仓库、个人设备或未经批准的外部系统；外部资产须经来源、许可和审批核验后方可引入。
```
