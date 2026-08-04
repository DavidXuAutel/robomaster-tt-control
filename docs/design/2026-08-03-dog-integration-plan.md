# 机器狗（宇树 B2 + 拓普视平台）对接方案 v2

日期：2026-08-03
状态：**可据此开工**（M6 起纯软件，不占设备）
取代：`2026-08-03-dog-platform-control-and-wam-integration.html`（已删除，含事实错误，勘误见 §9）

设备：宇树 Unitree B2，编号 `B2000397`，名称「遣通B2」
平台：广东拓普视 topsee《全地形智能巡检机器人系统》，云端 `:8001` / 离线一体机 `:8888`

## 材料与方法

| 材料 | 用途 |
|---|---|
| `docs/references/机器人巡检平台用户说明书/2026-08-03-dog-inspection-platform-user-manual.md` | 手册整合版，行为语义的权威来源 |
| `docs/references/机器狗文档/机器人模块.openapi.json` | 275 接口 / 154 schema，接口事实的权威来源 |
| `docs/references/机器狗文档/登录模块.openapi.json` | 5 个登录接口 |
| 平台实例 `112.94.22.203:8011` | 只读探活（`curl -k`），验证返回码与鉴权行为 |
| 平台界面截图 12 张 | UI 能力（含 OpenAPI 未导出的本体遥控） |

本方案经三个独立模型交叉评审（能力取舍 / 接口设计 / 红队反驳），关键接口事实由脚本逐条核验。**凡标注「未覆盖」者一律不得在代码中假设其存在。**

---

## 1. 决策摘要

1. **平台定位为语义层与台账层，不是控制底座。** 它提供 LiDAR 地图、teach-and-repeat 点位、拓扑派单、证据图片；不提供实时位姿、连续速度、置信度、气体标定。
2. **WAM 必须走宇树 DDS 直控。** 平台只有拓扑级接口，且手册专章讨论「控制延迟」，云端转发不可能支撑 10–50 Hz 闭环。
3. **两条通道分时互斥，不并行。** 手册锁死「机器狗只能由一个用户控制」，且本体控制必须在「手动巡检」模式下。
4. **模式切换目前无法自动化。** OpenAPI 275 个接口里没有手动/自动巡检切换接口 → 近期 WAM 段只能人工切模式，FSM 必须有显式的「等待人工」状态，不许假装已切换。
5. **到点判定是最大单点风险。** `sendNavigate` 返回裸 `Result` 无 taskId，而 `getCurrentByRobotId` 整个响应结构是围绕「自动巡检任务」设计的。**单点派单是否产生可查询任务，未被证实。** 这条不验证就不要写生产代码。
6. **状态字符串全部无枚举。** 到点判定必须三态（含 `UNKNOWN`）+ 多证据源，不许裸字符串比较后静默当 False。
7. **`dog_goal_id` 不能直接等于平台 `pointsId`。** 中间必须加一层稳定语义标签，`pointsId` 只作可失效缓存。
8. **气体标定无数据源，用人工台账，不得伪造。**
9. **abort 分三级**：任务级 `stopTask` / 运动级 DDS `StopMove`+`Damp` / 灾难级平台急停（≈断电，仅人工）。
10. **规避平台 80% 的业务能力。** 表计、红外测温、声纹、液位计、设备缺陷、巡检报表、告警确认工作流、充电房联动、机械臂、人脸识别、区域-楼层-设备树、cron 任务调度——一律不接。

---

## 2. 事实基线

### 2.1 已核验为真

| # | 事实 | 依据 |
|---|---|---|
| F1 | 成功返回码是 `code:0`，不是 200 | 实测 `GET /service/api/permission/free/security/rsa` 返回 `{"code":0,"message":"success","data":"<RSA公钥>"}`；手册与 OpenAPI 描述的「200 代表成功」不准 |
| F2 | 未登录返回 HTTP 401，body 是纯文本「未登录」而非 JSON | 实测 `GET /service/api/robot/state/getStateData` |
| F3 | 登录三步：取 `securityKey` 对应公钥 → RSA 加密密码 → `POST /permission/free/pc/login` | `登录模块.openapi.json`；第一步已实测可用 |
| F4 | Token 的 header 名、有效期、续期方式**文档完全未写** | `登录模块.openapi.json` 的 `securitySchemes` 为空 |
| F5 | 用户有**授权天数**，到期无法登录 | 手册 §2.2 |
| F6 | `getCurrentByRobotId` 返回 `ShowRobotTaskAllEntity`，字段为 `pointsId`、`currentState`、`totalState`、`taskId`、`taskItemId`、`autoInspectionName`、`rosLine`、`rosMapMan`、`rosPointsMan`、`keyName`、`lineId`、`mapId`、`taskStartDate` | 脚本核验 |
| F7 | `currentPointsId` / `inspectionRate` 属于 `RobotTask`，**仅** `getPagingRobotTask` 返回 | 脚本核验 |
| F8 | `sendNavigate` 响应是裸 `Result`（`data` 为空对象），**不返回任何任务标识** | 脚本核验 |
| F9 | 全库 7 个接口标记 `deprecated`：`analysis/getDevRunLine`、`archivesMan/getChannelData`、`archivesMan/getRtspUrl`、`archivesMan/getTopseeData`、`archivesMan/getWeather`、`robotTaskItem/testPoints`、`state/selectControllerUser` | 脚本核验 |
| F10 | `selectControllerUser` 描述为「02:空闲；04:忙碌」，返回 `ControllerUser{message, result}`，但该接口**已废弃** | 脚本核验 |
| F11 | `updateControllerUser` 参数为 `robotId`、`state`、`force`，三者描述都是含糊的「机器人状态」，**无枚举**，返回裸 `Result` | 脚本核验 |
| F12 | OpenAPI 全文检索「置信度 / confidence / 延迟 / latency」**命中 0 次** | 脚本核验 |
| F13 | OpenAPI 全文检索「标定 / 校准 / calibr」**命中 0 次**；`gasJson` 出现 5 次但均为不透明 blob | 脚本核验 |
| F14 | **没有**手动/自动巡检模式切换接口 | 脚本核验（按 `mode`/`manual`/`patrol`/`手动` 检索路径与 summary） |
| F15 | `mapUpdate` 的 `MapUpdateDomain` 支持 `action` 0 初始化 / 1 重定位 / 2 增量建图 / 3 结束 / 4 取消 / 5 切割 / 6 加载地图，另有 `mapType`（0 室内 / 1 室外）、`destKeyname`（地图编号）、`srcKeyname`（父地图）、`xyth`（`{'x':0,'y':0,'th':0}`）、`robotId`、`pointsId` | 脚本核验 |
| F16 | 平台**没有**对外 WebSocket / MQTT 事件通道；`taskAlarm/alarmCallback` 是平台侧接收 `Alarm` 的 POST 端点，**没有**回调注册、鉴权、重试的任何说明 | OpenAPI + 手册 |
| F17 | 导航被阻挡且未设检修区时，平台告警「导航超时，未找到有效路径」 | 手册 §5.8 |
| F18 | 平台急停「相当于断电」 | 手册 §4.2.1；§5.7 摔倒处理同样要求先切断动力 |
| F19 | 单一控制权：同一时刻仅一个控制人；释放方式是手动→自动巡检，或监控页「释放控制」 | 手册 §4.3.1、§5.2 |
| F20 | 云台与本体控制**均须**先进入「手动巡检」 | 手册 §4.2.1 |
| F21 | 本体控制能力：前后、左右旋、**横移**、身高、速度、站立/下蹲、爬坡步态（即全向 vx/vy/ωz） | 手册 §4.2.1 |
| F22 | 重定位后**置信度 ≥ 0.9** 才允许发任务；低于此值狗会乱走 | 手册 §4.4.1、§4.5.3、§5.5 |
| F23 | 点位是 teach-and-repeat：用本体+云台把狗开到目标位姿再保存；保存云台（水平/垂直/倍率/焦距）与本体（航向/俯仰/横滚/身高） | 手册 §4.4.2 |
| F24 | 相邻点位间距须 > 0.5 m，或旁有明显固定参照物 | 手册 §4.5.3、§5.4 |
| F25 | 建图要求：特征明显区域、玻璃前置实物遮挡、速度 ≤ 0.3 m/s、排除移动物体、直线 + 90° 转弯、起点闭环 | 手册 §4.4.1 |
| F26 | **同步** = 把 PC 端改动写入机器人地图源文件；**发布** = 机器人开机自动拉取最新改动。只编辑不同步/不发布，开机不会用新信息 | 手册 §4.4.1 |
| F27 | 视频源含可见光 / 红外热成像 / **深度相机** | 手册 §4.2 |
| F28 | 离线版连云端后，策略/楼层/气体/联动设备/设备/任务管理等**禁止新增**，以云端为配置主源 | 手册 §0、§4.1、§4.4.5 |
| F29 | 电量分层：低于两格建议停机，仅一格闪烁须立即停机 | 手册 §1.5 |
| F30 | 遥控档位软限位（低速档爬坡 <45°、台阶 <25 cm、越障 <40 cm）低于整机极限（爬坡 45°、台阶 40 cm、最大速度 5 m/s） | 手册 §3.4.4、附录 6.2 |
| F31 | 反光柱/充电桩移动后必须重新标定；灰尘削弱雷达反射会导致找不到桩 | 手册 §5.10 |
| F32 | B2 整机：站立 1098×450×645 mm、机身 60 kg、持续负载 40 kg、IP67、空载续航 5 h/20 km | 附录 6.2 |

### 2.2 文档未覆盖（禁止假设）

| # | 缺口 | 阻塞什么 |
|---|---|---|
| G1 | 单点 `sendNavigate` 是否产生可被 `getCurrentByRobotId` 查询的任务 | **整条慢通道**（最高优先级） |
| G2 | `currentState` / `robotState` / `resultState` / `status` 的取值枚举 | 到点与失败判定 |
| G3 | 实时位姿（平面 `x/y/th`）接口 | WAM 位姿源的备份、到点的距离判据 |
| G4 | 定位置信度接口（F12：全库 0 命中） | `≥0.9` 安全门禁无法自动化 |
| G5 | 控制延迟接口 | WAM 通道准入判据 |
| G6 | 手动/自动巡检模式切换接口（F14） | 双通道自动交接 |
| G7 | `updateControllerUser` 的 `state`/`force` 实际取值（F11） | 控制权抢占/释放 |
| G8 | Token header 名与续期（F4） | 长驻客户端 |
| G9 | 气体标定时间来源、是否能「立即采样一次」（F13） | `GasBackend` 两个方法 |
| G10 | 本体遥控（速度下发）的协议、URL、载荷、频率上限 | 备用快通道 |
| G11 | 第三方直连宇树 DDS 是否被边缘侧独占；释放平台控制后 `Move` 是否生效 | **方案 C 生死线** |
| G12 | 深度相机对外流的协议、编码、内参、时间戳、尺度 | 深度数据可用性 |
| G13 | `RosMapMan.mapSpot`（地图原点，string）的解析格式；`RosPoints.th` 单位（度/弧度） | 坐标对齐 |
| G14 | `RosPointsTag.tagFamily` 取值与识别算法 | 与我方 AprilTag 能否统一 |
| G15 | 现场实际部署形态（`:8001` 云端 vs `:8888` 离线一体机）与写权限（F28） | 自动化闭环是否成立 |
| G16 | `ShowRobotTaskAllEntity` 被 `getRobotMapAll`(×2) 与 `getCurrentByRobotId` 三个语义不同的接口共用为响应类型，疑似导出标注错误 | 地图导出脚本的解析假设 |

---

## 3. 能力取舍矩阵

判断标准只有一条：**是否服务于「MissionBrain 可编排的执行体」+「WAM 训练数据」这两个诉求。**

### 3.1 必用（不用就要重造且做不好）

| 能力 | 接口 / 依据 | 理由 |
|---|---|---|
| LiDAR SLAM 地图资产 | `mapMan/*`；手册 §4.4.1 | 厂商导航栈的权威地图，我们绝不自建稠密 SLAM |
| 建图 / 重定位 / 加载地图 | `POST mapMan/mapUpdate`，`action` 0–6 + `xyth`（F15） | **可程序化**，能进 preflight 自动化 |
| 虚拟墙 / 检修区 | `avoid/*`（`RosAvoidStrategy` 的 min/max X/Y/Height）；手册 §4.4.1 | 唯一的静态安全围栏；临时障碍靠检修区避免导航超时（F17） |
| 同步 / 发布 | `mapMan/syncMap`、`mapMan/updateState`（F26） | 地图改动不同步等于没改，必须进运维 SOP |
| teach-and-repeat 点位 | `robot_point/*`（ROS 位姿点）、`point/*`（巡检业务配置） | 我们唯一能指挥狗去的目标 |
| 单点拓扑派单 | `POST point/sendNavigate?robotId=&pointsId=` | `NavBackend.goto_goal()` 的首选落点（待 E1 验证） |
| 任务停止 / 暂停 / 返桩 | `robotTask/stopTask`、`pauseTask`、`back` | `NavBackend.cancel()` 与任务级 abort |
| 地图/线路/点位导出 | `GET mapMan/getRobotMapAll?robotId=` | 反向生成 SharedMap 绑定（注意 G16） |
| 视频取流 | `GET video/getStreamUrl`（**不要**用已废弃的 `archivesMan/getRtspUrl`，F9） | 本地视觉与证据留存 |

### 3.2 可用但需包装

| 能力 | 包装成什么 | 注意 |
|---|---|---|
| 控制人查询/修改 | `Arbiter` 的租约层 | `selectControllerUser` 已废弃（F10），改用 `updateControllerUser`，但取值待抓包（G7） |
| 告警查询 | `PerceptionBackend` 的旁路证据源 | `taskAlarm/getAlarmList`/`getAlarmHistory` 只读；`Alarm.srcImage`/`rstImage` 作 `evidence_uri`。**不驱动任务状态机** |
| 气体历史 | `GasBackend.sample()` 语义改为「窗口聚合查询」 | 没有「立即采样」；标定另走台账（§5.4） |
| 巡检结果记录 | 证据索引 | `robotTaskItem/getTaskLog` 的图片 URL 可用；识别值不用 |
| 自动巡检任务 | **仅**作 `sendNavigate` 失败时的 fallback 派单方式 | `addAutoInspection` + `executeNow` 能换来正规 taskId，代价是延迟与配置污染 |

### 3.3 明确规避（平台复杂但我们用不上）

| 规避对象 | 为什么 |
|---|---|
| 表计读数、红外测温、声纹监测、液位计、边界检测配置 | 电力/工业巡检业务，与我们的 `target_label` 语义无关；标注流程繁琐（表计要按序取圆周三点、液位计要按序点四角），接进来即被锁进厂商业务模型 |
| 设备缺陷统计、巡检分析报告、巡检效率分析、报表导出 | 纯运营报表 |
| 告警确认工作流（确认/撤销/批量/误报标记） | 人工运营流程，不是机器契约 |
| 任务 cron 调度、优先级、线路编排、回充策略 | 会与 MissionBrain 形成**双调度源**，超时/取消/幂等语义打架 |
| 区域 / 楼层 / 设备 / 联动设备（卷帘门、电梯、充电屋门）树 | 现场配置强耦合，且离线连云后禁止新增（F28） |
| 机械臂（J1–J6、教学回放）、人脸识别、声像仪 | 本机未配备或需求未覆盖；人脸识别手册正文缺失，属未文档化能力 |
| 应用版本管理 / 升级 / 重启 | 运维职能，不是集成职能 |
| 充电房管理 | 只用 `robotTask/back` 一键返桩即可 |

### 3.4 必须绕过（平台做不到）

| 需求 | 绕行方案 |
|---|---|
| 连续速度控制（WAM 动作输出） | 宇树 `SportClient.Move(vx,vy,vyaw)` API 1008 |
| 高频位姿 / 本体感受 | DDS `/sportmodestate`（position/velocity/yaw）、`/lowstate`（≈500 Hz，12 电机 + IMU + 足底力） |
| 运动级安全停 | `StopMove` 1003 / `Damp` 1001（**不是**平台急停，F18） |
| 姿态与步态 | `StandUp` 1004 / `StandDown` 1005 / `SwitchGait` 1011 / `BodyHeight` 1013 / `SpeedLevel` 1015 |
| 气体标定溯源 | 人工标定台账（§5.4） |
| 定位置信度门禁 | 近期人工确认 + 审计留痕（G4） |

### 3.5 三个陷阱

1. **把 `currentPointsId` / `inspectionRate` 当位姿用。** 它们是点位级离散进度，且不在 `getCurrentByRobotId` 里（F6/F7）。用来判到点勉强够，用来训 WAM 完全不够。
2. **把 `alarmCallback` 当成「可配置的 webhook」。** OpenAPI 里它是平台侧的接收端点，没有任何注册/鉴权/重试说明（F16）。近期只能轮询。
3. **把「深度相机可选」当成「能拿到可训练的标定深度流」。** 拿到一个播放 URL ≠ 拿到带内参和时间戳的深度帧（G12）。

---

## 4. 目标架构

### 4.1 三层分工

```
┌─────────────────────────────────────────────────────────┐
│ MissionBrain（任务层，分钟级）                            │
│  FSM: IDLE→SCOUTING→DOG_NAV→DOG_SEARCH→GAS_SAMPLE→…     │
│  事件契约 v1 冻结不动（events.py 一行不改）               │
└────────────────────────┬────────────────────────────────┘
                         │ dog.inspect / gas.sample / dog.abort
                         ▼
┌─────────────────────────────────────────────────────────┐
│ DogControlArbiter（仲裁层，新增）                         │
│  单命令所有者 + 租约 + preflight 门禁 + 看门狗            │
└──────────┬───────────────────────────────┬──────────────┘
           │ 慢通道（互斥）                 │ 快通道（互斥）
           ▼                               ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ TopseeNav/Perception/Gas │  │ UnitreeSportClient (DDS) │
│ HTTP 1–2 Hz 轮询          │  │ Move / 位姿 / StopMove    │
│ 拓扑派单、证据、气体历史   │  │ WAM 动作级闭环 10–50 Hz   │
└──────────────────────────┘  └──────────────────────────┘
```

关键约束：`Arbiter.topsee_cmd_enabled` 与 `Arbiter.unitree_cmd_enabled` **永不同时为真**。

### 4.2 Arbiter 状态机

```
                  acquire_for_mission
   ┌──────┐        ┌───────────┐   门禁通过   ┌─────────────┐
   │ IDLE │───────▶│ PREFLIGHT │────────────▶│ MISSION_NAV │
   └──────┘        └─────┬─────┘             └──────┬──────┘
      ▲                  │ 门禁失败                  │ arrived 且需 WAM 段
      │                  ▼                          ▼
      │              ┌───────┐          ┌───────────────────────────┐
      │              │ FAULT │          │ WAITING_HUMAN_MODE_SWITCH │
      │              └───┬───┘          └──────────┬────────────────┘
      │                  │                         │ 人工已切手动巡检 + DDS 探测通过
      │                  │                         ▼
      │                  │                  ┌────────────┐
      │                  │                  │ WAM_ACTIVE │
      │                  │                  └──────┬─────┘
      │                  │      完成 / abort / TTL  │
      │                  │                         ▼
      │                  │           ┌──────────────────────┐
      │                  └──────────▶│ HANDOVER_TO_MISSION  │
      │                              └──────────┬───────────┘
      │        ┌───────────┐                    │ 已释放
      └────────│ SAFE_HOLD │◀── 断网/崩溃/被抢 ──┘
               └───────────┘
```

与初版方案的最大区别：**`HANDOVER_TO_WAM` 被替换为 `WAITING_HUMAN_MODE_SWITCH`**。因为 F14 已核验没有模式切换接口，自动交接现在**做不到**，必须停在显式的等待人工态，不许假装切换成功后直接下发 `Move`。

### 4.3 preflight 门禁

| 检查 | 数据源 | 门槛 | 近期可自动化？ |
|---|---|---|---|
| 控制人空闲 | `updateControllerUser`（取值待 G7） | 空闲或成功抢占 | 待抓包 |
| 无冲突任务 | `getCurrentByRobotId` | 无执行中任务，否则先 `stopTask` | 是 |
| 定位置信度 | **无接口**（G4） | ≥ 0.9 | **否，人工确认 + 留痕** |
| 控制延迟 | **无接口**（G5） | 建议 >200 ms 告警、>500 ms 禁 WAM | 否 |
| 电量 | `state/getStateData` 的 `battery` | 高于两格（F29） | 是 |
| DDS 权限 | `probe_dds_authority()` | `StopMove` 有响应 | 是（需 G11 前置结论） |

置信度不达标时可程序化补救：`POST mapMan/mapUpdate` 带 `action=1`（重定位）+ `xyth`（F15）。但**重定位是否成功仍无法程序化读取**，只能人工看 UI。

### 4.4 abort 三级

| 级别 | 触发者 | 动作 | 禁止 |
|---|---|---|---|
| 任务级 | `MissionSupervisor` → `dog.abort()` | `Arbiter.force_release()` → `StopMove` + `stopTask` + 吊销租约 + best-effort 释放控制人 | — |
| 运动级 | Arbiter 看门狗 / WAM 本地危险 | DDS `StopMove` 1003，必要时 `Damp` 1001 | 不得越级发 `mission.abort` |
| 灾难级 | **人工** | 平台急停 / 遥控器 `L2+B` 阻尼 | **禁止程序调用**（≈断电，F18） |

保持现有不变量：`mission.abort` 的所有权仍**只属于** `MissionSupervisor`，`DogAdapter` 不监听该事件；WAM 只能调 `Arbiter.safe_hold()`。

### 4.5 安全不变量（写成 pytest 断言）

```python
# I1  单命令所有者
assert not (arbiter.topsee_cmd_enabled and arbiter.unitree_cmd_enabled)
# I2  无有效租约不得下发速度
assert unitree.last_move_token in (None, arbiter.lease_token)
# I3  WAM_ACTIVE 期间零次 sendNavigate
assert arbiter.state != "WAM_ACTIVE" or topsee.navigate_calls_in_state == 0
# I4  MISSION_NAV 期间零次 Move
assert arbiter.state != "MISSION_NAV" or unitree.move_calls_in_state == 0
# I5  命令看门狗超时后速度必须归零
assert unitree.cmd_after_watchdog == (0.0, 0.0, 0.0)
# I6  未确认人工切换模式，不得进入 WAM_ACTIVE
assert arbiter.state != "WAM_ACTIVE" or arbiter.human_mode_ack is True
# I7  abort 后租约必须吊销
assert (not dog.abort_called) or arbiter.lease_token is None
# I8  平台急停不在常规停止路径里
assert "platform_estop" not in arbiter.normal_stop_methods
# I9  置信度未达标不得通过 preflight
assert (not arbiter.preflight_ok) or arbiter.confidence_ack >= 0.9
# I10 SAFE_HOLD / FAULT / IDLE 下两通道使能均为假
assert arbiter.state not in ("SAFE_HOLD", "FAULT", "IDLE") or arbiter.no_owner
```

---

## 5. 契约变更（对现有代码的最小侵入）

**不变的**：`mission_brain/events.py`（含 `FORBIDDEN_KEYS`）、`mission_brain/brain.py`、`mission_brain/supervisor.py`、`adapters/dog_base.py`、`NavBackend`/`PerceptionBackend`/`GasBackend` 三个 Protocol 的**方法签名**。

以下是必须做的四处改动。

### 5.1 到点判定：三态 + 多证据 + 可上报故障

`NavBackend.is_arrived() -> bool` 签名不变，但 `TopseeNav` 内部维护三态并新增一个**可选**故障钩子：

```python
class NavStatus(str, Enum):
    ARRIVED = "arrived"
    EN_ROUTE = "en_route"
    UNKNOWN = "unknown"   # 平台字符串不在白名单内 → 不信任

class TopseeNav:
    def is_arrived(self) -> bool:
        """三条独立证据，任一为 ARRIVED 且不冲突才返回 True：
        1. currentState 命中可配置白名单（取值待 G2 实测）
        2. ShowRobotTaskAllEntity.pointsId == 目标 pointsId
        3. （可选）DDS 位姿与点位 x/y 距离 < arrive_radius_m
        UNKNOWN 连续计数超 unknown_tolerance 次 → 记录 nav fault。
        """

    def poll_fault(self) -> Optional[str]:
        """可选扩展钩子。返回 nav 失败原因或 None。
        取值：nav_status_unrecognized / nav_timeout / controller_lost /
              platform_unreachable / goal_not_in_active_map
        """
```

`adapters/dog_sdk.py` 的 `tick()` 里加一段向后兼容探测（**不改 Protocol**）：

```python
fault = getattr(self.nav, "poll_fault", lambda: None)()
if fault and not self._arrived_emitted:
    self._emit(make_event(EventType.DOG_INSPECT_FAILED, ...,
        payload={"region_id": ..., "stage": "nav", "reason": fault}))
```

理由：现状下平台字符串对不上就会静默返回 `False`，一路死等到 supervisor 的 `stage_timeout_s`，把「接口语义不匹配」误报成「狗走得慢」。F17 的「导航超时，未找到有效路径」也需要独立 reason。

### 5.2 SharedMap 加一层稳定绑定

`pointsId` 是平台自动生成的不透明串（现场形如 `快速打点-1785465994716`），地图重建或重打点后会失效，而 `SharedMap.from_dict()` 只校验 JSON 自洽，**不校验目标在平台侧是否存在** → 失效在软件层完全不可见。

```json
{
  "version": 2,
  "frame": "dog_map",
  "regions": {
    "region_x": {
      "region_id": "region_x",
      "dog_goal_label": "wp_region_x_staging",
      "drone_route_id": "route_region_x_scout",
      "anchor_ids": ["AX-01", "TAG-0"]
    }
  },
  "platform_binding": {
    "map_id": "<RosMapMan.mapId>",
    "map_version": "<RosMapMan.mapVersion>",
    "exported_at": "2026-08-03T14:00:00+08:00",
    "goals": {
      "wp_region_x_staging": {
        "points_id": "快速打点-1785465994716",
        "points_name": "…",
        "x": 12.4, "y": 5.1, "th": 0.52
      }
    }
  }
}
```

- `dog_goal_label` 是**我们的**语义资产，人写、稳定、进 git。
- `platform_binding` 是**缓存**，由导出脚本生成，带地图版本与时间戳。
- 每次现场重打点/重建地图后重跑导出脚本，脚本必须输出 diff 报告，人工确认没有 region 静默丢失。
- 加载时校验：`map_id` 与当前机器人加载的地图一致，且每个 label 都有绑定，否则拒绝启动。

### 5.3 HTTP 不得阻塞 tick 循环

现状：`EventBus.publish()` 是进程内同步调用，`MissionSupervisor` 约定单线程 tick，demo 里 `dog.tick()` / `brain.tick()` / `scout.process_frame()` 在同一个 10 Hz 循环里顺序执行。若 `is_arrived()` 直接同步 HTTP：轮询频率被 tick 放大到 10 Hz（超出 1–2 Hz 预算、可能触发网关限流），且一次卡顿会连带拖慢无人机侧的超时检测，**连 abort 路径本身都会被拖慢**。

规定：

- `TopseeClient` 内置**独立轮询线程**，按 1–2 Hz 刷新状态缓存。
- `is_arrived()` / `search_target()` 只读缓存，附带 `age_ms`；超过 `stale_ms` 一律返回 `UNKNOWN`。
- `cancel()` 走**同步 HTTP + 2 s 短超时**；超时后仍把本地状态置为「已尝试取消」，绝不无限阻塞调用方。

### 5.4 气体：标定走人工台账，不许伪造

`calibration_at()` 在平台侧无任何数据源（F13）。两个坏选项——返回当前时间（门禁形同虚设，安全后门）或返回很早时间（气检永久瘫痪）——都不接受。决策：

```yaml
# configs/mission/gas_calibration.yaml（运维维护，进 git）
sensors:
  - robot_id: B2000397
    device_id: gas_rs485
    channels: [CH4, H2S, CO, O2]
    calibrated_at: 2026-07-20T09:30:00+08:00
    calibrated_by: 张某
    certificate: docs/references/gas-cal/2026-07-20.pdf
```

- `TopseeGas.calibration_at()` 读该台账；台账缺失或过期 → 照现有 `dog_sdk.py` 逻辑发 `GAS_FAILED`，但 reason 细分为 `calibration_source_unavailable`（无台账）与 `calibration_stale`（台账过期）。
- `sample(window_s)` 语义如实改为「按 `[now-window_s, now]` 查 `gas/getGasHistory` 并聚合」，`GAS_COMPLETED` 的 payload 里注明数据来自历史聚合而非即时采样。
- 若客户要求真正的即时采样，只能接气体传感器的原始 Modbus/RS485 链路（绕过平台）。

---

## 6. WAM 数据采集规格（机器狗版 Episode）

对齐无人机侧 `EpisodeRecorder` 的既有约定：一目录一 episode，视频第 N 帧对应 `frames.csv` 第 N 行，**主时钟按深度帧 ts 变化计帧，不按控制循环计**。

### 6.1 目录布局

```
logs/dog_episodes/ep_YYYYMMDD_HHMMSS/
├── meta.json          # 含 T_map_from_odom、租约日志、outcome
├── frames.csv         # 逐帧主表
├── video_rgb.mp4      # 可见光（主观测）
├── video_thermal.mp4  # 可选，红外
├── depth/             # 深度帧（png 或 npy）
├── lowstate/          # /lowstate 降采样至 50 Hz
└── README.md          # 交付必附（对齐保证、动作列语义、ctrl_state 取值）
```

### 6.2 `frames.csv` 字段

| 字段 | 单位 / 坐标系 | 来源 | 频率 |
|---|---|---|---|
| `frame_id` | — | recorder，深度新帧递增 | 主时钟 |
| `t_mono_ms` / `t_utc_ms` | ms | host | 随帧 |
| `depth_ts` / `depth_age_ms` | ms | 深度源；age = host 收到时刻 − ts | 主时钟 |
| `rgb_path` / `has_rgb` | — | `video/getStreamUrl` 抽帧 | 随帧 |
| `depth_path` / `has_depth` | 深度相机光学系 | 平台深度通道或机载话题（G12） | 主时钟 |
| `pose_x` `pose_y` `pose_z` `pose_yaw` | m / rad，**odom(sport) 系** | DDS `/sportmodestate` | ≥10 Hz 取最近 |
| `vel_x` `vel_y` `vel_yaw` | m/s、rad/s，body 系 | `/sportmodestate` | 同上 |
| `pose_age_ms` | ms | recorder | 随帧 |
| `map_x` `map_y` `map_yaw` | m / rad，**平台地图系** | `T_map_from_odom · pose_odom`；变换无效则留空 | 同上 |
| `loc_confidence` | [0,1] | **人工录入或抓包后补**（G4） | 向前填充 |
| `bat_pct` `temp_c` | % / ℃ | `/lowstate` 或 `state/getStateData` | 1 Hz |
| `act_vx` `act_vy` `act_vyaw` | m/s、rad/s，body，**下发命令非实测** | WAM / teleop → `Move` | 与帧对齐 |
| `act_source` | `TELEOP`/`WAM`/`HOLD`/`MISSION_TOPO` | Arbiter | 随帧 |
| `ctrl_state` | `TELEOP`/`WAM`/`HOLD`/`MISSION_NAV`/`SAFE_HOLD` | Arbiter | 随帧 |
| `gait` `speed_level` `posture` `body_height_cmd` | 枚举 / int / m | 命令侧，事件时更新 | 事件 |
| `lease_state` | Arbiter 状态名 | Arbiter | 随帧 |
| `near_left` `near_mid` `near_right` | [0,1] | 本地深度分带 nearness（沿用无人机习惯） | 随深度帧 |

`/lowstate`（≈500 Hz，12 电机 + IMU + 足底力）降采样到 50 Hz 单独落盘，**不进 frames.csv 行内**。

### 6.3 时间对齐规则

1. 主时钟 = `depth_ts` 变化。`depth_ts` 未变则不新开 CSV 行（控制环可以更快，但不写帧）。
2. RGB 取与 `depth_ts` 差最小且 `|Δt| < sync_slack_ms`（建议 50 ms）的帧；超差则 `has_rgb=0`，深度行照写。
3. 动作写「该深度帧时刻之前最后一条已下发且未过期的 `Move`」；距今超过 `cmd_watchdog_s`（0.3 s）则动作记 0 且 `ctrl_state=HOLD`。
4. 位姿写 `/sportmodestate` 最近样本，并记 `pose_age_ms`。
5. **禁止**把三路抖动的流做「平均时间戳」当主键。

### 6.4 动作空间与无人机 `RcAxes` 对照

| 无人机 `RcAxes` | 机器狗 | 说明 |
|---|---|---|
| `pitch` > 0 前进 | `vx` | 语义一致 |
| `roll` 左右横移 | `vy` | B2 明确支持横移（F21） |
| `yaw` | `vyaw`（ωz） | 语义一致 |
| `throttle` 升降 | **无直接对应** | 改用 `body_height_cmd` + 站/蹲离散量 |
| — | `gait` / `speed_level` / `posture` | 无人机没有；用整数码或 one-hot 单列存，**不塞进连续动作向量** |

训练用连续动作 `a_t = [vx, vy, vyaw]`（SI 单位，非无量纲杆量），裁剪到安全盒。采集起步建议 `|v| ≤ 1.0 m/s`，远低于整机 5 m/s。

**重要安全缺口**：遥控器档位的软限位（爬坡 <45°、台阶 <25 cm、越障 <40 cm，F30）在 DDS `Move()` 直控路径上**不生效**。我们必须在自己的控制律里重新实现这层保护，否则 WAM 动作头可能让狗尝试超出安全限位的坡度或台阶。

### 6.5 坐标系与地图对齐

- 训练主源是 DDS `/sportmodestate` 的 odom 系；平台**没有**实时位姿接口（G3）。
- 平台地图系原点是 `RosMapMan.mapSpot`（string，解析格式待 G13），点位坐标是 `RosPoints.{x,y,th}`（`th` 单位待 G13）。
- 对齐流程：把狗停在已知点位 `P` → 同时记录 `pose_odom` 与 `P` 的平台坐标 → 估平面刚体变换 `T_map_from_odom`（x, y, yaw）→ 写入 `meta.json`。
- 定位丢失（置信度 < 0.9）时 `map_*` 置空，仍可用 odom 训短时序，但 README 必须显式声明 `odom_only`。

### 6.6 为什么 `FORBIDDEN_KEYS` 红线在这里依然正确

`events.py` 禁止 `global_pose` / `pose_xyz` / `point_cloud` / `covariance` / `video_b64` / `depth_b64` / `transform` / `T_world` 进事件总线。有人会认为「WAM 需要位姿，所以这条红线是设计矛盾」——不是：

1. `EventBus.publish()` 是同步调用，50–500 Hz 的位姿加图像会直接拖死确定性任务 FSM。
2. Brain 只需要 region / goal / evidence_uri 这一层语义；稠密几何属于 recorder 与 WAM runtime。
3. odom 未与地图对齐时冒充 `global_pose`，会让无人机 Scout 与狗的 goal 错配——这是最危险的失败模式。
4. 证据用 URI 传递（`Alarm.srcImage` 等 URL）已足够，不需要 base64。

**因此 WAM 数据流完全绕开 `mission_brain`，直接落盘。** 这条必须写进 `adapters/dog_unitree.py` 的模块 docstring：位姿数据只准写入 WAM 落盘通道，禁止经过 `mission_brain.events`。否则将来有人为了调试顺手发一个 `dog.debug` 事件带上 x/y/yaw，`validate_event()` 会在运行时直接炸掉整条 mission。

### 6.7 Episode 验收门槛

新建 `tools/dog_episode_check.py`（**不要**复用 wander 的 `episode_check.py`，它的 turn 事件门槛不适用，同 orbit 的教训）。

| 检查项 | 门槛 |
|---|---|
| `frame_match` | `len(frames.csv) == video_rgb 帧数` |
| `depth_p95_ok` | `depth_age_ms` p95 < 600 ms |
| `pose_coverage` | `pose_x/y/yaw` 非空占比 ≥ 95% |
| `action_nonzero` | `|a| > ε` 帧占比 ≥ 90%（HOLD 段可配置排除） |
| `single_owner_ok` | 租约日志无 `TOPSEE ∩ UNITREE` 重叠窗口 |
| `conf_declared` | 开录时置信度已记录；未达 0.9 则 `outcome` 不得标 `completed` |
| `cmd_watchdog_ok` | 无「连续 >0.5 s 无命令但速度异常大」的脏段 |
| `no_platform_estop` | meta 未把平台急停当正常结束 |
| `duration` | 有效运动 ≥ 120 s |
| `map_align_declared` | `meta.json` 声明 `T_map_from_odom` 来源或显式 `odom_only` |

---

## 7. 落地路线图

原则：**先只读、再写入、最后碰运动。** M6–M9 全是纯软件，不占设备、不依赖任何待确认项。

| 阶段 | 交付物 | 依赖 | 验收 |
|---|---|---|---|
| **M6** 只读探针 | `tools/topsee_probe.py`：登录、导出地图/点位/线路、抓状态字符串、跑 §8 的 E1–E7 并产出 `artifacts/topsee_probe_*.json` | 只读账号 + 现场网络 | E1/E2 有明确结论 |
| **M7** HTTP 适配 | `adapters/dog_topsee.py`：`TopseeClient`（轮询线程 + 短超时 cancel）、`TopseeNav`（三态 + `poll_fault`）；`tests/test_dog_topsee.py` 用本地 HTTP fixture | M6 的 E1/E2 结论 | 认证/重试/状态映射/取消/未知枚举失败路径全覆盖 |
| **M8** 语义绑定 | `SharedMap` v2 `platform_binding` + `tools/export_dog_bindings.py`（含 diff 报告）+ 加载期校验 | M6 导出样例 | 重打点后能检出 region 静默丢失 |
| **M9** 感知与气体 | `TopseePerception`（本地视觉优先，告警图作旁路证据）、`TopseeGas`（台账 + 窗口聚合）、`configs/mission/gas_calibration.yaml` | — | 无台账时如实发 `calibration_source_unavailable` |
| **M10** DDS 通道 | `adapters/dog_unitree.py`（位姿订阅 + `Move` + `StopMove`/`Damp` + `probe_dds_authority`） | **E4（G11）必须先通过** | 围栏内极小幅度 `Move` 有响应且无冲突 |
| **M11** 仲裁层 | `adapters/dog_arbiter.py` + `tests/test_dog_arbiter.py`（§4.5 的 I1–I10 全部固化为断言） | M7 + M10 | 10 条不变量全绿 |
| **M12** 狗版录制 | `tt_control` 之外新建狗版 recorder + `tools/dog_episode_check.py` | M10 | §6.7 门槛全过 |

### 7.1 G1 门禁升级

原 G1「狗单独 10 次到点成功」不足以覆盖新发现的风险，追加三条：

- **G1a**：`sendNavigate` → `getCurrentByRobotId` 的到点判定链在真机上闭环可用（E1 通过）；若走 fallback 派单方式，需说明并重测。
- **G1b**：实测状态字符串枚举表已归档，`is_arrived()` 的白名单来自实测而非猜测；`UNKNOWN` 超阈值能正确报 `nav_status_unrecognized`。
- **G1c**：`nav_timeout`（F17 的「未找到有效路径」）与「一直没到点」能被区分开，不再靠 `stage_timeout_s` 兜底掩盖。

### 7.2 运维 SOP 补充（手册要求但方案曾遗漏）

1. **地图改动后必须同步，需要开机自动生效则再发布**（F26）。现场重打点后只调新增点位接口而不同步/发布，狗开机仍在用旧地图。
2. **每次发任务前确认置信度**（F22）。长期趴在充电桩会丢定位（手册 §5.6）。
3. **反光柱/充电桩挪动后必须通知厂商重新标定**，每周用柔软干布擦拭（F31）。这直接决定 `robotTask/back` 一键返桩能否用。
4. **摔倒处理必须人工物理介入**：先切断动力 → 确认关节放松后抓机身扶正（禁止拉腿/传感器/摄像头）→ 重启自检 → 检查外壳/关节/线缆/雷达（手册 §5.7）。此状态**不允许程序自动重试**，Arbiter 须停在 `FAULT` 等人工 `reset_after_fault()`。
5. **平台账号有授权天数**（F5）。长驻自动化服务的登录失败要区分「临时网络错误」（可重试）与「授权过期」（须人工联系厂商），不能一律重试。
6. **确认现场部署形态**（G15）。若是离线一体机连云，多项配置禁止本地新增（F28），我们的角色可能退化为「只读 + 发单点导航」。

---

## 8. 开工前必做的验证实验

每条都是「最小成本推翻或确认一个关键假设」。**E1 不过就不要写 M7 的生产代码。**

| # | 假设 | 怎么验 | 预期 | 失败则怎么改 |
|---|---|---|---|---|
| **E1** | `sendNavigate` 之后 `getCurrentByRobotId` 能查到对应变化 | 只读账号先查基线 → 调一次 `sendNavigate` → 1 Hz 轮询直到肉眼确认到达，全程记录 `pointsId`/`currentState`/`totalState`/`taskId` | 到达前后返回值有可区分的变化 | 改用 `addAutoInspection` + `executeNow` 包成单点任务换取正规 taskId；或退化为 DDS 位姿距离判据（依赖 E4） |
| **E2** | `currentState`/`totalState` 有稳定有限取值集 | 在 E1 基础上制造：正常到达、无效 `pointsId`、中途 `stopTask`、断网、路径被挡 | 得到实测枚举表 | 字符串比较不可靠 → `is_arrived()` 改用距离判据，或暂停实现等厂商给枚举 |
| **E3** | 手动⇄自动巡检切换有可调用的 HTTP 接口 | Chrome F12 抓包，在 Web 端手动切两次模式 | 找到接口且能脱离页面调用 | 自动交接不可行，Arbiter 停在 `WAITING_HUMAN_MODE_SWITCH`（已按此设计） |
| **E4** | 释放平台控制后 DDS 稳定可用，边缘侧不再抢控制 | 空场地 + 安全围栏，`unitree_sdk2` 订阅 `/sportmodestate`，平台侧空闲时下发极小幅度 `Move`，观察响应与冲突 | 稳定订阅 + 控制成功无冲突 | 方案 C 不成立 → 回退到平台本体遥控接口（需 G10 抓包），或要求厂商显式禁用边缘侧自动控制 |
| **E5** | 置信度 / 延迟 / 实时位姿有 HTTP 或 WS 接口 | F12 抓包实时监控页与地图管理详情页 | 找到路径与刷新频率 | 置信度门禁改为带超时和审计留痕的显式人工确认步骤 |
| **E6** | 气体标定记录存在某处 | 查机器人档案完整选配表单；直接问厂商「标定记录存哪、有无 API」 | 找到字段或拿到厂商明确「没有」的答复 | 按 §5.4 走人工台账，作为客户已知悉的产品限制记录 |
| **E7** | 现场部署形态与写权限 | 直接问现场实施人员 | 确认 `:8001` 云端 or `:8888` 离线 | 若离线联云，所有「通过 API 新增点位/线路/任务」的自动化设想需重评 |
| **E8** | 深度相机能取到可用深度流 | `video/getStreamUrl` 逐通道取址，确认协议/编码/帧率/时间戳 | 拿到可解码流 | 深度只用无人机侧 Depth Anything，狗侧只录 RGB |
| **E9** | `mapMan/getRobotMapAll` 实际响应结构与标注一致（G16） | 只读调用一次，比对 `ShowRobotTaskAllEntity` 标注 | 结构可解析 | 导出脚本改用 `robot_point/*` 与 `mapMan` 的其它查询接口拼装 |

---

## 9. 勘误（对初版方案的修正）

| 初版说法 | 实际 | 影响 |
|---|---|---|
| `getCurrentByRobotId` 返回 `RobotTask`，字段 `currentPointsId` / `inspectionRate` | 返回 `ShowRobotTaskAllEntity`，字段是 `pointsId` / `currentState`（F6）；`currentPointsId`/`inspectionRate` 仅 `getPagingRobotTask` 返回（F7） | 到点判定的信息量**比初版说的还少**，只有一个无枚举字符串 |
| `robotTaskItem/testPoints` 可作 `goto_goal` 的备选落点 | 该接口**已 deprecated**（F9） | 不应作为主备选；备选改为「包成单点自动巡检任务」 |
| `archivesMan/getRtspUrl` 取视频流 | 该接口**已 deprecated**（F9） | 改用 `video/getStreamUrl` |
| `selectControllerUser` 可查控制人空闲 | **已 deprecated**（F9/F10） | 改用 `updateControllerUser`，但取值待抓包（G7） |
| 置信度、延迟可从状态接口读 | OpenAPI 全文 0 命中（F12） | 门禁近期无法自动化，必须改为人工确认 + 留痕 |
| `updateControllerUser` ≈ 模式交接 | 它只改**控制人**；「手动/自动巡检」是另一个概念且**无接口**（F14） | 必须拆成「控制权租约」与「巡检模式」两轴；自动交接近期不可行 |
| 「MissionBrain 一行不改，只新增适配器」 | 事件契约确实可冻结，但 `Supervisor → DogAdapter` 之间必须插入 Arbiter，且 `dog_sdk.py` 需加 `poll_fault` 探测（§5.1） | 改动量比初版估计的大，但仍属小改 |
| AprilTag 与平台 `RosPointsTag` 可共用编号空间 | `RosPointsTag` 描述用词是「二维码」，`tagFamily` 取值与识别算法在手册与 OpenAPI 中**均无说明**（G14）；我方 `tag36h11` 是给无人机空中识别用的，视角/高度/距离都不同 | **撤回该主张**，两套标记保持独立，直到厂商确认 |
| `dog_goal_id` 直接等于 `pointsId` | 会造成地图重建后静默失效（§5.2） | 加稳定语义标签中间层 |
| 分时双通道架构图 | 图里画成了确定的箭头，实际上模式切换接口不存在、DDS 权限未验证 | 状态机改为含 `WAITING_HUMAN_MODE_SWITCH`；E3/E4 列为阻塞级实验 |

初版 HTML 已删除，避免留下带已知错误的陈旧文档。

---

## 10. 待厂商确认清单（可直接发给拓普视）

**阻塞级（不答复就无法完成自动化）**

1. 单点 `POST /point/sendNavigate` 是否创建可查询任务？若是，用哪个接口查进度与到点？若不是，推荐的单点派单 + 到点判定方式是什么？
2. `currentState`、`totalState`、`robotState`、`resultState` 的完整取值枚举表。
3. 手动巡检 ⇄ 自动巡检的模式切换 HTTP 接口（路径、参数、返回）。
4. `PUT /state/updateControllerUser` 的 `state` 与 `force` 取值；Web 端「释放控制」按钮实际发的请求。
5. 定位置信度、控制延迟、实时位姿（平面 `x/y/th`）的查询接口与刷新频率。
6. 第三方程序直连宇树 DDS（`unitree_sdk2`）是否被边缘侧独占？释放平台控制权后 `SportClient.Move` 是否生效？是否有官方建议的共存方式？
7. 登录 Token 的 header 名、有效期、续期方式；账号授权天数到期前的提醒机制。

**重要级**

8. 气体传感器标定时间/状态是否有存储与接口？能否触发「立即采样一次」？
9. 深度相机对外流的协议、编码、内参、时间戳与尺度。
10. `RosMapMan.mapSpot` 的字符串格式；`RosPoints.th` 的单位（度还是弧度）。
11. `RosPointsTag.tagFamily` 的取值与对应识别算法。
12. 本体遥控（速度下发）的接口协议、载荷与频率上限。
13. `mapMan/getRobotMapAll` 的实际响应结构（OpenAPI 标注为 `ShowRobotTaskAllEntity`，疑似导出错误）。
14. `taskAlarm/alarmCallback` 能否配置回调地址？鉴权、重试、事件时序如何？
15. 现场部署形态确认：云端 `:8001` 还是离线一体机 `:8888`？离线连云后我们能否通过 API 新增点位/线路？

---

## 附录 A. 已核验接口速查

| 用途 | Method + Path | 备注 |
|---|---|---|
| RSA 公钥 | `GET /service/api/permission/free/security/rsa?securityKey=` | 已实测，返回 `code:0` |
| PC 登录 | `POST /service/api/permission/free/pc/login` | — |
| 实时状态列表 | `GET /service/api/robot/state/getStateData` | 未登录返回 401 纯文本 |
| 修改控制人 | `PUT /service/api/robot/state/updateControllerUser?robotId=&state=&force=` | 取值待确认 |
| ~~查控制人空闲~~ | ~~`PUT /service/api/robot/state/selectControllerUser`~~ | **deprecated**，描述 `02 空闲 / 04 忙碌` |
| 单点导航 | `POST /service/api/robot/point/sendNavigate?robotId=&pointsId=` | 返回裸 `Result`，无 taskId |
| 当前任务 | `GET /service/api/robot/instrument/robotTask/getCurrentByRobotId?robotId=` | → `ShowRobotTaskAllEntity` |
| 巡检结果分页 | `GET /service/api/robot/instrument/robotTask/getPagingRobotTask` | → `RobotTask`（含 `currentPointsId`/`inspectionRate`） |
| 停止 / 暂停任务 | `GET .../instrument/robotTask/stopTask?robotId=` / `pauseTask` | 任务级 abort |
| 一键返桩 | `GET /service/api/robot/instrument/robotTask/back?robotId=` | 依赖反光柱标定 |
| 离桩 | `POST /service/api/robot/instrument/robotTask/sendChargingLeave` | — |
| 建图/重定位/加载地图 | `POST /service/api/robot/mapMan/mapUpdate`（`MapUpdateDomain`） | `action` 0–6 + `xyth` + `mapType` |
| 地图/线路/点位导出 | `GET /service/api/robot/mapMan/getRobotMapAll?robotId=` | 响应标注疑似有误（G16） |
| 点位详情 | `GET /service/api/robot/point/getPointsById?pointsId=` | — |
| ROS 点位增改导出 | `POST/PUT /service/api/robot/robot_point/*` | `RosPointsShowEntity` |
| 视频流 | `GET /service/api/robot/video/getStreamUrl`（必填 `deviceId,ip,screenshot,stream,streamMode`） | 响应是未结构化 `JSONObject` |
| ~~RTSP 地址~~ | ~~`GET /service/api/robot/archivesMan/getRtspUrl`~~ | **deprecated**（`protocol: FLV\|WEBRTC`） |
| 气体历史 | `GET /service/api/robot/gas/getGasHistory` | 只有历史聚合 |
| 告警列表 / 历史 | `GET /service/api/robot/taskAlarm/getAlarmList` / `getAlarmHistory` | 只读，作旁路证据 |
| 告警回调 | `POST /service/api/robot/taskAlarm/alarmCallback` | 平台侧接收端点，注册方式未知 |
| 云台控制 | `GET /service/api/robot/thirdParty/camera/ptzcontrol`（必填 `command,deviceId,direction,speed`） | 须先进手动巡检 |
| 自动巡检任务 | `POST /service/api/robot/instrument/autoInspection/addAutoInspection` + `executeNow` | 仅作 E1 失败时的 fallback |
| ~~测试单独点位~~ | ~~`POST /service/api/robot/instrument/robotTaskItem/testPoints`~~ | **deprecated** |

## 附录 B. 宇树 B2 SDK 速查

| 能力 | 调用 | API ID |
|---|---|---|
| 连续速度 | `SportClient.Move(vx, vy, vyaw)` | 1008 |
| 相对位移 | `MoveToPos(x, y, yaw)` | 1036 |
| 停止 | `StopMove()` | 1003 |
| 软急停（阻尼） | `Damp()` | 1001 |
| 站立 / 卧倒 | `StandUp()` / `StandDown()` | 1004 / 1005 |
| 切步态 | `SwitchGait(gait)` | 1011 |
| 身高 | `BodyHeight(h)` | 1013 |
| 速度档 | `SpeedLevel(level)` | 1015 |
| 运动状态 | DDS `/sportmodestate`（position、velocity、yaw） | — |
| 本体感受 | DDS `/lowstate`（≈500 Hz，12 电机 + IMU + 足底力） | — |

## 附录 C. 参与本方案的评审

| 视角 | 模型 | 主要贡献 |
|---|---|---|
| 能力取舍 | GPT-5.6 Terra | 四象限矩阵；指出 `alarmCallback` 不是可配置 webhook、深度流「有 URL ≠ 有可训练深度」 |
| 接口设计 | Grok 4.5 | Arbiter 状态机与租约模型；`updateControllerUser` 与巡检模式必须拆成两轴；安全停用 `Damp`/`StopMove` 而非平台急停 |
| 红队反驳 | Claude Sonnet 5 | 找出 `getCurrentByRobotId` 响应结构错误、E1 致命假设、`tick()` 阻塞风险、`dog_goal_id` 强耦合、撤回 AprilTag 合并主张 |
| 综合与核验 | Claude Opus 5 | 逐条脚本核验上述指控；发现 `mapUpdate` 的 `action` 0–6 + `xyth` 可程序化建图/重定位 |

