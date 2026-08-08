# 拓普视平台接口清单（实现方案第一检索位）

> **用途**：狗侧任何「这个功能怎么调」的问题，**先查本文**，再查下列一手材料，
> 最后才考虑问厂商。本文是一手材料的索引 + 前端逆向补充，不是替代品。
>
> 编制日期：2026-08-08　适用平台版本：`3.3.8`（离线一体机形态）

## 0. 一手材料位置

| 材料 | 路径 | 覆盖内容 |
|---|---|---|
| 机器人模块 OpenAPI（275 接口） | `docs/references/机器狗文档/机器人模块.openapi.json` | 档案、任务、点位、地图、动作、告警、摄像头 |
| 登录模块 OpenAPI（5 接口） | `docs/references/机器狗文档/登录模块.openapi.json` | RSA 公钥、PC/APP 登录、登出、改密 |
| 平台用户手册（整合版） | `docs/references/机器人巡检平台用户说明书/2026-08-03-dog-inspection-platform-user-manual.md` | 业务语义、操作约束、安全口径、故障处理 |
| 真机实测修正 | `docs/handover/2026-08-07-dog-first-navigate-loop-handover.md` | 上述两者与真机不符之处（**冲突时以实测为准**） |

**检索优先级**：实测交接 > OpenAPI > 用户手册 > 前端逆向（本文 §3）> 问厂商。

---

## 1. 网络与部署形态（回答「上位机地址是什么」）

平台档案接口 `archivesMan/getDateById` 返回的 `localhostIp`（字段语义：**本体IP地址**）
即为集成商登记的上位机地址。实测该值 **与业务平台主机地址同一个**，即：

```
上位机 = 业务平台一体机 = 同一台 x86 工控机
  ├── :8888  业务平台 HTTP + WebSocket（Vue 前端 / OpenAPI / STOMP / 运控 WS）
  ├── :22    SSH
  ├── :1883  MQTT
  ├── :5672  AMQP（RabbitMQ）
  ├── :6379  Redis
  └── :3306  MySQL
```

要点：

1. **本项目实际取值不写入本仓库**（脱敏准则 6），从 `configs/dog/*.json` 与
   `archivesMan` 档案读取。下文一律以 `<平台主机>` 指代。
2. **宇树内网（`192.168.123.0/24`，B2 板载算力所在网段）从办公网不可路由**，
   `7400/7401`（DDS discovery）在 `<平台主机>` 上未开放。
   → **DDS 受阻不只是集成商政策问题，是网络拓扑的硬约束。**
      要走 DDS，必须先落地跨网段路由或在上位机内部署，两者都需集成商书面同意。
3. 手册 §5.3 / §5.9 出现的 `192.168.123.76` / `192.168.123.12` 是**在线云端版文档示例**，
   非本现场取值；档案 `configJson` 内 `light.ip` 字段与其 `rtsp_url` 网段亦不一致
   （字段陈旧，**以 `rtsp_url` 为准**）。
4. 部署形态判定：手册 §0 与附录 B 给出「云端/OpenAPI = `8001`，离线一体机 = `8888`」。
   本现场为 **离线一体机（`8888`）**，登录走 JSON + `tuopushi_edge_token` 头。

---

## 2. OpenAPI 已覆盖的关键接口

按我们的用途归类，完整清单见 JSON 本身。

| 用途 | 接口 | 备注 |
|---|---|---|
| 查机器人档案（含本体IP/型号/SN） | `GET /archivesMan/getDateById`、`getDateAll` | `robotNum` 即 SN，`robotId` 主键 |
| 点位导航派单 | `POST /point/sendNavigate` | 参数仅 `robotId` + `pointsId`；**返回不含任务标识** |
| 任务列表 / 结果 | `GET /instrument/robotTask/getPagingRobotTask` | 到达判据主路径之一 |
| 任务起停 | `stopTask` / `pauseTask` / `resumeTask` / `back`(一键返航) | |
| 控制权 | `PUT /state/selectControllerUser`（`02` 空闲 / `04` 忙碌）、`updateControllerUser` | **同一时刻仅一个控制人** |
| 实时状态 | `GET /state/getStateData` | **实测无位姿、无置信度字段** |
| 跟随 / 避障开关 | `POST /archivesMan/startFollow`、`startObstacle` | |
| 地图 / 点位 / 线路 | `mapMan/*`、`robot_point/*`、`line/*` | |
| 动作与动作组 | `robot_action/*`（`testPoints` 需 `actionId`+`pointsId`+`robotNum`） | 离散动作，非连续运控 |

**OpenAPI 里不存在的东西（重要否定结论）**：

- ❌ 没有任何**连续速度 / 遥操作**接口（无 velocity、无 twist、无 joystick）。
- ❌ 没有**位姿 / 置信度**查询接口。
- ❌ 没有 **MQ / 推送通道**的任何描述。
- ❌ 没有 §3 列出的 `model/*` 一族接口。

→ 结论：**厂商口头所说的「运控接口」不在他们提供的 OpenAPI 文档里。**

---

## 3. 前端逆向补充（未文档化，⚠️ 无厂商契约保证）

来源：`<平台主机>:8888` 的 Vue 打包产物（`main-*.js` + 懒加载 chunk）。
**这些接口没有官方文档，随版本升级可能变更，接入前必须让厂商书面确认。**

### 3.1 连续运控：WebSocket `/edg`

- 端点：`ws(s)://<平台主机>:8888/edg`　（实测握手返回 `101`，服务存活）
- 心跳：每 **30 s** 发 `{"data":"0000","value":null}`
- 移动指令（**ROS Twist 语义**）：

```jsonc
{
  "robotId": "<robotNum>",
  "data": "move-by-direction",
  "value": {
    "robotId": "<robotNum>",
    "twist": {
      "linear":  { "x": 0.3, "y": 0.0, "z": 0 },   // x 前后, y 横移
      "angular": { "x": 0, "y": 0, "z": 0.0 }      // z 转向
    }
  }
}
```

- 前端默认速度标量 **0.3**（滑条可调），按住方向键时以 **200 ms 周期（5 Hz）重复下发**。
- 松键 / 停止：**下发全零 twist**，前端侧无其它停止手段。
- 键位：`I`=前 `J`=左转 `K`=后 `L`=右转（keyCode 73/74/75/76）。

⚠️ **安全空白**：前端「停止」= 停止发包 + 补一帧零速。
**服务端是否有超时 deadman（断连/停发后自动停机）在前端代码中无任何证据。**
这正是 v3.4 方案标注的「deadman 未验证」项 —— 逆向没有解除该风险，反而坐实了它。

### 3.2 离散动作：`POST /service/model/action`

请求体 `{ "robotId": "<robotNum>", "type": <int> }`

| `type` | 语义 | 关联 `gait` |
|---|---|---|
| `0` | 下蹲 / 趴下 | — |
| `1` | 站立 | `0` |
| `3` | **急停** | — |
| `5` | 爬坡模式 | `2` |

手册附录 B 明确：**「急停 ≈ 断电」**。即 `type=3` 不是优雅减速停车，
而是切动力（狗会失去支撑）。安全设计中**不可**把它当作常规刹车使用。

### 3.3 其它未文档化 HTTP 接口

| 常量 | 路径 | 用途 |
|---|---|---|
| `ROBOT_BODY_HEIGHT` | `POST /service/model/body-height` | 身高档位，`{robotId, type}`，`type` ∈ {0,1,2} |
| `ROBOT_MAP_CORRECT_POSE` | `POST /service/model/map-correct-pose` | **前端真正用的重定位接口**（需 `robotId` + `mapId`） |
| `ROBOT_MAP_UPDATE` | `POST /service/free/mapUpdate` | 建图数据更新（**非重定位**，实测 `action=1` 假成功） |
| `VIDEO_POSITION` | `POST /service/camera/position` | 云台位姿 |
| `VIDEO_LiftingRod` | `POST /service/thirdParty/camera/liftingRod` | 升降杆 |

> 前端 axios `baseURL="/"`，路径前缀常量 `y="/service"`；而 OpenAPI 写作
> `/service/api/robot/...`。两种前缀在未带 token 时均返回 `401`（鉴权早于路由），
> **无法在未登录状态下区分**，需带 token 复验。

### 3.4 实时推送：STOMP over WebSocket

- 端点：`ws(s)://<平台主机>:8888/robot/mq/`　（实测握手返回 `101`）
- 协议：STOMP（RabbitMQ Web-STOMP，`.>` 通配符为 RabbitMQ topic 语法）
- 凭据：**前端硬编码**，见 §5 安全提示（本文不落明文）

订阅目的地：

| 目的地 | 内容 | 对我们的价值 |
|---|---|---|
| `/topic/service_robot_status` | 机器人实时状态 | 🎯 **解开 G4/G5**：含 `dogmode` `doggait` `bodyHeight` `footHeight` `follow` `obstacle`，疑含位姿（待抓包确认） |
| `/topic/service_robot_scan` | 激光扫描 | 🎯 感知输入 |
| `/topic/service_robot_velodyne_base_laser` | 点云 | 🎯 感知输入 |
| `/topic/service_robot_task_result` | 任务结果 | 🎯 **到点检测主路径**（替代 HTTP 轮询） |
| `/topic/service_robot_taskItem` | 任务航点项 | 进度粒度 |
| `/topic/service_robot_map_base64` | 地图底图 | |
| `/topic/service_robot_event.>` / `service_robot_alarm.>` | 事件 / 告警 | 失败判据交叉核对源 |
| `/topic/service_robot_obstacle_avoidance_topic` | 避障状态 | |
| `/topic/service_robot_camera_status`、`ros_camera` | 相机状态 | |
| `/topic/service_robot_temperature`、`topic_gas`、`topic_acoustic` | 测温 / 气体 / 声学 | |
| `/topic/service_follow_msg`、`service_robot_charging_room`、`service_robot_arm.>`、`file_queue` | 跟随 / 充电房 / 机械臂 / 文件 | |

> 交接文档 §G4/G5 记「大屏位姿走另一条实时通道（未逆向，疑 MQTT/WebSocket）」——
> **该通道已定位为本节的 STOMP 通道**；位姿字段是否在 `service_robot_status` 内需实际订阅确认。

#### 3.4.1 ⚠️ 头号陷阱：按机器人分发的主题必须拼 SN 后缀

前端 §3.4 常量表里的主题字符串**不是完整目的地**。订阅前会拼 `.<robotNum>`：

```js
subscribeToTopic([`${ROBOT_STATUS}.${robotNum}`, `${ROBOT_SCAN}.${robotNum}`, …])
// → /topic/service_robot_status.<机器人SN>
```

**订裸主题（不带后缀）会静默收到 0 帧，且不返回任何 `ERROR` 帧。**
初次实测正是踩了这个坑，一度误判为「平台没在推送」。
连 `/topic/#` 通配也收不到 —— 该 broker 不按 RabbitMQ `#` 语义匹配，**通配探测不可作为反证**。

| 主题 | 是否需 `.<robotNum>` |
|---|---|
| `service_robot_status` / `_scan` / `_velodyne_base_laser` / `_task_result` / `_map_base64` / `_taskItem` | ✅ **必须** |
| `service_robot_event.>` / `service_robot_alarm.>` / `_obstacle_avoidance_topic` | ❌ 按原样 |

#### 3.4.2 只读订阅实测结论（2026-08-08，已通过）

工具：`tools/topsee_mq_probe.py`（零依赖；帧类型白名单锁死为
`CONNECT`/`SUBSCRIBE`/心跳/`DISCONNECT`，**不具备下发控制指令的能力**）。

| 主题 | 帧率 | 活性校验 |
|---|---|---|
| `service_robot_status.<SN>` | **2.00 Hz** | `x`/`y`/`th`/`matchProb` 40/40 帧取值互异，**确认在变**（非静态回填） |
| `service_robot_scan.<SN>` | 2.00 Hz | 标准 ROS `sensor_msgs/LaserScan` |
| `service_robot_velodyne_base_laser.<SN>` | 2.05 Hz | 含位姿字段 |
| `service_robot_task_result.<SN>` | 事件型 | 20 s 内收到 1 帧 |
| `service_robot_event.>` / `alarm.>` | 事件型 | 采样窗口内无事件，属正常 |

### 3.4.3 `service_robot_status` 字段表（实测，2 Hz）

| 字段 | 含义 | 备注 |
|---|---|---|
| `position.x` / `.y` / `.z` | **地图系位姿** | 🎯 **G5 解开** |
| `position.th` | 朝向 | 实测跨度 −0.04 ~ 1.45，**与弧度一致**（交接文档记「`th` 单位不明」，此处得解） |
| `position.name_map` | 当前地图 ID | 可用于 `map-correct-pose` 的 `mapId` |
| `position.mode_map` | 定位/导航模式 | 实测 `"navigating"` |
| `position.device` | 定位源 | 实测 `"gps"` |
| `position.gps` / `.rtk` / `.uwb` | 其它定位源 | 本现场全 `0`，未配置 |
| **`matchProb`** | **定位置信度** | 🎯 **G4 解开**。即 UI「置信度」，实测 0.625~0.690 |
| `robotState` | 机器人状态 | `02` 空闲 / `03` 任务中 / `07` 建图 等 |
| `currentTaskId` | 当前任务 ID | 到点检测可用 |
| `dogmode` / `doggait` / `bodyHeight` / `footHeight` | 姿态与步态 | 与 §3.2 动作 `type` 对应 |
| `follow` / `obstacle` | 跟随 / 避障开关 | |
| **`controllerUserName`** / **`robotControlUser`** | **当前控制人** | 🎯 仲裁层可据此观测控制权归属 |
| `battery` / `isCharging` / `current` / `temperatures` | 电量与电气 | |
| `speed` / `rpy` / `motorangle` / `odometer` | 运动量 | |
| `delayTime` / `useCpu` / `useMemory` / `runningTime` / `runningDay` | 健康度 | |
| `time` | 设备时间（Unix 秒） | ⚠️ **仅秒级分辨率**，2 Hz 下每两帧共用一个值 |
| `connection` | 连接状态 | |

**工程结论**：

1. 🎯 **G4 / G5 彻底解开**。位姿与置信度都能拿到，**完全不依赖 DDS**。
   交接文档「无置信度、无实时位姿接口」的结论**仅对 HTTP 成立**，对 MQ 不成立。
2. ✅ **E4-S（只读影子部署）的数据前提已齐备**：位姿 + 激光 + 点云 + 状态 + 控制权。
3. ⚠️ **2 Hz 只够慢通道**。做影子部署、到点判定、评测足够；
   **不足以支撑 10 Hz 级闭环快通道**，不要拿它当 WAM 控制回路的状态源。
4. ⚠️ **`time` 秒级分辨率不可用于时序对齐**，必须以本地接收时刻（`t_mono`）为准 ——
   与 DDS 侧 `t_device` / `t_mono` 的处理口径保持一致。
5. ✅ `matchProb` 使人工 `ack_confidence` 可以自动化；注意手册要求发任务前 **≥ 0.9**，
   而实测常态在 **0.63~0.69**，低于门槛，这本身是需要现场处理的问题。

---

## 4. 仍需向厂商确认的问题

### 必答（阻塞架构）

1. **`/edg` 的 `move-by-direction` 是否为受支持的对外接口？**
   请提供该 WebSocket 协议的正式文档（全部 `data` 操作码、`value` 结构、鉴权方式、版本兼容承诺）。
   若不对外开放，请说明「运控接口」的正式形态是什么。
2. **deadman / 看门狗**：停止发送 `move-by-direction` 或 WebSocket 断开后，
   机器人是否在固定时间内自动停止？**超时是多少毫秒？** 有无官方测试记录？
   （这是我们能否上闭环的**唯一硬门槛**。）
3. **`model/action type=3`（急停）是否等同断电**？是否存在**优雅停车**（减速至零并保持站立）的指令？
   若无，我们的安全设计只能依赖「停发 + 零速」，需要问题 2 的答案兜底。
4. **STOMP 通道的对外授权**：`/robot/mq/` 是否允许第三方订阅？
   请为我们**单独开一个只读账号**（不复用前端内置账号）。
   字段与频率我们已实测清楚（§3.4.3），只需确认三点：
   `position` 的坐标系定义、`matchProb` 与 UI「置信度」是否同一口径、2 Hz 是否可调高。
5. **控制权与运控的关系**：`/edg` 下发运控前是否必须先 `updateControllerUser` 抢控 +
   切「手动巡检」？第三方持控期间平台侧如何**强制收回**？

### 次要（影响实现质量，不阻塞）

6. `service_robot_task_result` 的消息体结构，及其与 `getPagingRobotTask.result`
   在「成功」判定上的一致性（实测 HTTP 侧 `result=成功` 与告警表冲突过）。
7. `move-by-direction` 的速度上限与安全限幅在服务端还是客户端？超限如何处理？
8. `model/map-correct-pose` 的完整参数与 `mapId` 取值来源（用于自动化重定位）。
9. 平台版本升级时，§3 这些未文档化接口的兼容策略。

### 已可自查、无需再问

- 上位机地址 → `archivesMan/getDateById.localhostIp`（§1）
- 机器人 SN / 型号 → 同上 `robotNum` / `robotModel`
- MQ 主题清单与订阅方式 → §3.4 / §3.4.1（**注意 SN 后缀**）
- **是否有实时位姿与置信度 → 有**，`service_robot_status` @2 Hz（§3.4.3）
- **`th` 的单位 → 弧度**（§3.4.3，实测跨度佐证）
- 是否有连续运控 → 有，见 §3.1（要问的是「是否授权」，不是「有没有」）

---

## 5. 安全提示（须处置）

1. **平台前端把 MQ 账号密码硬编码在 JS 里**，任何能打开登录页的人都能拿到并订阅全部主题。
   → 我方接入时**不要复用该账号**，向厂商申请独立只读账号（§4 问题 4）。
2. 上位机对办公网暴露 `3306`(MySQL) / `6379`(Redis)。属客户网络治理问题，
   **我方不触碰**，但应在交付文档中书面提示客户。
3. **本仓库将公开**：`artifacts/` 与 `configs/dog/` 当前**未被 `.gitignore` 覆盖**，
   其中含现场内网地址与设备明文口令（摄像头 RTSP 凭据等）。
   提交前必须脱敏或加入忽略清单，否则违反 CLAUDE.md 脱敏准则 6。

---

## 6. 复现方法

前端逆向可复现（只读 HTTP GET，不触发任何动作）：

```bash
curl -s "http://<平台主机>:8888/" -o index.html          # 取资源清单
# 按 index.html 中 src/href 下载 /topsee-offline-app/js/*.js
# 在打包产物中检索：
#   Ea={...}                     → STOMP 目的地常量表
#   ce={...}                     → HTTP 接口常量表（含未文档化的 model/*）
#   "move-by-direction"          → 运控指令构造点
#   function Me(               → QuadrupedControl 控制逻辑
```

WebSocket 存活性检查（仅握手，不订阅、不发指令）：

```bash
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "http://<平台主机>:8888/edg"     # 期望 101
```
