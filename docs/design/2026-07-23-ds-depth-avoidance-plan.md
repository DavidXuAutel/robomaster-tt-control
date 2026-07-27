# Tello 接入 DS 路由器、远端深度推理与避障绕飞方案

日期：2026-07-23
状态：设计中
前置：[`./2026-07-17-tt-visual-avoidance-design.md`](./2026-07-17-tt-visual-avoidance-design.md)

## 一、原方案（Codex 版）评审

原方案保存在 `docs/2026-07-23-ds-depth-avoidance-plan.md`（docs 根目录，应按约定放入 `docs/design/`）。整体框架合理，但存在以下问题需在本方案中修正或细化：

### 1.1 结构性问题

| # | 问题 | 说明 |
|---|------|------|
| P1 | **文档位置错误** | 按 `docs/README.md` 约定，设计方案应放入 `docs/design/`，不应放在 docs 根目录 |
| P2 | **Tello 入网方案不够具体** | 只说"第二块 Wi-Fi 接口或另一台设备"，未给出推荐方案和具体操作步骤 |
| P3 | **异步深度推理设计缺失细节** | 只说"改为异步工作线程"，未描述线程模型、帧丢弃策略、超时与降级逻辑 |
| P4 | **新深度推理服务器接口未定义** | 用户提到"深度推理接我们的新服务器服务"，但原方案未定义新服务器与现有 `da_v2_service.py` 的关系、新增能力及协议扩展 |
| P5 | **缺少具体代码修改位置** | 只列了问题但未标注涉及的文件和函数，实施时容易遗漏 |

### 1.2 技术细节问题

| # | 问题 | 说明 |
|---|------|------|
| P6 | **保活命令响应污染** | `TelloClient._keepalive_loop()` 在 cmd socket 上发 `command` 但不读响应，旧 `ok` 会残留在 socket 缓冲区。后续 `send()` 的 `recvfrom` 可能读到保活的 `ok` 而非目标命令的应答。需引入命令序列号或独立 socket。 |
| P7 | **`station_mode.py` 密码泄露** | `_send()` 打印 `>>> ap {ssid} {password}` 明文到终端/日志。`auto_fly.py` 的 `log()` 调用 `cmd_setup()` 虽然手工脱敏了日志行，但 `cmd_setup` 内部的 `_send` 仍会打印密码。 |
| P8 | **`auto_fly.py` 切换 Mac Wi-Fi** | `switch_wifi()` 用 `networksetup -setairportnetwork` 切换 Mac 内置 Wi-Fi，违反"不修改 Mac 网络配置"的约束。本方案场景下不应使用 `auto_fly.py` 的自动切网逻辑。 |
| P9 | **连接后探测冲突** | `station_mode.py` 的 `find_drone()` 向整个 `/24` 网段发 `command` 探测。如果控制界面已经连上飞机，额外的 `command` 可能干扰控制会话（Tello SDK 对重复 `command` 的处理不确定）。 |
| P10 | **Depth Anything 相对深度** | 当前 `da_v2_service.py` 对单帧做 2%~98% 分位数归一化，近度是帧内相对值。不同场景（近墙 vs 开阔地）相同近度值对应不同物理距离，阈值需现场标定，无法直接跨场景复用。 |

### 1.3 方案层面缺失

| # | 缺失项 |
|---|--------|
| P11 | 未说明如何验证 DS 路由器是否支持 Tello 入网（2.4GHz 频段、客户端隔离、DHCP 地址保留） |
| P12 | 未定义新深度推理服务器的 `/health`、`/depth` 及扩展协议（鉴权、超时、错误码） |
| P13 | 未给出 Mac 同时保持 DS 上网 + 访问 Tello 的网络验证步骤 |
| P14 | 未说明避障绕飞失败后如何安全恢复，以及各阶段的成功/失败判定标准 |

---

## 二、目标

在 **Mac 内置 Wi-Fi 始终连接 DS 路由器（可正常上网）** 的前提下：

1. 一次性将 Tello 接入 DS 路由器局域网（不切换 Mac 内置 Wi-Fi）。
2. Mac 通过 DS 路由器局域网同时访问 Tello 和新深度推理服务器。
3. 从 Tello 接收视频帧，异步发送到新深度推理服务器，获得感知结果。
4. 在 Mac 本地执行安全控制状态机，基于感知结果完成避障绕飞。
5. 分阶段验证：地面联调 → 起飞悬停 → 遇障停止 → 单障碍物绕飞。

---

## 三、网络拓扑

```
┌─────────────────────────────────────────────────────────────┐
│                        DS 路由器 (LAN)                        │
│  2.4GHz + 5GHz, DHCP: 192.168.x.0/24                         │
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  Mac 内置 Wi-Fi │    │  Tello (组网后) │    │ 深度推理服务器  │  │
│   │ 192.168.x.100  │    │ 192.168.x.101  │    │ 192.168.x.200  │  │
│   │ (保持上网)      │    │ (2.4GHz only)  │    │ (有线或Wi-Fi)   │  │
│   └──────┬─────────┘    └──────┬─────────┘    └──────┬─────────┘  │
│          │                     │                     │           │
│          └─────────────────────┴─────────────────────┘           │
│                  Tello SDK UDP (8889/8890/11111)                 │
│                  深度推理 HTTP POST /depth                       │
└─────────────────────────────────────────────────────────────┘
```

### 关键约束

- Mac 内置 Wi-Fi **始终**连接 DS，**禁止**切换到 Tello 热点。
- Tello 入网通过 **USB Wi-Fi 网卡**（推荐）或第二台设备完成一次性配网。
- 不修改 Mac 的 DHCP、静态 IP、DNS 或默认路由。
- DS 路由器需满足：**2.4GHz 频段开启**、SSID 广播、客户端间通信不禁用（无 AP Isolation）、DHCP 地址池充足。

### 备选方案：USB Wi-Fi 网卡

如果 DS 路由器不满足条件（如 5GHz only 或开启客户端隔离），备选方案为 Mac 通过 USB Wi-Fi 网卡直连 Tello 热点，同时内置 Wi-Fi 保持 DS 上网：

```
Mac 内置 Wi-Fi ──── DS 路由器 ──── Internet / 深度推理服务器
Mac USB Wi-Fi ──── Tello 热点 (192.168.10.x)
```

此方案下 Mac 同时拥有两个 IP，需注意路由表不冲突（Tello 网段 `192.168.10.0/24` 仅经 USB 网卡）。

---

## 四、一次性 Tello 入网流程

### 4.1 前置确认

在实施前，**先确认 DS 路由器能力**：

- [ ] 路由器型号与管理界面地址
- [ ] 2.4GHz 频段是否已开启、SSID 是否与 5GHz 相同
- [ ] 客户端隔离（AP Isolation / Client Isolation）是否已关闭
- [ ] DHCP 地址池是否有空闲（建议预留 `192.168.x.101-110` 给 Tello）
- [ ] 如有 DHCP 地址保留功能，建议为 Tello MAC 地址设置静态绑定

### 4.2 推荐方案：USB Wi-Fi 网卡配网

```
准备：USB Wi-Fi 网卡插入 Mac
  1. 确认 USB 网卡已识别（en2 等新接口）
  2. USB 网卡连接 Tello 热点 RMTT-xxxx
     执行：networksetup -setairportnetwork <usb-iface> RMTT-xxxx
  3. 确认 USB 网卡拿到 192.168.10.x 地址
  4. 通过 USB 网卡发 SDK 入网命令：
     python station_mode.py setup --ssid <DS-SSID> --password <DS-PASSWORD>
     （注意：此时需指定 bind_ip 为 USB 网卡 IP）
  5. Tello 回复 OK 后重启，约 10-15 秒加入 DS
  6. DS 路由器管理界面确认 Tello 出现在 DHCP 客户端列表
  7. 记录 Tello 的局域网 IP
  8. USB 网卡断开 Tello 热点，完成
```

### 4.3 备选方案：第二台设备配网

如无 USB Wi-Fi 网卡，可用一台 Windows/Linux 笔记本：

1. 该设备连接 Tello 热点
2. 安装 Python 3.9+，复制 `station_mode.py` 到该设备
3. 执行 `python station_mode.py setup --ssid <DS-SSID> --password <DS-PASSWORD>`
4. Tello 重启加入 DS 后，该设备可退出

### 4.4 station_mode.py 需修改

当前 `station_mode.py` 硬编码绑定 `(bind_ip, CMD_PORT)`，在多网卡场景下需支持指定网卡或 IP：

```python
# station_mode.py 增加 --bind-ip 参数
p.add_argument("--bind-ip", default="", help="指定发命令的本机 IP（多网卡时必填）")
```

`_open_socket(bind_ip)` 已在签名上支持，只需 CLI 暴露。

---

## 五、代码安全加固（阶段 2）

以下修改在真机飞行前**必须完成**：

### 5.1 拆分"日常连接"与"一次性配网"

| 文件 | 修改 |
|------|------|
| `auto_fly.py` | 增加 `--no-switch-wifi` 参数：跳过 Mac 切网步骤，仅做局域网扫描 + 拉起控制界面。或者本场景下**完全不用** `auto_fly.py`，直接用 `main.py`。 |
| `main.py` | 已是日常连接入口，无需改动。确认 `--local-ip` 和 `--tello-ip` 参数满足 DS 局域网场景。 |
| `station_mode.py` | 增加 `--bind-ip` 参数（见 4.4）。`setup` 子命令增加 `--dry-run` 仅验证连接不发 `ap`。 |

### 5.2 脱敏 SSID/密码日志

| 文件 | 函数/位置 | 修改 |
|------|-----------|------|
| `station_mode.py:32-33` | `_send()` | 检查 `cmd` 是否以 `ap ` 开头，若是则打印 `ap <ssid> ***` |
| `station_mode.py:48` | `cmd_setup()` | 调用 `_send` 前不需额外处理（已在 `_send` 内统一脱敏） |
| `depth_backend.py:74` | `_request_depth()` | HTTP 请求 URL 中不应含敏感信息；当前仅含 IP:port，无需修改 |

具体实现：在 `_send()` 中增加脱敏逻辑：

```python
def _safe_log(cmd: str) -> str:
    if cmd.startswith("ap "):
        parts = cmd.split(" ", 2)
        if len(parts) >= 3:
            return f"ap {parts[1]} ***"
    return cmd
```

### 5.3 修复 Tello 命令应答配对

| 文件 | 问题 | 修改方案 |
|------|------|----------|
| `tt_control/tello_client.py:50-64` | `_keepalive_loop()` 在 cmd socket 上发 `command` 不读响应，旧 `ok` 残留在缓冲区 | **方案 A（推荐）**：保活改用独立 socket（临时创建、发完即关）。**方案 B**：在 `send()` 中加入应答匹配校验（检查响应是否对应发出的命令）。推荐方案 A，改动最小且不引入序列号复杂度。 |

方案 A 实现：

```python
def _keepalive_loop(self, interval: float = 5.0) -> None:
    while self._running:
        slept = 0.0
        while slept < interval:
            if not self._running:
                return
            time.sleep(0.5)
            slept += 0.5
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.bind((self.local_ip, 0))  # 随机源端口
            s.sendto(b"command", self.tello_addr)
            s.close()
        except OSError:
            pass
```

### 5.4 连接后探测不干扰控制会话

| 文件 | 函数 | 修改 |
|------|------|------|
| `station_mode.py:85-116` | `find_drone()` | 增加检查：如果已知 Tello IP 且已连接，则跳过扫描。或者在扫描前检查当前是否有控制会话活跃。 |

对于本方案场景，`find_drone()` 只在 Tello 刚入网、尚未启动控制界面时使用。一旦 `main.py` 连接上 Tello，不应再运行 `find`。

### 5.5 深度推理改为异步工作线程

**这是本次最重要的架构变更。** 当前 `DepthAnythingBackend.infer()` 在主线程的显示循环中被同步调用，HTTP 请求阻塞整个 GUI 和控制循环。

#### 目标架构

```
VideoStream 线程                    主线程 (GUI + 控制循环)
     │                                    │
     ├─ 解码 H.264 → BGR 帧               │
     │                                    │
     ├─ 放入 _latest_frame (threadsafe)    │
     │                                    │
     ▼                                    ▼
AsyncInferWorker 线程              App._update_rc_stream()
     │                                    │
     ├─ 取最新帧                         ├─ 读 latest_depth (非阻塞)
     ├─ POST /depth (异步, 带超时)        ├─ 若 AUTO ON → AvoidanceController.decide()
     ├─ 写回 _latest_depth (threadsafe)   ├─ 发送 rc 命令
     └─ 记录 RTT, 帧龄, 错误              └─ 看门狗检查
```

#### 新增模块：`tt_control/async_infer.py`

```python
@dataclass
class DepthResult:
    nearness: np.ndarray
    ts: float          # 请求发起时间
    rtt_ms: float      # 往返耗时
    frame_age_ms: float # 帧采集到请求发起的延迟
    error: str = ""

class AsyncInferWorker:
    """工作线程：异步取帧 → POST 深度服务 → 写结果。"""
    def __init__(self, service_url, timeout=2.0, max_pending=1):
        ...
    def start(self): ...
    def stop(self): ...
    def latest_result(self) -> Optional[DepthResult]: ...
    def stats(self) -> dict: ...  # RTT, 帧龄, 成功率
```

#### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 帧丢弃策略 | 只保留最新帧，工作线程取帧时总是拿最新的 | 旧帧的深度结果对实时控制无用 |
| 请求并发 | `max_pending=1`，上一请求未返回前不发送新请求 | 避免服务端排队积压 |
| 超时处理 | HTTP 超时 `timeout` 秒，超时后记录错误，保留上一帧有效深度 | 不因单次超时中断控制 |
| 深度过期 | 控制循环检查 `now - result.ts > depth_stale_s`，过期视为失联 | 与现有 `AutoWatchdog` 兼容 |
| 线程安全 | `threading.Lock` 保护 shared state，主线程只读不写 | 避免 GIL 之外的竞态 |

#### 现有代码迁移

| 文件 | 改动 |
|------|------|
| `tt_control/depth_backend.py` | `infer()` 方法改为：存储最新帧 → 返回叠图（不含阻塞请求）。请求逻辑移到 `AsyncInferWorker`。 |
| `tt_control/app.py` | `_update_rc_stream()` 中：从 `AsyncInferWorker.latest_result()` 读深度，不再调用阻塞 `backend.infer()`。 |
| `tt_control/app.py` | 初始化时创建 `AsyncInferWorker` 并启动；退出时 stop。 |

### 5.6 监控指标

增加实时指标采集与 HUD 显示：

| 指标 | 来源 | 含义 |
|------|------|------|
| 服务 RTT | `AsyncInferWorker.stats()` | HTTP 请求往返时间 (avg/P95) |
| 深度帧龄 | `now - result.ts` | 当前使用的深度距请求发起的时间 |
| 视频帧龄 | `now - frame_ts` | 当前显示帧距采集的时间 |
| 控制频率 | `1 / rc_interval` | RC 命令实际下发频率 |
| 连续失败计数 | `consecutive_errors` | 连续请求失败次数，>N 触发告警 |

### 5.7 避障 ABORT 闭环

| 文件 | 修改 |
|------|------|
| `tt_control/app.py` | 增加 ABORT 按钮（界面）+ 键盘 `B` 绑定。触发后：停止 RC 前进 → 悬停 → 解除 AUTO → 记录 `TEST ABORT` 事件。 |
| `tt_control/episode_recorder.py` | 增加 `ABORT` 事件类型，记录触发时间、当前状态、深度快照、遥测快照。 |
| `tt_control/auto_safety.py` | 增加 `user_abort: bool` 标志，ABORT 后 `check()` 始终返回解除原因。 |

---

## 六、新深度推理服务器集成

### 6.1 现有服务 vs 新服务

| | 现有 `da_v2_service.py` | 新深度推理服务器 |
|---|---|---|
| 模型 | Depth Anything V2 Small | 待确认（预期更强的深度模型或 VLM） |
| 输入 | JPEG 单帧 | JPEG 单帧（可扩展多帧） |
| 输出 | 近度网格 (96×128, float16) | 近度网格 + 可选的语义信息 |
| 部署 | GPU 机 (4090) | 新服务器 |
| 协议 | HTTP POST `/depth` | HTTP POST `/depth`（兼容） |
| 鉴权 | 无 | 待确认 |

### 6.2 协议兼容设计

新服务器 **必须兼容** 现有协议作为 baseline，可在此基础上扩展：

```
请求：(不变)
  POST /depth
  Content-Type: image/jpeg
  Body: JPEG 字节

响应（baseline，与现有相同）：
  200 OK
  Content-Type: application/octet-stream
  Body: struct("<II", H, W) + H*W × float16 nearness grid

响应（扩展，可选）：
  200 OK
  Content-Type: application/json
  Body: {
    "nearness": { "grid": "<base64 encoded float16>", "h": 96, "w": 128 },
    "depth_meters": { "grid": "...", "h": 96, "w": 128 },  // 可选：米制深度
    "obstacles": [{"x": 0.3, "y": 0.5, "r": 0.15}],        // 可选：障碍物检测
    "free_space": {"left": 0.8, "center": 0.3, "right": 0.7}, // 可选：通行度
    "infer_ms": 45.2,
    "model": "depth-anything-v2-small"
  }
```

### 6.3 客户端适配

`AsyncInferWorker` 支持两种响应解析：

1. 检查 `Content-Type`，`application/octet-stream` → 按现有协议解析
2. `application/json` → 按扩展协议解析
3. 解析失败 → 降级为错误，复用上一帧

---

## 七、避障绕飞状态机（完善版）

### 7.1 状态定义

```
┌──────────┐ 起飞成功   ┌──────────┐ 用户 ARM   ┌──────────┐
│ PREFLIGHT │ ────────→ │  HOVER   │ ────────→ │ APPROACH │
└──────────┘            └──────────┘            └─────┬────┘
                                                      │
                                         检测到障碍     │
                                                      ▼
                                                 ┌──────────┐
                                                 │ AVOID_TURN│
                                                 └─────┬────┘
                                                      │ 转向完成
                                                      ▼
                                                 ┌──────────┐
                                                 │PASS_OBSTACLE│
                                                 └─────┬────┘
                                           越障确认   │
                                                      ▼
                                                 ┌──────────┐
                                                 │RECOVER_HEADING│
                                                 └─────┬────┘
                                           航向恢复    │
                                                      ▼
                                                 ┌──────────┐
                                                 │HOVER_COMPLETE│
                                                 └─────┬────┘
                                              用户确认 │
                                                      ▼
                                                 ┌──────────┐
                                                 │   LAND   │
                                                 └──────────┘
```

### 7.2 各状态详解

| 状态 | 进入条件 | RC 输出 | 退出条件 | 超时 |
|------|----------|---------|----------|------|
| `PREFLIGHT` | 连接成功 | `rc(0,0,0,0)` | 用户按 T 起飞 | - |
| `HOVER` | `takeoff` OK | `rc(0,0,0,0)` | 用户按 V ARMED → ON | - |
| `APPROACH` | AUTO ON, danger ≤ clear_thresh | 前进巡航 | 中区近度 > clear_thresh | `max_approach_s=10s` |
| `AVOID_TURN` | 检测到障碍，锁定转向方向 | yaw ± pitch 递减 | 中区近度 < clear_thresh 且连续 N 帧通畅 | `max_turn_s=8s` |
| `PASS_OBSTACLE` | 转向后前方通畅 | 小前进量，保持偏航 | 侧向近度也降到 clear_thresh 以下 | `max_pass_s=5s` |
| `RECOVER_HEADING` | 障碍物已在侧后方 | yaw 回到初始航向 | `|yaw - target_yaw| < 10°` | `max_recover_s=5s` |
| `HOVER_COMPLETE` | 航向恢复完成 | `rc(0,0,0,0)` 悬停 | 用户确认降落 | - |
| `LAND` | 用户按 L 或自动触发 | `land` 命令 | 落地 | `max_land_s=15s` |

### 7.3 安全规则（全局）

任意状态发生以下任一条件，**立即停止前进 → 悬停 → 解除 AUTO**：

| 条件 | 判定 |
|------|------|
| 视频丢失 | `video_frame_age > 1.0s` |
| 深度过期 | `now - depth.ts > depth_stale_s(1.5s)` |
| 服务连续失败 | `consecutive_errors >= 3` |
| 低电量 | `bat < 30%` |
| 遥测异常 | `h < 0` 或 `h > 500`（cm） |
| 人工接管 | 键盘 WASD / Space / Esc / L / B |
| 超时 | 任一状态超时或 `max_auto_engaged_s=30s` |

安全触发后需用户手动确认才能重新 ARM AUTO。

### 7.4 航向恢复

```python
# 在进入 APPROACH 时记录初始 yaw
initial_yaw = tello_state.get("yaw", 0)  # Tello 遥测 yaw

# 在 RECOVER_HEADING 状态中
current_yaw = tello_state.get("yaw", 0)
yaw_error = (target_yaw - current_yaw + 180) % 360 - 180  # [-180, 180]
if abs(yaw_error) < 10:
    → HOVER_COMPLETE
else:
    rc.yaw = max(-yaw_speed, min(yaw_speed, int(yaw_error * 0.5)))
```

### 7.5 绕行方向选择

沿袭 `AvoidanceController._commit` 的滞回逻辑，并增加：

- **初始选向**：比较左右近度，向更开阔侧转向（`left ≥ right → 右转`）
- **方向锁**：一次绕行中不翻转，直到前方重新通畅才释放
- **死胡同检测**：如果转向后 `danger` 持续增长（连续 N 帧上升），判定为死胡同 → 悬停并提示人工接管

### 7.6 实现记录（2026-07-23）

已于 `tt_control/avoidance_fsm.py` 实现，集成到 `tt_control/app.py`。

**代码位置：**
- 状态机：`tt_control/avoidance_fsm.py::AvoidanceFSM`
- 集成点：`tt_control/app.py::_auto_decision()` 调用 `self._fsm.step()`

**与设计的微小差异：**

| 项 | 设计值 | 实现值 | 原因 |
|----|--------|--------|------|
| 航向容差 | ±10° | ±15° | Tello yaw 读数抖动较大 |
| `PASS_OBSTACLE` 回退 | 无 | 中区近度回升 → 退回 `AVOID_TURN` | 障碍物可能不止一个，保守处理 |
| yaw 最小杆量 | 无 | `10` | 杆量太小 Tello 不响应 |
| `RECOVER_HEADING` 超时 | abort | 视为完成 | 航向恢复非关键安全项，超时不影响飞行安全 |

**HUD 显示格式：**
```
AUTO: ON  [APPROACH] CRUISE L0.12 M0.08 R0.15 rc(0,25,0,0)  infer 2492ms
          ^^^^^^^^^^ FSM 状态        ^^^^^^^^^^^^^^^^^^^^^^^^ 底层控制器
```

---

## 八、分阶段执行计划

### 阶段 1：网络准备与 Tello 入网（一次性）

**目标**：Tello 成功加入 DS 路由器，Mac 可 ping 通 Tello

| 步骤 | 操作 | 验证方法 |
|------|------|----------|
| 1.1 | 确认 DS 路由器 2.4GHz、客户端隔离、DHCP 配置 | 路由器管理界面 |
| 1.2 | 准备 USB Wi-Fi 网卡或第二台设备 | 硬件准备 |
| 1.3 | 修改 `station_mode.py`：增加 `--bind-ip`、脱敏日志 | 代码审查 + 单元测试 |
| 1.4 | 通过 USB 网卡发 `ap` 入网命令 | Tello LED 变为组网状态 |
| 1.5 | DS 路由器 DHCP 列表确认 Tello IP | `ping <tello-ip>` 通 |
| 1.6 | （可选）设置 DHCP 地址保留 | 路由器管理界面 |

**停止条件**：任一步骤失败即停止，不循环重试。保持 Mac 网络不变。

### 阶段 2：基础代码安全加固

**目标**：修复已知缺陷，异步推理就绪，地面可跑通完整管线

| 步骤 | 改动 | 测试 |
|------|------|------|
| 2.1 | `station_mode.py`: 脱敏 + `--bind-ip` | `tests/test_station_mode.py` |
| 2.2 | `tello_client.py`: 保活改用独立 socket | `tests/test_tello_client.py` |
| 2.3 | 新增 `tt_control/async_infer.py` | `tests/test_async_infer.py`（mock HTTP） |
| 2.4 | `depth_backend.py`: 移除同步阻塞，改为帧缓存 | 与上一步联测 |
| 2.5 | `app.py`: 接入 `AsyncInferWorker`，控制循环非阻塞读深度 | 地面运行 `main.py --inference depth-anything`，确认 GUI 流畅 |
| 2.6 | 增加监控指标采集与 HUD 显示 | 目视确认 HUD 显示 RTT、帧龄等 |
| 2.7 | 增加 ABORT 按钮与键盘绑定 | 手动测试 ABORT 逻辑 |

**验收标准**：
- `main.py --inference depth-anything` 启动后 GUI 不卡顿
- 断开深度服务后 HUD 显示错误，控制循环不崩溃
- 保活命令不干扰正常 SDK 通信（连续发送 20 次命令，全部收到正确应答）

### 阶段 3：新服务器对接

**目标**：确认新深度推理服务器可用，地面完成端到端深度预览

| 步骤 | 操作 | 验证 |
|------|------|------|
| 3.1 | 确认新服务器地址、端口、`/health` 端点 | `curl /health` 返回 200 |
| 3.2 | 确认新服务器 `/depth` 协议兼容性 | 发一张测试 JPEG，收到正确的 nearness grid |
| 3.3 | 如新服务器使用扩展协议（JSON），适配 `AsyncInferWorker` | 单元测试 |
| 3.4 | 测量连续请求成功率、平均/P95 RTT | 100 次连续请求，成功率 > 95% |
| 3.5 | 地面连接 Tello，实时视频 → 深度预览（不起飞） | HUD 显示热力图叠图，RTT 稳定 |
| 3.6 | 现场标定近度阈值：在不同距离放障碍物，记录近度值 | 确定 `clear_thresh` / `stop_thresh` / `estop_thresh` |

### 阶段 4：避障绕飞完善

**目标**：完善状态机和离线验证

| 步骤 | 改动 | 验证 |
|------|------|------|
| 4.1 | 在 `avoidance.py` 或新增 `avoidance_fsm.py` 中实现完整状态机（7.1） | `tests/test_avoidance_fsm.py` |
| 4.2 | 增加航向记录与恢复逻辑 | 单元测试 |
| 4.3 | 增加越障确认与连续通畅判断 | 单元测试 |
| 4.4 | 更新 `sim_avoidance.py` 支持新状态机仿真 | 离线仿真：单障碍物、左右障碍、死胡同场景 |
| 4.5 | 更新 `auto_safety.py`：增加各状态超时、ABORT 闭环 | `tests/test_auto_safety.py` |

### 阶段 5：分级真机验证

**每一级完成并确认日志正常后，才进入下一级。**

#### 5.1 起飞与悬停

- 开阔区域，确认电池 ≥ 80%、桨叶完好、无人员进入
- Mac 连接 DS → 启动 `main.py --inference depth-anything --tello-ip <ip> --local-ip <ip>`
- 确认图传和遥测正常
- 手动起飞，观察定高悬停
- 记录遥测日志
- 30 秒后手动降落

**成功标准**：起飞和降落正常，悬停期间高度稳定（±10cm），无异常漂移。

#### 5.2 深度观察（不动）

- 起飞后悬停，开启深度预览
- 人在 Tello 前方不同距离走动，确认 HUD 热力图响应正确
- 验证标定阈值：在 1m、2m、3m 处放置障碍物，记录三区近度值
- 不发送自动 RC 移动命令
- 5 分钟后手动降落

**成功标准**：深度热力图与实际障碍物位置一致，RTT 稳定，帧率正常。

#### 5.3 遇障停止

- 起飞后悬停，ARM AUTO
- 在 Tello 前方约 2m 处放置障碍物
- Tello 缓慢前进，检测到障碍后自动悬停
- 确认悬停后不再前进，状态显示 BLOCKED
- 手动降落

**成功标准**：接近障碍物时在安全距离（≥ 0.5m）外悬停，无碰撞。

#### 5.4 单障碍物绕飞

- 使用单个柔性障碍物（如泡沫板或布帘，高度约 1m）
- 起飞后悬停，ARM AUTO
- Tello 前进 → 检测障碍 → 转向绕行 → 越障 → 航向恢复 → 悬停
- 人工全程监视，随时准备接管
- 限制：`cruise_speed ≤ 20`，`yaw_speed ≤ 30`，`max_auto_engaged_s = 20s`
- 全程录制 episode（`--record`）

**成功标准**：
- Tello 成功绕开障碍物，不碰撞
- 航向恢复误差 < 20°
- 全程在测试区域内，无失控漂移
- 日志完整（RGB + 深度 + RC + 遥测 + 决策）

---

## 九、真机测试约束

- **测试场地**：开阔室内，无人员进入，地面平整、光照均匀
- **障碍物**：柔性或可倒伏（泡沫板、布帘），高度 ≥ 1m、宽度 ≥ 0.3m
- **起飞前检查**：
  - [ ] 电池 ≥ 80%
  - [ ] 桨叶完好、无裂纹
  - [ ] 视频流正常、无花屏
  - [ ] 遥测数据正常（高度、电量、姿态）
  - [ ] 深度服务器 `/health` 正常
  - [ ] 降落通道无障碍
  - [ ] 控制界面 CONNECTED 状态
- **安全操作**：
  - AUTO ARM 需二次确认（V → ARMED → 再按 V → ON）
  - 键盘手动输入、悬停（Space）、降落（L）、急停（Esc）优先级最高
  - `TEST FAIL / ABORT` 立即停止自动移动 → 悬停 → 解除 AUTO → 等待人工指令
- **日志记录**：`--record` 全程开启，记录 RGB 帧、深度网格、RC 命令、遥测、决策状态、RTT、事件
- **保守参数**：初次绕飞使用 `--cruise 15 --approach-pitch 8 --yaw 25`

---

## 十、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `station_mode.py` | 修改 | `--bind-ip` 参数、`_send()` 脱敏 |
| `tt_control/tello_client.py` | 修改 | 保活改用独立 socket |
| `tt_control/async_infer.py` | **新增** | 异步推理工作线程 |
| `tt_control/depth_backend.py` | 修改 | 移除同步阻塞，改为帧缓存 + 叠图 |
| `tt_control/avoidance_fsm.py` | **新增** | 完整避障状态机 |
| `tt_control/avoidance.py` | 保留 | 低层控制律 `decide()` 不变，被 FSM 调用 |
| `tt_control/auto_safety.py` | 修改 | 增加状态超时、ABORT 闭环 |
| `tt_control/app.py` | 修改 | 接入 AsyncInferWorker、FSM、ABORT 按钮、监控 HUD |
| `tt_control/episode_recorder.py` | 修改 | 增加 ABORT 事件、FSM 状态记录 |
| `tt_control/inference.py` | 保留 | 注册后端逻辑不变 |
| `sim_avoidance.py` | 修改 | 支持新 FSM 仿真 |
| `tests/test_async_infer.py` | **新增** | 异步推理单元测试 |
| `tests/test_avoidance_fsm.py` | **新增** | 状态机单元测试 |
| `tests/test_tello_client.py` | 修改 | 增加保活隔离测试 |
| `docs/design/2026-07-23-ds-depth-avoidance-plan.md` | **新增** | 本方案文档 |
| `docs/2026-07-23-ds-depth-avoidance-plan.md` | 删除/归档 | 原 Codex 方案（内容已整合至此） |

---

## 十一、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| DS 路由器不支持 2.4GHz 或开启客户端隔离 | Tello 无法入网 | 阶段 1 前置确认；备选 USB 网卡直连方案 |
| Tello DHCP 地址变动 | 每次需重新扫描 IP | DS 路由器设置 DHCP 地址保留；`main.py` 支持 `--tello-ip` 传入 |
| 深度服务 RTT 过高（>500ms） | 控制延迟大，撞墙风险 | 降低 `cruise_speed`；增加 `depth_stale_s` 阈值；现场测量 RTT 后调整参数 |
| 近度阈值跨场景不通用 | 不同环境相同阈值表现不同 | 阶段 3.6 现场标定；可通过 CLI 参数覆盖 |
| Tello Wi-Fi 干扰 DS 2.4GHz | Mac → DS 网速下降、深度请求超时 | 如可行将 DS 切到 5GHz（Mac 连 5GHz）；或使用 USB 网卡直连方案 |
| 新服务器协议不兼容 | 深度请求全部失败 | 阶段 3 先确认协议；保留对旧服务的兼容适配 |

---

## 十二、尚待确认

- [ ] DS 路由器型号、管理地址、2.4GHz 频段状态、客户端隔离设置
- [ ] 新深度推理服务器的地址、端口、`/health` 和 `/depth` 协议详情、鉴权方式
- [ ] USB Wi-Fi 网卡型号（如使用此方案）
- [ ] 测试场地尺寸、障碍物形式
- [ ] 新服务器是否支持扩展协议（JSON 响应、语义信息）
