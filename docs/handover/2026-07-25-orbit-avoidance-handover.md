# Tello 视觉避障 + POI 环绕飞行：开发交接文档

日期：2026-07-25
状态：✅ 环绕飞行已验证可用，继续迭代中

## 一、功能概述

半自动视觉避障系统，基于 Depth Anything V2 深度估计实现：

| 功能 | 状态 | 说明 |
|------|------|------|
| **半自动巡航** | ✅ | 飞机直飞前进，遇障自动绕行 |
| **POI 环绕** | ✅ | 检测到椅子后，以椅子为中心绕圈飞行，保持固定距离 |
| **视觉伺服居中** | ✅ | 实时追踪椅子在画面中的位置，偏了停横移、yaw 拉回中央 |
| **安全保护** | ✅ | 太近悬停、椅子丢失超时悬停、低电量/高度异常保护 |

## 二、快速启动（Agent 可直接执行）

### 环境

- Mac：Apple M5，IP `192.168.0.103`
- Tello：IP `192.168.0.100`（station mode，已配置入同一路由器）
- Python：`.venv/bin/python`（Python 3.11，所有依赖已安装）
- 深度推理服务 venv：`server/.venv/bin/python`（torch 2.13, MPS 可用）

### 启动步骤

```bash
# 1. 启动本地深度推理服务（M5 MPS，端口 8890）
server/.venv/bin/python server/da_v2_service.py --host 0.0.0.0 --port 8890 --grid 96x128 &

# 2. 启动控制界面（指向本地推理服务）
.venv/bin/python main.py \
  --tello-ip 192.168.0.100 \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --depth-service http://127.0.0.1:8890/depth \
  -v
```

### 飞行操作

1. 点击 **CONNECT** 连接飞机
2. 点击 **TAKEOFF** 或按 `T` 起飞
3. 按 `V` 第一次 → AUTO **ARMED**（待命）
4. 按 `V` 第二次 → AUTO **ON**（开始自主飞行）
5. 飞机会自动前进，检测到障碍物后切入 ORBIT 环绕
6. 按 **SPACE** 悬停（同时解除 AUTO），按 **L** 降落

## 三、核心代码结构

```
tt_control/
├── avoidance.py        ← 控制律：AvoidanceController + OrbitController
├── avoidance_fsm.py    ← 状态机：AvoidanceFSM（APPROACH → ORBIT → ...）
├── app.py              ← GUI + 参数配置（AvoidParams / FsmParams 入口）
├── depth_backend.py    ← 深度客户端：DepthAnythingBackend（POST JPEG → nearness）
├── async_infer.py      ← 异步推理线程：AsyncInferWorker
├── tello_client.py     ← Tello UDP 协议客户端
├── status.py           ← 飞机在线探测（ping + UDP probe）
└── control.py          ← RcAxes、键盘映射、HELP_TEXT

server/
└── da_v2_service.py    ← 深度推理服务端（Flask-less，标准库 http.server）

station_mode.py         ← Tello 组网工具（setup / find）
main.py                 ← 入口
```

### 关键类关系

```
App (app.py)
 ├── TelloClient          UDP 命令 + 状态流
 ├── VideoStream          H.264 图传解码
 ├── DepthAnythingBackend 异步深度推理客户端
 │    └── AsyncInferWorker  工作线程（JPEG→POST→nearness grid）
 ├── AvoidanceController  底层控制律（nearness → RcAxes）
 ├── OrbitController      POI 环绕控制律（视觉伺服居中）
 └── AvoidanceFSM         上层状态机（FSM → 组合上述控制器）
```

## 四、ORBIT 环绕控制律（核心设计）

### 4.1 视觉伺服优先策略

```
每帧:
  chair_pos = 重心法算椅子在画面中的水平位置 [-1, 1]

  ┌─ |pos| < 0.12 (居中，phase="strafe")
  │   → yaw = pos × 200 (微调)
  │   → roll = 10 (慢慢横移环绕)
  │
  ├─ |pos| ≥ 0.12 (偏了，phase="center")
  │   → yaw = pos × 200 (全力拉回，max ±50)
  │   → roll = 0 (停横移！让 yaw 先把椅子拉回中央)
  │
  └─ pos = None (椅子丢了，phase="search")
      → yaw = 0, roll = 0 (等待)
```

### 4.2 椅子位置检测（重心法）

`OrbitController._chair_horizontal(nearness)`：
- 取深度图中部水平带（30%~80%）
- 每列中位 nearness，减去 0.10 背景噪声作为权重
- 计算加权重心 → 归一化到 [-1, 1]

### 4.3 距离保持

```
dist_err = target_nearness - M (中区 nearness)
|err| ≤ deadband(0.05)  → pitch = 0 (不调整)
|err| > deadband        → pitch = err × gain(2.5) × 20, 钳位 ±max_pitch(35)
```

### 4.4 当前参数（OrbitParams，在 avoidance.py）

| 参数 | 值 | 说明 |
|------|----|------|
| `direction` | 1 | 环绕方向（1=顺时针，-1=逆时针） |
| `target_nearness` | 0.55 | 目标距离（~1m） |
| `distance_deadband` | 0.05 | 距离死区 |
| `orbit_roll` | 10 | 环绕横移速度（居中时） |
| `yaw_centering_gain` | 200 | yaw 比例增益（pos × gain = yaw 杆量） |
| `centering_deadband` | 0.12 | 居中门槛（|pos|小于此值允许横移） |
| `pitch_distance_gain` | 2.5 | 距离控制增益 |
| `max_yaw` | 50 | yaw 上限 |
| `max_pitch` | 35 | pitch 上限 |
| `danger_thresh` | 0.78 | 太近触发悬停保护 |
| `lost_thresh` | 0.10 | 椅子丢失判定（nearness 低于此值） |

## 五、状态机流程（AvoidanceFSM）

```
PREFLIGHT → (起飞) → HOVER
                       │
                  V 两次 AUTO ON
                       ▼
                   APPROACH
                    │     │
                    │     ├─ 通畅 → CRUISE (pitch=30 前进)
                    │     │
                    │     └─ M ≥ orbit_enter_nearness(0.35)
                    │           │
                    │           ▼
                    │       ORBIT (在 APPROACH 内运行)
                    │        │
                    │        ├─ 正常 → 持续环绕
                    │        ├─ DANGER (M > 0.78) → HOVER
                    │        └─ LOST 超时 (5s) → HOVER
                    │
                    └─ 超时 (max_auto_engaged_s=120s) → HOVER
```

注：`orbit_mode=True` 时，ORBIT 在 APPROACH 状态内部运行。非环绕模式会走 AVOID_TURN → PASS_OBSTACLE → RECOVER_HEADING 路径。

## 六、调试排障记录

### 问题 1：20 秒超时掐断环绕 ✅ 已修复
- **现象**：环绕 20s 后 `AUTO watchdog disengage: approach >20s`
- **原因**：ORBIT 在 APPROACH 状态内运行，`max_approach_s=20s` 超时检查在 `_orbit_active` 检查之前
- **修复**：`avoidance_fsm.py:259`，环绕激活时跳过 APPROACH 超时

### 问题 2：刚进环绕就被 DANGER 踢出 ✅ 已修复
- **现象**：环绕 2 秒后 M 从 0.50 冲到 0.65+ 触发 DANGER
- **原因**：`orbit_enter_nearness=0.50` 离 `danger_thresh=0.65` 太近，没缓冲
- **修复**：`orbit_enter_nearness` 降到 0.35，`danger_thresh` 提到 0.78

### 问题 3：环绕中椅子偏出视野 ✅ 已修复
- **现象**：M 值 0.38→0.24，椅子越来越远；yaw 只有 ±5-6，转不动
- **原因**：roll=15 恒定横移，yaw 太弱追不上，椅子滑出视野
- **修复**：重新设计为视觉伺服优先——椅子偏离时停横移(roll=0)，yaw 加大到 ±50 全力拉回；居中后才慢慢横移(roll=10)

### 问题 4：yaw 振荡 ✅ 已修复
- **现象**：yaw 频繁翻转 +8↔-8
- **原因**：`yaw_centering_gain=3.0` 导致小误差立即饱和，`min_yaw=8` 强制钳位
- **修复**：改用直接增益 `yaw = pos × 200`，取消 min_yaw 钳位

## 七、本地推理服务

- **模型**：`depth-anything/Depth-Anything-V2-Small-hf`
- **设备**：Apple M5 MPS（Metal Performance Shaders）
- **端口**：8890
- **协议**：POST JPEG → binary nearness grid (float16, H×W)
- **服务端代码**：`server/da_v2_service.py`
- **客户端代码**：`tt_control/async_infer.py` + `tt_control/depth_backend.py`
- **默认服务 URL**：`https://depth.david-x.com/depth`（需要时用 `--depth-service` 覆盖）

## 八、继续开发/测试的方向

1. **环绕稳定性**：调整 `target_nearness`、`orbit_roll`、`centering_deadband` 微调
2. **改变环绕方向**：修改 `OrbitParams.direction = -1` 逆时针
3. **支持远程推理**：VPN 或其他方式访问 `10.229.20.125:8899` 或 `depth.david-x.com`
4. **多障碍物处理**：当前只追踪中区最大物体，多障碍场景未覆盖
5. **录制回放**：`--record` 参数可录制 episode 数据到 `logs/episodes/`
6. **测试**：`pytest tests/test_avoidance.py -v` 验证控制律

## 九、监控命令

```bash
# 找飞机
python3 station_mode.py find

# 查看实时日志
tail -f /tmp/claude-*/tasks/*.output 2>/dev/null

# 查看推理服务状态
curl http://127.0.0.1:8890/health

# 跑单元测试
.venv/bin/pytest tests/test_avoidance.py -v
```
