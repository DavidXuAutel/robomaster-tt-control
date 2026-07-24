# 三会话代码合并集成说明 (2026-07-18)

把三个并行开发的功能合并进服务器同一份代码,回归 + 端到端测试通过。

## 合并的三块功能
1. **离线仿真测试台**(会话A/本次):`sim_drone.py`(SimDrone/SimVideo)、`sim_runner.py`、`trajectory_plot.py`(轨迹实线)、`mujoco_twin.py` 无头模式 + 采样 bug 修复。`--sim` 启用。
2. **深度视觉避障**(会话B):`tt_control/avoidance.py`(控制律)、`tt_control/depth_backend.py`(连 GPU 深度服务)、`server/da_v2_service.py`(Depth Anything V2,已在 4090 :8899 常驻)、`offline_avoidance.py`、`sim_avoidance.py`。`--inference depth-anything` + `V` 键 OFF→ARMED→ON。
3. **手势控制**(会话C):`tt_control/gesture.py`(MediaPipe Hands:食指上抬=起飞、响指=降落)。`--gesture` 启用。

## 关键合并决策:自动控制统一
会话A 的"policy AUTO(G 键)"与会话B 的"深度避障(V 键)"**概念重叠**(都是视觉→自动 RC),且都占用 `self._auto`。按"统一"方案:**避障作为唯一真机自动路径**(V 键),A 的 policy-AUTO/G 退役(它本是给"别人的模型"占位,避障就是那个模型)。A 的**离线仿真台/轨迹实线/孪生修复保留**(与避障正交,用于离线开发与验证)。`policy.py`+`sim_runner.py` 保留为离线仿真驱动(测试用)。

三条控制路径现并存于 app.py:真机手动(TelloClient)、避障自动(AvoidanceController)、离线仿真(SimDrone)。

## 冲突解决
- app.py:git 三方合并(base=7-16),仅 `_auto`/`_toggle_auto` 冲突 → 按统一方案取避障版,叠加 sim 分支。
- config/main:加法合并(gesture/depth 参数 + `--sim`)。
- control.py:取会话B(`v`=auto_toggle)。inference/status:取会话B(depth/gesture 后端 + 跨平台 ping,超集)。
- 原文件备份:`*.orig-20260718`、`.merge-backup-mine/`。

## 测试状态
- **回归 pytest:18 项全绿**(control/sim_drone/policy/trajectory/integration + 新增 avoidance)。
- **E2E**:
  - 避障 2D 仿真 `sim_avoidance.py`:min_clearance>1m 无碰撞(WANDER=无目标反应式已知特性)。
  - 深度感知服务 + DepthAnythingBackend + 控制律:真服务(:8899)端到端跑通。
  - App 离线 sim 粘合层:SimDrone 连接/起飞/图传/关闭 OK。
  - sim_runner 闭环:正常。

## 已知事项
- **手势需 mediapipe**,但 mediapipe 要 numpy≥2,与主 .venv 锁定的 numpy 1.21(mujoco/matplotlib 编译依赖)**冲突**。故**手势须用独立 venv 跑**(仿照 `~/da_v2_service/.venv` 模式),或在已装 mediapipe 的本机跑。缺 mediapipe 时 `--gesture` 会优雅报错并给安装指引,不崩。
- `offline_avoidance.py` 需真机视频源;深度链路已用合成帧对真服务验证过。

## 运行速查
    python main.py --sim                              # 离线仿真(无飞机)
    python -m tt_control.sim_runner 250 mock          # 无头闭环 + 轨迹
    python sim_avoidance.py --scenario slalom          # 避障控制律 2D 验证
    python main.py --inference depth-anything          # 真机+深度避障(V 键 ARM/ON)
    python main.py --gesture                           # 真机+手势(需 mediapipe 独立 venv)
