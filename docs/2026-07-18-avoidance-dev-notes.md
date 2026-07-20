# RoboMaster TT 视觉避障 —— 离线开发说明 (2026-07-18)

按大众思路(真机优先):**真机自己飞 → 仿真记录整条轨迹并导出实线 → 证明真机↔仿真匹配 → 现场部署"别人的模型"跑真机**。
本次在**无飞机**条件下,把除"真模型"外的整条链路开发好并离线测通;到场把 IO 切回真机即可跑。

## 已完成(离线可测)
- 仿真无人机(可替换真机)+ 合成图传,驱动同一套 App/孪生/策略代码。
- 自动控制通路:帧+状态 → 策略 → RcAxes → 下发,含 AUTO 挂载/解除、手动随时接管、超时看门狗。
- 轨迹"实线"导出(大众要的交付物):俯视 XY 实线 + 高度-时间图。
- MuJoCo 孪生新增无头记录模式(服务器无显示器也能录轨迹)。
- 一套 pytest(14 项全绿)。

## 未完成(需外部输入)
- **"别人的模型"未接入**:其输入输出/调用方式未知 → 预留 `ExternalModelPolicy` 适配点(见 policy.py)。到手后仅需实现该 adapter。若是单目深度类模型(如 Depth Anything),按"深度图→最近障碍方位→避让杆量"映射即可复用现有思路。
- **真机实飞**:需飞机在场 + 场地。服务器双网卡可行(有线保 SSH、WiFi 连飞机热点),到场用 nmcli 连接即可。

## 新增文件
- tt_control/sim_drone.py     : SimDrone / SimVideo(鸭子类型兼容真机 IO)
- tt_control/policy.py        : Policy 协议 + MockAvoidPolicy / ScriptedPolicy / ExternalModelPolicy(stub)+ create_policy
- tt_control/trajectory_plot.py: 轨迹 CSV → 实线 PNG(含 CLI)
- tt_control/sim_runner.py    : 无头闭环会话 run_sim_session(测试台/演示)
- tests/                      : test_control / test_sim_drone / test_policy / test_trajectory / test_integration_sim

## 改动文件(均已备份为 *.orig-20260718)
- tt_control/config.py   : AppConfig 增 sim / auto
- tt_control/app.py      : _connect 增 sim 分支;__init__ 增 policy;AUTO 挂载/解除 + 手动接管 + 看门狗;循环内调用策略;UI 显示 SIM/AUTO
- main.py                : 增 --sim / --policy
- tt_control/control.py  : 增按键 g = auto_toggle
- tt_control/mujoco_twin.py : 增 headless 记录模式;**修复 _append_point 采样锚点漂移 bug**(慢速/悬停时原逻辑只记 1 点 → 现按固定锚点每 ≥2cm 记一点)

## 如何运行
### 现在(离线,公司网即可)
    # 无头闭环 + 出实线图
    .venv/bin/python -m tt_control.sim_runner 250 mock
    .venv/bin/python -m tt_control.trajectory_plot logs/trajectories/traj_*.csv out.png
    # 带界面(在有显示器的 :1 上):SIM + 自动策略 + 孪生窗口
    #   T 起飞 → G 挂载 AUTO(任意手动键/SPACE/ESC 立即接管)→ L 降落
    DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority .venv/bin/python main.py --sim --policy mock --mujoco
    # 测试
    .venv/bin/python -m pytest tests/ -q

### 到场(真机)
    # 1. 服务器 WiFi 连飞机热点(有线保 SSH):nmcli dev wifi connect TELLO-XXXX
    # 2. Phase 1 匹配验证:手动飞,孪生记录
    .venv/bin/python main.py --mujoco
    #    降落后导实线:python -m tt_control.trajectory_plot logs/trajectories/traj_*.csv
    # 3. Phase 2:实现 ExternalModelPolicy 后
    .venv/bin/python main.py --policy external --mujoco   # T 起飞 → G 挂载

## 安全
- AUTO 仅在起飞后可挂载;手动杆量 / SPACE 悬停 / ESC 急停 / 降落 任一即刻解除 AUTO。
- 单次 AUTO 超过 30s 自动解除并悬停(看门狗)。
- 不带 --sim / 不带 --policy 时,行为与改动前完全一致(真机手动飞行路径未受影响)。

## 与 GitHub 对账
本目录为服务器手工拷贝(非 git)。以上"新增/改动文件"即本次全部变更;原文件见同名 *.orig-20260718。
后续接 git 时以此清单为准 merge 到分支。
