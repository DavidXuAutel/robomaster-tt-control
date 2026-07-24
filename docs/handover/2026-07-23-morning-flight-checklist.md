# 明早开飞清单 · 新服务器(2026-07-23)

服务器:`a26125@10.229.20.110`(自带 4090)。**已全部就绪**:代码 + 主 venv(55 pytest 绿)+ 深度服务 + 仿真栈。
分工:**校正测试我(SSH)驱动脚本、你摆卡看飞机;避障测试你在控制台跑 main.py 按键、我看日志出报告。** 安全铁律:人在飞机旁,手能断电。

---

## 0. 开工前(~2 分钟)

1. **深度服务在跑?**(避障要用)控制台或让我跑:`~/da_v2_service/start.sh` → 应回 `{"ok": true ... cuda: true}`。
2. **WiFi 切到飞机**(控制台,权限只能你连):`~/tello-wifi.sh` → 等"✓ 飞机可达";若掉线重连同命令。
3. 飞机满电、桨叶保护圈、软地面、场地无人。

---

## A. 校正测试(先做:验新服务器上"~1.5s 滞后"是否消失)

> 背景:旧服务器满载,疑似 CPU 抢占拖慢了位置遥测 → 位移比命令滞后 ~1.5s、VMAX 测不准。新服务器空闲,重测看是否恢复。

- 摆 **1 张卡**,飞机放其上、机头对准火箭。
- 告诉我 → 我跑:`calibrate_flight.py --axis fwd --stick 20 --dur 2.0`(起飞→前进→后退→降落,自动)。
- 我立刻看 pos 轨迹判断:
  - **动作与位移对齐了 + 飞得更远** → 大众猜想成立;接着**多卡长线**测准 VMAX(卡:同向、中心距 ~40cm、排成 ~1.2m 直线),我跑 `calib_vmax.py`。
  - 仍滞后/只走 ~35cm → 是飞机/垫子固有,记录归档,VMAX 用 ~0.6 定性值。

---

## B. 避障测试(主目标:验能力边界 + 采世界模型数据)

- 挡板:**大、不透明、有纹理**;正前偏一侧留空档;距起飞点 3~4m。
- **你在控制台本机跑**(GUI + 录制):
  ```
  cd ~/Projects/robomaster-tt-control
  .venv/bin/python main.py --inference depth-anything --depth-service http://127.0.0.1:8899/depth --record --mujoco
  ```
- 键序:`C` 连接 → 看电量/图传 → `T` 起飞悬停 → `V`(ARMED)→ `V`(ON 接管)。
- 观察并逐项记:靠近**减速** → 到阈值**朝空侧拐弯 TURN(不横跳)** → 宽墙堵住**原地悬停 BLOCKED**。
- **失败降级链**(AUTO ON 时逐条,低空小范围):
  1. 我在 SSH 端 `pkill` 深度服务(或你挡镜头)→ 预期 HUD `depth stale`、**自动悬停解除**;(测完我重启 `start.sh`)
  2. 保持不动等 **>30s** → 预期自动解除;
  3. 按 `WASD` → 预期瞬间夺权。
- 急停:`SPACE` 悬停 / `ESC` 急停 / `L` 降落 —— 手别离开键盘。
- 每场景 ≥5 次看一致;数据自动落 `logs/episodes/ep_*/`。

---

## 收尾

- 核对 `logs/episodes/ep_*/`:`video.mp4` 能播、`frames.csv` 行数正常、`meta.json`(outcome/action_source/bat 合理)。
- 直传回来或留服务器,我出汇总(能力边界实测结论 + 数据交付)。

---

## 常用命令备忘

| 用途 | 命令 |
|---|---|
| 起/查深度服务 | `~/da_v2_service/start.sh` |
| 连飞机 WiFi | `~/tello-wifi.sh`(断:`off`,回办公网:`office`) |
| 校正飞行(我跑) | `.venv/bin/python calibrate_flight.py --axis fwd --stick 20 --dur 2.0` |
| 校正报告(我跑) | `.venv/bin/python calib_vmax.py logs/calib/<最新>.csv --stick 20` |
| 避障+录制(你控制台跑) | `.venv/bin/python main.py --inference depth-anything --depth-service http://127.0.0.1:8899/depth --record --mujoco` |

> 深度服务与代码环境重启后仍在(setsid 常驻);**只要服务器不关机,明早直接从第 0 步开始即可。**
