# 真机验证 Runbook —— 手势起飞(优先)  2026-07-19

## 现状与前提
- 操作:人在 4090 服务器控制台(显示器 :1 + 键盘),飞机在服务器附近飞。
- 手势环境:独立 `~/gesture_venv`(mediapipe/numpy2),**不是**主 .venv。
- **手势对着飞机前置摄像头做**(后端处理的是 Tello 图传帧)。
- 无 Mission Pad → 目标1(采数/孪生)本次不做;目标2(避障)见末尾「后续」。

## 安全总则(务必)
- 空间:≥2m 见方空地,**头顶 ≥1.5m 无遮挡**(手势起飞会升到 50cm)。
- 电量 >50%,飞机放平整地面。
- **全程一只手放键盘:`ESC`=急停停桨、`L`=降落、`SPACE`=悬停。** 建议两人:A 看护键盘,B 做手势。
- 远离人/宠物/易碎物;室内低速首飞。

## Phase 0 — 准备(不起飞)
0.1 飞机开机、放平地、清场。
0.2 服务器 WiFi 连飞机热点(有线网口保 SSH 不断):
```
nmcli dev wifi rescan
nmcli dev wifi connect TELLO-XXXXXX          # 换成飞机实际热点名
ip -brief addr show wlx6c1ff783eb62          # 应出现 192.168.10.x
ping -c 2 192.168.10.1                        # 通
```
✅ 通过:wifi 拿到 192.168.10.x 且 ping 通;SSH(走有线 eno1)不断。
0.3 环境自检:
```
cd ~/Projects/robomaster-tt-control
~/gesture_venv/bin/python -c "import mediapipe,av,cv2,tt_control.gesture; print('env ok')"
```

## Phase 1 — 手势识别「干跑」(飞机不动,只验识别)
```
cd ~/Projects/robomaster-tt-control
DISPLAY=:1 ~/gesture_venv/bin/python gesture_dryrun.py
```
1.1 桌面弹出窗口,显示飞机图传 + "DRY-RUN NO FLIGHT"。
1.2 **空场**对着飞机前摄(无手):HUD 显示 "no hand",无 FIRED 打印,飞机纹丝不动(观察 ~20s,排除误触发)。
1.3 对前摄做**食指竖起、其余手指蜷曲**并保持:HUD 进度条涨满 → 终端打印 `>>> FIRED: takeoff_80`(飞机**仍不动**)。
1.4 做**拇指+中指捏合再快速弹开**(响指):打印 `>>> FIRED: land`。
✅ 通过:两手势都能稳定 FIRED,空场不误触发。`ESC` 退出。
识别不灵:手距摄像头 0.3–1m、光线充足、背景简洁;仍差则告诉我调 `hold_frames`/阈值。

## Phase 2 — 真机手势起飞(真飞行!)
2.1 再次确认:清场、头顶净空、电量、键盘手就位。
2.2 起 App:
```
cd ~/Projects/robomaster-tt-control
DISPLAY=:1 ~/gesture_venv/bin/python main.py --gesture -v
```
2.3 点 `CONNECT`(或按 `C`):状态灯 → CONNECTED,出现图传,HUD 显示 "Gesture ON"。
2.4 **空场确认不误触发**(HUD "no hand",飞机不动)。
2.5 **确认头顶净空后**,对前摄做食指竖起并保持 → 进度条满 → **飞机起飞并升到 ~50cm 悬停**。
    ✅ 通过:手势稳定触发起飞、悬停在 ~50cm。异常立即 `ESC`(停桨)或 `L`(降落)。
2.6 降落:做响指手势 → 降落;或直接按 `L`。

## Phase 3 — 记录
- 记:需保持几秒触发、有无误触发、起飞高度、异常。
- 需要日志给我看:上面命令已带 `-v`,可加 `> ~/gesture_run.log 2>&1` 事后回看。

## 出问题怎么办
- 连不上飞机:确认 wifi 在 192.168.10.x、飞机电量、重开飞机再 `nmcli ... connect`。
- 图传黑屏:多为 streamon 失败 → 断开重连;确认在 192.168.10 网段。
- **误触发起飞**:立刻 `ESC`;简化背景、增大 `hold_frames`(告诉我改)。
- 想中止:`ESC` 停桨(会直接掉落,仅紧急用)/ `L` 正常降落 / 关窗口前先降落。

## 后续(本次不做,备查)
- 目标1 采数/孪生:**需 Mission Pad**。有了之后,用**主 .venv**(非手势 venv):
  `DISPLAY=:1 .venv/bin/python main.py --mujoco` → 垫子上方飞 → 降落后 `.venv/bin/python -m tt_control.trajectory_plot logs/trajectories/traj_*.csv`。
- 目标2 避障:用**主 .venv**:`DISPLAY=:1 .venv/bin/python main.py --inference depth-anything -v`
  → 起飞后按 `V`(OFF→ARMED→再按→ON)→ 前方放障碍观察绕行。depth 服务已在 :8899 常驻。**先小范围、随时 ESC/WASD 接管**。
- 注意:避障(主 .venv/numpy1)与手势(gesture_venv/numpy2)**不能同一进程**,分开跑。
