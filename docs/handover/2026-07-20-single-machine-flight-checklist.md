# RoboMaster TT 视觉避障 — 单机（服务器一肩挑）挡板首飞傻瓜 Checklist

日期：2026-07-20
适用场景：**场景 A — 一台带 WiFi + 屏幕键盘 + GPU 的工作站，同时跑感知服务 + 控制程序 + 直连飞机**
设计依据：[`../design/2026-07-17-tt-visual-avoidance-design.md`](../design/2026-07-17-tt-visual-avoidance-design.md)
配套通用清单：[`./2026-07-20-real-flight-test-checklist.md`](./2026-07-20-real-flight-test-checklist.md)

> 本清单专为"服务器就在现场、一台机全干"写。感知走本地回环 `127.0.0.1`，无线链路只剩"这台机 WiFi 连飞机"一条。
> 铁律：**人必须在这台机的键盘旁边，手放急停。飞行操作只用本机本地键盘，不要靠远程 SSH/VNC 遥控起飞。**

---

## 阶段 0 — 出发前（在工位，不接飞机）

- [ ] 拉最新代码到服务器，装好环境（感知服务依赖 `requirements-avoidance.txt`）
- [ ] 控制律纯软件自证：
  - `python sim_avoidance.py --scenario wall`（正前方一堵墙）
  - `python sim_avoidance.py --scenario slalom`（连续障碍）
  - 通过：`outcome=CLEARED`、`min_clearance>0`
- [ ] 单测全绿：`python -m pytest tests/ -q`
- [ ] 电池充满 + 备用电池；螺旋桨/桨叶保护圈完好

## 阶段 1 — 感知服务自检（还不飞，服务器本地）

在服务器上开**终端 1**，起感知服务（前台留着看日志）：

```bash
python server/da_v2_service.py --host 127.0.0.1 --port 8899
```

- [ ] 开**终端 2** 健康检查：`curl http://127.0.0.1:8899/health` 返回 `ok:true`
- [ ] （可选）拿一段"走向挡板"的录像离线回放，确认靠近时热力图变红、决策由 CRUISE→TURN/BLOCKED：
  - `python offline_avoidance.py --source clip.mp4 --service http://127.0.0.1:8899/depth --out out.mp4 --show`

> 注意服务地址路径是 `/depth`，端口 `8899`，和控制程序 `--depth-service` 要一致。

## 阶段 2 — 场地与挡板

- [ ] 场地：开阔、软地面（草地最佳）、无人、镜头别正对强光
- [ ] 挡板：**大块、不透明、有纹理**（纸箱 / 泡沫板 / 深色布幕）；**别用玻璃或纯白反光墙**（深度模型看不清）
- [ ] 想看"拐弯"：挡板放正前方**偏一点**，留出一侧空档
- [ ] 想看"停住"：用够宽的墙把左右都堵上
- [ ] 挡板距起飞点约 **3~4 米**

## 阶段 3 — 网络（单机最简，一步）

- [ ] 服务器 WiFi 连上飞机热点 `TELLO-xxxx`（本机进入 `192.168.10.x` 网段）
- [ ] 感知走本地回环，无需额外配置

## 阶段 4 — 定点航线热身（可选，先不避障）

- [ ] `python fly_real_mission.py --sim`（离线走一遍）
- [ ] `python fly_real_mission.py --speed-cms 30`（真机飞固定航线，电量 <25% 脚本会拦）
- [ ] 降落后核对 `logs/` 下轨迹 PNG

## 阶段 5 — 挡板避障首飞（核心）

**终端 1** 保持感知服务在跑。开**终端 3** 起控制程序：

```bash
python main.py --inference depth-anything --depth-service http://127.0.0.1:8899/depth --mujoco
```

飞行时序（**记死这套按键，全程本机本地键盘操作**）：

1. [ ] `T` — 起飞，原地悬停几秒
2. [ ] 看 HUD：**无** `waiting depth service`（感知在线）、左/中/右近度有数
3. [ ] `V` — 进入 `ARMED`（预备，还不动）
4. [ ] 再 `V` — 进入 `ON`，避障接管，开始缓慢前飞朝挡板
5. [ ] 观察预期行为：
   - 靠近 → 近度升高 → **减速**（pitch 随近度线性减小）
   - 到 `clear_thresh(0.45)` → **朝更空一侧偏航拐弯**（方向锁定不横跳）
   - 左中右全 `>stop_thresh(0.70)` → **原地悬停 BLOCKED**
6. [ ] 首飞先只放**一个**挡板，验证"减速+拐开"就收；成功再上多障碍

## 安全与急停（手别离开键盘）

| 键 | 效果 |
|---|---|
| `WASD/QE/RF` | 立即人工接管，覆盖避障 |
| `SPACE` | 悬停并关闭 AUTO |
| `ESC` | 急停（停桨），关闭 AUTO |
| `L` | 降落，关闭 AUTO |

- AUTO 仅在**已起飞 + 深度后端在线**时可挂载。
- **AUTO 看门狗（已接入）**：单次 AUTO 超 30s、或感知失联超 1.5s 会自动悬停解除 AUTO，HUD 显示原因。**但它是兜底，人工接管才是第一保障。**

## 异常处置

- 图传卡死 / 感知长时间无返回 → `SPACE` 悬停 → `L` 降落 → 查终端 1 感知服务日志。
- 位移指令返回非 `ok` → 悬停后重发。
- 任一异常无法恢复 → `ESC` 急停兜底。

## 收尾归档

- [ ] `logs/` 下日志 / 遥测 CSV / 轨迹 PNG / 标注视频已保存
- [ ] 现场参数（感知 RTT、实际生效阈值、结论）回填本清单或 dev-notes
