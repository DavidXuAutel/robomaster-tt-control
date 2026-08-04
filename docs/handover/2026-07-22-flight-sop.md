# 现场飞行 SOP —— 校正飞行 + 避障采集飞行

日期:2026-07-22 · 适用:RoboMaster TT 真机测试(服务器一肩挑:感知+控制+直连飞机)
前置:所有在线准备已完成(代码已同步服务器且 55 项 pytest 绿、感知服务 `:8899` 在跑且 CUDA、网络就绪)。

> **安全铁律:** 人必须在**服务器物理控制台**旁,手放急停键。**飞行按键在服务器本机键盘按,不靠 SSH/VNC 盲飞。** 我(Claude)SSH 在线只负责看日志、跑校正报告、必要时重启感知服务——**不代按飞行键**。

---

## 通用准备(每次飞行前)

- [ ] 电池充满 + 备用;桨叶保护圈完好;场地开阔、软地面(草地最佳)、无人。
- [ ] 登录服务器**物理显示器**的桌面会话(main.py 要开 OpenCV 窗口)。
- [ ] 飞机上电,等自检完成。
- [ ] 服务器 WiFi 网卡连飞机热点:
      ```
      nmcli device wifi rescan
      nmcli device wifi connect TELLO-XXXX      # 换成实际 SSID
      ip -4 -o addr show wlx6c1ff783eb62         # 确认进入 192.168.10.x
      ```
- [ ] 确认 `eno1` 仍在(`ip -4 -o addr show eno1` → 10.229.20.125),我这边 SSH 不断。

---

## 飞行 A · 校正飞行(目标 2a,先做,拿仿真可信度证据)

**目的:** 标定"仿真=真机可信替身"。**单卡、机头对准、全程不转向。**

- [ ] 地上摆 **1 张** Mission Pad,飞机放其正上方,**机头对准火箭方向**。
- [ ] 服务器本机终端启动(不接感知服务):
      ```
      cd ~/Projects/robomaster-tt-control
      .venv/bin/python main.py --record --mujoco
      ```
- [ ] 键序(**本机键盘**,全程不按 Q/E):
      1. `T` — 起飞,悬停 ~3s;
      2. `W` — 前进,**到 HUD 仍显示 pad 锁定的最远处就松手**(别丢卡,HUD 有 `pad m1 ...` 字样);
      3. `S` — 后退回起点附近;
      4. `L` — 降落。
- [ ] 落地后告诉我 episode 名(或我直接看 `logs/episodes/` 最新),我在线跑:
      ```
      .venv/bin/python sim_match_report.py logs/episodes/ep_<最新>
      ```
- [ ] **验收:** `endpoint_error_m` 小、`mae_m` 小、`path_len_ratio≈1`;`match_report.png` 两条线贴合。
- [ ] 重复 **3~5 次**,确认稳定。

---

## 飞行 B · 避障采集飞行(目标一验证 + 目标 2b 采数据,同批完成)

**目的:** 验证避障能力边界 + 采世界模型视频数据。**可不用 pad。**

- [ ] 挡板放正前方**偏一侧、留出空档**(先验"拐弯"),距起飞点 **3~4m**;挡板**大、不透明、有纹理**(禁玻璃/白墙/反光)。
- [ ] 服务器本机终端启动(接感知服务):
      ```
      cd ~/Projects/robomaster-tt-control
      .venv/bin/python main.py --inference depth-anything \
          --depth-service http://127.0.0.1:8899/depth --record --mujoco
      ```
- [ ] 键序 + 观察:
      1. `T` — 起飞,悬停;
      2. 看 HUD:**无** `waiting depth service`、左/中/右近度有数;
      3. `V` — 进 `ARMED`(预备,不动);
      4. 再 `V` — 进 `ON`(接管,朝挡板缓飞);
      5. 观察预期行为:靠近→**减速**(pitch 随近度降)→ 到阈值**朝空侧拐弯(TURN)、方向不横跳**;
      6. 换"够宽的墙把左中右都堵上" → 验 **BLOCKED 原地悬停**。
- [ ] **急停随时可用**:`SPACE` 悬停 / `ESC` 急停停桨 / `L` 降落 / `WASD` 瞬间人工夺权。

### 失败降级链实测(目标一,逐条,建议在低空小范围做)

- [ ] **感知失联**:AUTO `ON` 时,让我在 SSH 端 `pkill -f da_v2_service`(或现场用手挡住镜头)→ 预期 HUD 显 `depth stale`、**AUTO 自动悬停解除**回 ARMED。(测完告诉我,我重启感知服务)
- [ ] **挂载超时**:AUTO `ON` 后保持不动等 **>30s** → 预期 **AUTO 自动解除悬停**。
- [ ] **人工夺权**:AUTO `ON` 时按 `WASD` → 预期立即接管,松手超时后交回。

- [ ] 每种场景飞 **≥5 次**看行为一致(重复性)。
- [ ] **验收:** 三行为成立 + 三条降级链都触发 + 数据完整落盘。

---

## 收尾归档(每次飞行后)

- [ ] 核对 `logs/episodes/ep_*/`:`video.mp4` 能播、`frames.csv` 行数正常、`meta.json` 的 `outcome/action_source/battery` 合理;看门狗触发那次 `notes.auto_disengage` 有记因。
- [ ] 现场回填:感知 RTT、实际生效阈值、最小净空、每种行为结论。
- [ ] episode 直传回来(或留服务器),我出汇总:能力边界实测结论 + 校正报告,交接大众。

---

## 现场异常处置速查

| 现象 | 处置 |
|---|---|
| 图传卡死 / 感知长时间无返回 | `SPACE` 悬停 → `L` 降落 → 查感知服务日志 `~/da_v2_service/service.log` |
| 位移/RC 指令返回非 ok | 悬停后重发;`fly_real_mission` 类已带自动重试 |
| AUTO 该拐不拐 / 冲挡板 | 立即 `SPACE`/`ESC`;挡板换更大更有纹理的,或调低 `cruise_speed` |
| 任一异常无法恢复 | `ESC` 急停兜底 |
