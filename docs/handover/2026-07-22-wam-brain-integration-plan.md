---
title: 无人机接入 WAM 大脑 · 对接方案
type: integration-plan
audience: 大众(世界模型 / WAM 训练方)
date: 2026-07-22
status: 待对齐(接口契约需双方确认)
related:
  - docs/design/2026-07-21-single-drone-solidify-and-data-delivery-design.md
  - docs/references/2026-07-20-scoutxwam-world-model-analysis.md
  - tt_control/policy.py (ExternalModelPolicy 接缝)
---

# 无人机接入 WAM 大脑 · 对接方案

## 0. 一句话

无人机控制端以**瘦 HTTP 客户端**方式调用大众 WAM 的推理端点:**发当前观测(图像+本体状态)→ 收动作 → 映射为遥控杆量下发**。WAM 当"大脑",视觉避障当"兜底",键盘随时可夺权。与现有的深度感知微服务(`da_v2_service:8899`)完全同构,复用同一套瘦客户端 + 远端 GPU 推理的模式。

---

## 1. 架构与拓扑

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  大众 WAM 推理服务(GPU)      │        │  无人机控制端(我方新服务器      │
│  POST /act  GET /health      │◀──────▶│  10.229.20.110)               │
│  世界-动作模型               │  HTTP  │   ExternalModelPolicy(瘦客户端)│
│  观测 → 动作(+可选预测/价值)  │  局域网 │      │ 收动作 → 映射 RcAxes        │
└─────────────────────────────┘        │      ▼                        │
                                        │   RC 仲裁: 键盘>WAM>避障>悬停   │
                                        │      │ UDP 8889                │
                                        │      ▼  WiFi(192.168.10.x)    │
                                        │   RoboMaster TT 无人机          │
                                        └──────────────────────────────┘
```

- **部署**:WAM 跑在大众的 GPU 机(或现有 4090);控制端跑在我方服务器,已验证能跨机访问 10.229.x 内网的 GPU 服务(现有深度服务即如此)。
- **调用**:控制回路按固定频率(默认 ~15Hz)把观测 POST 给 `/act`,拿回动作。
- **可退化**:WAM 不可达/超时/失联 → 自动回退到视觉避障(若深度在线)或悬停,人工键盘永远第一优先。

---

## 2. 推理接口契约(核心 · 需大众确认)

> 若大众的 WAM 已有自己的 API,把它的 schema 发我们,我方 adapter 直接适配即可——下面是**建议契约**,便于最省事对接。

### 2.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/act` | 输入观测,返回动作 |
| `GET` | `/health` | 返回 `{"ok":true,"model":"...","expects_depth":false}` |

地址由大众提供,形如 `http://<wam-host>:<port>/act`;我方用 `--wam-service http://<wam-host>:<port>/act` 指向它(与 `--depth-service` 同理)。

### 2.2 请求(我方 → WAM)

`Content-Type: application/json`:
```json
{
  "proto": 1,
  "t_ms": 1699999999123,
  "obs": {
    "rgb_jpeg_b64": "<前视相机 960x720 JPEG 的 base64>",
    "nearness_96x128_b64": "<可选:DA-V2 近度网格 float16,若 WAM 要深度>"
  },
  "state": {
    "height_cm": 80, "pitch": 0.0, "roll": 0.0, "yaw": 5.0,
    "pad_id": 1, "pos_x_cm": 0, "pos_y_cm": 0, "pos_z_cm": 80,
    "bat": 80
  },
  "instruction": "向前巡视",        // 可选:语言指令
  "goal": null                      // 可选:目标(点/图像/描述)
}
```
> **本体遥测告警(重要)**:本台 TT 的 **速度遥测 `vgx/vgy/vgz` 恒为 0(硬件不报)**,故请求里不含可信速度;WAM 请**基于视觉 + 自己下发的动作**推断运动,勿依赖测量速度。位姿 `pos_*` 仅在 Mission Pad 锁定(`pad_id≥0`)时可信。

### 2.3 响应(WAM → 我方)

**推荐:归一化动作**(嵌入体无关,规避本机米制标定不确定性):
```json
{
  "action": { "vx": 0.3, "vy": 0.0, "vz": 0.0, "wyaw": 0.1 },   // 各轴 [-1,1]
  "action_seq": [ {...}, {...} ],       // 可选:动作序列(见 §4 延迟)
  "horizon_dt_s": 0.1,                  // action_seq 每步时长
  "confidence": 0.9,                    // 可选
  "pred": { "depth_96x128_b64": "..." } // 可选:预测观测/重建(用于闭环校验)
}
```

**轴与单位约定**(机体系):
| 字段 | 含义 | 范围 | 映射到杆量 |
|---|---|---|---|
| `vx` | 前进(+前) | [-1,1] | `pitch = vx*100` |
| `vy` | 横移(+右) | [-1,1] | `roll  = vy*100` |
| `vz` | 升降(+上) | [-1,1] | `throttle = vz*100` |
| `wyaw` | 偏航(+顺时针/右转) | [-1,1] | `yaw = wyaw*100` |

> 归一化 = "满杆的百分比"。**这样无需依赖 VMAX 米制标定**(本机标定不确定,见校正结论)。若大众更愿输出**物理速度(m/s)**,我方也可用标定常数换算,但归一化更省事、更稳。

---

## 3. 我方适配(`ExternalModelPolicy` → `RcAxes`)

`tt_control/policy.py::ExternalModelPolicy.decide(frame, state)`:
1. `frame`(BGR)编码 JPEG + `state` 组装成 §2.2 请求;
2. POST 到 `/act`(标准库 urllib,超时~200ms);
3. 取 `action`(或 `action_seq` 首步)→ 映射为 `RcAxes`(§2.3 表)+ clamp 到 [-100,100];
4. 交给 RC 仲裁下发。失败/超时 → 返回 `None`(上层按 §5 退化)。

接入后用 `main.py --inference depth-anything --wam-service <url>` 或专门开关启用;首次仍需 `V` 键 `ARMED→ON` 确认。

---

## 4. 延迟预算与轨迹模式

- 控制回路 RC 下发 ~15Hz(每 ~66ms 一拍)。**单步推理建议 ≤150ms**(与深度服务 ~106ms 同量级),否则控制会滞后。
- **若 WAM 单步推理较慢(想象/rollout 重)**:返回**动作序列 `action_seq`**(如未来 5 步 @ `horizon_dt_s`),我方在两次推理之间**开环执行**该序列,把推理频率与控制频率解耦。这是应对世界模型推理耗时的推荐方式。
- 我方对 WAM 响应设**新鲜度看门狗**:超过阈值(默认 1.5s)无有效动作 → 视为失联,退化。

---

## 5. 安全与仲裁

优先级(高→低):**键盘 > WAM > 避障 > 悬停**。
- **首次挂载**:地面不驱动;起飞后按 `V` 确认 `ARMED→ON` 才接管;
- **看门狗**(复用 `auto_safety.AutoWatchdog`):WAM 失联/超时、或单次挂载超时 → 自动悬停解除,HUD 显因;
- **退化链**:WAM 不可用 → 若深度在线切避障兜底,否则悬停;
- **人工**:`WASD` 瞬时夺权、`SPACE` 悬停、`ESC` 急停、`L` 降落,永远第一优先;
- 全程 episode 录制(视频+动作+状态),WAM 接管段与人工/避障段以 `ctrl_state` 区分。

---

## 6. 数据飞轮回路(闭环)

WAM 驱动飞行 → 我方录制训练级 episode(`video.mp4 + frames.csv`,格式见单机数据交付设计文档)→ 回流给大众做**具身特定后训练**(把 WAM 微调到无人机本体)→ 更强的 WAM 再驱动。`frames.csv` 的 `act_*` 即 WAM 下发的动作,`video.mp4` 即对应观测,天然是"(观测,动作,下一观测)"训练三元组。

---

## 7. 推理地址与联调步骤

1. **大众提供**:`http://<wam-host>:<port>/act`(+ `/health`),按 §2 契约(或给我方它现有 schema)。
2. **连通性**:我方服务器 `curl http://<wam-host>:<port>/health` 应 `ok:true`(内网可达,已验证能到 4090 服务)。
3. **Dry-run**:先不飞——用录像/合成帧离线打接口,确认 request/response 通、动作方向合理(类似 `offline_avoidance` 的离线回放)。
4. **仿真闭环**:`ExternalModelPolicy` 接进 `sim_runner`/`--sim`,在仿真里验"观测→WAM→动作→机体"闭环不发散。
5. **真机**:挡板场地,`ARMED→ON`,小范围验 WAM 接管 + 退化链 + 人工夺权,再放开。

---

## 8. 待大众确认项(对接前拍板)

1. **WAM 输出形态**:归一化动作(推荐)还是物理速度?单步还是动作序列(+`horizon_dt`)?
2. **WAM 需要哪些输入**:仅 RGB?还要深度(我方可附 DA-V2 近度网格)?要多视角吗(本机只有单前视)?要语言指令/目标吗?
3. **单步推理延迟**大概多少?据此定单步还是轨迹模式。
4. **接口形式**:用本文契约,还是我方适配大众 WAM 现有 API(给 schema 即可)。
5. **坐标/正方向**是否与本文约定一致(机体系 +x 前/+y 右/+z 上/+yaw 右转)。

---

## 附:参考适配实现(拿到端点后即可落地)

```python
# tt_control/policy.py::ExternalModelPolicy.decide 的参考实现
import base64, json, urllib.request, cv2, numpy as np
from tt_control.control import RcAxes

def decide(self, frame, state):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    req = {
        "proto": 1,
        "obs": {"rgb_jpeg_b64": base64.b64encode(buf).decode()},
        "state": {k: state.get(k) for k in
                  ("height_cm","pitch","roll","yaw","pad_id","pos_x_cm","pos_y_cm","pos_z_cm","bat")},
    }
    r = urllib.request.Request(self.model_ref, data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(r, timeout=0.2) as resp:
            a = json.loads(resp.read())["action"]
    except Exception:
        return None                      # 失败 → 上层退化
    clamp = lambda v: max(-100, min(100, int(round(v * 100))))
    return RcAxes(roll=clamp(a["vy"]), pitch=clamp(a["vx"]),
                  throttle=clamp(a["vz"]), yaw=clamp(a["wyaw"]))
```
