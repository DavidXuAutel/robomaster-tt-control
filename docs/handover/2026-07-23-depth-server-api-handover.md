# 深度推理服务器 — 远程访问与调用指南

日期：2026-07-23
面向：大众团队（外部使用方）
前置：需具备 Python 3.9+ 及 opencv-python、numpy

---

## 一、服务器地址

深度推理服务通过 Cloudflare Tunnel 暴露为公网 HTTPS 域名，无需 VPN 或内网 IP：

| 端点 | URL | 方法 |
|------|-----|------|
| 健康检查 | `https://depth.david-x.com/health` | GET |
| 深度推理 | `https://depth.david-x.com/depth` | POST |

> 源站 GPU 服务器位于内网（Depth Anything V2 Small，CUDA），通过 cloudflared 隧道对外。所有请求走 443 端口标准 HTTPS，无需特殊防火墙规则。

---

## 二、鉴权与请求要求

### 2.1 Cloudflare 防护

接口**无需 API Key / Token**，但 Cloudflare 开启了 **Browser Integrity Check**。Python 默认的 `Python-urllib/x` User-Agent 会被判为 bot，直接返回 `HTTP 403`（error code 1010）。

**解决办法：请求时必须携带浏览器样式的 User-Agent 头。**

```http
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36
```

### 2.2 长期建议

如果你的部署环境中请求量较大，建议在 Cloudflare 后台对该 hostname **关闭 Browser Integrity Check**（安全 → BIC → 关）。此功能主要面向网页防爬，对 API 接口不适用。关掉后无需伪造 UA 头。

---

## 三、接口协议

### 3.1 GET /health

**响应：**
```json
{
  "ok": true,
  "model": "depth-anything/Depth-Anything-V2-Small-hf",
  "cuda": true,
  "grid": [96, 128]
}
```

| 字段 | 含义 |
|------|------|
| `ok` | 服务是否正常 |
| `model` | 模型名称 |
| `cuda` | 是否使用 GPU 推理 |
| `grid` | 近度网格尺寸 `[H, W]`，当前固定 `[96, 128]` |

### 3.2 POST /depth

**请求：**
```
POST /depth
Content-Type: image/jpeg
User-Agent: Mozilla/5.0 ... Chrome/125.0 Safari/537.36

<JPEG 原始字节>
```

- 输入为单帧 JPEG 编码的 RGB 图像
- 建议 JPEG 质量 80（平衡画质与传输体积）
- 无分辨率硬性要求，服务端会自动缩放处理

**响应：**
```
200 OK
Content-Type: application/octet-stream
X-Infer-Ms: 42.5

<binary grid>
```

**Binary 格式：**

| 偏移 | 长度 | 类型 | 含义 |
|------|------|------|------|
| 0 | 4 | uint32 LE | 网格高度 H |
| 4 | 4 | uint32 LE | 网格宽度 W |
| 8 | H×W×2 | float16 LE × (H×W) | 近度网格，行优先 |

**近度语义：**
- 值域约 `[0, 1]`，**值越大表示越近 / 越挡路**
- 每个像素值为该区域相对于全帧的相对近度
- 服务端对单帧做 2%–98% 分位归一化，因此同一场景不同时刻的值可能略有漂移
- 不可直接当作米制距离使用，需现场标定阈值

**响应头：**
| 头 | 含义 |
|----|------|
| `X-Infer-Ms` | 服务端 GPU 推理耗时（ms），不含网络传输 |

---

## 四、客户端示例

### 4.1 Python 最简示例

```python
import struct
import urllib.request
import cv2
import numpy as np

SERVICE_URL = "https://depth.david-x.com/depth"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

def infer_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """输入 BGR 图像 (H,W,3)，返回近度网格 (96,128) float32。"""
    ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("JPEG 编码失败")

    req = urllib.request.Request(
        SERVICE_URL,
        data=jpeg.tobytes(),
        headers={"Content-Type": "image/jpeg", "User-Agent": UA},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        raw = resp.read()

    h, w = struct.unpack("<II", raw[:8])
    return np.frombuffer(raw[8:], dtype=np.float16).reshape(h, w).astype(np.float32)


# 使用
frame = cv2.imread("test.jpg")
nearness = infer_depth(frame)
print(f"网格: {nearness.shape}, 范围: [{nearness.min():.3f}, {nearness.max():.3f}]")
```

### 4.2 curl 测试

```bash
# 健康检查
curl -sS -H "User-Agent: Mozilla/5.0 ... Chrome/125.0 Safari/537.36" \
  https://depth.david-x.com/health

# 深度推理（需实际 JPEG 文件）
curl -sS -X POST \
  -H "Content-Type: image/jpeg" \
  -H "User-Agent: Mozilla/5.0 ... Chrome/125.0 Safari/537.36" \
  --data-binary @test.jpg \
  https://depth.david-x.com/depth \
  -o nearness.bin
```

### 4.3 使用本仓库客户端（推荐）

如果你的环境和本仓库代码一致，可直接使用已封装的异步客户端：

```bash
# 启动 Tello 控制界面 + 深度后端
python main.py \
  --tello-ip <飞机IP> \
  --local-ip <本机IP> \
  --inference depth-anything \
  --depth-service https://depth.david-x.com/depth \
  -v
```

或单独使用推理模块：

```python
from tt_control.async_infer import AsyncInferWorker

worker = AsyncInferWorker("https://depth.david-x.com/depth", timeout=5.0)
worker.start()
worker.feed_frame(frame)         # 主线程喂帧
result = worker.latest_result()  # 主线程读结果（非阻塞）
# result.nearness  → np.ndarray (96,128) float32
# result.rtt_ms    → 往返耗时
# result.frame_age_ms → 帧延迟
worker.stop()
```

---

## 五、集成避障的最小示例

```python
from tt_control.avoidance import AvoidanceController, AvoidParams
from tt_control.avoidance_fsm import AvoidanceFSM

# 1. 创建异步推理客户端
worker = AsyncInferWorker("https://depth.david-x.com/depth", timeout=5.0)
worker.start()

# 2. 创建避障状态机
params = AvoidParams(cruise_speed=20, yaw_speed=30, approach_pitch=10)
controller = AvoidanceController(params)
fsm = AvoidanceFSM(controller=controller)

# 3. 控制循环（伪代码）
while flying:
    frame = tello.read_video_frame()
    worker.feed_frame(frame)

    result = worker.latest_result()
    nearness = result.nearness if result else None

    decision = fsm.step(nearness, tello.telemetry, auto_on=True)
    if decision.abort_reason:
        tello.hover()
        break
    tello.rc(*decision.axes.as_tuple())

worker.stop()
```

---

## 六、性能参考

| 指标 | 典型值 | 说明 |
|------|--------|------|
| GPU 推理耗时 | 40–50 ms | `X-Infer-Ms` 响应头 |
| 公网往返 RTT | 1.0–3.0 s | Cloudflare 隧道延迟，波动取决于网络路径 |
| 近度网格 | 96×128 | 固定尺寸，约 24 KB/帧 |
| 推理频率 | ~0.3–0.5 Hz | 受 RTT 限制，适合低速避障 |

> RTT 较高是因为走公网 Cloudflare 隧道。如你的实验环境与 GPU 服务器在同一内网，可直连内网 IP（默认 `http://10.229.20.125:8899/depth`），RTT 可降至 5–20 ms。

---

## 七、故障排查

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `HTTP 403` + `error code: 1010` | 未带 User-Agent 头 | 添加浏览器 UA 头（见 2.1） |
| `EOF in violation of protocol` | 本地代理/VPN 劫持了域名 DNS | 关闭 Clash/Surge/WARP 或将 `david-x.com` 加入直连白名单 |
| 请求超时 | RTT 波动或服务端负载 | 增大 timeout（建议 5s），检查 `/health` |
| 近度值异常（全 0 或全 1） | 输入图像过暗/过曝 | 检查 Tello 图传质量，确保光照均匀 |

---

## 八、联系方式

如有问题，通过 Ou Xuedong 联系或在本仓库提 issue。
