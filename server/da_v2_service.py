#!/usr/bin/env python3
"""Depth Anything V2 Small 推理微服务(部署在 GPU 机,默认 4090)。

协议(与 tt_control/depth_backend.py 对应):
  POST /depth   body = JPEG 字节, Content-Type: image/jpeg
      → 200, body = struct("<II", H, W) + H*W 个 float16
        (按帧分位数归一化的「近度网格」,值越大越近)
  GET  /health  → 200 JSON, 模型/设备信息

仅用标准库 http.server + torch/transformers + pillow + numpy(独立 venv)。
用法: python da_v2_service.py --host 0.0.0.0 --port 8890 --grid 96x128
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("da_v2")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

_PIPE = None
_GRID = (96, 128)  # (H, W)
_LO_HI = (2.0, 98.0)  # 归一化分位数


def load_model() -> None:
    global _PIPE
    device = 0 if torch.cuda.is_available() else -1
    logger.info("loading %s on %s ...", MODEL_ID, "cuda:0" if device == 0 else "cpu")
    _PIPE = pipeline("depth-estimation", model=MODEL_ID, device=device)
    logger.info("model ready (cuda=%s)", torch.cuda.is_available())


def infer_nearness(jpeg: bytes) -> np.ndarray:
    """JPEG → 近度网格(float32, 0..1, 越大越近)。"""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    out = _PIPE(img)
    depth = np.asarray(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    # DA-V2 输出为视差/逆深度:值越大越近。分位数归一化到 0..1。
    lo = np.percentile(depth, _LO_HI[0])
    hi = np.percentile(depth, _LO_HI[1])
    near = np.clip((depth - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    # 缩放到固定网格(用 PIL 双线性,避免额外依赖)
    gh, gw = _GRID
    small = np.asarray(
        Image.fromarray((near * 255).astype(np.uint8)).resize((gw, gh), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return small


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音默认访问日志
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            body = json.dumps(
                {"ok": True, "model": MODEL_ID, "cuda": torch.cuda.is_available(), "grid": _GRID}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/depth"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        jpeg = self.rfile.read(n)
        try:
            t0 = time.time()
            near = infer_nearness(jpeg)
            dt = (time.time() - t0) * 1000.0
        except Exception as e:  # noqa: BLE001
            logger.exception("infer failed")
            self.send_error(500, str(e))
            return
        h, w = near.shape
        payload = struct.pack("<II", h, w) + near.astype(np.float16).tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Infer-Ms", f"{dt:.1f}")
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    global _GRID
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--grid", default="96x128", help="近度网格 HxW")
    args = p.parse_args()
    gh, gw = (int(x) for x in args.grid.lower().split("x"))
    _GRID = (gh, gw)

    load_model()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("serving on %s:%d (grid=%s)", args.host, args.port, _GRID)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
