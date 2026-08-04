#!/usr/bin/env python3
"""离线避障验证(第一阶段):不连飞机、不起飞。

把录制视频 / 图片序列 / 摄像头 喂给 DA-V2 感知后端 + AvoidanceController,
把「深度热力图 + 左中右分区 + 此刻会输出的 RC」叠回画面,输出标注 mp4,
并打印逐帧 RC 决策日志。用于在真机前离线看效果、调阈值。

在公司内网(能访问 4090)即可运行,例如:
  python offline_avoidance.py --source tt_twin.mp4 --out out.mp4
  python offline_avoidance.py --source frames/ --show
  python offline_avoidance.py --source 0 --show          # 本机摄像头
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from typing import Iterator, Optional

import cv2
import numpy as np

from tt_control.avoidance import AvoidanceController, AvoidParams
from tt_control.depth_backend import DEFAULT_SERVICE, DepthAnythingBackend

logger = logging.getLogger("offline_avoid")


def frames_from_source(source: str) -> Iterator[np.ndarray]:
    """视频文件 / 图片目录 / 摄像头索引 → 逐帧 BGR。"""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        while cap.isOpened():
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()
        return
    if os.path.isdir(source):
        paths = sorted(
            p
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for p in glob.glob(os.path.join(source, ext))
        )
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def main() -> int:
    p = argparse.ArgumentParser(description="离线视觉避障验证")
    p.add_argument("--source", required=True, help="视频文件 / 图片目录 / 摄像头索引(如 0)")
    p.add_argument("--service", default=DEFAULT_SERVICE, help="DA-V2 服务地址")
    p.add_argument("--out", default="", help="输出标注视频路径(.mp4);留空不写")
    p.add_argument("--show", action="store_true", help="实时窗口预览")
    p.add_argument("--fps", type=float, default=20.0, help="输出视频帧率")
    p.add_argument("--cruise", type=int, default=AvoidParams.cruise_speed)
    p.add_argument("--yaw", type=int, default=AvoidParams.yaw_speed)
    p.add_argument("--stop-thresh", type=float, default=AvoidParams.stop_thresh)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    params = AvoidParams(
        cruise_speed=args.cruise, yaw_speed=args.yaw, stop_thresh=args.stop_thresh
    )
    controller = AvoidanceController(params)
    backend = DepthAnythingBackend(service_url=args.service, controller=controller)

    writer: Optional[cv2.VideoWriter] = None
    n = 0
    states: dict[str, int] = {}
    t_start = time.time()
    rtts: list[float] = []
    for frame in frames_from_source(args.source):
        n += 1
        t0 = time.time()
        annotated = backend.infer(frame)
        rtts.append((time.time() - t0) * 1000.0)

        depth = backend.latest_depth()
        if depth is not None:
            d = controller.decide(depth.nearness)
            states[d.state] = states.get(d.state, 0) + 1
            if n % 5 == 1 or args.verbose:
                logger.info("frame %04d  %s", n, d.as_hud())

        if args.out:
            if writer is None:
                out_dir = os.path.dirname(os.path.abspath(args.out))
                os.makedirs(out_dir, exist_ok=True)
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法打开视频写出: {args.out}")
            writer.write(annotated)
        if args.show:
            cv2.imshow("offline avoidance", annotated)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    dur = time.time() - t_start
    logger.info("done: %d frames in %.1fs (%.1f fps)", n, dur, n / dur if dur else 0)
    if rtts:
        logger.info(
            "perception rtt: mean %.0fms  p50 %.0fms  p95 %.0fms",
            float(np.mean(rtts)),
            float(np.percentile(rtts, 50)),
            float(np.percentile(rtts, 95)),
        )
    if states:
        logger.info("decision histogram: %s", states)
    if args.out:
        logger.info("annotated video -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
