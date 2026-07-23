"""飞行 episode 逐帧同步录制器：视频(MP4) + 轨迹数据(frames.csv) + 深度网格。

产出训练级 episode 目录（对齐 docs/design/2026-07-21 数据格式规范）：

    logs/episodes/ep_<stamp>/
      ├── meta.json      回合级元数据（用途/尺度锚/电量/结局/视频信息...）
      ├── video.mp4      H.264(CRF18)视频；第 N 帧 <-> frames.csv 第 N 行（严格 1:1）
      ├── frames.csv     逐帧轨迹：时间戳 + 位姿 + 动作 + 决策 + 深度引用
      └── depth/000001.npy  96×128 float16 近度网格（有深度时；按 ts 去重不重复落盘）

为什么视频不是"只能回看"：微调的是**世界模型**（学"动作→未来观测"），RGB 视频就是
模型的核心训练输入（视频经 VAE 编码后预测未来帧/深度），不是附属审阅材料。

为什么用 MP4 而非逐帧图片：世界模型的视频 VAE 本就吃视频序列；MP4 在同等时长下体量
远小于逐帧 JPEG、文件数少、便于跨机传输(DLP 友好)，是 DROID/LeRobot 等主流机器人数据
集的通行做法。训练所需的"帧↔动作严格对齐"由 frames.csv 的行号=视频帧号 + 时间戳保证。

同步保证：单调时钟 `t_mono_ms` 为对齐基准；视频帧与 CSV 行严格 1:1；深度按 `ts` 去重并
记录 `depth_age_ms`(配对深度相对本帧的滞后)，供校正时诚实核算与筛查不同步坏帧。

编码走 PyAV 的 file-object 通路(项目已依赖 av)，绕开 cv2/中文路径的 C++ 写文件限制；
若编码器初始化失败则回退逐帧 JPEG，保证数据不丢。
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:  # 视频编码首选 PyAV（H.264 + CRF，且 file-object 通路对中文路径安全）
    import av
except ImportError:  # pragma: no cover
    av = None

try:  # 回退：逐帧 JPEG 时用 cv2 编码
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)

# frames.csv 列（顺序即写入顺序），对齐设计文档 §4.1
FRAME_FIELDS = [
    "frame_id", "t_mono_ms", "video_frame",
    "rgb_path", "depth_path", "has_depth", "depth_rtt_ms", "depth_age_ms",
    "pad_id", "pos_x_cm", "pos_y_cm", "pos_z_cm", "yaw_deg", "height_cm",
    "vgx", "vgy", "vgz", "pitch_deg", "roll_deg", "bat_pct",
    "act_roll", "act_pitch", "act_throttle", "act_yaw", "ctrl_state",
    "near_left", "near_mid", "near_right",
]

# 由控制律输出的自动决策状态（用于统计 action_source）
_AUTO_STATES = {"CRUISE", "TURN_L", "TURN_R", "BLOCKED", "STOP"}


def _num(state: dict, key: str):
    """从 SDK state（值为字符串）取数值；缺失/非法返回空串以便 CSV 留空。"""
    try:
        return float(state.get(key))
    except (TypeError, ValueError, AttributeError):
        return ""


class EpisodeRecorder:
    """一次飞行 = 一个 episode 目录。按固定频率同步落盘视频+轨迹+状态。"""

    def __init__(
        self,
        directory: str | Path,
        meta_base: Optional[dict] = None,
        record_hz: float = 10.0,
        jpeg_quality: int = 80,
        crf: int = 18,
    ) -> None:
        directory = Path(directory)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.dir = directory / f"ep_{stamp}"
        (self.dir / "depth").mkdir(parents=True, exist_ok=True)
        self.episode_id = self.dir.name

        self.record_hz = float(record_hz)
        self._interval = 1.0 / record_hz if record_hz > 0 else 0.0
        self._nominal_fps = max(1, int(round(record_hz)) or 10)
        self._jpeg_q = int(jpeg_quality)
        self._crf = int(crf)
        self._meta_base = dict(meta_base or {})

        self._csv = (self.dir / "frames.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=FRAME_FIELDS)
        self._writer.writeheader()
        self._csv.flush()

        self._lock = threading.Lock()
        self._closed = False
        self._n = 0
        self._n_depth = 0
        self._last_cap = float("-inf")  # 保证第一帧一定被采
        self._t0: Optional[float] = None
        self._t_last = 0.0
        self._bat_first: Any = None
        self._bat_last: Any = None
        self._depth_grid: Optional[list[int]] = None
        self._last_depth_ts: Optional[float] = None
        self._last_depth_rel = ""
        self._n_manual = 0
        self._n_auto = 0
        self._outcome = "completed"
        self._abort_reason: Optional[str] = None
        self._notes: dict = {}

        # 视频编码状态（首帧惰性初始化，拿到分辨率后再建流）
        self._vfile = None
        self._vcontainer = None
        self._vstream = None
        self._vframes = 0
        self._vw = 0
        self._vh = 0
        self._video_mode = "mp4" if av is not None else "jpeg"
        if self._video_mode == "jpeg":
            (self.dir / "rgb").mkdir(parents=True, exist_ok=True)

    def due(self, t_mono: float) -> bool:
        """是否到达采样时刻（限流，避免 30fps 全存爆盘）。"""
        return self._interval <= 0.0 or (t_mono - self._last_cap) >= self._interval

    def _ensure_video(self, h: int, w: int) -> None:
        if self._video_mode != "mp4" or self._vcontainer is not None:
            return
        try:
            self._vfile = (self.dir / "video.mp4").open("wb")
            self._vcontainer = av.open(self._vfile, mode="w", format="mp4")
            st = self._vcontainer.add_stream("libx264", rate=self._nominal_fps)
            st.width = w
            st.height = h
            st.pix_fmt = "yuv420p"
            st.options = {"crf": str(self._crf), "preset": "veryfast"}
            self._vstream = st
            self._vw, self._vh = w, h
        except Exception as e:  # 编码器不可用 → 回退逐帧 JPEG，不丢数据
            logger.warning("MP4 编码器初始化失败，回退逐帧 JPEG：%s", e)
            self._video_mode = "jpeg"
            (self.dir / "rgb").mkdir(parents=True, exist_ok=True)
            try:
                if self._vcontainer is not None:
                    self._vcontainer.close()
            except Exception:
                pass
            if self._vfile is not None:
                self._vfile.close()
            self._vfile = self._vcontainer = self._vstream = None

    def _write_rgb(self, rgb: np.ndarray, fid: int):
        """把一帧写入视频(返回视频帧号)或逐帧 JPEG(返回相对路径)。"""
        h, w = rgb.shape[:2]
        self._ensure_video(h, w)
        if self._video_mode == "mp4" and self._vstream is not None:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="bgr24")
            for pkt in self._vstream.encode(frame):
                self._vcontainer.mux(pkt)
            idx = self._vframes
            self._vframes += 1
            return idx, ""  # (video_frame, rgb_path)
        # JPEG 回退
        rel = f"rgb/{fid:06d}.jpg"
        if cv2 is not None:
            ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q])
            if ok:
                with (self.dir / rel).open("wb") as f:
                    f.write(buf.tobytes())
                return "", rel
        return "", ""

    def capture(
        self,
        *,
        t_mono: float,
        rgb: np.ndarray,
        depth: Any = None,
        depth_rtt_ms: float = 0.0,
        state: Optional[dict] = None,
        act: Any = None,
        ctrl_state: str = "",
        zones: Optional[Sequence[float]] = None,
    ) -> bool:
        """落盘一帧。返回 True 表示已记录，False 表示被限流跳过或已关闭。"""
        with self._lock:
            if self._closed or not self.due(t_mono):
                return False
            self._last_cap = t_mono
            if self._t0 is None:
                self._t0 = t_mono
            self._t_last = t_mono
            self._n += 1
            fid = self._n

            video_frame, rgb_rel = self._write_rgb(rgb, fid)

            # --- 深度（同一深度帧不重复落盘，按 ts 去重）+ 滞后核算 ---
            depth_rel = ""
            has_depth = 0
            depth_age_ms: Any = ""
            near = getattr(depth, "nearness", None) if depth is not None else None
            if near is not None:
                has_depth = 1
                ts = getattr(depth, "ts", None)
                if ts is not None:
                    depth_age_ms = round(max(0.0, (time.time() - ts)) * 1000.0, 1)
                if ts is not None and ts == self._last_depth_ts:
                    depth_rel = self._last_depth_rel  # 复用上一份，标记共享
                else:
                    depth_rel = f"depth/{fid:06d}.npy"
                    grid = np.asarray(near, dtype=np.float16)
                    with (self.dir / depth_rel).open("wb") as f:
                        np.save(f, grid)
                    self._last_depth_ts = ts
                    self._last_depth_rel = depth_rel
                    self._n_depth += 1
                    if self._depth_grid is None:
                        self._depth_grid = list(grid.shape)

            # --- 状态 / 电量 ---
            st = state or {}
            bat = _num(st, "bat")
            if bat != "":
                if self._bat_first is None:
                    self._bat_first = bat
                self._bat_last = bat

            # --- 动作 / 决策 ---
            a, b, c, d = act.as_tuple() if act is not None else (0, 0, 0, 0)
            if ctrl_state == "MANUAL":
                self._n_manual += 1
            elif ctrl_state in _AUTO_STATES:
                self._n_auto += 1

            zl = zm = zr = ""
            if zones is not None:
                zl, zm, zr = (round(float(v), 4) for v in zones)

            self._writer.writerow({
                "frame_id": fid,
                "t_mono_ms": int(round((t_mono - self._t0) * 1000)),
                "video_frame": video_frame,
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "has_depth": has_depth,
                "depth_rtt_ms": round(float(depth_rtt_ms), 1),
                "depth_age_ms": depth_age_ms,
                "pad_id": _num(st, "mid"),
                "pos_x_cm": _num(st, "x"),
                "pos_y_cm": _num(st, "y"),
                "pos_z_cm": _num(st, "z"),
                "yaw_deg": _num(st, "yaw"),
                "height_cm": _num(st, "h"),
                "vgx": _num(st, "vgx"),
                "vgy": _num(st, "vgy"),
                "vgz": _num(st, "vgz"),
                "pitch_deg": _num(st, "pitch"),
                "roll_deg": _num(st, "roll"),
                "bat_pct": bat,
                "act_roll": a,
                "act_pitch": b,
                "act_throttle": c,
                "act_yaw": d,
                "ctrl_state": ctrl_state,
                "near_left": zl,
                "near_mid": zm,
                "near_right": zr,
            })
            self._csv.flush()
            return True

    def set_outcome(self, outcome: str, abort_reason: Optional[str] = None) -> None:
        with self._lock:
            self._outcome = outcome
            self._abort_reason = abort_reason

    def note(self, **kw: Any) -> None:
        with self._lock:
            self._notes.update(kw)

    @property
    def n_frames(self) -> int:
        return self._n

    def _action_source(self) -> str:
        if self._n_manual and self._n_auto:
            return "mixed"
        if self._n_auto:
            return "avoidance"
        if self._n_manual:
            return "manual"
        return "unknown"

    def _close_video(self) -> None:
        if self._vstream is not None:
            try:
                for pkt in self._vstream.encode():  # flush 编码器
                    self._vcontainer.mux(pkt)
            except Exception:  # pragma: no cover
                logger.exception("flush video failed")
        if self._vcontainer is not None:
            try:
                self._vcontainer.close()
            except Exception:  # pragma: no cover
                pass
        if self._vfile is not None:
            try:
                self._vfile.close()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> Path:
        """写出 meta.json 并关闭。可重复调用（幂等）。"""
        with self._lock:
            if self._closed:
                return self.dir
            self._closed = True
            self._close_video()
            duration_ms = int(round((self._t_last - (self._t0 or self._t_last)) * 1000))
            if self._video_mode == "mp4":
                video_meta = {
                    "file": "video.mp4", "codec": "h264", "crf": self._crf,
                    "nominal_fps": self._nominal_fps, "frames": self._vframes,
                    "width": self._vw, "height": self._vh,
                    "note": "第 N 帧 <-> frames.csv 第 N 行；真实时刻见 t_mono_ms",
                }
            else:
                video_meta = {"mode": "jpeg_frames", "dir": "rgb/"}
            meta = {
                "episode_id": self.episode_id,
                "created": datetime.now().astimezone().isoformat(),
                "record_hz": self.record_hz,
                **self._meta_base,
                "action_source": self._meta_base.get("action_source") or self._action_source(),
                "n_frames": self._n,
                "n_depth_frames": self._n_depth,
                "depth_grid": self._depth_grid,
                "frames_manual": self._n_manual,
                "frames_auto": self._n_auto,
                "video": video_meta,
                "t_start_ms": 0,
                "t_end_ms": duration_ms,
                "battery_start_pct": self._bat_first,
                "battery_end_pct": self._bat_last,
                "outcome": self._outcome,
                "abort_reason": self._abort_reason,
                "notes": self._notes,
            }
            try:
                self._csv.close()
            except Exception:  # pragma: no cover
                pass
            (self.dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return self.dir
