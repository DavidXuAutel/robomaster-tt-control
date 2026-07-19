"""纯视觉手势控制：张开手掌上抬起飞、响指动作降落。"""

from __future__ import annotations

import math
import pathlib
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from tt_control.inference import InferenceBackend, InferenceEvent
from tt_control.gesture_profile import GestureProfile, LandmarkSequence, load_latest

logger = logging.getLogger(__name__)

Point3 = tuple[float, float, float]

# MediaPipe Hand Landmarker 的骨架连接。
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _distance(a: Point3, b: Point3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mean_point(points: Sequence[Point3], indices: Sequence[int]) -> Point3:
    n = float(len(indices))
    return (
        sum(points[i][0] for i in indices) / n,
        sum(points[i][1] for i in indices) / n,
        sum(points[i][2] for i in indices) / n,
    )


@dataclass
class GestureThresholds:
    # Tello 720p H.264 实测张掌分数约 0.52-0.59；动态轨迹仍负责二次确认。
    open_palm_score: float = 0.50
    raise_min_duration: float = 0.45
    raise_max_window: float = 1.20
    raise_min_distance: float = 0.12
    raise_max_x_drift: float = 0.18
    raise_monotonic_ratio: float = 0.70
    snap_pinch_ratio: float = 0.45
    snap_min_pinch_duration: float = 0.08
    snap_release_ratio: float = 0.72
    snap_max_duration: float = 0.65
    snap_min_middle_motion: float = 0.52
    snap_max_palm_distance: float = 1.15
    snap_max_middle_extension: float = 0.95
    event_cooldown: float = 3.0


@dataclass
class _RaiseSample:
    timestamp: float
    center: Point3
    scale: float


class GestureSequenceDetector:
    """只依赖关键点的可测试时序检测器，不依赖 MediaPipe 运行时。"""

    def __init__(self, thresholds: GestureThresholds | None = None) -> None:
        self.t = thresholds or GestureThresholds()
        self._raise_samples: deque[_RaiseSample] = deque()
        self._last_open_palm = 0.0
        self._snap_primed_at = 0.0
        self._snap_middle_ref: tuple[float, float] | None = None
        self._snap_candidate_at = 0.0
        self._snap_candidate_ref: tuple[float, float] | None = None
        self._last_event = {"takeoff": -1e9, "land": -1e9}
        self.phase = "waiting"

    def reset(self) -> None:
        self._raise_samples.clear()
        self._last_open_palm = 0.0
        self._snap_primed_at = 0.0
        self._snap_middle_ref = None
        self._snap_candidate_at = 0.0
        self._snap_candidate_ref = None
        self.phase = "waiting"

    def update(
        self,
        timestamp: float,
        landmarks: Sequence[Point3],
        open_palm_score: float,
    ) -> list[InferenceEvent]:
        if len(landmarks) < 21:
            self.reset()
            return []

        palm = _mean_point(landmarks, (0, 5, 9, 13, 17))
        scale = max(
            _distance(landmarks[5], landmarks[17]),
            _distance(landmarks[0], landmarks[9]) * 0.75,
            1e-4,
        )
        events: list[InferenceEvent] = []
        takeoff = self._update_raise(timestamp, palm, scale, open_palm_score)
        if takeoff:
            events.append(takeoff)
        land = self._update_snap(timestamp, landmarks, palm, scale)
        if land:
            events.append(land)
        return events

    def _update_raise(
        self,
        timestamp: float,
        palm: Point3,
        scale: float,
        open_palm_score: float,
    ) -> InferenceEvent | None:
        if open_palm_score >= self.t.open_palm_score:
            self._last_open_palm = timestamp
            self._raise_samples.append(_RaiseSample(timestamp, palm, scale))
            while (
                self._raise_samples
                and timestamp - self._raise_samples[0].timestamp > self.t.raise_max_window
            ):
                self._raise_samples.popleft()
        elif timestamp - self._last_open_palm > 0.18:
            self._raise_samples.clear()

        if len(self._raise_samples) < 5:
            return None
        first, last = self._raise_samples[0], self._raise_samples[-1]
        duration = last.timestamp - first.timestamp
        if duration < self.t.raise_min_duration:
            self.phase = "open palm"
            return None

        up_distance = first.center[1] - last.center[1]
        x_drift = abs(last.center[0] - first.center[0])
        scales = [sample.scale for sample in self._raise_samples]
        scale_ratio = max(scales) / max(min(scales), 1e-4)
        pairs = zip(self._raise_samples, list(self._raise_samples)[1:])
        monotonic = sum(b.center[1] <= a.center[1] + 0.006 for a, b in pairs)
        monotonic_ratio = monotonic / max(len(self._raise_samples) - 1, 1)
        self.phase = f"palm up {up_distance:.2f}"

        if (
            up_distance >= self.t.raise_min_distance
            and x_drift <= self.t.raise_max_x_drift
            and scale_ratio <= 1.8
            and monotonic_ratio >= self.t.raise_monotonic_ratio
            and timestamp - self._last_event["takeoff"] >= self.t.event_cooldown
        ):
            self._last_event["takeoff"] = timestamp
            self._raise_samples.clear()
            self.phase = "TAKEOFF gesture"
            confidence = min(1.0, open_palm_score * (up_distance / self.t.raise_min_distance))
            return InferenceEvent("takeoff", confidence, f"open-palm rise {up_distance:.2f}")
        return None


    def _update_snap(
        self,
        timestamp: float,
        landmarks: Sequence[Point3],
        palm: Point3,
        scale: float,
    ) -> InferenceEvent | None:
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        pinch = _distance(thumb_tip, middle_tip) / scale
        middle_extension = _distance(middle_tip, middle_mcp) / scale
        middle_palm_distance = _distance(middle_tip, palm) / scale
        middle_rel = (
            (middle_tip[0] - palm[0]) / scale,
            (middle_tip[1] - palm[1]) / scale,
        )

        if self._snap_middle_ref is None:
            if pinch <= self.t.snap_pinch_ratio and middle_extension >= 0.60:
                if self._snap_candidate_at <= 0.0:
                    self._snap_candidate_at = timestamp
                    self._snap_candidate_ref = middle_rel
                    self.phase = "snap pinch"
                elif timestamp - self._snap_candidate_at >= self.t.snap_min_pinch_duration:
                    self._snap_primed_at = timestamp
                    self._snap_middle_ref = self._snap_candidate_ref or middle_rel
                    self.phase = "snap primed"
            else:
                self._snap_candidate_at = 0.0
                self._snap_candidate_ref = None
            return None

        elapsed = timestamp - self._snap_primed_at
        if elapsed > self.t.snap_max_duration:
            self._snap_primed_at = 0.0
            self._snap_middle_ref = None
            return None

        motion = math.hypot(
            middle_rel[0] - self._snap_middle_ref[0],
            middle_rel[1] - self._snap_middle_ref[1],
        )
        down_motion = middle_rel[1] - self._snap_middle_ref[1]
        self.phase = f"snap move {motion:.2f}"
        snapped = (
            pinch >= self.t.snap_release_ratio
            and motion >= self.t.snap_min_middle_motion
            and down_motion >= 0.35
            and middle_palm_distance <= self.t.snap_max_palm_distance
            and middle_extension <= self.t.snap_max_middle_extension
        )
        if snapped and timestamp - self._last_event["land"] >= self.t.event_cooldown:
            self._last_event["land"] = timestamp
            self._snap_primed_at = 0.0
            self._snap_middle_ref = None
            self._snap_candidate_at = 0.0
            self._snap_candidate_ref = None
            self.phase = "LAND gesture"
            confidence = min(1.0, 0.65 + 0.25 * motion)
            return InferenceEvent("land", confidence, f"visual snap motion {motion:.2f}")
        return None


class GuidedTrainingSession:
    """用倒计时切分十次动作，避免让用户手工逐条保存。"""

    TARGET_SAMPLES = 10
    _CAPTURE_SECONDS = {"takeoff": 1.35, "land": 1.05, "none": 1.10}

    def __init__(self, label: str, started_at: float) -> None:
        if label not in self._CAPTURE_SECONDS:
            raise ValueError(f"unknown training label: {label}")
        self.label = label
        self.samples: list[LandmarkSequence] = []
        self.phase = "prepare"
        self.phase_started_at = started_at
        self._current: list[list[Point3]] = []
        self.completed = False
        self.last_capture_ok: bool | None = None

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def status(self) -> str:
        number = min(self.count + 1, self.TARGET_SAMPLES)
        if self.completed:
            return f"TRAIN {self.label}: complete {self.count}/{self.TARGET_SAMPLES}"
        if self.phase == "capture":
            return f"TRAIN {self.label}: RECORD {number}/{self.TARGET_SAMPLES}"
        suffix = " - retry, keep hand visible" if self.last_capture_ok is False else ""
        return f"TRAIN {self.label}: GET READY {number}/{self.TARGET_SAMPLES}{suffix}"

    def update(self, timestamp: float, landmarks: Sequence[Point3] | None) -> None:
        if self.completed:
            return
        elapsed = timestamp - self.phase_started_at
        prepare_seconds = 1.15 if self.count == 0 else 0.80
        if self.phase == "prepare":
            if elapsed >= prepare_seconds:
                self.phase = "capture"
                self.phase_started_at = timestamp
                self._current = []
            return

        if landmarks is not None and len(landmarks) >= 21:
            self._current.append(list(landmarks))
        if elapsed < self._CAPTURE_SECONDS[self.label]:
            return

        # 低帧率下 6 帧仍足够做 DTW；不足则不计数并重试当前轮。
        if len(self._current) >= 6:
            self.samples.append(self._current)
            self.last_capture_ok = True
        else:
            self.last_capture_ok = False
        self._current = []
        if self.count >= self.TARGET_SAMPLES:
            self.completed = True
            return
        self.phase = "prepare"
        self.phase_started_at = timestamp


class MediaPipeGestureBackend(InferenceBackend):
    """MediaPipe 关键点 + 默认规则或个人 DTW 动态手势模板。"""

    def __init__(
        self,
        model_path: str | pathlib.Path | None = None,
        profile_dir: str | pathlib.Path | None = None,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "手势后端需要 mediapipe；请运行 pip install -r requirements.txt"
            ) from exc

        path = pathlib.Path(model_path) if model_path else (
            pathlib.Path(__file__).parent / "assets" / "gesture_recognizer.task"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少手势模型 {path}；请按 README 下载 gesture_recognizer.task"
            )

        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._mp = mp
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._detector = GestureSequenceDetector()
        self._events: deque[InferenceEvent] = deque()
        self._status = "Gesture: waiting for hand"
        self._last_timestamp_ms = 0
        self._last_logged_status = ""
        self._last_status_log = 0.0
        self._profile_dir = pathlib.Path(profile_dir) if profile_dir else (
            pathlib.Path(__file__).resolve().parent.parent / "gesture_profiles"
        )
        try:
            self._profile, self._profile_path = load_latest(self._profile_dir)
        except Exception as exc:
            logger.warning("gesture profile load failed: %s", exc)
            self._profile, self._profile_path = None, None
        self._training_session: GuidedTrainingSession | None = None
        self._training_samples: dict[str, list[LandmarkSequence]] = {
            "takeoff": [], "land": [], "none": [],
        }
        self._training_summary = ""
        self._live_landmarks: deque[tuple[float, list[Point3]]] = deque()
        self._last_match_at = 0.0
        self._match_candidate = ""
        self._match_hits = 0
        self._profile_armed = True
        self._hand_absent_since: float | None = None
        self._profile_last_event = {"takeoff": -1e9, "land": -1e9}
        if self._profile_path:
            logger.info("loaded gesture profile: %s", self._profile_path)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        timestamp = timestamp_ms / 1000.0
        if not result.hand_landmarks:
            if self._training_session:
                self._training_session.update(timestamp, None)
                self._status = self._training_session.status
                if self._training_session.completed:
                    self._finish_training()
            else:
                self._detector.reset()
                self._live_landmarks.clear()
                self._mark_hand_absent(timestamp)
                prefix = "Gesture profile" if self._profile else "Gesture"
                self._status = f"{prefix}: waiting for hand"
            self._log_status()
            return frame

        hand = result.hand_landmarks[0]
        landmarks = [(point.x, point.y, point.z) for point in hand]
        self._hand_absent_since = None
        open_score = 0.0
        label = "None"
        if result.gestures and result.gestures[0]:
            category = result.gestures[0][0]
            label = category.category_name or "None"
            if label == "Open_Palm":
                open_score = float(category.score)

        if self._training_session:
            self._training_session.update(timestamp, landmarks)
            self._status = self._training_session.status
            if self._training_session.completed:
                self._finish_training()
        elif self._profile:
            armed_before = self._profile_armed
            rule_events = self._detector.update(timestamp, landmarks, open_score)
            event_count = len(self._events)
            self._update_profile_match(timestamp, landmarks)
            # 个性化 DTW 是主通道；经过现场硬化的视觉响指规则作为降落兜底。
            # 两者仍共用离手重置和 App 层 dry-run/飞行安全门控。
            if (
                armed_before
                and self._profile_armed
                and len(self._events) == event_count
                and any(event.kind == "land" for event in rule_events)
            ):
                self._events.append(InferenceEvent(
                    "land", 0.75, "hybrid visual-snap rule fallback"
                ))
                self._profile_armed = False
                self._status = "Gesture profile: LAND detected (hybrid)"
        else:
            for event in self._detector.update(timestamp, landmarks, open_score):
                self._events.append(event)
            self._status = f"Gesture: {label} {open_score:.2f} | {self._detector.phase}"
        self._log_status()
        self._draw_hand(frame, landmarks)
        return frame

    def _mark_hand_absent(self, timestamp: float) -> None:
        """连续看不到手 0.4 秒才重新布防，忽略 MediaPipe 的单帧漏检。"""
        if not self._profile:
            return
        if self._hand_absent_since is None:
            self._hand_absent_since = timestamp
        elif timestamp - self._hand_absent_since >= 0.40:
            self._profile_armed = True

    def _update_profile_match(self, timestamp: float, landmarks: list[Point3]) -> None:
        if self._live_landmarks and timestamp - self._live_landmarks[-1][0] > 0.30:
            self._live_landmarks.clear()
        self._live_landmarks.append((timestamp, landmarks))
        while self._live_landmarks and timestamp - self._live_landmarks[0][0] > 2.0:
            self._live_landmarks.popleft()
        if (
            len(self._live_landmarks) < 8
            or timestamp - self._live_landmarks[0][0] < 0.45
            or timestamp - self._last_match_at < 0.12
        ):
            self._status = "Gesture profile: observing motion"
            return
        self._last_match_at = timestamp
        assert self._profile is not None
        results = self._profile.nearest([points for _, points in self._live_landmarks])
        if not results:
            self._status = "Gesture profile: no templates"
            return

        best = results[0]
        ratio = best.distance / max(best.threshold, 1e-6)
        raw = {item.label: item.distance for item in results}
        self._status = (
            f"Profile raw T{raw.get('takeoff', math.inf):.2f} "
            f"L{raw.get('land', math.inf):.2f} N{raw.get('none', math.inf):.2f} "
            f"| {best.label} {ratio:.2f}"
        )
        if not self._profile_armed:
            self._match_candidate = ""
            self._match_hits = 0
            self._status += " | remove hand to rearm"
            return
        none_result = next((item for item in results if item.label == "none"), None)
        none_distance = none_result.distance if none_result else math.inf
        other_actions = [
            item.distance
            for item in results if item.label not in (best.label, "none")
        ]
        ambiguous_action = bool(other_actions) and min(other_actions) <= best.distance + 0.08
        accepted = (
            best.label in ("takeoff", "land")
            and ratio <= 1.25
            and best.distance + 0.10 < none_distance
            and not ambiguous_action
        )
        if not accepted:
            self._match_candidate = ""
            self._match_hits = 0
            return
        if self._match_candidate == best.label:
            self._match_hits += 1
        else:
            self._match_candidate = best.label
            self._match_hits = 1
        self._status += f" | confirm {self._match_hits}/2"
        if self._match_hits < 2:
            return
        if timestamp - self._profile_last_event[best.label] < 3.0:
            return

        confidence = max(0.55, min(0.99, 1.0 - 0.45 * ratio))
        detail = f"personal DTW {best.label} distance={best.distance:.3f}"
        self._events.append(InferenceEvent(best.label, confidence, detail))
        self._profile_last_event[best.label] = timestamp
        self._profile_armed = False
        self._status = f"Gesture profile: {best.label.upper()} detected"
        self._live_landmarks.clear()
        self._match_candidate = ""
        self._match_hits = 0

    def _finish_training(self, cancelled: bool = False) -> str:
        session = self._training_session
        if not session:
            return self._training_summary
        if session.samples:
            # 再次训练同一动作时替换旧的内存样本；已落盘 profile 不会被覆盖。
            self._training_samples[session.label] = list(session.samples)
        state = "stopped" if cancelled else "complete"
        self._training_summary = (
            f"TRAIN {session.label}: {state} {session.count}/{session.TARGET_SAMPLES}"
        )
        self._training_session = None
        self._status = self._training_summary
        return self._training_summary

    @property
    def training_supported(self) -> bool:
        return True

    @property
    def active_training_label(self) -> str:
        return self._training_session.label if self._training_session else ""

    def toggle_training(self, label: str) -> str:
        if label not in ("takeoff", "land", "none"):
            return f"Unknown training label: {label}"
        if self._training_session:
            previous = self._training_session.label
            message = self._finish_training(cancelled=True)
            if previous == label:
                return message
        self._events.clear()
        self._detector.reset()
        self._live_landmarks.clear()
        self._profile_armed = True
        self._hand_absent_since = None
        self._training_session = GuidedTrainingSession(label, time.monotonic())
        self._status = self._training_session.status
        return f"Training {label}: follow GET READY / RECORD, 10 rounds"

    def save_training_profile(self) -> str:
        if self._training_session:
            self._finish_training(cancelled=True)
        counts = {label: len(samples) for label, samples in self._training_samples.items()}
        missing = [label for label, count in counts.items() if count < 3]
        if missing:
            summary = ", ".join(f"{label}={counts[label]}" for label in counts)
            return f"Need at least 3 each before save ({summary})"
        profile = GestureProfile.build(self._training_samples)
        path = profile.save_new(self._profile_dir)
        self._profile, self._profile_path = profile, path
        self._live_landmarks.clear()
        self._profile_armed = True
        self._hand_absent_since = None
        self._training_summary = f"Profile saved: {path.name}"
        self._status = self._training_summary
        logger.info("gesture profile saved: %s counts=%s", path, counts)
        return self._training_summary

    def _log_status(self) -> None:
        """dry-run/现场标定时提供节流后的阶段信息。"""
        now = time.monotonic()
        if self._status != self._last_logged_status and now - self._last_status_log >= 0.5:
            logger.info("%s", self._status)
            self._last_logged_status = self._status
            self._last_status_log = now

    def _draw_hand(self, frame: np.ndarray, landmarks: Sequence[Point3]) -> None:
        h, w = frame.shape[:2]
        pts = [
            (max(0, min(w - 1, int(p[0] * w))), max(0, min(h - 1, int(p[1] * h))))
            for p in landmarks
        ]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 80), 2, cv2.LINE_AA)
        for i, point in enumerate(pts):
            color = (0, 180, 255) if i in (4, 12) else (255, 180, 40)
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)

    def drain_events(self) -> list[InferenceEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    @property
    def status_text(self) -> str:
        return self._status

    def close(self) -> None:
        self._recognizer.close()
