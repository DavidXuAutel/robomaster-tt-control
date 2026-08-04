"""少样本动态手势模板：关键点归一化、DTW、自动阈值与本地 profile。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

Point3 = tuple[float, float, float]
LandmarkFrame = Sequence[Point3]
LandmarkSequence = Sequence[LandmarkFrame]
FeatureSequence = list[list[float]]

LABELS = ("takeoff", "land", "none")
PROFILE_VERSION = 1


def _distance(a: Point3, b: Point3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _palm_center(frame: LandmarkFrame) -> Point3:
    ids = (0, 5, 9, 13, 17)
    return (
        sum(frame[i][0] for i in ids) / len(ids),
        sum(frame[i][1] for i in ids) / len(ids),
        sum(frame[i][2] for i in ids) / len(ids),
    )


def sequence_features(sequence: LandmarkSequence) -> FeatureSequence:
    """将关键点序列转换为旋转/尺度归一化手形 + 掌心全局轨迹。"""
    if not sequence:
        return []
    start_palm = _palm_center(sequence[0])
    start_scale = max(_distance(sequence[0][5], sequence[0][17]), 1e-4)
    output: FeatureSequence = []
    for frame in sequence:
        if len(frame) < 21:
            continue
        palm = _palm_center(frame)
        scale = max(
            _distance(frame[5], frame[17]),
            _distance(frame[0], frame[9]) * 0.75,
            1e-4,
        )
        # 将掌横轴（食指 MCP -> 小指 MCP）旋转到统一方向。
        axis_x = frame[17][0] - frame[5][0]
        axis_y = frame[17][1] - frame[5][1]
        angle = math.atan2(axis_y, axis_x)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        features: list[float] = []
        for point in frame:
            dx = (point[0] - palm[0]) / scale
            dy = (point[1] - palm[1]) / scale
            features.extend((dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a))

        # 全局轨迹不能随手形归一化掉；起飞主要依赖这一部分。
        features.extend((
            (palm[0] - start_palm[0]) / start_scale,
            (palm[1] - start_palm[1]) / start_scale,
            _distance(frame[4], frame[12]) / scale,
            _distance(frame[12], palm) / scale,
            _distance(frame[12], frame[9]) / scale,
        ))
        output.append(features)
    return output


def feature_weights(size: int) -> list[float]:
    # 42 个局部 xy 关键点 + global x/y + pinch/middle-palm/middle-extension。
    weights = [0.35] * min(42, size)
    tail = [1.6, 2.8, 1.5, 1.3, 1.2]
    weights.extend(tail[: max(0, size - len(weights))])
    return weights[:size]


def dtw_distance(a: FeatureSequence, b: FeatureSequence) -> float:
    """带 Sakoe-Chiba 窗口的归一化多变量 DTW 距离。"""
    if not a or not b:
        return math.inf
    n, m = len(a), len(b)
    dims = min(len(a[0]), len(b[0]))
    weights = feature_weights(dims)
    band = max(abs(n - m), int(max(n, m) * 0.40), 2)
    inf = (math.inf, 0)
    prev = [inf] * (m + 1)
    prev[0] = (0.0, 0)
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        start, stop = max(1, i - band), min(m, i + band)
        for j in range(start, stop + 1):
            local = math.sqrt(sum(
                weights[k] * (a[i - 1][k] - b[j - 1][k]) ** 2
                for k in range(dims)
            ))
            predecessor = min(prev[j], cur[j - 1], prev[j - 1], key=lambda item: item[0])
            cur[j] = (predecessor[0] + local, predecessor[1] + 1)
        prev = cur
    cost, length = prev[m]
    return cost / max(length, 1)


def _quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return math.inf
    index = min(len(clean) - 1, max(0, round((len(clean) - 1) * q)))
    return clean[index]


@dataclass(frozen=True)
class MatchResult:
    label: str
    distance: float
    threshold: float
    score: float


@dataclass
class GestureProfile:
    templates: dict[str, list[FeatureSequence]]
    thresholds: dict[str, float]
    created_at: str

    @classmethod
    def build(cls, samples: dict[str, list[LandmarkSequence]]) -> "GestureProfile":
        templates = {
            label: [features for seq in samples.get(label, [])
                    if len(features := sequence_features(seq)) >= 5]
            for label in LABELS
        }
        thresholds: dict[str, float] = {}
        for label, class_templates in templates.items():
            positives: list[float] = []
            negatives: list[float] = []
            for i, template in enumerate(class_templates):
                same = [
                    dtw_distance(template, other)
                    for j, other in enumerate(class_templates) if i != j
                ]
                if same:
                    positives.append(min(same))
                for other_label, other_templates in templates.items():
                    if other_label != label:
                        negatives.extend(dtw_distance(template, other) for other in other_templates)
            positive_edge = _quantile(positives, 0.90)
            if not math.isfinite(positive_edge):
                positive_edge = 0.10
            candidate = max(0.035, positive_edge * 1.35 + 0.01)
            negative_edge = min(negatives, default=math.inf)
            if math.isfinite(negative_edge) and negative_edge > positive_edge:
                candidate = min(candidate, positive_edge + 0.45 * (negative_edge - positive_edge))
            thresholds[label] = candidate
        return cls(
            templates=templates,
            thresholds=thresholds,
            created_at=datetime.now().astimezone().isoformat(),
        )

    def nearest(self, live: LandmarkSequence) -> list[MatchResult]:
        results: list[MatchResult] = []
        for label, templates in self.templates.items():
            if not templates:
                continue
            # 同时尝试多种尾窗长度：用户实时动作可能比训练时快或慢，
            # DTW 处理局部速度差，尾窗搜索负责找到动作真正的起点。
            distances: list[float] = []
            for template in templates:
                window_sizes = {
                    min(len(live), max(6, round(len(template) * factor)))
                    for factor in (0.80, 1.00, 1.20, 1.50, 1.80)
                }
                distances.append(min(
                    dtw_distance(sequence_features(live[-window_size:]), template)
                    for window_size in window_sizes
                ))
            distance = min(distances)
            threshold = self.thresholds.get(label, 0.1)
            score = max(0.0, min(1.0, 1.0 - distance / max(threshold, 1e-6)))
            results.append(MatchResult(label, distance, threshold, score))
        # 各类阈值只用于“是否足够像该类”，不能用于类别排序。尤其 NONE
        # 样本通常更杂、阈值更宽；若按 distance/threshold 排序，它会不合理地
        # 抢走本来与 LAND 更接近的动作。
        return sorted(results, key=lambda item: item.distance)

    def save_new(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        path = directory / f"profile_{stamp}.json"
        payload = {
            "version": PROFILE_VERSION,
            "created_at": self.created_at,
            "thresholds": self.thresholds,
            "templates": self.templates,
        }
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GestureProfile":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != PROFILE_VERSION:
            raise ValueError(f"不支持的手势 profile 版本: {payload.get('version')}")
        return cls(
            templates=payload["templates"],
            thresholds={k: float(v) for k, v in payload["thresholds"].items()},
            created_at=payload["created_at"],
        )


def load_latest(directory: str | Path) -> tuple[GestureProfile | None, Path | None]:
    paths = sorted(Path(directory).glob("profile_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return None, None
    path = paths[-1]
    return GestureProfile.load(path), path
