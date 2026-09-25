"""纯 OpenCV 主体初始化、光流传播和中心降级跟踪后端。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError, ConfigurationError

from .keyframe_selector import interval_scene_spans, reinitialization_frames
from .prompt_generator import (
    observation_points,
    observation_primary_points,
    observation_recommended_crop_center,
    observation_recommended_crop_confidence,
    subject_observations,
    subject_point,
    subject_point_frames,
)
from .subject_selector import _clamp_box, select_subject_box
from .track_monitor import track_is_valid

if TYPE_CHECKING:
    from .mask_geometry import MaskEvidence


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """初始的帧级跟踪点，用于后续最终裁剪框计算"""
    frame: int
    subject_box: list[float]
    confidence: float
    source: str
    object_count: int = 1
    object_ids: tuple[int, ...] = ()
    # 仅 SAM2 后端提供。使用紧凑面积密度图而非完整分辨率 Mask，避免长视频内存膨胀。
    mask_evidence: MaskEvidence | None = None
    primary_object_ids: tuple[int, ...] = ()
    primary_focus_points: tuple[tuple[float, float], ...] = ()
    object_importance: tuple[tuple[int, float], ...] = ()
    composition_mode: str = "single_focus"
    # Stage 3.5 v3 帧级构图建议。坐标在原始帧像素空间中，供候选与弱评分共用。
    recommended_crop_center: tuple[float, float] | None = None
    recommended_crop_confidence: float = 0.0


class SubjectTracker(Protocol):
    def track(
        self,
        video_path: Path,
        interval: dict[str, Any],
        frame_size: tuple[int, int],
        scenes: list[dict[str, Any]],
        visualization_dir: Path | None = None,
    ) -> list[TrackPoint]:
        """
        Returns:
            按帧升序排列的list[TrackPoint]

        """
        ...


class CenterSubjectTracker:
    """不解码视频的 Qwen 点线性插值后端，并以固定画面中心作为最终兜底。

    名称保留为 ``center`` 以兼容现有 CLI/config，但行为不再是无条件使用画面中心：同一镜头中只要存在有效 Stage 3.5 点，就在相邻点之间逐帧线性插值；
    首点之前和末点之后保持最近锚点。只有该镜头完全没有有效点时才使用固定中心。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def _valid_rows(interval: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
        """找出镜头内至少含一个有效目标点的观察。"""
        return [
            row for row in subject_observations(interval)
            if start <= int(row.get("frame", -1)) < end and observation_points(row)
        ]

    @staticmethod
    def _recommendation_rows(interval: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
        return [
            row for row in subject_observations(interval)
            if start <= int(row.get("frame", -1)) < end
            and observation_recommended_crop_center(row) is not None
        ]

    @staticmethod
    def _recommendation_at(
        rows: list[dict[str, Any]], frame: int, frame_size: tuple[int, int]
    ) -> tuple[tuple[float, float], float]:
        """在同一镜头内线性插值 Qwen 构图中心；镜头无推荐时回退画面中心。"""

        frame_w, frame_h = frame_size
        if not rows:
            return (frame_w * 0.5, frame_h * 0.5), 0.0
        frames = [int(row["frame"]) for row in rows]
        left_index = max(0, min(len(rows) - 1, bisect_right(frames, frame) - 1))
        left = rows[left_index]
        left_point = observation_recommended_crop_center(left)
        assert left_point is not None
        left_confidence = observation_recommended_crop_confidence(left)
        if frame <= frames[0] or left_index + 1 >= len(rows):
            point, confidence = left_point, left_confidence
        else:
            right = rows[left_index + 1]
            right_point = observation_recommended_crop_center(right)
            assert right_point is not None
            alpha = (frame - frames[left_index]) / max(1, frames[left_index + 1] - frames[left_index])
            point = (
                left_point[0] + alpha * (right_point[0] - left_point[0]),
                left_point[1] + alpha * (right_point[1] - left_point[1]),
            )
            right_confidence = observation_recommended_crop_confidence(right)
            confidence = left_confidence + alpha * (right_confidence - left_confidence)
        return (point[0] * frame_w, point[1] * frame_h), confidence

    def _point_box(self, point: tuple[float, float], frame_size: tuple[int, int]) -> list[float]:
        width, height = frame_size
        box_w = width * float(self.config.get("initial_width_ratio", 0.28))
        box_h = height * float(self.config.get("initial_height_ratio", 0.38))
        center_x, center_y = point[0] * width, point[1] * height
        return _clamp_box(
            (
                center_x - box_w * 0.5,
                center_y - box_h * 0.5,
                center_x + box_w * 0.5,
                center_y + box_h * 0.5,
            ),
            width,
            height,
        )

    def _observation_box(self, row: dict[str, Any], frame_size: tuple[int, int]) -> list[float]:
        boxes = [self._point_box(point, frame_size) for point in observation_primary_points(row)]
        if not boxes:
            return self._point_box((0.5, 0.5), frame_size)
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    def track(self, video_path: Path, interval: dict[str, Any], frame_size: tuple[int, int], scenes: list[dict[str, Any]], visualization_dir: Path | None = None) -> list[TrackPoint]:
        del video_path, visualization_dir
        output: list[TrackPoint] = []
        for span_start, span_end in interval_scene_spans(interval, scenes):
            rows = self._valid_rows(interval, span_start, span_end)
            recommendation_rows = self._recommendation_rows(interval, span_start, span_end)
            if not rows:
                # 该镜头没有任何可用空间观测时，才退回真正的画面中心。
                center_box = self._point_box((0.5, 0.5), frame_size)
                for frame in range(span_start, span_end):
                    recommended_center, recommended_confidence = self._recommendation_at(
                        recommendation_rows, frame, frame_size
                    )
                    output.append(TrackPoint(
                        frame, list(center_box), 0.0, "center_fallback",
                        recommended_crop_center=recommended_center,
                        recommended_crop_confidence=recommended_confidence,
                    ))
                continue

            cursor = 0
            for frame in range(span_start, span_end):
                while cursor + 1 < len(rows) and int(rows[cursor + 1]["frame"]) <= frame: # 经典防御性编程，可以跳过重复的rows帧
                    cursor += 1
                left = rows[cursor]
                left_frame = int(left["frame"])
                left_box = self._observation_box(left, frame_size)
                if frame < int(rows[0]["frame"]):
                    box = left_box
                    source = "qwen_anchor_hold"
                elif cursor + 1 < len(rows):
                    right = rows[cursor + 1]
                    right_frame = int(right["frame"])
                    right_box = self._observation_box(right, frame_size)
                    alpha = (frame - left_frame) / max(1, right_frame - left_frame) # 纯线性插值，alpha为区间比值
                    box = [
                        left_box[index] + alpha * (right_box[index] - left_box[index])
                        for index in range(4)
                    ]
                    source = "qwen_anchor" if frame == left_frame else "qwen_linear"
                else:
                    box = left_box
                    source = "qwen_anchor" if frame == left_frame else "qwen_anchor_hold"
                confidence = max(
                    (float(target.get("confidence", 0.0)) for target in left.get("targets", [])),
                    default=0.85,
                )
                primary_points = observation_primary_points(left)
                pixel_primary_points = tuple(
                    (point[0] * frame_size[0], point[1] * frame_size[1]) for point in primary_points
                )
                recommended_center, recommended_confidence = self._recommendation_at(
                    recommendation_rows, frame, frame_size
                )
                output.append(TrackPoint(
                    frame, list(box), confidence, source,
                    max(1, len(observation_points(left))), (), None, (), pixel_primary_points, (),
                    str(left.get("composition_mode", "group_focus" if len(primary_points) > 1 else "single_focus")),
                    recommended_center, recommended_confidence,
                ))
        return output


class OpenCVSubjectTracker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def _features(gray: np.ndarray, box: list[float], maximum: int) -> np.ndarray | None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        mask[max(0, y1):min(gray.shape[0], y2), max(0, x1):min(gray.shape[1], x2)] = 255
        return cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=maximum, qualityLevel=0.01, minDistance=5, blockSize=7)

    def track(self, video_path: Path, interval: dict[str, Any], frame_size: tuple[int, int], scenes: list[dict[str, Any]], visualization_dir: Path | None = None) -> list[TrackPoint]:
        del visualization_dir
        if not video_path.is_file():
            raise ArtifactValidationError(f"源视频不存在: {video_path}")
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        resets = reinitialization_frames(interval, scenes, int(self.config.get("reinitialize_every_frames", 0)))
        # 每个 Stage 3.5 有效采样点都是新的空间观测；在这些帧主动校正光流漂移。
        resets.update(subject_point_frames(interval))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ArtifactValidationError(f"OpenCV 无法打开视频: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        previous_gray: np.ndarray | None = None
        previous_box: list[float] | None = None
        previous_points: np.ndarray | None = None
        output: list[TrackPoint] = []
        maximum_points = int(self.config.get("max_corners", 120))
        try:
            for frame_index in range(start, end):
                ok, frame = capture.read()
                if not ok:
                    raise ArtifactValidationError(f"视频在 frame={frame_index} 提前结束")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_index in resets or previous_gray is None or previous_box is None:
                    point = subject_point(interval, frame_index)
                    box, confidence, source = select_subject_box(frame, point, self.config)
                    points = self._features(gray, box, maximum_points)
                else:
                    if previous_points is None or len(previous_points) < 4:
                        previous_points = self._features(previous_gray, previous_box, maximum_points)
                    box, confidence, source = list(previous_box), 0.0, "optical_flow"
                    points = None
                    if previous_points is not None and len(previous_points):
                        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points, None, winSize=(21, 21), maxLevel=3)
                        if tracked is not None and status is not None:
                            valid = status.reshape(-1).astype(bool)
                            old, new = previous_points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid]
                            if len(new) >= 4:
                                delta = np.median(new - old, axis=0)
                                box = [previous_box[0] + float(delta[0]), previous_box[1] + float(delta[1]), previous_box[2] + float(delta[0]), previous_box[3] + float(delta[1])]
                                width, height = frame_size
                                box_w, box_h = box[2] - box[0], box[3] - box[1]
                                cx = min(max((box[0] + box[2]) * 0.5, box_w * 0.5), width - box_w * 0.5)
                                cy = min(max((box[1] + box[3]) * 0.5, box_h * 0.5), height - box_h * 0.5)
                                box = [cx - box_w * 0.5, cy - box_h * 0.5, cx + box_w * 0.5, cy + box_h * 0.5]
                                confidence = float(len(new) / max(len(previous_points), 1))
                                points = new.reshape(-1, 1, 2).astype(np.float32)
                    if not track_is_valid(previous_box, box, frame_size, confidence, self.config):
                        point = subject_point(interval, frame_index)
                        box, confidence, source = select_subject_box(frame, point, self.config)
                        points = self._features(gray, box, maximum_points)
                        source = "reinitialized_" + source
                output.append(TrackPoint(frame_index, list(box), confidence, source))
                previous_gray, previous_box, previous_points = gray, list(box), points
        finally:
            capture.release()
        return output


def build_subject_tracker(config: dict[str, Any], visualization_config: dict[str, Any] | None = None) -> SubjectTracker:
    backend = str(config.get("backend", "opencv")).lower()
    if backend == "opencv":
        return OpenCVSubjectTracker(config)
    if backend == "center":
        return CenterSubjectTracker(config)
    if backend == "sam2":
        from .sam2_adapter import SAM2SubjectTracker
        return SAM2SubjectTracker(config, visualization_config or {})
    raise ConfigurationError(f"未知 Stage 4 tracking.backend: {backend}")
