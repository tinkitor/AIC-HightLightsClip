"""Grounding DINO 多目标锚定 + SAM2 镜头内窗口传播。"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError, ConfigurationError
from video_highlight.contracts.schema_versions import STAGE4_SCHEMA_VERSION

from .grounding_selector import (
    GroundedDetection,
    ScoredDetection,
    box_iou,
    select_detections,
    stabilize_object_assignments,
)
from .keyframe_selector import interval_scene_spans
from .mask_geometry import build_mask_evidence
from .prompt_generator import observation_points, observation_primary_points, subject_observations
from .propagation_visualizer import PropagationVisualizer, VisualizationError
from .subject_selector import select_subject_box
from .subject_tracker import CenterSubjectTracker, TrackPoint


@dataclass(frozen=True, slots=True)
class AnchorWindow:
    """一个镜头内由当前 Stage 3.5 观察覆盖的左闭右开传播窗口。"""

    start: int
    end: int
    observation: dict[str, Any] | None
    is_qwen_anchor: bool

    @property
    def points(self) -> list[tuple[float, float]]:
        return [] if self.observation is None else observation_points(self.observation)

    @property
    def grounding_phrases(self) -> list[str]:
        if self.observation is None:
            return []
        values = self.observation.get("grounding_phrases", [])
        return [str(value) for value in values if str(value).strip()]

    @property
    def group_mode(self) -> str:
        if self.observation is None:
            return "single"
        return str(self.observation.get("group_mode", "multiple" if len(self.points) > 1 else "single"))

    @property
    def point(self) -> tuple[float, float] | None:
        """保留旧测试和可视化使用的群体中心属性。"""

        points = self.points
        if not points:
            return None
        return (
            (min(point[0] for point in points) + max(point[0] for point in points)) * 0.5,
            (min(point[1] for point in points) + max(point[1] for point in points)) * 0.5,
        )


@dataclass(slots=True)
class SAM2RecoveryGate:
    """SAM2 Mask 缺失后的连续帧恢复确认状态机。

    正常传播时有效 Mask 立即通过；一旦出现空 Mask，后续重新出现的 Mask 必须连续
    ``required_frames`` 帧有效才恢复使用。确认期的前几帧仍输出 fallback，且任何再次
    缺失都会把计数清零，避免零星恢复帧立即造成构图跳变。
    """

    required_frames: int
    recovering: bool = False
    valid_streak: int = 0

    def observe(self, valid: bool) -> bool:
        if not valid:
            self.recovering = True
            self.valid_streak = 0
            return False
        if not self.recovering:
            return True
        self.valid_streak += 1
        if self.valid_streak < max(1, int(self.required_frames)):
            return False
        self.recovering = False
        self.valid_streak = 0
        return True


def anchor_windows(interval: dict[str, Any], span_start: int, span_end: int) -> list[AnchorWindow]:
    """用每条 Stage 3.5 观察建立窗口，镜头起点使用最近观察作为虚拟锚点。"""

    rows = [
        row for row in subject_observations(interval)
        if span_start <= int(row.get("frame", -1)) < span_end
        and (
            observation_points(row)
            or any(str(value).strip() for value in row.get("grounding_phrases", []))
        )
    ]
    by_frame = {int(row["frame"]): row for row in rows}
    boundaries = [span_start, *sorted(frame for frame in by_frame if frame > span_start), span_end]
    windows: list[AnchorWindow] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start in by_frame:
            observation = by_frame[start]
            is_qwen_anchor = True
        elif rows:
            observation = min(rows, key=lambda row: abs(int(row["frame"]) - start))
            is_qwen_anchor = False
        else:
            observation = None
            is_qwen_anchor = False
        windows.append(AnchorWindow(start, end, observation, is_qwen_anchor))
    return windows


def update_grounding_phrase_memory(
    current_phrases: list[str],
    phrase_last_seen: dict[str, int],
    anchor_index: int,
    ttl_anchors: int,
    maximum: int = 12,
) -> list[str]:
    """合并当前和近期 Grounding 短语，避免一次 Qwen 漏词立刻停止检测旧主体。"""

    current: list[str] = []
    seen: set[str] = set()
    for value in current_phrases:
        phrase = str(value).strip().lower().rstrip(". ")
        if phrase and phrase not in seen:
            current.append(phrase)
            seen.add(phrase)
            phrase_last_seen[phrase] = anchor_index
    ttl = max(0, int(ttl_anchors))
    for phrase, last_seen in list(phrase_last_seen.items()):
        if anchor_index - last_seen > ttl:
            del phrase_last_seen[phrase]
    historical = sorted(
        (phrase for phrase in phrase_last_seen if phrase not in seen),
        key=lambda phrase: (-phrase_last_seen[phrase], phrase),
    )
    return [*current, *historical][:max(1, int(maximum))]


def _union_box(boxes: list[list[float] | tuple[float, ...]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _mask_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def associate_targets_to_objects(
    observation: dict[str, Any] | None,
    assignments: list[tuple[int, GroundedDetection]],
    frame_size: tuple[int, int],
) -> dict[int, dict[str, Any]]:
    """把本帧 Qwen 语义目标映射到已稳定的 SAM2 object_id。"""

    if not observation or not assignments:
        return {}
    width, height = frame_size
    primary_ids = {str(value) for value in observation.get("primary_target_ids", [])}
    targets = [target for target in observation.get("targets", []) if isinstance(target, dict)]
    targets = [target for target in targets if isinstance(target.get("subject_point"), list) and len(target["subject_point"]) == 2]
    targets.sort(key=lambda target: (
        str(target.get("target_id")) not in primary_ids and target.get("role") != "primary",
        -float(target.get("importance", 0.5)),
        -float(target.get("confidence", 0.0)),
    ))
    output: dict[int, dict[str, Any]] = {}
    unused = {object_id for object_id, _ in assignments}
    detections = dict(assignments)
    for target in targets:
        if not unused:
            break
        point = target["subject_point"]
        px, py = float(point[0]) * width, float(point[1]) * height
        target_tokens = set(str(target.get("grounding_phrase", "")).lower().replace(".", "").split())
        choices: list[tuple[float, int]] = []
        for object_id in unused:
            detection = detections[object_id]
            box = detection.box_xyxy
            center_x, center_y = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
            distance = math.hypot(px - center_x, py - center_y) / max(1.0, math.hypot(width, height))
            outside = 0.0 if box[0] <= px <= box[2] and box[1] <= py <= box[3] else 0.5
            phrase_tokens = set(str(detection.phrase).lower().replace(".", "").split())
            semantic_bonus = 0.15 * len(target_tokens & phrase_tokens) / max(1, len(target_tokens | phrase_tokens))
            choices.append((outside + distance - semantic_bonus, object_id))
        if not choices:
            continue
        object_id = min(choices)[1]
        unused.discard(object_id)
        target_id = str(target.get("target_id", ""))
        focus = target.get("focus_point", point)
        output[object_id] = {
            "target_id": target_id,
            "role": "primary" if target_id in primary_ids or target.get("role") == "primary" else "supporting",
            "is_primary": target_id in primary_ids or target.get("role") == "primary",
            "importance": max(0.0, min(1.0, float(target.get("importance", 0.5)))),
            "confidence": max(0.0, min(1.0, float(target.get("confidence", 0.0)))),
            "focus_point": (float(focus[0]), float(focus[1])),
        }
    return output


def update_primary_object_state(
    current: tuple[int, ...],
    challenger: tuple[int, ...],
    challenger_streak: int,
    proposed: tuple[int, ...],
    active_ids: set[int],
    confirm_anchors: int = 2,
) -> tuple[tuple[int, ...], tuple[int, ...], int, bool]:
    """镜头内主主体迟滞：新提议需连续出现，已消失的主主体则立即切换。"""

    current = tuple(sorted(value for value in current if value in active_ids))
    proposed = tuple(sorted(value for value in proposed if value in active_ids))
    if not current:
        return proposed, (), 0, bool(proposed)
    if not proposed or proposed == current:
        return current, (), 0, False
    if proposed == challenger:
        challenger_streak += 1
    else:
        challenger, challenger_streak = proposed, 1
    if challenger_streak >= max(1, int(confirm_anchors)):
        return proposed, (), 0, True
    return current, challenger, challenger_streak, False


class SAM2SubjectTracker:
    """用 Grounding DINO 选择多个语义对象，并在每个锚点窗口用 SAM2 传播。"""

    def __init__(self, config: dict[str, Any], visualization_config: dict[str, Any] | None = None) -> None:
        checkpoint = Path(str(config.get("checkpoint", "")))
        model_config = str(config.get("model_config", ""))
        if not checkpoint.is_file() or not model_config:
            raise ConfigurationError("SAM2 后端要求有效的 tracking.checkpoint 和 model_config")
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as error:
            raise ConfigurationError("SAM2 后端需要安装官方 sam2 包和兼容的 PyTorch") from error
        self.torch = torch
        self.device = str(config.get("device", "cuda"))
        self.amp_dtype = str(config.get("amp_dtype", "bfloat16"))
        self.predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=self.device)
        self.config = config
        self.visualization_config = visualization_config or {}
        grounding_config = config.get("grounding", {})
        self.grounding_config = grounding_config if isinstance(grounding_config, dict) else {}
        self.grounder = None
        if bool(self.grounding_config.get("enabled", False)):
            from .grounding_dino_adapter import GroundingDINOAdapter

            self.grounder = GroundingDINOAdapter(self.grounding_config)
        self.last_grounding_records: list[dict[str, Any]] = []

    def _contexts(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(self.torch.inference_mode())
        if self.device.startswith("cuda") and self.amp_dtype != "none":
            stack.enter_context(self.torch.autocast("cuda", dtype=getattr(self.torch, self.amp_dtype)))
        else:
            stack.enter_context(nullcontext())
        return stack

    @staticmethod
    def _write_scene_frames(video_path: Path, start: int, end: int, directory: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ArtifactValidationError(f"OpenCV 无法打开视频: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            for local_index, frame_index in enumerate(range(start, end)):
                ok, frame = capture.read()
                if not ok:
                    raise ArtifactValidationError(f"SAM2 镜头解码在 frame={frame_index} 提前结束")
                if not cv2.imwrite(str(directory / f"{local_index:06d}.jpg"), frame):
                    raise ArtifactValidationError(f"SAM2 临时帧写入失败: frame={frame_index}")
        finally:
            capture.release()

    @staticmethod
    def _logits_to_mask(logits: Any, frame_size: tuple[int, int]) -> np.ndarray:
        mask = (logits.detach().float().cpu().numpy().squeeze() > 0).astype(np.uint8)
        width, height = frame_size
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    def _active_masks(
        self,
        object_ids: Any,
        logits: Any,
        active_ids: set[int],
        frame_size: tuple[int, int],
        include_object_masks: bool = False,
    ) -> tuple[np.ndarray, dict[int, list[float]]] | tuple[
        np.ndarray, dict[int, list[float]], dict[int, np.ndarray]
    ]:
        width, height = frame_size
        union = np.zeros((height, width), dtype=np.uint8)
        boxes: dict[int, list[float]] = {}
        masks: dict[int, np.ndarray] = {}
        ids = [
            int(value.detach().cpu().item()) if hasattr(value, "detach")
            else int(value.item()) if hasattr(value, "item")
            else int(value)
            for value in object_ids
        ]
        for index, object_id in enumerate(ids):
            if object_id not in active_ids:
                continue
            mask = self._logits_to_mask(logits[index], frame_size)
            box = _mask_box(mask)
            if box is None:
                continue
            union |= mask
            boxes[object_id] = box
            if include_object_masks:
                masks[object_id] = mask
        if include_object_masks:
            return union, boxes, masks
        return union, boxes

    @staticmethod
    def _track_point(
        frame: int,
        mask: np.ndarray,
        source: str,
        object_ids: tuple[int, ...] = (),
        object_masks: dict[int, np.ndarray] | None = None,
        mask_grid_max_side: int = 160,
        primary_object_ids: tuple[int, ...] = (),
        object_focus_offsets: dict[int, tuple[float, float]] | None = None,
        object_importance: dict[int, float] | None = None,
        composition_mode: str = "single_focus",
        object_boxes: dict[int, list[float]] | None = None,
    ) -> TrackPoint | None:
        box = _mask_box(mask)
        if box is None:
            return None
        area = float(mask.sum())
        box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        focus_points: list[tuple[float, float]] = []
        for object_id in primary_object_ids:
            object_box = (object_boxes or {}).get(object_id)
            if object_box is None:
                continue
            offset = (object_focus_offsets or {}).get(object_id, (0.5, 0.5))
            focus_points.append((
                object_box[0] + offset[0] * (object_box[2] - object_box[0]),
                object_box[1] + offset[1] * (object_box[3] - object_box[1]),
            ))
        return TrackPoint(
            frame, box, min(1.0, area / box_area), source,
            max(1, len(object_ids)), object_ids,
            build_mask_evidence(mask, object_masks, mask_grid_max_side),
            tuple(value for value in primary_object_ids if value in object_ids),
            tuple(focus_points),
            tuple(sorted((int(key), float(value)) for key, value in (object_importance or {}).items() if key in object_ids)),
            composition_mode,
        )

    def _anchor_is_valid(
        self,
        prediction: TrackPoint | None,
        mask: np.ndarray,
        points: list[tuple[float, float]],
        selected_boxes: list[tuple[float, ...]],
        object_boxes: dict[int, list[float]],
        frame_size: tuple[int, int],
    ) -> bool:
        if prediction is None:
            return False
        width, height = frame_size
        if points:
            hits = 0
            for point in points:
                px = min(width - 1, max(0, int(round(point[0] * (width - 1)))))
                py = min(height - 1, max(0, int(round(point[1] * (height - 1)))))
                hits += int(mask[py, px] > 0)
            if hits / len(points) < float(self.config.get("anchor_min_point_coverage", 0.5)):
                # 多对象并集中心可能离任一真实主体都很远；使用最近的单对象 Mask
                # 框中心判断，避免 Qwen 暂时只返回一个点时误判整个稳定对象组。
                candidate_boxes = list(object_boxes.values()) or [prediction.subject_box]
                nearest = min(
                    math.hypot(
                        (box[0] + box[2]) * 0.5 - point[0] * width,
                        (box[1] + box[3]) * 0.5 - point[1] * height,
                    )
                    for point in points
                    for box in candidate_boxes
                ) / max(1.0, math.hypot(width, height))
                if nearest > float(self.config.get("anchor_max_center_distance_ratio", 0.20)):
                    return False
        grounded_union = _union_box([list(box) for box in selected_boxes])
        if grounded_union is not None:
            grounded_area = max(1.0, (grounded_union[2] - grounded_union[0]) * (grounded_union[3] - grounded_union[1]))
            x1 = max(grounded_union[0], prediction.subject_box[0])
            y1 = max(grounded_union[1], prediction.subject_box[1])
            x2 = min(grounded_union[2], prediction.subject_box[2])
            y2 = min(grounded_union[3], prediction.subject_box[3])
            coverage = max(0.0, x2 - x1) * max(0.0, y2 - y1) / grounded_area
            if coverage < float(self.config.get("grounding_min_coverage", 0.10)):
                return False
        return True

    def _fallback_detections(
        self, frame: np.ndarray, points: list[tuple[float, float]]
    ) -> list[GroundedDetection]:
        if points:
            return [
                GroundedDetection(tuple(select_subject_box(frame, point, self.config)[0]), 0.5, "qwen target")
                for point in points
            ]
        box, confidence, source = select_subject_box(frame, None, self.config)
        return [GroundedDetection(tuple(box), confidence, source)]

    def _resolve_detections(
        self,
        frame: np.ndarray,
        window: AnchorWindow,
        previous_boxes: dict[int, list[float]],
        frame_size: tuple[int, int],
        grounding_phrases: list[str] | None = None,
    ) -> tuple[list[GroundedDetection], list[GroundedDetection], list[ScoredDetection], str, str | None]:
        raw: list[GroundedDetection] = []
        scored: list[ScoredDetection] = []
        error_message: str | None = None
        effective_phrases = window.grounding_phrases if grounding_phrases is None else grounding_phrases
        if self.grounder is not None and effective_phrases:
            try:
                raw = self.grounder.detect(frame, effective_phrases)
                selected, scored = select_detections(
                    raw,
                    window.points,
                    previous_boxes,
                    window.group_mode,
                    frame_size,
                    self.grounding_config,
                )
                if selected:
                    return selected, raw, scored, "grounding_dino", None
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
                if str(self.grounding_config.get("error_policy", "qwen")) == "error":
                    raise
        return self._fallback_detections(frame, window.points), raw, scored, "qwen_fallback", error_message

    @staticmethod
    def _point_assignments(
        points: list[tuple[float, float]],
        assignments: list[tuple[int, GroundedDetection]],
        frame_size: tuple[int, int],
    ) -> dict[int, list[tuple[float, float]]]:
        width, height = frame_size
        output: dict[int, list[tuple[float, float]]] = {object_id: [] for object_id, _ in assignments}
        for point in points:
            px, py = point[0] * width, point[1] * height
            choices: list[tuple[float, int]] = []
            for object_id, detection in assignments:
                box = detection.box_xyxy
                contains = 0.0 if box[0] <= px <= box[2] and box[1] <= py <= box[3] else 1.0
                center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
                choices.append((contains + math.hypot(px - center[0], py - center[1]), object_id))
            if choices:
                output[min(choices)[1]].append(point)
        return output

    def _add_anchor_prompts(
        self,
        state: Any,
        local_frame: int,
        absolute_frame: int,
        assignments: list[tuple[int, GroundedDetection]],
        points: list[tuple[float, float]],
        frame_size: tuple[int, int],
        source_prefix: str,
        primary_object_ids: tuple[int, ...] = (),
        object_focus_offsets: dict[int, tuple[float, float]] | None = None,
        object_importance: dict[int, float] | None = None,
        composition_mode: str = "single_focus",
    ) -> tuple[TrackPoint, np.ndarray, dict[int, list[float]], dict[int, np.ndarray]]:
        object_ids: Any = []
        logits: Any = []
        active_ids = {object_id for object_id, _ in assignments}
        for object_id, detection in assignments:
            _, object_ids, logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_frame,
                obj_id=object_id,
                box=np.asarray(detection.box_xyxy, dtype=np.float32),
            )
        union_mask, object_boxes, object_masks = self._active_masks(
            object_ids, logits, active_ids, frame_size, include_object_masks=True
        )
        ordered_ids = tuple(sorted(active_ids))
        prediction = self._track_point(
            absolute_frame,
            union_mask,
            f"sam2_{source_prefix}_anchor",
            ordered_ids,
            object_masks,
            int(self.config.get("mask_grid_max_side", 160)),
            primary_object_ids,
            object_focus_offsets,
            object_importance,
            composition_mode,
            object_boxes,
        )
        selected_boxes = [detection.box_xyxy for _, detection in assignments]
        if self._anchor_is_valid(
            prediction, union_mask, points, selected_boxes, object_boxes, frame_size
        ):
            assert prediction is not None
            return prediction, union_mask, object_boxes, object_masks

        # 将每个 Qwen 正点追加到离它最近的已分配对象，不清除已有框提示。
        width, height = frame_size
        for object_id, assigned_points in self._point_assignments(points, assignments, frame_size).items():
            if not assigned_points:
                continue
            coordinates = np.asarray(
                [[point[0] * width, point[1] * height] for point in assigned_points], dtype=np.float32
            )
            labels = np.ones((len(assigned_points),), dtype=np.int32)
            _, object_ids, logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_frame,
                obj_id=object_id,
                points=coordinates,
                labels=labels,
                clear_old_points=False,
            )
        union_mask, object_boxes, object_masks = self._active_masks(
            object_ids, logits, active_ids, frame_size, include_object_masks=True
        )
        prediction = self._track_point(
            absolute_frame,
            union_mask,
            f"sam2_{source_prefix}_corrected",
            ordered_ids,
            object_masks,
            int(self.config.get("mask_grid_max_side", 160)),
            primary_object_ids,
            object_focus_offsets,
            object_importance,
            composition_mode,
            object_boxes,
        )
        if self._anchor_is_valid(
            prediction, union_mask, points, selected_boxes, object_boxes, frame_size
        ):
            assert prediction is not None
            return prediction, union_mask, object_boxes, object_masks
        raise RuntimeError(f"SAM2 多目标锚点 Mask 校验失败: frame={absolute_frame}")

    @staticmethod
    def _fallback_point(point: TrackPoint, reason: str = "window_fallback") -> TrackPoint:
        return TrackPoint(
            point.frame,
            list(point.subject_box),
            min(0.5, point.confidence),
            f"sam2_{reason}_{point.source}",
            point.object_count,
            point.object_ids,
            point.mask_evidence,
            point.primary_object_ids,
            point.primary_focus_points,
            point.object_importance,
            point.composition_mode,
            point.recommended_crop_center,
            point.recommended_crop_confidence,
        )

    @staticmethod
    def _detection_dict(detection: GroundedDetection) -> dict[str, Any]:
        return {
            "box_xyxy": [float(value) for value in detection.box_xyxy],
            "score": float(detection.score),
            "phrase": detection.phrase,
        }

    def _track_scene(
        self,
        video_path: Path,
        interval: dict[str, Any],
        frame_size: tuple[int, int],
        span_start: int,
        span_end: int,
        fallback_by_frame: dict[int, TrackPoint],
        visualization_dir: Path | None,
        source_fps: float,
        scene_index: int,
    ) -> list[TrackPoint]:
        with tempfile.TemporaryDirectory(prefix="stage4-sam2-scene-") as directory_name:
            directory = Path(directory_name)
            self._write_scene_frames(video_path, span_start, span_end, directory)
            scene_visualization_dir = None
            if visualization_dir is not None:
                scene_visualization_dir = visualization_dir / str(interval["interval_id"]) / f"scene_{scene_index:03d}_{span_start}_{span_end}"
            visualizer = PropagationVisualizer(
                self.visualization_config, scene_visualization_dir, source_fps, span_start
            )
            state = self.predictor.init_state(
                video_path=str(directory),
                offload_video_to_cpu=bool(self.config.get("offload_video_to_cpu", True)),
                offload_state_to_cpu=bool(self.config.get("offload_state_to_cpu", False)),
                async_loading_frames=bool(self.config.get("async_loading_frames", False)),
            )
            results: dict[int, TrackPoint] = {}
            previous_object_boxes: dict[int, list[float]] = {}
            object_missing_anchors: dict[int, int] = {}
            phrase_last_seen: dict[str, int] = {}
            next_object_id = 1
            primary_object_ids: tuple[int, ...] = ()
            primary_challenger: tuple[int, ...] = ()
            primary_challenger_streak = 0
            object_focus_offsets: dict[int, tuple[float, float]] = {}
            object_importance: dict[int, float] = {}
            composition_mode = "single_focus"

            def apply_fallback(
                window: AnchorWindow,
                only_frame: int | None = None,
                reason: str = "window_fallback",
            ) -> None:
                frame_range = range(only_frame, only_frame + 1) if only_frame is not None else range(window.start, window.end)
                for frame_index in frame_range:
                    prediction = self._fallback_point(fallback_by_frame[frame_index], reason)
                    results[frame_index] = prediction
                    visualizer.save(
                        frame_index,
                        directory / f"{frame_index - span_start:06d}.jpg",
                        prediction,
                        qwen_point=window.point if frame_index == window.start else None,
                        qwen_points=window.points if frame_index == window.start else None,
                        qwen_focus_points=(
                            observation_primary_points(window.observation)
                            if frame_index == window.start and window.observation is not None else None
                        ),
                        is_anchor=frame_index == window.start and window.is_qwen_anchor,
                        is_fallback=True,
                    )

            try:
                with self._contexts():
                    for anchor_index, window in enumerate(anchor_windows(interval, span_start, span_end)):
                        recovery_gate = SAM2RecoveryGate(
                            int(self.config.get("recovery_confirm_frames", 3))
                        )
                        local_start = window.start - span_start
                        frame = cv2.imread(str(directory / f"{local_start:06d}.jpg"))
                        if frame is None:
                            apply_fallback(window)
                            continue
                        grounding_source = "unresolved"
                        carried_ids: list[int] = []
                        expired_ids: list[int] = []
                        effective_phrases = update_grounding_phrase_memory(
                            window.grounding_phrases,
                            phrase_last_seen,
                            anchor_index,
                            int(self.grounding_config.get("phrase_memory_anchors", 3)),
                            int(self.grounding_config.get("max_grounding_phrases", 12)),
                        )
                        try:
                            selected, raw, scored, grounding_source, grounding_error = self._resolve_detections(
                                frame,
                                window,
                                previous_object_boxes,
                                frame_size,
                                effective_phrases,
                            )
                            assignments, next_object_id, next_missing_anchors, carried_ids, expired_ids = stabilize_object_assignments(
                                selected,
                                previous_object_boxes,
                                object_missing_anchors,
                                next_object_id,
                                frame_size,
                                self.grounding_config,
                            )
                            active_ids = {object_id for object_id, _ in assignments}
                            target_mappings = associate_targets_to_objects(
                                window.observation, assignments, frame_size
                            )
                            proposed_primary = tuple(sorted(
                                object_id for object_id, metadata in target_mappings.items()
                                if metadata["is_primary"]
                            ))
                            if not proposed_primary and not primary_object_ids and target_mappings:
                                proposed_primary = (max(
                                    target_mappings,
                                    key=lambda object_id: (
                                        target_mappings[object_id]["importance"],
                                        target_mappings[object_id]["confidence"],
                                    ),
                                ),)
                            next_primary, next_challenger, next_streak, primary_switched = update_primary_object_state(
                                primary_object_ids,
                                primary_challenger,
                                primary_challenger_streak,
                                proposed_primary,
                                active_ids,
                                int(self.config.get("primary_switch_confirm_anchors", 2)),
                            )
                            next_focus_offsets = dict(object_focus_offsets)
                            next_importance = dict(object_importance)
                            frame_width, frame_height = frame_size
                            assignment_boxes = {object_id: detection.box_xyxy for object_id, detection in assignments}
                            for object_id, metadata in target_mappings.items():
                                box = assignment_boxes[object_id]
                                focus_x = metadata["focus_point"][0] * frame_width
                                focus_y = metadata["focus_point"][1] * frame_height
                                next_focus_offsets[object_id] = (
                                    max(0.0, min(1.0, (focus_x - box[0]) / max(1.0, box[2] - box[0]))),
                                    max(0.0, min(1.0, (focus_y - box[1]) / max(1.0, box[3] - box[1]))),
                                )
                                next_importance[object_id] = float(metadata["importance"])
                            next_composition_mode = "group_focus" if len(next_primary) > 1 else "single_focus"
                            anchor_prediction, anchor_mask, anchor_object_boxes, anchor_object_masks = self._add_anchor_prompts(
                                state,
                                local_start,
                                window.start,
                                assignments,
                                window.points,
                                frame_size,
                                grounding_source,
                                next_primary,
                                next_focus_offsets,
                                next_importance,
                                next_composition_mode,
                            )
                            # 只有超过 keepalive 的对象才过期；其余对象按 ID 增量更新，
                            # 避免一次 Qwen/Grounding 漏检覆盖掉整个历史对象集合。
                            object_missing_anchors = next_missing_anchors
                            for object_id in expired_ids:
                                previous_object_boxes.pop(object_id, None)
                                next_focus_offsets.pop(object_id, None)
                                next_importance.pop(object_id, None)
                            for object_id in list(previous_object_boxes):
                                if object_id not in active_ids:
                                    previous_object_boxes.pop(object_id, None)
                            for object_id, detection in assignments:
                                previous_object_boxes.setdefault(
                                    object_id, [float(value) for value in detection.box_xyxy]
                                )
                            previous_object_boxes.update(anchor_object_boxes)
                            primary_object_ids = next_primary
                            primary_challenger = next_challenger
                            primary_challenger_streak = next_streak
                            object_focus_offsets = next_focus_offsets
                            object_importance = next_importance
                            composition_mode = next_composition_mode
                            self.last_grounding_records.append({
                                "schema_version": STAGE4_SCHEMA_VERSION,
                                "video_id": interval["video_id"],
                                "interval_id": interval["interval_id"],
                                "scene_index": scene_index,
                                "frame": window.start,
                                "window_end": window.end,
                                "is_qwen_anchor": window.is_qwen_anchor,
                                "group_mode": window.group_mode,
                                "qwen_targets": [] if window.observation is None else list(window.observation.get("targets", [])),
                                "qwen_points": [list(point) for point in window.points],
                                "grounding_phrases": window.grounding_phrases,
                                "effective_grounding_phrases": effective_phrases,
                                "raw_detections": [self._detection_dict(row) for row in raw],
                                "scored_detections": [{
                                    **self._detection_dict(row.detection),
                                    "total_score": row.total_score,
                                    "point_score": row.point_score,
                                    "temporal_score": row.temporal_score,
                                } for row in scored],
                                "selected_objects": [{
                                    "object_id": object_id,
                                    "is_temporal_carry": object_id in carried_ids,
                                    "missing_anchor_count": object_missing_anchors.get(object_id, 0),
                                    **self._detection_dict(detection),
                                } for object_id, detection in assignments],
                                "primary_object_ids": list(primary_object_ids),
                                "primary_switched": primary_switched,
                                "primary_challenger_ids": list(primary_challenger),
                                "primary_challenger_streak": primary_challenger_streak,
                                "composition_mode": composition_mode,
                                "target_object_mappings": [
                                    {"object_id": object_id, **metadata}
                                    for object_id, metadata in sorted(target_mappings.items())
                                ],
                                "carried_object_ids": carried_ids,
                                "expired_object_ids": expired_ids,
                                "source": grounding_source,
                                "error": grounding_error,
                                "status": "accepted",
                            })
                            results[window.start] = anchor_prediction
                            visualizer.save(
                                window.start,
                                directory / f"{local_start:06d}.jpg",
                                anchor_prediction,
                                mask=anchor_mask,
                                qwen_point=window.point,
                                qwen_points=window.points,
                                qwen_focus_points=(
                                    observation_primary_points(window.observation)
                                    if window.observation is not None else None
                                ),
                                prompt_box=_union_box([list(detection.box_xyxy) for _, detection in assignments]),
                                grounding_objects=[*({
                                    "kind": "raw",
                                    "box_xyxy": list(detection.box_xyxy),
                                    "score": detection.score,
                                    "phrase": detection.phrase,
                                } for detection in raw), *({
                                    "kind": "selected",
                                    "object_id": object_id,
                                    "box_xyxy": list(detection.box_xyxy),
                                    "score": detection.score,
                                    "phrase": detection.phrase,
                                } for object_id, detection in assignments)],
                                object_masks=anchor_object_masks,
                                is_anchor=window.is_qwen_anchor,
                                is_fallback=False,
                            )
                            for local_frame, object_ids, logits in self.predictor.propagate_in_video(
                                state,
                                start_frame_idx=local_start,
                                max_frame_num_to_track=max(0, window.end - window.start - 1),
                                reverse=False,
                            ):
                                absolute_frame = span_start + int(local_frame)
                                if not window.start <= absolute_frame < window.end:
                                    continue
                                union_mask, object_boxes, object_masks = self._active_masks(
                                    object_ids,
                                    logits,
                                    active_ids,
                                    frame_size,
                                    include_object_masks=True,
                                )
                                prediction = self._track_point(
                                    absolute_frame,
                                    union_mask,
                                    f"sam2_{grounding_source}_anchor" if absolute_frame == window.start else f"sam2_{grounding_source}_propagated",
                                    tuple(sorted(active_ids)),
                                    object_masks,
                                    int(self.config.get("mask_grid_max_side", 160)),
                                    primary_object_ids,
                                    object_focus_offsets,
                                    object_importance,
                                    composition_mode,
                                    object_boxes,
                                )
                                if prediction is None:
                                    recovery_gate.observe(False)
                                    continue
                                if not recovery_gate.observe(True):
                                    apply_fallback(
                                        window,
                                        only_frame=absolute_frame,
                                        reason="recovery_wait",
                                    )
                                    continue
                                results[absolute_frame] = prediction
                                previous_object_boxes.update(object_boxes)
                                visualizer.save(
                                    absolute_frame,
                                    directory / f"{int(local_frame):06d}.jpg",
                                    prediction,
                                    mask=union_mask,
                                    qwen_point=window.point if absolute_frame == window.start else None,
                                    qwen_points=window.points if absolute_frame == window.start else None,
                                    qwen_focus_points=(
                                        observation_primary_points(window.observation)
                                        if absolute_frame == window.start and window.observation is not None else None
                                    ),
                                    grounding_objects=[*({
                                        "kind": "raw",
                                        "box_xyxy": list(detection.box_xyxy),
                                        "score": detection.score,
                                        "phrase": detection.phrase,
                                    } for detection in raw), *({
                                        "kind": "selected",
                                        "object_id": object_id,
                                        "box_xyxy": list(detection.box_xyxy),
                                        "score": detection.score,
                                        "phrase": detection.phrase,
                                    } for object_id, detection in assignments)] if absolute_frame == window.start else None,
                                    object_masks=object_masks,
                                    is_anchor=absolute_frame == window.start and window.is_qwen_anchor,
                                    is_fallback=False,
                                )
                        except VisualizationError:
                            raise
                        except Exception as error:
                            self.last_grounding_records.append({
                                "schema_version": STAGE4_SCHEMA_VERSION,
                                "video_id": interval["video_id"],
                                "interval_id": interval["interval_id"],
                                "scene_index": scene_index,
                                "frame": window.start,
                                "window_end": window.end,
                                "qwen_targets": [] if window.observation is None else list(window.observation.get("targets", [])),
                                "qwen_points": [list(point) for point in window.points],
                                "grounding_phrases": window.grounding_phrases,
                                "effective_grounding_phrases": effective_phrases,
                                "carried_object_ids": carried_ids,
                                "expired_object_ids": expired_ids,
                                "source": grounding_source,
                                "status": "window_fallback",
                                "error": f"{type(error).__name__}: {error}",
                            })
                            apply_fallback(window)
                            continue
                        for frame_index in range(window.start, window.end):
                            if frame_index not in results:
                                apply_fallback(window, only_frame=frame_index)
            finally:
                if hasattr(self.predictor, "reset_state"):
                    self.predictor.reset_state(state)
                visualizer.close()
            return [results[frame] for frame in range(span_start, span_end)]

    def track(
        self,
        video_path: Path,
        interval: dict[str, Any],
        frame_size: tuple[int, int],
        scenes: list[dict[str, Any]],
        visualization_dir: Path | None = None,
    ) -> list[TrackPoint]:
        if not video_path.is_file():
            raise ArtifactValidationError(f"源视频不存在: {video_path}")
        self.last_grounding_records = []
        fallback = CenterSubjectTracker(self.config).track(Path(), interval, frame_size, scenes)
        fallback_by_frame = {point.frame: point for point in fallback}
        capture = cv2.VideoCapture(str(video_path))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        capture.release()
        output: list[TrackPoint] = []
        for scene_index, (span_start, span_end) in enumerate(interval_scene_spans(interval, scenes)):
            output.extend(
                self._track_scene(
                    video_path,
                    interval,
                    frame_size,
                    span_start,
                    span_end,
                    fallback_by_frame,
                    visualization_dir,
                    source_fps,
                    scene_index,
                )
            )
        expected = list(range(int(interval["start_frame"]), int(interval["end_frame"])))
        if [point.frame for point in output] != expected:
            raise RuntimeError("SAM2 镜头窗口没有完整覆盖高光区间")
        return output
