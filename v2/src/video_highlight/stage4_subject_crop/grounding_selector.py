"""Grounding DINO 多框的空间筛选、去重和跨锚点对象 ID 关联。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class GroundedDetection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    phrase: str


@dataclass(frozen=True, slots=True)
class ScoredDetection:
    detection: GroundedDetection
    total_score: float
    point_score: float
    temporal_score: float


def box_iou(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(1e-9, left_area + right_area - intersection)


def _center_score(
    left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float],
    frame_size: tuple[int, int]
) -> float:
    left_center = ((float(left[0]) + float(left[2])) * 0.5, (float(left[1]) + float(left[3])) * 0.5)
    right_center = ((float(right[0]) + float(right[2])) * 0.5, (float(right[1]) + float(right[3])) * 0.5)
    diagonal = max(1.0, math.hypot(*frame_size))
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) / diagonal
    return math.exp(-distance / 0.12)


def _association_score(
    detection_box: tuple[float, ...] | list[float],
    previous_box: tuple[float, ...] | list[float],
    frame_size: tuple[int, int],
) -> float:
    """检测框与历史对象框的统一时序关联分。"""

    return (
        0.65 * box_iou(detection_box, previous_box)
        + 0.35 * _center_score(detection_box, previous_box, frame_size)
    )


def _point_score(
    box: tuple[float, ...], points: list[tuple[float, float]], frame_size: tuple[int, int]
) -> float:
    if not points:
        return 0.5
    width, height = frame_size
    box_center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    diagonal = max(1.0, math.hypot(width, height))
    scores: list[float] = []
    for point in points:
        px, py = point[0] * width, point[1] * height
        if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
            scores.append(1.0)
        else:
            distance = math.hypot(px - box_center[0], py - box_center[1]) / diagonal
            scores.append(math.exp(-distance / 0.10))
    return max(scores)


def _deduplicate(detections: list[GroundedDetection], threshold: float) -> list[GroundedDetection]:
    output: list[GroundedDetection] = []
    for detection in sorted(detections, key=lambda row: row.score, reverse=True):
        if any(box_iou(detection.box_xyxy, kept.box_xyxy) >= threshold for kept in output):
            continue
        output.append(detection)
    return output


def score_detections(
    detections: list[GroundedDetection],
    qwen_points: list[tuple[float, float]],
    previous_boxes: dict[int, list[float]],
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> list[ScoredDetection]:
    detection_weight = float(config.get("detection_weight", 0.40))
    point_weight = float(config.get("point_weight", 0.35))
    temporal_weight = float(config.get("temporal_weight", 0.25))
    rows: list[ScoredDetection] = []
    for detection in _deduplicate(detections, float(config.get("nms_iou", 0.85))):
        point = _point_score(detection.box_xyxy, qwen_points, frame_size)
        temporal = max(
            (
                _association_score(detection.box_xyxy, previous, frame_size)
                for previous in previous_boxes.values()
            ),
            default=0.5,
        )
        total = detection_weight * detection.score + point_weight * point + temporal_weight * temporal
        rows.append(ScoredDetection(detection, total, point, temporal))
    return sorted(rows, key=lambda row: row.total_score, reverse=True)


def select_detections(
    detections: list[GroundedDetection],
    qwen_points: list[tuple[float, float]],
    previous_boxes: dict[int, list[float]],
    group_mode: str,
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[list[GroundedDetection], list[ScoredDetection]]:
    """选择单个或多个主体框；multiple 模式优先让不同 Qwen 点认领不同框。"""

    scored = score_detections(detections, qwen_points, previous_boxes, frame_size, config)
    threshold = float(config.get("selection_threshold", 0.45))
    maximum = max(1, int(config.get("max_objects", 8)))
    eligible = [row for row in scored if row.total_score >= threshold]
    if not eligible and not previous_boxes:
        return [], scored
    selected: list[GroundedDetection] = []
    used: set[int] = set()
    width, height = frame_size
    if group_mode != "multiple" and eligible:
        selected.append(eligible[0].detection)
        used.add(scored.index(eligible[0]))
    else:
        # 每个 Qwen 点优先认领一个包含它或离它最近的独立检测框。
        for point in qwen_points:
            px, py = point[0] * width, point[1] * height
            choices: list[tuple[float, int, ScoredDetection]] = []
            for index, row in enumerate(scored):
                if index in used or row.total_score < threshold:
                    continue
                box = row.detection.box_xyxy
                individual_point_score = _point_score(box, [point], frame_size)
                if individual_point_score < float(config.get("claim_min_point_score", 0.20)):
                    continue
                contains = 1.0 if box[0] <= px <= box[2] and box[1] <= py <= box[3] else 0.0
                center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
                distance = math.hypot(px - center[0], py - center[1]) / max(1.0, math.hypot(width, height))
                choices.append((contains * 2.0 + row.total_score - distance, index, row))
            if choices:
                _, index, row = max(choices, key=lambda value: value[0])
                used.add(index)
                selected.append(row.detection)

    # 当前 Qwen 即使从 multiple 瞬时退化成 single，仍保留与历史对象空间匹配的
    # Grounding 框。Qwen 是有噪声的观察，不是删除历史主体的指令。
    retention_association = float(
        config.get("retention_association_threshold", config.get("association_threshold", 0.30))
    )
    retention_detection = float(config.get("retention_detection_threshold", 0.25))
    for previous_box in previous_boxes.values():
        if len(selected) >= maximum:
            break
        if any(
            _association_score(detection.box_xyxy, previous_box, frame_size)
            >= retention_association
            for detection in selected
        ):
            continue
        choices = [
            (
                _association_score(row.detection.box_xyxy, previous_box, frame_size),
                index,
                row,
            )
            for index, row in enumerate(scored)
            if index not in used and row.detection.score >= retention_detection
        ]
        if choices and max(choices, key=lambda value: value[0])[0] >= retention_association:
            _, index, row = max(choices, key=lambda value: value[0])
            used.add(index)
            selected.append(row.detection)

    if group_mode == "multiple":
        # Qwen 可能漏掉群体中的成员；允许加入语义得分高且空间关系合理的额外框。
        minimum_point = float(config.get("multiple_min_point_score", 0.35))
        for index, row in enumerate(scored):
            if len(selected) >= maximum:
                break
            if index in used or row.total_score < threshold:
                continue
            if not qwen_points or row.point_score >= minimum_point:
                selected.append(row.detection)
                used.add(index)
    return selected[:maximum], scored


def associate_object_ids(
    detections: list[GroundedDetection],
    previous_boxes: dict[int, list[float]],
    next_object_id: int,
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[list[tuple[int, GroundedDetection]], int]:
    """将本锚点多框贪心匹配到上一窗口对象，未匹配框分配新 ID。"""

    assigned: list[tuple[int, GroundedDetection]] = []
    available = set(previous_boxes)
    threshold = float(config.get("association_threshold", 0.30))
    for detection in detections:
        matches = [
            (
                _association_score(detection.box_xyxy, previous_boxes[obj_id], frame_size),
                obj_id,
            )
            for obj_id in available
        ]
        if matches and max(matches)[0] >= threshold:
            _, object_id = max(matches)
            available.remove(object_id)
        else:
            object_id = next_object_id
            next_object_id += 1
        assigned.append((object_id, detection))
    return assigned, next_object_id


def stabilize_object_assignments(
    detections: list[GroundedDetection],
    previous_boxes: dict[int, list[float]],
    previous_missing_anchors: dict[int, int],
    next_object_id: int,
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[
    list[tuple[int, GroundedDetection]],
    int,
    dict[int, int],
    list[int],
    list[int],
]:
    """关联当前检测，并让短暂漏检的历史对象以原 ID/末框继续存活。

    ``missing_anchors`` 只表示连续多少个锚点没有获得新的检测框；carry 对象仍会
    作为当前窗口的活跃 SAM2 对象。超过 ``object_keepalive_anchors`` 后才过期。
    """

    assigned, next_object_id = associate_object_ids(
        detections, previous_boxes, next_object_id, frame_size, config
    )
    matched_ids = {object_id for object_id, _ in assigned}
    missing_anchors = {object_id: 0 for object_id in matched_ids}
    carried_ids: list[int] = []
    expired_ids: list[int] = []
    keepalive = max(0, int(config.get("object_keepalive_anchors", 2)))
    maximum = max(1, int(config.get("max_objects", 8)))
    decay = float(config.get("carry_score_decay", 0.75))

    candidates: list[tuple[int, int, list[float]]] = []
    for object_id, box in previous_boxes.items():
        if object_id in matched_ids:
            continue
        missed = int(previous_missing_anchors.get(object_id, 0)) + 1
        if missed <= keepalive:
            candidates.append((missed, object_id, box))
        else:
            expired_ids.append(object_id)

    # 优先保留缺失次数更少的对象；达到 max_objects 后其余对象显式过期。
    for missed, object_id, box in sorted(candidates):
        if len(assigned) >= maximum:
            expired_ids.append(object_id)
            continue
        score = 0.5 * max(0.0, min(1.0, decay)) ** missed
        assigned.append((
            object_id,
            GroundedDetection(tuple(float(value) for value in box), score, "temporal carry"),
        ))
        missing_anchors[object_id] = missed
        carried_ids.append(object_id)

    return assigned, next_object_id, missing_anchors, carried_ids, expired_ids
