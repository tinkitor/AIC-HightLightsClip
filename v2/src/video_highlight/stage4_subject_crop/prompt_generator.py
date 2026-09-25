"""读取 Stage 3.5 多主体与构图观察，并兼容旧版单主体点。"""

from __future__ import annotations

from typing import Any


def _valid_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return point if 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0 else None


def subject_observations(interval: dict[str, Any]) -> list[dict[str, Any]]:
    """返回按帧排序的 v2 观察；内存旧数据会即时适配但不修改调用方对象。"""

    rows = interval.get("subject_observations")
    if isinstance(rows, list):
        return sorted((row for row in rows if isinstance(row, dict)), key=lambda row: int(row["frame"]))
    legacy = interval.get("subject_points", [])
    output: list[dict[str, Any]] = []
    for row in legacy if isinstance(legacy, list) else []:
        if not isinstance(row, dict):
            continue
        point = _valid_point(row.get("subject_point"))
        targets = [] if point is None else [{
            "target_id": "primary",
            "description": str(interval.get("subject") or "primary subject"),
            "grounding_phrase": "main subject",
            "subject_point": list(point),
            "focus_point": list(point),
            "role": "primary",
            "importance": 1.0,
            "confidence": float(row.get("confidence", 0.0)),
            "visibility": str(row.get("visibility", "visible")),
        }]
        output.append({
            **row,
            "group_mode": "single",
            "composition_mode": "single_focus",
            "primary_target_ids": ["primary"] if targets else [],
            "grounding_phrases": [],
            "targets": targets,
        })
    return sorted(output, key=lambda row: int(row["frame"]))


def observation_points(row: dict[str, Any]) -> list[tuple[float, float]]:
    """返回一条观察里所有合法可见目标点。"""

    points: list[tuple[float, float]] = []
    for target in row.get("targets", []):
        if not isinstance(target, dict):
            continue
        point = _valid_point(target.get("subject_point"))
        if point is not None:
            points.append(point)
    return points


def observation_recommended_crop_center(row: dict[str, Any]) -> tuple[float, float] | None:
    """读取 Stage 3.5 v3 的帧级 Qwen 推荐构图中心。"""

    return _valid_point(row.get("recommended_crop_center"))


def observation_recommended_crop_confidence(row: dict[str, Any]) -> float:
    """读取并限制 Qwen 推荐构图置信度；旧产物默认没有推荐。"""

    try:
        confidence = float(row.get("recommended_crop_confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def observation_primary_points(row: dict[str, Any], focus: bool = True) -> list[tuple[float, float]]:
    """返回主主体点；旧产物没有主次字段时兼容回退到全部主体点。"""

    primary_ids = {str(value) for value in row.get("primary_target_ids", [])}
    points: list[tuple[float, float]] = []
    for target in row.get("targets", []):
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id", ""))
        if target_id not in primary_ids and target.get("role") != "primary":
            continue
        point = _valid_point(target.get("focus_point" if focus else "subject_point"))
        if point is None and focus:
            point = _valid_point(target.get("subject_point"))
        if point is not None:
            points.append(point)
    return points or observation_points(row)


def subject_observation(interval: dict[str, Any], frame: int | None = None) -> dict[str, Any] | None:
    rows = subject_observations(interval)
    if not rows:
        return None
    target = int(interval["start_frame"]) if frame is None else int(frame)
    return min(rows, key=lambda row: abs(int(row["frame"]) - target))


def subject_points(interval: dict[str, Any], frame: int | None = None) -> list[tuple[float, float]]:
    row = subject_observation(interval, frame)
    return [] if row is None else observation_points(row)


def subject_point(interval: dict[str, Any], frame: int | None = None) -> tuple[float, float] | None:
    """兼容单点跟踪器：多目标观察返回所有点的包围中心。"""

    points = subject_points(interval, frame)
    if not points:
        return None
    return (
        (min(point[0] for point in points) + max(point[0] for point in points)) * 0.5,
        (min(point[1] for point in points) + max(point[1] for point in points)) * 0.5,
    )


def subject_point_frames(interval: dict[str, Any]) -> set[int]:
    """返回至少包含一个合法目标点的采样帧号。"""

    return {
        int(row["frame"])
        for row in subject_observations(interval)
        if observation_points(row)
    }
