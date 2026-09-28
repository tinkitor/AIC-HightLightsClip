"""解析 Qwen 的单帧精简响应，并展开为 Stage 3.5 持久化结构。"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError

from .frame_sampler import SampledFrame


def _join_error_prediction(
    errors: list[dict[str, Any]], item: dict[str, Any], reason: str
) -> None:
    errors.append({"error_reason": reason, **item})


def _normalize_axis(value: float, size: int) -> float:
    """兼容旧响应中的归一化、千分制和像素坐标。"""

    if value <= 1.5:
        normalized = value
    elif value <= 1000.0:
        normalized = value / 1000.0
    elif size > 0:
        normalized = value / float(size)
    else:
        normalized = value / 1000.0
    return max(0.0, min(1.0, normalized))


def _normalize_point(value: Any, frame: SampledFrame) -> list[float] | None:
    """解析旧版宽松坐标格式。"""

    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("坐标点非 [x,y]")
    point = [float(value[0]), float(value[1])]
    if not all(math.isfinite(axis) and axis >= 0.0 for axis in point):
        raise ValueError(f"坐标点含非法坐标: {point}")
    return [
        _normalize_axis(point[0], frame.width),
        _normalize_axis(point[1], frame.height),
    ]


def _normalize_millipoint(value: Any) -> list[float] | None:
    """严格按精简 schema 的 0..1000 坐标解析，避免把整数 1 误当归一化 1.0。"""

    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("坐标点非 [x,y]")
    point = [float(value[0]), float(value[1])]
    if not all(math.isfinite(axis) and 0.0 <= axis <= 1000.0 for axis in point):
        raise ValueError(f"千分制坐标超出 0..1000: {point}")
    return [point[0] / 1000.0, point[1] / 1000.0]


def _phrases(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for raw in value:
        phrase = str(raw).strip().lower().rstrip(". ")
        if phrase and phrase not in seen:
            output.append(phrase)
            seen.add(phrase)
    return output[:8]


def _target_id_from_phrase(phrase: str, position: int) -> str:
    target_id = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return target_id[:32] or f"target_{position + 1}"


def _legacy_targets(item: dict[str, Any]) -> list[dict[str, Any]]:
    """把旧版单点响应适配为多目标结构，便于读取旧产物和回归测试。"""

    if "subject_point" not in item:
        return []
    return [
        {
            "target_id": "primary",
            "description": "primary subject",
            "grounding_phrase": "main subject",
            "subject_point": item.get("subject_point"),
            "confidence": item.get("confidence", 0.0),
            "visibility": item.get("visibility", "not_found"),
        }
    ]


def _parse_targets(
    item: dict[str, Any],
    frame: SampledFrame,
    errors: list[dict[str, Any]],
    index: int,
    lean_item: bool,
) -> list[dict[str, Any]]:
    raw_targets = item.get("targets")
    if raw_targets is None:
        raw_targets = _legacy_targets(item)
    if not isinstance(raw_targets, list):
        _join_error_prediction(errors, item, f"sample_index={index} 的 targets 非数组")
        return []

    targets: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    limit = 4 if lean_item else 12
    for position, raw in enumerate(raw_targets[:limit]):
        if not isinstance(raw, dict):
            _join_error_prediction(errors, item, f"sample_index={index} 的 target[{position}] 非对象")
            continue
        phrase = str(raw.get("grounding_phrase", "")).strip().lower().rstrip(". ")
        if not phrase:
            phrase = "main subject"
        focus_phrase = str(raw.get("focus_phrase") or phrase).strip().lower().rstrip(". ")
        if not focus_phrase:
            focus_phrase = phrase
        base_id = (
            _target_id_from_phrase(phrase, position)
            if lean_item
            else str(raw.get("target_id") or f"target_{position}").strip()[:40]
        ) or f"target_{position}"
        target_id = base_id
        suffix = 2
        while target_id in used_ids:
            target_id = f"{base_id[:36]}_{suffix}"
            suffix += 1
        used_ids.add(target_id)

        try:
            point = (
                _normalize_millipoint(raw.get("point"))
                if lean_item
                else _normalize_point(raw.get("subject_point"), frame)
            )
        except (TypeError, ValueError) as error:
            _join_error_prediction(errors, raw, f"sample_index={index}/{target_id}: {error}")
            point = None
        if lean_item:
            focus_point = point
            role = "primary" if raw.get("primary") is True else "supporting"
            confidence = 0.90 if role == "primary" else 0.80
            importance = 1.0 if role == "primary" else 0.5
            visibility = "visible" if point is not None else "not_found"
            description = phrase
        else:
            try:
                focus_point = _normalize_point(
                    raw.get("focus_point", raw.get("subject_point")), frame
                )
            except (TypeError, ValueError) as error:
                _join_error_prediction(
                    errors, raw, f"sample_index={index}/{target_id} focus_point: {error}"
                )
                focus_point = point
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                confidence = 0.0
            visibility = str(raw.get("visibility", "not_found"))
            if visibility not in {"visible", "occluded", "not_found"}:
                visibility = "not_found"
            if visibility == "visible" and point is None:
                visibility = "not_found"
            role = str(raw.get("role", "supporting")).strip().lower()
            if role not in {"primary", "supporting"}:
                role = "supporting"
            try:
                importance = float(
                    raw.get("importance", 1.0 if role == "primary" else 0.5)
                )
            except (TypeError, ValueError):
                importance = 1.0 if role == "primary" else 0.5
            if not math.isfinite(importance):
                importance = 1.0 if role == "primary" else 0.5
            importance = max(0.0, min(1.0, importance))
            description = str(raw.get("description", "")).strip()[:80]

        targets.append(
            {
                "target_id": target_id,
                "description": description,
                "grounding_phrase": phrase[:80],
                "focus_phrase": focus_phrase[:80],
                "subject_point": point,
                "focus_point": focus_point,
                "role": role,
                "importance": importance,
                "confidence": confidence,
                "visibility": visibility,
            }
        )
    return targets


def _legal_crop_center(
    anchor: list[float] | None,
    metadata: dict[str, Any] | None,
    frame: SampledFrame,
) -> list[float] | None:
    """把模型给出的理想中心限制在最大目标比例裁剪框的合法中心范围内。"""

    if anchor is None:
        return None
    metadata = metadata or {}
    try:
        source_width = float(metadata.get("display_width", metadata.get("width", frame.width)))
        source_height = float(metadata.get("display_height", metadata.get("height", frame.height)))
    except (TypeError, ValueError):
        source_width, source_height = float(frame.width), float(frame.height)
    if source_width <= 0 or source_height <= 0:
        source_width, source_height = float(frame.width), float(frame.height)

    ratio = metadata.get("targetRatioWH", [16, 9])
    try:
        target_width, target_height = float(ratio[0]), float(ratio[1])
    except (TypeError, ValueError, IndexError):
        target_width, target_height = 16.0, 9.0
    if target_width <= 0 or target_height <= 0:
        target_width, target_height = 16.0, 9.0

    if source_width / source_height >= target_width / target_height:
        crop_height = source_height
        crop_width = crop_height * target_width / target_height
    else:
        crop_width = source_width
        crop_height = crop_width * target_height / target_width
    half_width = min(0.5, crop_width / source_width / 2.0)
    half_height = min(0.5, crop_height / source_height / 2.0)
    return [
        max(half_width, min(1.0 - half_width, float(anchor[0]))),
        max(half_height, min(1.0 - half_height, float(anchor[1]))),
    ]


def _empty_prediction(reason: str) -> dict[str, Any]:
    return {
        "group_mode": "multiple",
        "composition_mode": "single_focus",
        "recommended_crop_center": None,
        "recommended_crop_confidence": 0.0,
        "primary_target_ids": [],
        "grounding_phrases": [],
        "targets": [],
        "reason": reason,
    }


def parse_predictions(
    text: str,
    frames: list[SampledFrame],
    use_batch: bool = False,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    """解析精简单帧响应，同时保留旧 predictions 包装格式的兼容读取。"""

    if not frames:
        return [], [], []
    try:
        root = json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(f"Stage 3.5 响应不是合法 JSON: {error}") from error
    if not isinstance(root, dict):
        raise ArtifactValidationError("Stage 3.5 响应根节点必须是对象")

    lean_root = "predictions" not in root
    if lean_root:
        if use_batch or len(frames) != 1:
            raise ArtifactValidationError("精简 Stage 3.5 响应只支持单帧解析")
        predictions: list[Any] = [{**root, "sample_index": frames[0].sample_index}]
    else:
        predictions = root.get("predictions")
        if not isinstance(predictions, list):
            raise ArtifactValidationError("Stage 3.5 响应缺少 predictions 数组")

    by_sample = {frame.sample_index: frame for frame in frames}
    ordered_indices = [frame.sample_index for frame in frames]
    errors: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ArtifactValidationError("prediction 必须是对象")
        index = int(item.get("sample_index", -1))
        if index not in by_sample:
            _join_error_prediction(errors, item, f"未知 sample_index: {index}")
            continue
        if index in by_index:
            _join_error_prediction(errors, item, f"重复 sample_index: {index}")
            continue

        lean_item = lean_root or "crop_anchor" in item
        targets = _parse_targets(item, by_sample[index], errors, index, lean_item)
        target_ids = {str(target["target_id"]) for target in targets}
        if lean_item:
            primary_ids = [
                str(target["target_id"])
                for target in targets
                if target["role"] == "primary" and target["subject_point"] is not None
            ]
        else:
            raw_primary_ids = item.get("primary_target_ids", [])
            primary_ids: list[str] = []
            if isinstance(raw_primary_ids, list):
                for value in raw_primary_ids:
                    target_id = str(value).strip()
                    if target_id in target_ids and target_id not in primary_ids:
                        primary_ids.append(target_id)
            for target in targets:
                if target["role"] == "primary" and target["target_id"] not in primary_ids:
                    primary_ids.append(target["target_id"])
            if not primary_ids:
                visible = [target for target in targets if target["subject_point"] is not None]
                if visible:
                    primary_ids = [
                        max(
                            visible,
                            key=lambda target: (
                                float(target["importance"]),
                                float(target["confidence"]),
                            ),
                        )["target_id"]
                    ]

        if lean_item and not primary_ids:
            targets = []
            target_ids = set()
        primary_set = set(primary_ids)
        for target in targets:
            if target["target_id"] in primary_set:
                target["role"] = "primary"
                target["importance"] = max(0.75, float(target["importance"]))

        phrases = [] if lean_item else _phrases(item.get("grounding_phrases"))
        for target in targets:
            phrase = target["grounding_phrase"]
            if phrase not in phrases:
                phrases.append(phrase)
        group_mode = "multiple" if len(targets) > 1 else "single"
        composition_mode = "group_focus" if len(primary_ids) > 1 else "single_focus"

        if lean_item:
            try:
                raw_anchor = _normalize_millipoint(item.get("crop_anchor"))
            except (TypeError, ValueError) as error:
                _join_error_prediction(
                    errors, item, f"sample_index={index} crop_anchor: {error}"
                )
                raw_anchor = None
            recommended_crop_center = (
                _legal_crop_center(raw_anchor, metadata, by_sample[index])
                if primary_ids
                else None
            )
            if recommended_crop_center is None:
                recommended_crop_confidence = 0.0
            else:
                recommended_crop_confidence = 0.85 if len(primary_ids) > 1 else 0.90
        else:
            try:
                recommended_crop_center = _normalize_point(
                    item.get("recommended_crop_center"), by_sample[index]
                )
            except (TypeError, ValueError) as error:
                _join_error_prediction(
                    errors, item, f"sample_index={index} recommended_crop_center: {error}"
                )
                recommended_crop_center = None
            try:
                recommended_crop_confidence = float(
                    item.get("recommended_crop_confidence", 0.0)
                )
            except (TypeError, ValueError):
                recommended_crop_confidence = 0.0
            if not math.isfinite(recommended_crop_confidence):
                recommended_crop_confidence = 0.0
            recommended_crop_confidence = max(
                0.0, min(1.0, recommended_crop_confidence)
            )
            if recommended_crop_center is None:
                recommended_crop_confidence = 0.0

        by_index[index] = {
            "group_mode": group_mode,
            "composition_mode": composition_mode,
            "recommended_crop_center": recommended_crop_center,
            "recommended_crop_confidence": recommended_crop_confidence,
            "primary_target_ids": primary_ids[:4],
            "grounding_phrases": phrases[:8],
            "targets": targets,
            "reason": str(item.get("reason", ""))[:160],
        }
        if not use_batch:
            break

    missing = [index for index in ordered_indices if index not in by_index]
    for index in missing:
        by_index[index] = _empty_prediction("model_omitted_sample")
    return [by_index[index] for index in ordered_indices], missing, errors
