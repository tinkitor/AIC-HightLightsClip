"""解析 Qwen 的逐帧多主体观察并补齐遗漏采样帧。"""

from __future__ import annotations

import json
import math
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError

from .frame_sampler import SampledFrame


def _join_error_prediction(
    errors: list[dict[str, Any]], item: dict[str, Any], reason: str
) -> None:
    errors.append({"error_reason": reason, **item})


def _normalize_axis(value: float, size: int) -> float:
    """兼容归一化、千分制和像素坐标，最终限制到 ``[0, 1]``。"""

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


def _legacy_targets(item: dict[str, Any]) -> list[dict[str, Any]]:
    """把旧版单点响应适配为 v2 单目标结构，便于滚动升级模型端。"""

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
    item: dict[str, Any], frame: SampledFrame, errors: list[dict[str, Any]], index: int
) -> list[dict[str, Any]]:
    raw_targets = item.get("targets")
    if raw_targets is None:
        raw_targets = _legacy_targets(item)
    if not isinstance(raw_targets, list):
        _join_error_prediction(errors, item, f"sample_index={index} 的 targets 非数组")
        return []

    targets: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for position, raw in enumerate(raw_targets[:12]):
        if not isinstance(raw, dict):
            _join_error_prediction(errors, item, f"sample_index={index} 的 target[{position}] 非对象")
            continue
        base_id = str(raw.get("target_id") or f"target_{position}").strip()[:40] or f"target_{position}"
        target_id = base_id
        suffix = 2
        while target_id in used_ids:
            target_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(target_id)
        try:
            point = _normalize_point(raw.get("subject_point"), frame)
        except (TypeError, ValueError) as error:
            _join_error_prediction(errors, raw, f"sample_index={index}/{target_id}: {error}")
            point = None
        try:
            focus_point = _normalize_point(raw.get("focus_point", raw.get("subject_point")), frame)
        except (TypeError, ValueError) as error:
            _join_error_prediction(errors, raw, f"sample_index={index}/{target_id} focus_point: {error}")
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
        phrase = str(raw.get("grounding_phrase", "")).strip().lower().rstrip(". ")
        if not phrase:
            phrase = "main subject"
        role = str(raw.get("role", "supporting")).strip().lower()
        if role not in {"primary", "supporting"}:
            role = "supporting"
        try:
            importance = float(raw.get("importance", 1.0 if role == "primary" else 0.5))
        except (TypeError, ValueError):
            importance = 1.0 if role == "primary" else 0.5
        if not math.isfinite(importance):
            importance = 1.0 if role == "primary" else 0.5
        importance = max(0.0, min(1.0, importance))
        targets.append(
            {
                "target_id": target_id,
                "description": str(raw.get("description", "")).strip()[:80],
                "grounding_phrase": phrase[:80],
                "subject_point": point,
                "focus_point": focus_point,
                "role": role,
                "importance": importance,
                "confidence": confidence,
                "visibility": visibility,
            }
        )
    return targets


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
    text: str, frames: list[SampledFrame], use_batch: bool = False
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    """解析结构化响应；局部 target 错误只丢弃该 target，不丢弃整个区间。"""

    if not frames:
        return [], [], []
    try:
        root = json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(f"Stage 3.5 响应不是合法 JSON: {error}") from error
    predictions = root.get("predictions") if isinstance(root, dict) else None
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
        targets = _parse_targets(item, by_sample[index], errors, index)
        target_ids = {str(target["target_id"]) for target in targets}
        raw_primary_ids = item.get("primary_target_ids", [])
        primary_ids = []
        if isinstance(raw_primary_ids, list):
            for value in raw_primary_ids:
                target_id = str(value).strip()
                if target_id in target_ids and target_id not in primary_ids:
                    primary_ids.append(target_id)
        for target in targets:
            if target["role"] == "primary" and target["target_id"] not in primary_ids:
                primary_ids.append(target["target_id"])
        # 兼容旧模型输出：若存在可见目标但没有主次字段，选择 importance/confidence
        # 最高者作为暂定主主体，后续 Stage 4 仍会做镜头内时序防抖。
        if not primary_ids:
            visible = [target for target in targets if target["subject_point"] is not None]
            if visible:
                primary_ids = [max(
                    visible,
                    key=lambda target: (float(target["importance"]), float(target["confidence"])),
                )["target_id"]]
        primary_set = set(primary_ids)
        for target in targets:
            if target["target_id"] in primary_set:
                target["role"] = "primary"
                target["importance"] = max(0.75, float(target["importance"]))
        phrases = _phrases(item.get("grounding_phrases"))
        for target in targets:
            phrase = target["grounding_phrase"]
            if phrase not in phrases:
                phrases.append(phrase)
        group_mode = str(item.get("group_mode", "multiple" if len(targets) > 1 else "single"))
        if group_mode not in {"single", "multiple"}:
            group_mode = "multiple" if len(targets) > 1 else "single"
        if group_mode == "single" and len(targets) > 1:
            group_mode = "multiple"
        composition_mode = str(item.get(
            "composition_mode", "group_focus" if len(primary_ids) > 1 else "single_focus"
        ))
        if composition_mode not in {"single_focus", "group_focus"}:
            composition_mode = "group_focus" if len(primary_ids) > 1 else "single_focus"
        if len(primary_ids) > 1:
            composition_mode = "group_focus"
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
            recommended_crop_confidence = float(item.get("recommended_crop_confidence", 0.0))
        except (TypeError, ValueError):
            recommended_crop_confidence = 0.0
        if not math.isfinite(recommended_crop_confidence):
            recommended_crop_confidence = 0.0
        recommended_crop_confidence = max(0.0, min(1.0, recommended_crop_confidence))
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
            "reason": str(item.get("reason", "")),
        }
        if not use_batch:
            break

    missing = [index for index in ordered_indices if index not in by_index]
    for index in missing:
        by_index[index] = _empty_prediction("model_omitted_sample")
    return [by_index[index] for index in ordered_indices], missing, errors
