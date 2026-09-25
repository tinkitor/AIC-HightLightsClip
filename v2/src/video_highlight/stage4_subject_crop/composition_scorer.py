"""主体覆盖、中心性、留白和尺度的逐帧构图代价。"""

from __future__ import annotations

import math
from typing import Any

from .boundary_limiter import legal_crop_from_state
from .mask_geometry import MaskCropMetrics, MaskEvidence, mask_crop_metrics


def _intersection_area(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """
    计算a,b两个矩形的相交面积，未相交则为零
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax1 + aw, bx2) - max(ax1, bx1)) * max(0.0, min(ay1 + ah, by2) - max(ay1, by1))


def composition_cost(
    crop: tuple[float, float, float, float],
    subject_box: list[float] | tuple[float, float, float, float],
    frame_size: tuple[int, int],
    config: dict[str, Any],
    mask_evidence: MaskEvidence | None = None,
    primary_object_ids: tuple[int, ...] = (),
    primary_focus_points: tuple[tuple[float, float], ...] = (),
    recommended_crop_center: tuple[float, float] | None = None,
    recommended_crop_confidence: float = 0.0,
) -> float:
    """计算单帧构图代价；SAM2 轨迹优先使用真实 Mask，其他后端兼容旧框评分。"""
    x, y, width, height = crop
    sx1, sy1, sx2, sy2 = map(float, subject_box)
    subject_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    uncovered = 1.0 - min(1.0, _intersection_area(crop, (sx1, sy1, sx2, sy2)) / subject_area)
    subject_cx, subject_cy = (sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5
    normalized_dx = abs(subject_cx - (x + width * 0.5)) / max(width, 1.0)
    normalized_dy = abs(subject_cy - (y + height * 0.5)) / max(height, 1.0)
    frame_width = max(1.0, float(frame_size[0]))
    zoom_cost = 1.0 - min(1.0, width / frame_width)
    qwen_recommendation_cost = 0.0
    if recommended_crop_center is not None:
        # 先把 Qwen 点按当前候选尺度合法化，边缘主体不会因为模型给出几何上
        # 不可实现的中心而被额外惩罚。该项权重刻意较弱，只负责打破近似平局。
        ideal = legal_crop_from_state(
            recommended_crop_center[0], recommended_crop_center[1], width, frame_size,
            (width, height),
        )
        ideal_cx, ideal_cy = ideal[0] + ideal[2] * 0.5, ideal[1] + ideal[3] * 0.5
        crop_cx, crop_cy = x + width * 0.5, y + height * 0.5
        distance = math.hypot(
            (crop_cx - ideal_cx) / max(width, 1.0),
            (crop_cy - ideal_cy) / max(height, 1.0),
        )
        confidence = max(0.0, min(1.0, float(recommended_crop_confidence)))
        qwen_recommendation_cost = (
            float(config.get("qwen_recommended_center_weight", 0.05)) * confidence * distance
        )
    metrics = mask_crop_metrics(
        crop,
        mask_evidence,
        float(config.get("boundary_band_ratio", 0.03)),
        primary_object_ids,
    )
    if metrics is not None:
        if primary_object_ids:
            weights = config.get("primary_mask_weights", {})
            safe_x = config.get("primary_safe_zone_x", [0.35, 0.65])
            safe_y = config.get("primary_safe_zone_y", [0.30, 0.70])
            focus_costs: list[float] = []
            for point_x, point_y in primary_focus_points:
                normalized_x = (point_x - x) / max(width, 1.0)
                normalized_y = (point_y - y) / max(height, 1.0)
                dx = max(float(safe_x[0]) - normalized_x, 0.0, normalized_x - float(safe_x[1]))
                dy = max(float(safe_y[0]) - normalized_y, 0.0, normalized_y - float(safe_y[1]))
                focus_costs.append(dx + dy)
            primary_center_cost = sum(focus_costs) / len(focus_costs) if focus_costs else normalized_dx + normalized_dy
            return qwen_recommendation_cost + (
                float(weights.get("primary_coverage", 0.28)) * (1.0 - metrics.primary_coverage)
                + float(weights.get("min_primary_coverage", 0.18)) * (1.0 - metrics.min_primary_coverage)
                + float(weights.get("primary_center", 0.24)) * primary_center_cost
                + float(weights.get("primary_boundary_cut", 0.12)) * metrics.primary_boundary_cut
                + float(weights.get("supporting_coverage", 0.07)) * (1.0 - metrics.supporting_coverage)
                + float(weights.get("iou", 0.06)) * (1.0 - metrics.iou)
                + float(weights.get("coverage", 0.03)) * (1.0 - metrics.coverage)
                + float(weights.get("zoom", 0.02)) * zoom_cost
            )
        weights = config.get("mask_weights", {})
        return qwen_recommendation_cost + (
            float(weights.get("iou", 0.25)) * (1.0 - metrics.iou)
            + float(weights.get("coverage", 0.25)) * (1.0 - metrics.coverage)
            + float(weights.get("object_coverage", 0.20)) * (1.0 - metrics.object_coverage)
            + float(weights.get("min_object_coverage", 0.15)) * (1.0 - metrics.min_object_coverage)
            + float(weights.get("boundary_cut", 0.10)) * metrics.boundary_cut
            + float(weights.get("centering", 0.03)) * (normalized_dx + normalized_dy)
            + float(weights.get("zoom", 0.02)) * zoom_cost
        )
    weights = config.get("weights", {})
    return qwen_recommendation_cost + (
        float(weights.get("uncovered", 0.55)) * uncovered
        + float(weights.get("centering", 0.25)) * (normalized_dx + normalized_dy)
        + float(weights.get("zoom", 0.20)) * zoom_cost
    )


def composition_metrics(
    crop: tuple[float, float, float, float],
    mask_evidence: MaskEvidence | None,
    config: dict[str, Any],
    primary_object_ids: tuple[int, ...] = (),
) -> MaskCropMetrics | None:
    """供平滑保护与诊断复用同一套 Mask 几何定义。"""

    return mask_crop_metrics(
        crop,
        mask_evidence,
        float(config.get("boundary_band_ratio", 0.03)),
        primary_object_ids,
    )
