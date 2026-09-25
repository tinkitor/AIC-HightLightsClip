"""围绕逐帧主体框生成多尺度、多偏移的合法目标比例构图。"""

from __future__ import annotations

from typing import Any

from .boundary_limiter import legal_crop_from_state, maximum_crop_width
from .mask_geometry import MaskEvidence, mask_object_centers, ranked_mask_centers


def generate_crop_candidates(
    subject_box: list[float] | tuple[float, float, float, float],
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
    motion: tuple[float, float],
    config: dict[str, Any],
    mask_evidence: MaskEvidence | None = None,
    primary_object_ids: tuple[int, ...] = (),
    primary_focus_points: tuple[tuple[float, float], ...] = (),
    recommended_crop_center: tuple[float, float] | None = None,
    recommended_only: bool = False,
) -> list[tuple[float, float, float, float]]:
    frame_w, frame_h = frame_size
    target_w, target_h = target_ratio
    x1, y1, x2, y2 = map(float, subject_box)
    subject_w, subject_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    margins = config.get("subject_margins", [0.20, 0.20, 0.15, 0.30])
    left, right, top, bottom = (float(value) for value in margins)
    # 同时满足边距与宽高比例约束
    required_width = max(
        subject_w * (1.0 + left + right),
        subject_h * (1.0 + top + bottom) * target_w / target_h,
    )
    max_width = maximum_crop_width(frame_size, target_ratio)
    if recommended_only:
        # 纯 center 方案的契约是“Qwen 推荐构图中心是唯一候选”。缺失推荐时
        # 仍生成唯一的画面中心最大框，避免退回旧 subject_point 或多候选搜索。
        candidate_center = recommended_crop_center or (frame_w * 0.5, frame_h * 0.5)
        return [legal_crop_from_state(
            candidate_center[0], candidate_center[1], float(max_width), frame_size, target_ratio
        )]
    # 无 Mask 的固定最大框仍保持旧行为。SAM2 提供 Mask 时，即使尺度固定，也需要
    # 在所有合法位置中搜索真实主体覆盖更高的中心，不能退回外接框几何中心。
    if bool(config.get("fixed_maximum", False)):
        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        base = legal_crop_from_state(center_x, center_y, float(max_width), frame_size, target_ratio)
        if mask_evidence is None and not primary_focus_points and recommended_crop_center is None:
            return [base]
        widths = [float(max_width)]
    else:
        scales = sorted({float(value) for value in config.get("scales", [0.55, 0.70, 0.85, 1.0])})
        widths = sorted({max(1.0, min(float(max_width), max_width * scale)) for scale in scales if scale > 0.0})
        # 外接框很宽不等于 Mask 像素在整个宽度上均匀分布。存在 Mask 时保留所有
        # 合法尺度，让 IoU/覆盖率决定应当紧凑取景还是扩大取景。
        if mask_evidence is None:
            widths = [value for value in widths if value + 1e-6 >= required_width] or [float(max_width)]
        if float(max_width) not in widths:
            widths.append(float(max_width))

    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    offset_values = [float(value) for value in config.get("offsets", [-0.12, 0.0, 0.12])]
    look_ahead = float(config.get("motion_look_ahead", 2.0))
    candidates: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for width in widths:
        height = width * target_h / target_w
        centers = [
            (
                center_x + offset_x * width + motion[0] * look_ahead,
                center_y + offset_y * height + motion[1] * look_ahead,
            )
            for offset_x in offset_values
            for offset_y in offset_values
        ]
        if recommended_crop_center is not None:
            centers.insert(0, recommended_crop_center)
        # 无论 fixed_maximum 开关如何，都显式加入主主体中心候选。focus_point 是
        # Qwen 给出的构图关注点；Mask 质心则是 SAM2 在当前帧的时序校正。
        primary_centers = list(primary_focus_points)
        primary_centers.extend(mask_object_centers(mask_evidence, primary_object_ids))
        if primary_centers:
            centers.insert(0, (
                sum(point[0] for point in primary_centers) / len(primary_centers),
                sum(point[1] for point in primary_centers) / len(primary_centers),
            ))
            for point in reversed(primary_centers):
                centers.insert(0, point)
        centers.append((center_x, center_y))
        centers.extend(ranked_mask_centers(
            mask_evidence,
            (width, height),
            maximum=max(1, int(config.get("mask_top_k_per_scale", 8))),
            nms_ratio=float(config.get("mask_candidate_nms_ratio", 0.18)),
        ))
        for candidate_center_x, candidate_center_y in centers:
            crop = legal_crop_from_state(
                candidate_center_x,
                candidate_center_y,
                width,
                frame_size,
                target_ratio,
            )
            key = (round(crop[0]), round(crop[1]), round(crop[2]))
            if key not in seen:
                candidates.append(crop)
                seen.add(key)
    return candidates
