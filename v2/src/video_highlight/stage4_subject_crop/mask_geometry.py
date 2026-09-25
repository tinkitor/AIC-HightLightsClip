"""SAM2 Mask 的紧凑空间表示、裁剪质量计算和候选中心搜索。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MaskCropMetrics:
    iou: float
    coverage: float
    object_coverage: float
    min_object_coverage: float
    boundary_cut: float
    primary_coverage: float = 0.0
    min_primary_coverage: float = 0.0
    supporting_coverage: float = 0.0
    primary_boundary_cut: float = 0.0


@dataclass(frozen=True, slots=True)
class MaskEvidence:
    """以低分辨率面积密度图保存逐帧 Mask，避免保留全分辨率二值图。"""

    frame_size: tuple[int, int]
    union_density: np.ndarray
    # 保留 SAM2 object_id，构图阶段才能把叙事主主体和辅助主体区别对待。
    object_densities: tuple[tuple[int, np.ndarray], ...] = ()

    @property
    def grid_size(self) -> tuple[int, int]:
        return int(self.union_density.shape[1]), int(self.union_density.shape[0])


def _density_grid(mask: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    # 先用 uint8 的 0/255 表示二值面积，再由 INTER_AREA 直接生成 0..255 的块覆盖率；
    # 避免逐对象创建全分辨率 float32 临时数组。
    binary = np.where(mask > 0, np.uint8(255), np.uint8(0))
    return cv2.resize(binary, grid_size, interpolation=cv2.INTER_AREA)


def build_mask_evidence(
    union_mask: np.ndarray,
    object_masks: dict[int, np.ndarray] | None = None,
    max_side: int = 160,
) -> MaskEvidence | None:
    """把联合/逐对象 Mask 压缩成面积密度图，供构图阶段近似真实像素积分。"""

    if union_mask.ndim != 2 or not np.any(union_mask):
        return None
    height, width = union_mask.shape
    limit = max(16, int(max_side))
    scale = min(1.0, limit / max(width, height))
    grid_size = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    union_density = _density_grid(union_mask, grid_size)
    densities: list[tuple[int, np.ndarray]] = []
    for object_id in sorted(object_masks or {}):
        mask = object_masks[object_id]
        if mask.shape == union_mask.shape and np.any(mask):
            densities.append((int(object_id), _density_grid(mask, grid_size)))
    return MaskEvidence((width, height), union_density, tuple(densities))


def _grid_bounds(
    crop: tuple[float, float, float, float], evidence: MaskEvidence
) -> tuple[int, int, int, int]:
    frame_w, frame_h = evidence.frame_size
    grid_w, grid_h = evidence.grid_size
    x, y, width, height = crop
    x1 = max(0, min(grid_w - 1, int(math.floor(x * grid_w / frame_w))))
    y1 = max(0, min(grid_h - 1, int(math.floor(y * grid_h / frame_h))))
    x2 = max(x1 + 1, min(grid_w, int(math.ceil((x + width) * grid_w / frame_w))))
    y2 = max(y1 + 1, min(grid_h, int(math.ceil((y + height) * grid_h / frame_h))))
    return x1, y1, x2, y2


def _area(density: np.ndarray, bounds: tuple[int, int, int, int] | None = None) -> float:
    if bounds is None:
        return float(density.sum(dtype=np.float64) / 255.0)
    x1, y1, x2, y2 = bounds
    return float(density[y1:y2, x1:x2].sum(dtype=np.float64) / 255.0)


def mask_crop_metrics(
    crop: tuple[float, float, float, float],
    evidence: MaskEvidence | None,
    boundary_band_ratio: float = 0.03,
    primary_object_ids: tuple[int, ...] | list[int] = (),
) -> MaskCropMetrics | None:
    """计算裁剪框和真实 Mask 密度的 IoU、覆盖率、逐对象覆盖率与切边风险。"""

    if evidence is None:
        return None
    bounds = _grid_bounds(crop, evidence)
    x1, y1, x2, y2 = bounds
    mask_area = _area(evidence.union_density)
    if mask_area <= 0.0:
        return None
    intersection = _area(evidence.union_density, bounds)
    crop_area = float(max(1, (x2 - x1) * (y2 - y1)))
    iou = intersection / max(1e-6, mask_area + crop_area - intersection)
    coverage = intersection / mask_area

    primary_ids = {int(value) for value in primary_object_ids}
    object_coverages: list[float] = []
    primary_coverages: list[float] = []
    supporting_coverages: list[float] = []
    primary_boundary_cuts: list[float] = []
    for object_id, density in evidence.object_densities:
        object_area = _area(density)
        if object_area > 0.0:
            object_coverage_value = _area(density, bounds) / object_area
            object_coverages.append(object_coverage_value)
            if object_id in primary_ids:
                primary_coverages.append(object_coverage_value)
            else:
                supporting_coverages.append(object_coverage_value)
    object_coverage = float(np.mean(object_coverages)) if object_coverages else coverage
    min_object_coverage = min(object_coverages, default=coverage)

    band = max(1, int(round(min(x2 - x1, y2 - y1) * max(0.0, boundary_band_ratio))))
    outer = (
        max(0, x1 - band),
        max(0, y1 - band),
        min(evidence.grid_size[0], x2 + band),
        min(evidence.grid_size[1], y2 + band),
    )
    inner = (min(x2, x1 + band), min(y2, y1 + band), max(x1, x2 - band), max(y1, y2 - band))
    inner_area = 0.0 if inner[2] <= inner[0] or inner[3] <= inner[1] else _area(evidence.union_density, inner)
    boundary_cut = max(0.0, _area(evidence.union_density, outer) - inner_area) / mask_area
    for object_id, density in evidence.object_densities:
        if object_id not in primary_ids:
            continue
        object_area = _area(density)
        if object_area <= 0.0:
            continue
        object_inner = 0.0 if inner[2] <= inner[0] or inner[3] <= inner[1] else _area(density, inner)
        primary_boundary_cuts.append(max(0.0, _area(density, outer) - object_inner) / object_area)
    primary_coverage = float(np.mean(primary_coverages)) if primary_coverages else (0.0 if primary_ids else object_coverage)
    min_primary_coverage = min(primary_coverages, default=0.0 if primary_ids else min_object_coverage)
    supporting_coverage = float(np.mean(supporting_coverages)) if supporting_coverages else coverage
    primary_boundary_cut = max(primary_boundary_cuts, default=1.0 if primary_ids else boundary_cut)
    return MaskCropMetrics(
        iou, coverage, object_coverage, min_object_coverage, boundary_cut,
        primary_coverage, min_primary_coverage, supporting_coverage, primary_boundary_cut,
    )


def mask_object_centers(
    evidence: MaskEvidence | None,
    object_ids: tuple[int, ...] | list[int],
) -> list[tuple[float, float]]:
    """返回指定 SAM2 对象的 Mask 质心（原图像素坐标）。"""

    if evidence is None:
        return []
    wanted = {int(value) for value in object_ids}
    frame_w, frame_h = evidence.frame_size
    grid_w, grid_h = evidence.grid_size
    centers: list[tuple[float, float]] = []
    for object_id, density in evidence.object_densities:
        if object_id not in wanted:
            continue
        weights = density.astype(np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        xs = (np.arange(grid_w, dtype=np.float64) + 0.5) * frame_w / grid_w
        ys = (np.arange(grid_h, dtype=np.float64) + 0.5) * frame_h / grid_h
        centers.append((float((weights.sum(axis=0) * xs).sum() / total), float((weights.sum(axis=1) * ys).sum() / total)))
    return centers


def mask_centers_by_object(evidence: MaskEvidence | None) -> dict[int, tuple[float, float]]:
    """返回所有对象的 Mask 质心，供诊断产物与可视化标注使用。"""

    if evidence is None:
        return {}
    output: dict[int, tuple[float, float]] = {}
    frame_w, frame_h = evidence.frame_size
    grid_w, grid_h = evidence.grid_size
    xs = (np.arange(grid_w, dtype=np.float64) + 0.5) * frame_w / grid_w
    ys = (np.arange(grid_h, dtype=np.float64) + 0.5) * frame_h / grid_h
    for object_id, density in evidence.object_densities:
        weights = density.astype(np.float64)
        total = float(weights.sum())
        if total > 0.0:
            output[object_id] = (
                float((weights.sum(axis=0) * xs).sum() / total),
                float((weights.sum(axis=1) * ys).sum() / total),
            )
    return output


def ranked_mask_centers(
    evidence: MaskEvidence | None,
    crop_size: tuple[float, float],
    maximum: int = 8,
    nms_ratio: float = 0.18,
) -> list[tuple[float, float]]:
    """在密度图上搜索 Mask 覆盖高的候选中心，同时兼顾每个独立对象。"""

    if evidence is None or maximum <= 0:
        return []
    frame_w, frame_h = evidence.frame_size
    grid_w, grid_h = evidence.grid_size
    crop_w, crop_h = crop_size
    kernel_w = max(1, min(grid_w, int(round(crop_w * grid_w / frame_w))))
    kernel_h = max(1, min(grid_h, int(round(crop_h * grid_h / frame_h))))

    densities = (evidence.union_density, *(density for _, density in evidence.object_densities))
    score = np.zeros((grid_h, grid_w), dtype=np.float32)
    for index, density in enumerate(densities):
        total = float(density.sum(dtype=np.float64))
        if total <= 0.0:
            continue
        covered = cv2.boxFilter(
            density.astype(np.float32),
            ddepth=-1,
            ksize=(kernel_w, kernel_h),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        normalized = covered / total
        score += normalized if index == 0 else normalized / max(1, len(evidence.object_densities))

    # 多取一些峰值用于 NMS；合法化和去重由候选生成器统一处理。
    order = np.argsort(score, axis=None)[::-1]
    centers: list[tuple[float, float]] = []
    for flat_index in order:
        gy, gx = np.unravel_index(int(flat_index), score.shape)
        center = ((gx + 0.5) * frame_w / grid_w, (gy + 0.5) * frame_h / grid_h)
        if any(
            abs(center[0] - other[0]) < crop_w * nms_ratio
            and abs(center[1] - other[1]) < crop_h * nms_ratio
            for other in centers
        ):
            continue
        centers.append(center)
        if len(centers) >= maximum:
            break
    return centers
