"""SAM2 传播过程的低频率调试图与可选视频输出。"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError

from .subject_tracker import TrackPoint


class VisualizationError(ArtifactValidationError):
    """可视化产物读取、编码或写入失败。"""


def _read_image(path: Path) -> np.ndarray | None:
    """绕开 Windows OpenCV 对非 ASCII 路径支持不完整的问题。"""

    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def _write_jpeg(path: Path, image: np.ndarray) -> bool:
    """使用 ``imencode + tofile`` 向中文路径安全写入 JPEG。"""

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


_OBJECT_COLORS = (
    (60, 210, 60), (255, 120, 0), (0, 190, 255), (220, 80, 220),
    (255, 200, 80), (80, 180, 255), (180, 255, 80), (255, 100, 160),
)

# OpenCV 使用 BGR。该颜色专门表示 Stage 3.5 的帧级 Qwen 推荐构图中心。
_QWEN_RECOMMENDED_COLOR = (180, 0, 180)


def _object_color(object_id: int) -> tuple[int, int, int]:
    return _OBJECT_COLORS[abs(int(object_id)) % len(_OBJECT_COLORS)]


def _mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"]))


def _draw_tag(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.52,
    thickness: int | None = None,
) -> None:
    """用不透明黑底绘制短标签，避免文字与画面纹理混在一起。"""

    thickness = max(1, int(thickness if thickness is not None else (1 if scale < 0.6 else 2)))
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(0, min(image.shape[1] - text_w - 8, int(origin[0])))
    y = max(text_h + 6, min(image.shape[0] - baseline - 3, int(origin[1])))
    cv2.rectangle(image, (x, y - text_h - 5), (x + text_w + 7, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_round_marker(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    border_thickness: int,
    text: str | None = None,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """绘制带白边的实心圆点，并按底色自动选择圆内文字颜色。"""

    radius = max(4, int(radius))
    border_thickness = max(1, int(border_thickness))
    cv2.circle(image, center, radius, fill_color, -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (255, 255, 255), border_thickness, cv2.LINE_AA)
    if text is None:
        return
    text = str(text)
    font_scale = max(0.34, min(0.70, radius / 18.0 * 0.55))
    font_thickness = max(1, border_thickness)
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    maximum_width = radius * 1.55
    if text_width > maximum_width:
        font_scale = max(0.22, font_scale * maximum_width / text_width)
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
    origin = (
        int(round(center[0] - text_width * 0.5)),
        int(round(center[1] + text_height * 0.5)),
    )
    blue, green, red = fill_color
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    text_color = (0, 0, 0) if luminance >= 150 else (255, 255, 255)
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        text_color, font_thickness, cv2.LINE_AA,
    )


def _append_sidebar(
    image: np.ndarray,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
    panel_width: int,
    font_scale: float,
    font_thickness: int,
    configured_line_height: int,
) -> np.ndarray:
    panel = np.full((image.shape[0], panel_width, 3), 18, dtype=np.uint8)
    scale = float(font_scale)
    line_height = max(12, int(configured_line_height))
    y = line_height
    maximum = max(1, (image.shape[0] - 8) // line_height)
    for text, color in list(lines)[:maximum]:
        cv2.putText(panel, str(text)[:54], (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(font_thickness)), cv2.LINE_AA)
        y += line_height
    cv2.line(panel, (0, 0), (0, image.shape[0] - 1), (95, 95, 95), 2)
    return np.hstack((image, panel))


def _append_decision_panel(
    image: np.ndarray,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
    panel_height: int,
    font_scale: float,
    font_thickness: int,
    configured_line_height: int,
) -> np.ndarray:
    panel = np.full((panel_height, image.shape[1], 3), 12, dtype=np.uint8)
    scale = float(font_scale)
    line_height = max(12, int(configured_line_height))
    columns = 2
    column_width = image.shape[1] // columns
    rows_per_column = max(1, (panel_height - 12) // line_height)
    for index, (text, color) in enumerate(lines[:rows_per_column * columns]):
        column, row = divmod(index, rows_per_column)
        cv2.putText(
            panel, str(text)[:72], (12 + column * column_width, 8 + (row + 1) * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(font_thickness)), cv2.LINE_AA,
        )
    cv2.line(panel, (0, 0), (image.shape[1] - 1, 0), (95, 95, 95), 2)
    return np.vstack((image, panel))


def _encode_images(frame_dir: Path, output_path: Path, fps: float) -> None:
    images = sorted(frame_dir.glob("frame_*.jpg"))
    if not images:
        return
    first = _read_image(images[0])
    if first is None:
        raise VisualizationError(f"无法读取可视化视频首帧: {images[0]}")
    height, width = first.shape[:2]
    descriptor, temporary_name = tempfile.mkstemp(prefix="stage4-viz-", suffix=".mp4")
    os.close(descriptor)
    temporary_video = Path(temporary_name)
    temporary_video.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), max(0.1, fps), (width, height)
    )
    if not writer.isOpened():
        temporary_video.unlink(missing_ok=True)
        raise VisualizationError(f"无法创建 Stage 4 可视化视频: {output_path}")
    try:
        for path in images:
            image = _read_image(path)
            if image is None:
                raise VisualizationError(f"无法读取可视化视频帧: {path}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()
    output_path.unlink(missing_ok=True)
    shutil.move(str(temporary_video), str(output_path))


def annotate_final_composition(
    visualization_root: Path,
    interval_id: str,
    crops: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    target_ratio: tuple[float, float],
    config: dict[str, Any],
    source_fps: float,
) -> None:
    """第二遍补画构图候选、DP 结果、最终裁剪框及可比较的质量指标。"""

    if not bool(config.get("enabled", False)) or not bool(config.get("annotate_composition", True)):
        return
    interval_dir = visualization_root / str(interval_id)
    if not interval_dir.is_dir():
        return
    crop_by_frame = {int(row["frame"]): row for row in crops}
    track_by_frame = {int(row["frame"]): row for row in tracks}
    touched_dirs: set[Path] = set()
    for image_path in interval_dir.rglob("frame_*.jpg"):
        try:
            frame_index = int(image_path.stem.split("_")[-1])
        except ValueError:
            continue
        crop_row, track = crop_by_frame.get(frame_index), track_by_frame.get(frame_index)
        if crop_row is None or track is None:
            continue
        image = _read_image(image_path)
        if image is None:
            raise VisualizationError(f"无法读取构图可视化帧: {image_path}")
        height, width = image.shape[:2]
        frame_size = track.get("frame_size_wh", [width, height])
        main_width = min(width, int(frame_size[0]))
        main_height = min(height, int(frame_size[1]))
        decision = track.get("composition_decision", {})
        recommended = track.get("recommended_crop_center_xy")
        recommended_center: tuple[int, int] | None = None
        if isinstance(recommended, (list, tuple)) and len(recommended) == 2:
            try:
                recommended_center = (
                    min(main_width - 1, max(0, int(round(float(recommended[0]))))),
                    min(main_height - 1, max(0, int(round(float(recommended[1]))))),
                )
            except (TypeError, ValueError):
                recommended_center = None
        try:
            recommended_confidence = float(track.get("recommended_crop_confidence", 0.0))
        except (TypeError, ValueError):
            recommended_confidence = 0.0
        drawn_candidates = decision.get("top_local_candidates", [])[:max(
            0, int(config.get("max_drawn_candidate_centers", 3))
        )]
        for candidate in drawn_candidates:
            center = candidate.get("center_xy", [])
            if len(center) != 2:
                continue
            xy = int(round(center[0])), int(round(center[1]))
            _draw_round_marker(
                image, xy,
                int(config.get("candidate_marker_radius", 6)),
                int(config.get("candidate_marker_thickness", 2)),
                str(candidate.get("rank", "")),
            )
        optimized = decision.get("optimized_bbox_xywh", [])
        optimized_center: tuple[int, int] | None = None
        if len(optimized) == 4:
            optimized_center = (int(round(optimized[0] + optimized[2] * 0.5)), int(round(optimized[1] + optimized[3] * 0.5)))
        x, y, crop_width = (float(value) for value in crop_row["bboxes"])
        crop_height = crop_width * target_ratio[1] / target_ratio[0]
        final_box = (x, y, x + crop_width, y + crop_height)
        PropagationVisualizer._draw_box(
            image, final_box, (0, 255, 0), int(config.get("final_crop_box_thickness", 4))
        )
        final_center = (int(round(x + crop_width * 0.5)), int(round(y + crop_height * 0.5)))
        _draw_round_marker(
            image, final_center,
            int(config.get("final_marker_size", 6)),
            int(config.get("final_marker_thickness", 4)),
            None,
            (0, 255, 0),
        )
        dp_overlaps_output = False
        if optimized_center is not None:
            dp_radius = int(config.get("dp_marker_size", 6))
            final_radius = int(config.get("final_marker_size", 6))
            dp_overlaps_output = math.hypot(
                optimized_center[0] - final_center[0], optimized_center[1] - final_center[1]
            ) <= max(dp_radius, final_radius) + 2
            if dp_overlaps_output:
                # 两个中心重合时，绿色输出点会遮住 DP。使用橙色同心环表示 DP
                # 位于同一坐标，不移动标记，避免可视化产生错误位置。
                cv2.circle(
                    image, optimized_center, max(dp_radius, final_radius) + 5,
                    (0, 165, 255), max(2, int(config.get("dp_marker_thickness", 3))), cv2.LINE_AA,
                )
                _draw_tag(
                    image, "DP", (optimized_center[0] + max(dp_radius, final_radius) + 8, optimized_center[1]),
                    (0, 165, 255), float(config.get("tag_font_scale", 0.52)),
                    int(config.get("tag_font_thickness", 1)),
                )
            else:
                _draw_round_marker(
                    image, optimized_center, dp_radius,
                    int(config.get("dp_marker_thickness", 3)), "DP", (0, 165, 255),
                )
        if recommended_center is not None:
            recommended_radius = int(config.get("qwen_recommended_marker_size", 6))
            recommended_thickness = int(config.get("qwen_recommended_marker_thickness", 1))
            overlap_radius = recommended_radius
            overlaps_decision = math.hypot(
                recommended_center[0] - final_center[0], recommended_center[1] - final_center[1]
            ) <= max(recommended_radius, int(config.get("final_marker_size", 6))) + 2
            if optimized_center is not None:
                overlaps_decision = overlaps_decision or math.hypot(
                    recommended_center[0] - optimized_center[0],
                    recommended_center[1] - optimized_center[1],
                ) <= max(recommended_radius, int(config.get("dp_marker_size", 6))) + 2
                overlap_radius = max(overlap_radius, int(config.get("dp_marker_size", 6)))
            if overlaps_decision:
                # 推荐点经常与 DP/最终中心重合。此时用紫色外环保留内层决策点颜色，
                # 同时在环外标注 R，避免任何一类信息被覆盖。
                ring_radius = max(overlap_radius, int(config.get("final_marker_size", 6))) + 10
                cv2.circle(
                    image, recommended_center, ring_radius, _QWEN_RECOMMENDED_COLOR,
                    max(1, recommended_thickness), cv2.LINE_AA,
                )
                _draw_tag(
                    image, "R", (recommended_center[0] + ring_radius + 4, recommended_center[1]),
                    _QWEN_RECOMMENDED_COLOR, float(config.get("tag_font_scale", 0.52)),
                    int(config.get("tag_font_thickness", 1)),
                )
            else:
                _draw_round_marker(
                    image, recommended_center, recommended_radius, recommended_thickness,
                    "R", _QWEN_RECOMMENDED_COLOR,
                )
        metrics = track.get("mask_crop_metrics", {})
        summary = f"output local_cost={float(decision.get('final_local_cost', 0.0)):.3f}"
        if metrics:
            summary += (
                f" IoU={float(metrics.get('iou', 0.0)):.3f} Pcov={float(metrics.get('primary_coverage', 0.0)):.3f} "
                f"Pmin={float(metrics.get('min_primary_coverage', 0.0)):.3f} "
                f"Scov={float(metrics.get('supporting_coverage', 0.0)):.3f} Pcut={float(metrics.get('primary_boundary_cut', 0.0)):.3f}"
            )
        else:
            summary += " no-mask"
        decision_lines: list[tuple[str, tuple[int, int, int]]] = [
            ("DECISION  C=local candidates  DP=temporal path  green=crop output", (255, 255, 255)),
            (summary, (0, 255, 0)),
            (f"frame area={main_width}x{main_height} candidates={int(decision.get('candidate_count', 0))}", (190, 190, 190)),
        ]
        if optimized_center is not None:
            decision_lines.extend([
                (f"DP center=({optimized_center[0]},{optimized_center[1]}) local={float(decision.get('optimized_local_cost', 0.0)):.4f}", (0, 165, 255)),
                (f"output center=({final_center[0]},{final_center[1]}) overlap={dp_overlaps_output}", (0, 255, 0)),
            ])
        for candidate in decision.get("top_local_candidates", []):
            center = candidate.get("center_xy", [0.0, 0.0])
            decision_lines.append((
                f"C{candidate.get('rank')}: center=({float(center[0]):.1f},{float(center[1]):.1f}) local={float(candidate.get('local_cost', 0.0)):.4f}",
                (215, 215, 215),
            ))
        sidebar_width = width - main_width
        if sidebar_width > 0:
            sidebar_scale = float(config.get("sidebar_font_scale", 0.56))
            sidebar_thickness = max(1, int(config.get("sidebar_font_thickness", 1)))
            sidebar_line_height = max(12, int(config.get("sidebar_line_height", 26)))
            sidebar_lines = [
                "COMPOSITION CENTERS",
                "R=Qwen recommended crop center",
                (
                    f"R=({recommended_center[0]},{recommended_center[1]}) conf={recommended_confidence:.3f}"
                    if recommended_center is not None else "R=unavailable conf=0.000"
                ),
                (
                    f"DP=({optimized_center[0]},{optimized_center[1]})"
                    if optimized_center is not None else "DP=unavailable"
                ),
                f"OUTPUT=({final_center[0]},{final_center[1]})",
                f"DP/output overlap={dp_overlaps_output}",
            ]
            block_height = sidebar_line_height * len(sidebar_lines) + 16
            block_top = max(0, main_height - block_height)
            cv2.rectangle(image, (main_width, block_top), (width - 1, main_height - 1), (12, 12, 12), -1)
            for line_index, line in enumerate(sidebar_lines):
                cv2.putText(
                    image, line, (main_width + 12, block_top + 8 + (line_index + 1) * sidebar_line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, sidebar_scale,
                    _QWEN_RECOMMENDED_COLOR if line.startswith("R=") else
                    (0, 165, 255) if line.startswith("DP") else
                    (0, 255, 0) if line.startswith("OUTPUT") else (235, 235, 235),
                    sidebar_thickness, cv2.LINE_AA,
                )
        image = _append_decision_panel(
            image,
            decision_lines,
            max(120, int(config.get("decision_panel_height", 170))),
            float(config.get("decision_font_scale", 0.52)),
            int(config.get("decision_font_thickness", 1)),
            int(config.get("decision_line_height", 26)),
        )
        if not _write_jpeg(image_path, image):
            raise VisualizationError(f"构图可视化图片写入失败: {image_path}")
        touched_dirs.add(image_path.parent)
    if bool(config.get("write_video", False)):
        output_fps = min(source_fps if source_fps > 0 else 30.0, float(config.get("sample_fps", 2.0)))
        for frame_dir in touched_dirs:
            output_dir = frame_dir.parent if frame_dir.name == ".video_frames" else frame_dir
            _encode_images(frame_dir, output_dir / "propagation.mp4", output_fps)
    for frame_dir in touched_dirs:
        if frame_dir.name == ".video_frames":
            shutil.rmtree(frame_dir, ignore_errors=True)


class PropagationVisualizer:
    """按配置采样并持久化一个镜头内的 SAM2 传播可视化。

    采样只决定哪些帧被绘制，不改变 SAM2 的逐帧传播。锚点和窗口回退帧可绕过
    普通采样频率强制保存。视频在镜头完成后由最终图片按绝对帧号排序编码，因而
    同一帧先产生 SAM2 结果、后因窗口异常改为回退时，不会在视频中出现重复帧。
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path | None,
        source_fps: float,
        span_start: int,
    ) -> None:
        self.config = config
        self.source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0
        self.span_start = span_start
        self.enabled = bool(config.get("enabled", False)) and output_dir is not None
        self.save_images = bool(config.get("save_images", True))
        self.write_video = bool(config.get("write_video", False))
        self.output_dir = output_dir
        self._video_frame_dir: Path | None = None
        if self.enabled and (self.save_images or self.write_video):
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.write_video and not self.save_images:
                self._video_frame_dir = output_dir / ".video_frames"
                self._video_frame_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return self.enabled and (self.save_images or self.write_video)

    def should_save(self, frame: int, *, is_anchor: bool, is_fallback: bool) -> bool:
        """判断绝对帧是否满足普通采样或强制保留条件。"""

        if not self.active:
            return False
        if is_anchor and bool(self.config.get("always_save_anchor_frames", True)):
            return True
        if is_fallback and bool(self.config.get("always_save_fallback_frames", True)):
            return True
        sample_fps = float(self.config.get("sample_fps", 2.0))
        # 用采样桶而不是 round(fps/sample_fps) 固定步长，可更准确处理 29.97 FPS。
        relative = max(0, frame - self.span_start)
        current_bucket = math.floor(relative * sample_fps / self.source_fps + 1e-9)
        if relative == 0:
            return True
        previous_bucket = math.floor((relative - 1) * sample_fps / self.source_fps + 1e-9)
        return current_bucket > previous_bucket

    @staticmethod
    def _draw_box(
        image: np.ndarray,
        box: Sequence[float],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def _render(
        self,
        frame: np.ndarray,
        prediction: TrackPoint,
        mask: np.ndarray | None,
        qwen_point: tuple[float, float] | None,
        qwen_points: Sequence[tuple[float, float]] | None,
        qwen_focus_points: Sequence[tuple[float, float]] | None,
        prompt_box: Sequence[float] | None,
        grounding_objects: Sequence[dict[str, Any]] | None,
        object_masks: dict[int, np.ndarray] | None,
        is_anchor: bool,
        is_fallback: bool,
    ) -> np.ndarray:
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        flags = ["ANCHOR"] if is_anchor else []
        if is_fallback:
            flags.append("FALLBACK")
        detail_lines: list[tuple[str, tuple[int, int, int]]] = [
            ("TRACKING EVIDENCE", (255, 255, 255)),
            (f"frame={prediction.frame} {' '.join(flags)}", (220, 220, 220)),
            (f"source={prediction.source}", (180, 210, 255)),
            (f"confidence={prediction.confidence:.3f}", (180, 210, 255)),
            (f"objects={list(prediction.object_ids)} primary={list(prediction.primary_object_ids)}", (0, 255, 255)),
            ("Q=qwen center (sidebar only)  F=qwen focus", (190, 190, 190)),
            ("D/G=Grounding boxes; center markers hidden", (190, 190, 190)),
            ("M=per-object mask centroid", (190, 190, 190)),
        ]
        if object_masks:
            alpha = float(self.config.get("mask_alpha", 0.35))
            for object_id, object_mask in sorted(object_masks.items()):
                if object_mask.shape != (height, width):
                    object_mask = cv2.resize(object_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                selected = object_mask.astype(bool)
                if not np.any(selected):
                    continue
                color = np.asarray(_object_color(object_id), dtype=np.float32)
                pixels = canvas[selected].astype(np.float32)
                canvas[selected] = np.clip(pixels * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
                centroid = _mask_centroid(object_mask)
                if centroid is not None:
                    marker_color = (0, 255, 255) if object_id in prediction.primary_object_ids else _object_color(object_id)
                    _draw_round_marker(
                        canvas, centroid,
                        int(self.config.get("mask_marker_size", 6)),
                        int(self.config.get("mask_marker_thickness", 3)),
                        f"M{object_id}",
                        marker_color,
                    )
                    detail_lines.append((f"M{object_id}: center=({centroid[0]},{centroid[1]})", marker_color))
        elif mask is not None:
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            selected = mask.astype(bool)
            if np.any(selected):
                alpha = float(self.config.get("mask_alpha", 0.35))
                color = np.asarray([60, 210, 60], dtype=np.float32)
                pixels = canvas[selected].astype(np.float32)
                canvas[selected] = np.clip(pixels * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        if prompt_box is not None and bool(self.config.get("draw_prompt_union", False)):
            self._draw_box(canvas, prompt_box, (0, 215, 255), 2)
        raw_seen = 0
        raw_limit = max(0, int(self.config.get("max_raw_grounding_boxes", 12)))
        raw_index = 0
        for row in grounding_objects or ():
            box = row.get("box_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            kind = str(row.get("kind", "selected"))
            if kind == "raw":
                raw_seen += 1
                if raw_seen > raw_limit:
                    continue
                raw_index += 1
            object_id = int(row.get("object_id", -1))
            color = (150, 150, 150) if kind == "raw" else _object_color(object_id)
            self._draw_box(
                canvas, box, color,
                int(self.config.get("raw_detection_box_thickness", 1))
                if kind == "raw" else int(self.config.get("selected_detection_box_thickness", 3)),
            )
            prefix = f"D{raw_index}" if kind == "raw" else f"G{object_id}"
            # Grounding DINO 框中心不参与最终构图中心的表达，取消中心圆点；
            # 短标签放在框的左上角，仅用于区分多个检测框。
            _draw_tag(
                canvas,
                prefix,
                (int(round(float(box[0]))), int(round(float(box[1]))) - 4),
                color,
                float(self.config.get("tag_font_scale", 0.52)),
                int(self.config.get("tag_font_thickness", 1)),
            )
            detail_lines.append((
                f"{prefix}: {str(row.get('phrase', ''))[:22]} score={float(row.get('score', 0.0)):.3f}",
                color,
            ))
        if is_fallback or bool(self.config.get("draw_subject_union", False)):
            self._draw_box(
                canvas, prediction.subject_box,
                (0, 0, 255) if is_fallback else (255, 120, 0),
                int(self.config.get("fallback_box_thickness", 2)),
            )
            x1, y1, x2, y2 = prediction.subject_box
            center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
            _draw_round_marker(
                canvas, center,
                int(self.config.get("fallback_marker_size", 6)),
                int(self.config.get("fallback_marker_thickness", 2)),
                "FB" if is_fallback else "U",
                (0, 0, 255) if is_fallback else (255, 120, 0),
            )
        all_qwen_points = list(qwen_points or ())
        if not all_qwen_points and qwen_point is not None:
            all_qwen_points = [qwen_point]
        for point_index, point in enumerate(all_qwen_points):
            # subject_point 仅保留在右侧证据栏供核对，不再叠加到主画面。
            detail_lines.append((f"Q{point_index + 1}: ({point[0]:.3f},{point[1]:.3f})", (255, 0, 255)))
        for focus_index, point in enumerate(qwen_focus_points or ()):
            fx = min(width - 1, max(0, int(round(point[0] * (width - 1)))))
            fy = min(height - 1, max(0, int(round(point[1] * (height - 1)))))
            _draw_round_marker(
                canvas, (fx, fy),
                int(self.config.get("focus_marker_size", 6)),
                int(self.config.get("focus_marker_thickness", 3)),
                f"F{focus_index + 1}",
                (0, 255, 255),
            )
            detail_lines.append((f"F{focus_index + 1}: ({point[0]:.3f},{point[1]:.3f})", (0, 255, 255)))
        panel_width = max(340, min(520, int(self.config.get("sidebar_width", max(340, width * 0.30)))))
        return _append_sidebar(
            canvas, detail_lines, panel_width,
            float(self.config.get("sidebar_font_scale", 0.56)),
            int(self.config.get("sidebar_font_thickness", 1)),
            int(self.config.get("sidebar_line_height", 26)),
        )

    def save(
        self,
        frame_index: int,
        source_frame_path: Path,
        prediction: TrackPoint,
        *,
        mask: np.ndarray | None = None,
        qwen_point: tuple[float, float] | None = None,
        qwen_points: Sequence[tuple[float, float]] | None = None,
        qwen_focus_points: Sequence[tuple[float, float]] | None = None,
        prompt_box: Sequence[float] | None = None,
        grounding_objects: Sequence[dict[str, Any]] | None = None,
        object_masks: dict[int, np.ndarray] | None = None,
        is_anchor: bool = False,
        is_fallback: bool = False,
    ) -> None:
        if not self.should_save(frame_index, is_anchor=is_anchor, is_fallback=is_fallback):
            return
        frame = _read_image(source_frame_path)
        if frame is None:
            raise VisualizationError(f"无法读取 SAM2 可视化源帧: {source_frame_path}")
        canvas = self._render(
            frame, prediction, mask, qwen_point, qwen_points, qwen_focus_points, prompt_box,
            grounding_objects, object_masks, is_anchor, is_fallback,
        )
        assert self.output_dir is not None
        target_dir = self.output_dir if self.save_images else self._video_frame_dir
        assert target_dir is not None
        target = target_dir / f"frame_{frame_index:06d}.jpg"
        if not _write_jpeg(target, canvas):
            raise VisualizationError(f"SAM2 可视化图片写入失败: {target}")

    def close(self) -> None:
        """按最终帧图编码镜头调试视频，并清理仅供编码使用的隐藏图片。"""

        if not self.active or not self.write_video:
            return
        assert self.output_dir is not None
        frame_dir = self.output_dir if self.save_images else self._video_frame_dir
        assert frame_dir is not None
        images = sorted(frame_dir.glob("frame_*.jpg"))
        if not images:
            if self._video_frame_dir is not None:
                shutil.rmtree(self._video_frame_dir, ignore_errors=True)
            return
        first = _read_image(images[0])
        if first is None:
            raise VisualizationError(f"无法读取可视化视频首帧: {images[0]}")
        height, width = first.shape[:2]
        output_fps = min(self.source_fps, float(self.config.get("sample_fps", 2.0)))
        # VideoWriter 在 Windows 下同样可能无法创建中文路径。先编码到系统临时
        # ASCII 路径，完成后再由 pathlib 移动到正式输出目录。
        descriptor, temporary_name = tempfile.mkstemp(prefix="stage4-sam2-viz-", suffix=".mp4")
        os.close(descriptor)
        temporary_video = Path(temporary_name)
        temporary_video.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(temporary_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(0.1, output_fps),
            (width, height),
        )
        if not writer.isOpened():
            temporary_video.unlink(missing_ok=True)
            raise VisualizationError(f"无法创建 SAM2 可视化视频: {self.output_dir / 'propagation.mp4'}")
        try:
            for path in images:
                image = _read_image(path)
                if image is None:
                    raise VisualizationError(f"无法读取可视化视频帧: {path}")
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(image)
        finally:
            writer.release()
        final_video = self.output_dir / "propagation.mp4"
        final_video.unlink(missing_ok=True)
        shutil.move(str(temporary_video), str(final_video))
        if self._video_frame_dir is not None and not bool(self.config.get("annotate_composition", True)):
            shutil.rmtree(self._video_frame_dir, ignore_errors=True)
