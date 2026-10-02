"""Stage 3.5 Qwen 版本化离线可视化。

本模块只读取 Stage 1 和 Stage 3.5 已完成产物，不参与模型推理，也不会修改任何
上游文件。它复用 Stage 3.5 的精确取帧实现，在对应原始视频帧上绘制：

* ``stage3.5.v1`` 的单主体点 ``S``；
* 新版精简协议中的目标点 ``T#``（主体点和焦点不同时区分 ``S#`` 与 ``F#``）；
* Qwen 原始构图锚点 ``A``、合法化中心 ``R``及对应裁剪框；
* 右侧证据栏中的主次关系、坐标、置信度、语义和相邻帧位移；
* 区间级推荐中心稳定性统计。

输出按 ``<output>/<video_id>/<interval_id>/`` 组织，并为每个视频写入
``manifest.json``，便于未来把其他阶段的可视化接入同一顶层工具。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from video_highlight.common.atomic_io import read_jsonl, write_json
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.stage3_5_subject_point.frame_sampler import SampledFrame, sample_interval


_COLORS: tuple[tuple[int, int, int], ...] = (
    (66, 211, 255),
    (80, 220, 120),
    (255, 145, 70),
    (210, 100, 255),
    (255, 210, 80),
    (90, 170, 255),
    (220, 220, 70),
    (180, 120, 255),
)

_RECOMMENDED_COLOR = (180, 0, 180)
_RAW_ANCHOR_COLOR = (0, 140, 255)
_FOCUS_COLOR = (0, 215, 255)
_PRIMARY_COLOR = (0, 255, 255)
_TEXT_COLOR = (225, 225, 225)
_V1_SCHEMA_VERSION = "stage3.5.v1"

_DEFAULT_VISUALIZATION_CONFIG: dict[str, Any] = {
    "sidebar_width": 460,
    "sidebar_font_size": 18,
    "sidebar_line_height": 25,
    "sidebar_max_targets": 8,
    "font_path": "",
    "marker_radius": 10,
    "marker_border_thickness": 2,
    "primary_ring_thickness": 2,
    "recommended_marker_radius": 11,
    "recommended_box_thickness": 2,
    "raw_anchor_marker_radius": 11,
    "raw_anchor_link_thickness": 1,
    "focus_link_thickness": 1,
    "group_span_thickness": 1,
    "draw_subject_points": True,
    "draw_focus_points": True,
    "draw_recommended_crop": True,
    "draw_raw_crop_anchor": True,
    "draw_focus_links": True,
}


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"缺少 JSON 文件: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 根节点不是对象: {path}")
    return value


def _safe_component(value: str) -> str:
    """将产物 ID 转成 Windows/Unix 都可用的目录名。"""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return (cleaned or "unnamed")[:160]


def _ascii(value: Any, maximum: int = 96) -> str:
    """OpenCV Hershey 字体只可靠支持 ASCII，非 ASCII 用问号显式替代。"""

    return str(value).encode("ascii", errors="replace").decode("ascii")[:maximum]


def _target_color(target_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(target_id.encode("utf-8")).digest()
    return _COLORS[int.from_bytes(digest[:2], "big") % len(_COLORS)]


def _normalized_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(axis) and 0.0 <= axis <= 1.0 for axis in (x, y)):
        return None
    return x, y


def _millipoint(value: Any) -> tuple[float, float] | None:
    """解析模型原始 0..1000 整数坐标；不使用旧坐标格式的启发式判断。"""

    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(axis) and 0.0 <= axis <= 1000.0 for axis in (x, y)):
        return None
    return x / 1000.0, y / 1000.0


def _load_raw_crop_anchors(
    stage3_5_video: Path,
) -> dict[tuple[str, int], tuple[float, float] | None]:
    """按 response_id 关联原始唯一构图中心；兼容 v3 crop_anchor。"""

    requests_path = stage3_5_video / "requests.jsonl"
    responses_path = stage3_5_video / "raw_responses.jsonl"
    if not requests_path.is_file() or not responses_path.is_file():
        return {}

    raw_by_response: dict[str, tuple[float, float] | None] = {}
    for response in read_jsonl(responses_path):
        response_id = str(response.get("response_id") or "")
        if not response_id:
            continue
        try:
            payload = json.loads(str(response.get("text", "")))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "predictions" in payload:
            continue
        raw_by_response[response_id] = _millipoint(
            payload.get("composition_center", payload.get("crop_anchor"))
        )

    anchors: dict[tuple[str, int], tuple[float, float] | None] = {}
    for request in read_jsonl(requests_path):
        response_id = str(request.get("response_id") or "")
        if response_id not in raw_by_response:
            continue
        timeline = request.get("timeline")
        if not isinstance(timeline, list) or len(timeline) != 1:
            continue
        sample = timeline[0]
        if not isinstance(sample, dict):
            continue
        anchors[
            (str(request.get("interval_id", "")), int(sample.get("sample_index", -1)))
        ] = raw_by_response[response_id]
    return anchors


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, quality))]
    )
    if not ok:
        raise ArtifactValidationError(f"可视化 JPEG 编码失败: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as error:
        raise ArtifactValidationError(f"可视化 JPEG 写入失败: {path}: {error}") from error


def _visualization_config(value: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(_DEFAULT_VISUALIZATION_CONFIG)
    if isinstance(value, dict):
        config.update(value)
    return config


def _pixel_point(
    point: tuple[float, float], width: int, height: int
) -> tuple[int, int]:
    return (
        min(width - 1, max(0, int(round(point[0] * (width - 1))))),
        min(height - 1, max(0, int(round(point[1] * (height - 1))))),
    )


def _draw_round_marker(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    border_thickness: int,
    text: str,
    fill_color: tuple[int, int, int],
) -> None:
    """沿用 Stage 4 的白边实心圆标记，但只在圆内保留短编号。"""

    radius = max(5, int(radius))
    border_thickness = max(1, int(border_thickness))
    cv2.circle(image, center, radius, fill_color, -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (255, 255, 255), border_thickness, cv2.LINE_AA)
    font_scale = max(0.26, min(0.58, radius / 11.0 * 0.42))
    font_thickness = max(1, border_thickness)
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    if text_width > radius * 1.65:
        font_scale = max(0.20, font_scale * radius * 1.65 / text_width)
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
    blue, green, red = fill_color
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    text_color = (0, 0, 0) if luminance >= 150 else (255, 255, 255)
    origin = (
        int(round(center[0] - text_width * 0.5)),
        int(round(center[1] + text_height * 0.5)),
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        text_color, font_thickness, cv2.LINE_AA,
    )


def _font_candidates(configured: str) -> list[Path]:
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        Path(value)
        for value in (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    return candidates


def _append_sidebar(
    image: np.ndarray,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
    config: dict[str, Any],
) -> np.ndarray:
    """追加与 Stage 4 相同的信息侧栏；可用 Pillow 时完整显示中文。"""

    panel_width = max(300, int(config.get("sidebar_width", 460)))
    panel = np.full((image.shape[0], panel_width, 3), 18, dtype=np.uint8)
    font_size = max(10, int(config.get("sidebar_font_size", 18)))
    line_height = max(font_size + 3, int(config.get("sidebar_line_height", 25)))
    maximum = max(1, (image.shape[0] - 10) // line_height)
    visible_lines = list(lines)
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = None
        for candidate in _font_candidates(str(config.get("font_path", ""))):
            if candidate.is_file():
                font = ImageFont.truetype(str(candidate), font_size)
                break
        if font is None:
            font = ImageFont.load_default()
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        drawer = ImageDraw.Draw(pil_image)
        wrapped_lines: list[tuple[str, tuple[int, int, int]]] = []
        maximum_text_width = panel_width - 24
        for text, color in visible_lines:
            current = ""
            for character in str(text):
                candidate = current + character
                bounds = drawer.textbbox((0, 0), candidate, font=font)
                if current and bounds[2] - bounds[0] > maximum_text_width:
                    wrapped_lines.append((current, color))
                    current = character
                else:
                    current = candidate
            wrapped_lines.append((current, color))
        y = 4
        for text, bgr in wrapped_lines[:maximum]:
            drawer.text((12, y), text, font=font, fill=(bgr[2], bgr[1], bgr[0]))
            y += line_height
        panel = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except (ImportError, OSError):
        scale = max(0.35, font_size / 32.0)
        y = line_height
        for text, color in visible_lines[:maximum]:
            cv2.putText(
                panel, _ascii(text, 64), (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA,
            )
            y += line_height
    cv2.line(panel, (0, 0), (0, image.shape[0] - 1), (95, 95, 95), 2)
    return np.hstack((image, panel))


def _crop_geometry(
    metadata: dict[str, Any] | None,
    rendered_width: int,
    rendered_height: int,
) -> dict[str, Any]:
    metadata = metadata or {}
    source_width = float(metadata.get("display_width", metadata.get("width", rendered_width)))
    source_height = float(metadata.get("display_height", metadata.get("height", rendered_height)))
    ratio = metadata.get("targetRatioWH", [16, 9])
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 2:
        ratio = [16, 9]
    try:
        target_width, target_height = float(ratio[0]), float(ratio[1])
    except (TypeError, ValueError):
        target_width, target_height = 16.0, 9.0
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        source_width, source_height = float(rendered_width), float(rendered_height)
        target_width, target_height = 16.0, 9.0
    if source_width / source_height >= target_width / target_height:
        crop_height = source_height
        crop_width = crop_height * target_width / target_height
    else:
        crop_width = source_width
        crop_height = crop_width * target_height / target_width
    normalized_width = min(1.0, crop_width / source_width)
    normalized_height = min(1.0, crop_height / source_height)
    return {
        "target_ratio": (target_width, target_height),
        "normalized_size": (normalized_width, normalized_height),
        "rendered_size": (
            max(1, int(round(normalized_width * rendered_width))),
            max(1, int(round(normalized_height * rendered_height))),
        ),
        "legal_x": (normalized_width * 0.5, 1.0 - normalized_width * 0.5),
        "legal_y": (normalized_height * 0.5, 1.0 - normalized_height * 0.5),
    }


def _point_text(value: tuple[float, float] | None) -> str:
    return "null" if value is None else f"({value[0]:.3f},{value[1]:.3f})"


def _recommended_delta(
    current: tuple[float, float] | None,
    previous_observation: dict[str, Any] | None,
) -> float | None:
    previous = _normalized_point(
        previous_observation.get("recommended_crop_center")
        if isinstance(previous_observation, dict) else None
    )
    if current is None or previous is None:
        return None
    return math.hypot(current[0] - previous[0], current[1] - previous[1])


def _observation_schema_version(
    rows: Sequence[dict[str, Any]], fallback: str | None = None
) -> str:
    """返回单个观察文件的版本；不允许把不同协议混在一次可视化中。"""

    if not rows:
        if fallback:
            return fallback
        raise ArtifactValidationError("Stage 3.5 观察文件为空")
    versions = {str(row.get("schema_version", "")).strip() for row in rows}
    if "" in versions:
        raise ArtifactValidationError("Stage 3.5 观察缺少 schema_version")
    if len(versions) != 1:
        raise ArtifactValidationError(
            f"Stage 3.5 观察包含多个 schema_version: {sorted(versions)}"
        )
    return next(iter(versions))


def _load_versioned_observations(
    stage3_5_video: Path,
) -> tuple[list[dict[str, Any]], str, str]:
    """按产物版本选择观察文件，当前额外兼容 ``stage3.5.v1``。"""

    observations_path = stage3_5_video / "subject_observations.jsonl"
    if not observations_path.is_file():
        observations_path = stage3_5_video / "subject_points.jsonl"
    if not observations_path.is_file():
        raise ArtifactValidationError(
            "Stage 3.5 缺少 subject_observations.jsonl 或 subject_points.jsonl: "
            f"{stage3_5_video}"
        )
    rows = read_jsonl(observations_path)
    run_schema_version = ""
    if not rows:
        stage3_5_root = stage3_5_video.parent.parent
        for filename in ("run_manifest.json", "resolved_config.json"):
            candidate = stage3_5_root / filename
            if candidate.is_file():
                run_schema_version = str(
                    _read_object(candidate).get("schema_version", "")
                ).strip()
                if run_schema_version:
                    break
    schema_version = _observation_schema_version(
        rows,
        run_schema_version
        or (
            _V1_SCHEMA_VERSION
            if observations_path.name == "subject_points.jsonl"
            else None
        ),
    )
    if observations_path.name == "subject_points.jsonl" and schema_version != _V1_SCHEMA_VERSION:
        raise ArtifactValidationError(
            "subject_points.jsonl 目前只支持 stage3.5.v1，"
            f"实际为 {schema_version}: {observations_path}"
        )
    return rows, schema_version, observations_path.name


def render_stage3_5_v1_observation(
    frame_bgr: np.ndarray,
    observation: dict[str, Any],
    *,
    interval_metadata: dict[str, Any] | None = None,
    previous_observation: dict[str, Any] | None = None,
    visualization_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """绘制 ``stage3.5.v1`` 的单主体点，不虚构后续版本的构图字段。"""

    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ArtifactValidationError("可视化输入帧必须是 HxWx3 BGR 图像")
    config = _visualization_config(visualization_config)
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    subject = str(observation.get("subject", ""))
    visibility = str(observation.get("visibility", "unknown"))
    point = _normalized_point(observation.get("subject_point"))
    color = _target_color(subject or "subject")
    if point is not None and bool(config.get("draw_subject_points", True)):
        point_xy = _pixel_point(point, width, height)
        radius = max(5, int(config.get("marker_radius", 10)))
        thickness = max(1, int(config.get("marker_border_thickness", 2)))
        if visibility == "visible":
            _draw_round_marker(canvas, point_xy, radius, thickness, "S", color)
        else:
            cv2.circle(canvas, point_xy, radius, color, thickness, cv2.LINE_AA)
            cv2.drawMarker(
                canvas, point_xy, color, cv2.MARKER_TILTED_CROSS,
                radius * 2, thickness, cv2.LINE_AA,
            )

    previous_point = _normalized_point(
        previous_observation.get("subject_point")
        if isinstance(previous_observation, dict) else None
    )
    delta = (
        math.hypot(point[0] - previous_point[0], point[1] - previous_point[1])
        if point is not None and previous_point is not None else None
    )
    confidence = float(observation.get("confidence", 0.0) or 0.0)
    detail_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("STAGE 3.5 V1 SUBJECT POINT", (255, 255, 255)),
        (
            f"frame={int(observation.get('frame', -1))} "
            f"sample={int(observation.get('sample_index', -1))} "
            f"time={float(observation.get('timestamp_sec', 0.0)):.3f}s",
            _TEXT_COLOR,
        ),
        (
            f"interval={observation.get('interval_id', '')} "
            f"status={observation.get('status', 'unknown')}",
            _TEXT_COLOR,
        ),
        (f"subject={subject}", _PRIMARY_COLOR),
        (f"visibility={visibility}", color),
        (f"S={_point_text(point)}  confidence={confidence:.2f}", color),
        (f"delta_from_previous={delta:.4f}" if delta is not None else "delta_from_previous=n/a", _TEXT_COLOR),
        (f"coordinate_space={observation.get('coordinate_space', '')}", _TEXT_COLOR),
        ("S=Qwen subject point", (180, 180, 180)),
    ]
    reason = str(observation.get("reason", "")).strip()
    interval_reason = str((interval_metadata or {}).get("reason", "")).strip()
    if reason:
        detail_lines.append((f"Qwen reason: {reason}", (210, 210, 255)))
    if interval_reason and interval_reason != reason:
        detail_lines.append((f"Stage 3 reason: {interval_reason}", (210, 255, 210)))
    if point is None:
        detail_lines.append(("NO VALID QWEN SUBJECT POINT", (80, 80, 255)))
    return _append_sidebar(canvas, detail_lines, config)


def render_stage3_5_observation(
    frame_bgr: np.ndarray,
    observation: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    interval_metadata: dict[str, Any] | None = None,
    previous_observation: dict[str, Any] | None = None,
    visualization_config: dict[str, Any] | None = None,
    draw_group_center: bool = False,
) -> np.ndarray:
    """按 Stage 4 的主画面 + 右侧证据栏样式绘制一条 Stage 3.5 观察。"""

    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ArtifactValidationError("可视化输入帧必须是 HxWx3 BGR 图像")
    config = _visualization_config(visualization_config)
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    targets = observation.get("targets", [])
    if not isinstance(targets, list):
        raise ArtifactValidationError("Stage 3.5 observation.targets 必须是数组")

    marker_radius = max(5, int(config.get("marker_radius", 10)))
    border_thickness = max(1, int(config.get("marker_border_thickness", 2)))
    maximum_targets = max(0, int(config.get("sidebar_max_targets", 8)))
    primary_ids = {str(value) for value in observation.get("primary_target_ids", [])}
    primary_focus_pixels: list[tuple[int, int]] = []
    detail_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("STAGE 3.5 QWEN SEMANTICS + CENTER", (255, 255, 255)),
        (
            f"frame={int(observation.get('frame', -1))} "
            f"sample={int(observation.get('sample_index', -1))} "
            f"time={float(observation.get('timestamp_sec', 0.0)):.3f}s",
            _TEXT_COLOR,
        ),
        (
            f"interval={observation.get('interval_id', '')} "
            f"status={observation.get('status', 'unknown')}",
            _TEXT_COLOR,
        ),
        (f"highlight_subject={observation.get('subject', '')}", _PRIMARY_COLOR),
        (
            f"mode={observation.get('group_mode', 'unknown')} / "
            f"{observation.get('composition_mode', 'unknown')}",
            _TEXT_COLOR,
        ),
        (f"primary={list(primary_ids)}", _PRIMARY_COLOR),
    ]

    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id") or f"target_{index}")
        is_primary = target_id in primary_ids or target.get("role") == "primary"
        color = _target_color(target_id)
        visibility = str(target.get("visibility", "visible"))
        subject = _normalized_point(target.get("subject_point"))
        focus = _normalized_point(target.get("focus_point"))
        subject_xy = _pixel_point(subject, width, height) if subject is not None else None
        focus_xy = _pixel_point(focus, width, height) if focus is not None else None
        if (
            subject_xy is not None and focus_xy is not None
            and bool(config.get("draw_focus_links", True))
            and subject_xy != focus_xy
        ):
            cv2.line(
                canvas, subject_xy, focus_xy, color,
                max(1, int(config.get("focus_link_thickness", 1))), cv2.LINE_AA,
            )
        same_point = subject_xy is not None and subject_xy == focus_xy
        if subject_xy is not None and bool(config.get("draw_subject_points", True)):
            _draw_round_marker(
                canvas, subject_xy, marker_radius, border_thickness,
                f"T{index + 1}" if same_point and bool(config.get("draw_focus_points", True)) else f"S{index + 1}",
                color,
            )
        if focus_xy is not None and bool(config.get("draw_focus_points", True)) and not same_point:
            _draw_round_marker(
                canvas, focus_xy, marker_radius, border_thickness,
                f"F{index + 1}", _FOCUS_COLOR,
            )
        if is_primary:
            primary_xy = focus_xy or subject_xy
            if primary_xy is not None:
                cv2.circle(
                    canvas, primary_xy, marker_radius + 5, _PRIMARY_COLOR,
                    max(1, int(config.get("primary_ring_thickness", 2))), cv2.LINE_AA,
                )
                primary_focus_pixels.append(primary_xy)
        confidence = float(target.get("confidence", 0.0) or 0.0)
        importance = float(target.get("importance", 0.5) or 0.0)
        phrase = str(target.get("grounding_phrase", ""))
        if index < maximum_targets:
            detail_lines.append((
                f"{index + 1} [{'P' if is_primary else 'S'}] {target_id}  {visibility}",
                color,
            ))
            if subject is not None or focus is not None:
                # 仅旧 v2/v3 产物存在目标坐标；v4 只展示语义。
                detail_lines.append((
                    (
                        f"  T={_point_text(subject)}"
                        if same_point
                        else f"  S={_point_text(subject)}  F={_point_text(focus)}"
                    ),
                    color,
                ))
            detail_lines.append((
                f"  imp={importance:.2f} conf={confidence:.2f} phrase={phrase}",
                color,
            ))
            description = str(target.get("description", "")).strip()
            if description and description != phrase:
                detail_lines.append((f"  description={description}", color))

    if primary_focus_pixels and draw_group_center:
        xs = [point[0] for point in primary_focus_pixels]
        ys = [point[1] for point in primary_focus_pixels]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        padding = marker_radius + 5
        cv2.rectangle(
            canvas,
            (max(0, x1 - padding), max(0, y1 - padding)),
            (min(width - 1, x2 + padding), min(height - 1, y2 + padding)),
            (235, 235, 235),
            max(1, int(config.get("group_span_thickness", 1))),
            cv2.LINE_AA,
        )

    geometry = _crop_geometry(metadata, width, height)
    raw_anchor_present = bool(observation.get("_raw_crop_anchor_present", False))
    raw_anchor = _normalized_point(observation.get("_raw_crop_anchor"))
    recommended = _normalized_point(observation.get("recommended_crop_center"))
    recommended_confidence = float(observation.get("recommended_crop_confidence", 0.0) or 0.0)
    raw_anchor_xy = _pixel_point(raw_anchor, width, height) if raw_anchor is not None else None
    recommended_xy = (
        _pixel_point(recommended, width, height) if recommended is not None else None
    )
    if (
        raw_anchor_xy is not None
        and recommended_xy is not None
        and bool(config.get("draw_raw_crop_anchor", True))
        and raw_anchor_xy != recommended_xy
    ):
        cv2.line(
            canvas,
            raw_anchor_xy,
            recommended_xy,
            _RAW_ANCHOR_COLOR,
            max(1, int(config.get("raw_anchor_link_thickness", 1))),
            cv2.LINE_AA,
        )
    geometry_legal = False
    contract_valid = (
        (recommended is None and not primary_ids and recommended_confidence == 0.0)
        or (recommended is not None and bool(primary_ids) and recommended_confidence > 0.0)
    )
    if recommended is not None:
        geometry_legal = (
            geometry["legal_x"][0] - 1e-6 <= recommended[0] <= geometry["legal_x"][1] + 1e-6
            and geometry["legal_y"][0] - 1e-6 <= recommended[1] <= geometry["legal_y"][1] + 1e-6
        )
        if bool(config.get("draw_recommended_crop", True)):
            crop_width, crop_height = geometry["rendered_size"]
            x1 = int(round(recommended_xy[0] - crop_width * 0.5))
            y1 = int(round(recommended_xy[1] - crop_height * 0.5))
            x2 = x1 + crop_width - 1
            y2 = y1 + crop_height - 1
            cv2.rectangle(
                canvas,
                (max(0, x1), max(0, y1)),
                (min(width - 1, x2), min(height - 1, y2)),
                _RECOMMENDED_COLOR if geometry_legal and contract_valid else (0, 0, 255),
                max(1, int(config.get("recommended_box_thickness", 2))),
                cv2.LINE_AA,
            )
        recommended_radius = max(5, int(config.get("recommended_marker_radius", 11)))
        overlaps_subject = any(
            math.hypot(recommended_xy[0] - point[0], recommended_xy[1] - point[1])
            <= marker_radius * 1.5
            for point in primary_focus_pixels
        )
        if overlaps_subject:
            cv2.circle(
                canvas, recommended_xy, marker_radius + 9, _RECOMMENDED_COLOR,
                border_thickness + 1, cv2.LINE_AA,
            )
        else:
            _draw_round_marker(
                canvas, recommended_xy, recommended_radius, border_thickness,
                "C", _RECOMMENDED_COLOR,
            )
    if raw_anchor_xy is not None and bool(config.get("draw_raw_crop_anchor", True)):
        raw_radius = max(5, int(config.get("raw_anchor_marker_radius", 11)))
        if (
            recommended_xy is not None
            and math.hypot(
                raw_anchor_xy[0] - recommended_xy[0],
                raw_anchor_xy[1] - recommended_xy[1],
            ) <= max(raw_radius, int(config.get("recommended_marker_radius", 11)))
        ):
            cv2.circle(
                canvas,
                raw_anchor_xy,
                max(raw_radius, int(config.get("recommended_marker_radius", 11))) + 6,
                _RAW_ANCHOR_COLOR,
                border_thickness + 1,
                cv2.LINE_AA,
            )
        else:
            _draw_round_marker(
                canvas,
                raw_anchor_xy,
                raw_radius,
                border_thickness,
                "Q",
                _RAW_ANCHOR_COLOR,
            )

    delta = _recommended_delta(recommended, previous_observation)
    anchor_to_recommended = (
        math.hypot(raw_anchor[0] - recommended[0], raw_anchor[1] - recommended[1])
        if raw_anchor is not None and recommended is not None else None
    )
    ratio_width, ratio_height = geometry["target_ratio"]
    crop_width, crop_height = geometry["rendered_size"]
    detail_lines[6:6] = [
        (
            f"Q(raw)={_point_text(raw_anchor)}"
            if raw_anchor_present else "Q(raw)=unavailable (legacy output)",
            _RAW_ANCHOR_COLOR,
        ),
        (
            f"C(legal)={_point_text(recommended)} conf={recommended_confidence:.3f} "
            f"geometry={'n/a' if recommended is None else ('legal' if geometry_legal else 'illegal')}",
            _RECOMMENDED_COLOR if geometry_legal or recommended is None else (80, 80, 255),
        ),
        (
            f"Q_to_C={'n/a' if anchor_to_recommended is None else f'{anchor_to_recommended:.4f}'}",
            _RAW_ANCHOR_COLOR,
        ),
        (
            f"C_contract={'valid' if contract_valid else 'INVALID'}",
            _TEXT_COLOR if contract_valid else (80, 80, 255),
        ),
        (f"C_delta={'n/a' if delta is None else f'{delta:.4f}'}", _RECOMMENDED_COLOR),
        (
            f"crop={crop_width}x{crop_height} ratio={ratio_width:g}:{ratio_height:g}",
            _RECOMMENDED_COLOR,
        ),
        (
            f"legal_x=[{geometry['legal_x'][0]:.3f},{geometry['legal_x'][1]:.3f}] "
            f"legal_y=[{geometry['legal_y'][0]:.3f},{geometry['legal_y'][1]:.3f}]",
            _TEXT_COLOR,
        ),
        ("Q=raw composition center  C=legal crop center", (180, 180, 180)),
    ]
    phrases = observation.get("grounding_phrases", [])
    if isinstance(phrases, list):
        detail_lines.append((f"grounding={', '.join(str(value) for value in phrases)}", (190, 255, 255)))
    qwen_reason = str(observation.get("reason", "")).strip()
    stage3_reason = str((interval_metadata or {}).get("reason", "")).strip()
    if qwen_reason:
        detail_lines.append((f"Qwen reason: {qwen_reason}", (210, 210, 255)))
    if stage3_reason and stage3_reason != qwen_reason:
        detail_lines.append((f"Stage 3 reason: {stage3_reason}", (210, 255, 210)))
    if not targets:
        detail_lines.append(("NO QWEN TARGET", (80, 80, 255)))
    if not contract_valid:
        detail_lines.append(("WARNING: primary/recommended/confidence inconsistent", (80, 80, 255)))
    if len(targets) > maximum_targets:
        detail_lines.append((f"... {len(targets) - maximum_targets} targets omitted in sidebar", (80, 80, 255)))
    return _append_sidebar(canvas, detail_lines, config)


def _source_path(
    metadata: dict[str, Any], video_id: str, paths_config: dict[str, Any]
) -> Path:
    candidate = Path(str(metadata.get("source_path", "")))
    if candidate.is_file():
        return candidate
    video_root = paths_config.get("video_root")
    if video_root:
        fallback = Path(str(video_root)) / f"{video_id}.mp4"
        if fallback.is_file():
            return fallback
    raise ArtifactValidationError(
        f"源视频不存在: metadata.source_path={candidate}; video_id={video_id}"
    )


def _validate_observations(
    rows: list[dict[str, Any]], video_id: str, frame_count: int,
    *,
    schema_version: str | None = None,
) -> None:
    schema_version = _observation_schema_version(rows, schema_version)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        interval_id = str(row.get("interval_id", ""))
        sample_index = int(row.get("sample_index", -1))
        frame = int(row.get("frame", -1))
        key = interval_id, sample_index
        if str(row.get("video_id")) != video_id:
            raise ArtifactValidationError(f"观察 video_id 与目录不一致: {row.get('video_id')}")
        if not interval_id or key in seen:
            raise ArtifactValidationError(f"观察 interval/sample 重复或为空: {key}")
        if not 0 <= frame < frame_count:
            raise ArtifactValidationError(f"观察帧越界: {interval_id}/{frame}")
        seen.add(key)
        if schema_version == _V1_SCHEMA_VERSION:
            point = row.get("subject_point")
            if point is not None and _normalized_point(point) is None:
                raise ArtifactValidationError(f"观察中心点非法: {key}")
            try:
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError) as error:
                raise ArtifactValidationError(f"观察置信度非法: {key}") from error
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ArtifactValidationError(f"观察置信度越界: {key}")
            coordinate_space = str(row.get("coordinate_space", "normalized_xy"))
            if coordinate_space != "normalized_xy":
                raise ArtifactValidationError(
                    f"stage3.5.v1 仅支持 normalized_xy 坐标: {key}/{coordinate_space}"
                )
            continue
        targets = row.get("targets")
        if not isinstance(targets, list):
            raise ArtifactValidationError(f"观察 targets 非数组: {key}")
        recommended = row.get("recommended_crop_center")
        if recommended is not None and _normalized_point(recommended) is None:
            raise ArtifactValidationError(f"观察推荐裁剪中心非法: {key}")
        try:
            recommended_confidence = float(row.get("recommended_crop_confidence", 0.0))
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(f"观察推荐裁剪置信度非法: {key}") from error
        if not math.isfinite(recommended_confidence) or not 0.0 <= recommended_confidence <= 1.0:
            raise ArtifactValidationError(f"观察推荐裁剪置信度越界: {key}")
        if recommended is None and recommended_confidence != 0.0:
            raise ArtifactValidationError(f"推荐中心为空但置信度非零: {key}")
        target_ids: set[str] = set()
        for target in targets:
            if not isinstance(target, dict):
                raise ArtifactValidationError(f"观察 target 非对象: {key}")
            target_id = str(target.get("target_id", "")).strip()
            if not target_id or target_id in target_ids:
                raise ArtifactValidationError(f"观察 target_id 为空或重复: {key}/{target_id}")
            target_ids.add(target_id)
            if schema_version == "stage3.5.v4" and (
                "subject_point" in target or "focus_point" in target
            ):
                raise ArtifactValidationError(
                    f"stage3.5.v4 target 不允许包含坐标: {key}/{target_id}"
                )
            point = target.get("subject_point")
            if point is not None and _normalized_point(point) is None:
                raise ArtifactValidationError(f"观察中心点非法: {key}/{target.get('target_id')}")
            focus = target.get("focus_point", point)
            if focus is not None and _normalized_point(focus) is None:
                raise ArtifactValidationError(f"观察构图点非法: {key}/{target.get('target_id')}")
        primary_ids = row.get("primary_target_ids", [])
        if not isinstance(primary_ids, list):
            raise ArtifactValidationError(f"观察 primary_target_ids 非数组: {key}")
        dangling = [str(value) for value in primary_ids if str(value) not in target_ids]
        if dangling:
            raise ArtifactValidationError(f"观察 primary_target_ids 引用不存在目标: {key}/{dangling}")


def _interval_v1_statistics(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [_normalized_point(row.get("subject_point")) for row in observations]
    valid_points = [point for point in points if point is not None]
    jumps = [
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(points, points[1:])
        if previous is not None and current is not None
    ]
    return {
        "subject_point_count": len(valid_points),
        "subject_point_null_count": len(points) - len(valid_points),
        "subject_point_mean_jump": sum(jumps) / len(jumps) if jumps else None,
        "subject_point_max_jump": max(jumps) if jumps else None,
    }


def _interval_statistics(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    centers = [_normalized_point(row.get("recommended_crop_center")) for row in observations]
    valid_centers = [point for point in centers if point is not None]
    raw_centers = [_normalized_point(row.get("_raw_crop_anchor")) for row in observations]
    valid_raw_centers = [point for point in raw_centers if point is not None]
    jumps = [
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(centers, centers[1:])
        if previous is not None and current is not None
    ]
    target_counts = [
        len(row.get("targets", [])) if isinstance(row.get("targets"), list) else 0
        for row in observations
    ]
    primary_sequences = [
        tuple(str(value) for value in row.get("primary_target_ids", []))
        if isinstance(row.get("primary_target_ids"), list) else ()
        for row in observations
    ]
    recommended_without_primary_count = sum(
        center is not None and not primary
        for center, primary in zip(centers, primary_sequences)
    )
    primary_without_recommended_count = sum(
        center is None and bool(primary)
        for center, primary in zip(centers, primary_sequences)
    )
    return {
        "raw_crop_anchor_count": len(valid_raw_centers),
        "raw_crop_anchor_exact_middle_count": sum(
            abs(point[0] - 0.5) <= 1e-9 and abs(point[1] - 0.5) <= 1e-9
            for point in valid_raw_centers
        ),
        "raw_anchor_legalization_change_count": sum(
            raw is not None
            and legal is not None
            and math.hypot(raw[0] - legal[0], raw[1] - legal[1]) > 1e-9
            for raw, legal in zip(raw_centers, centers)
        ),
        "recommended_center_count": len(valid_centers),
        "recommended_center_null_count": len(centers) - len(valid_centers),
        "recommended_center_exact_middle_count": sum(
            abs(point[0] - 0.5) <= 1e-9 and abs(point[1] - 0.5) <= 1e-9
            for point in valid_centers
        ),
        "recommended_center_mean_jump": (
            sum(jumps) / len(jumps) if jumps else None
        ),
        "recommended_center_max_jump": max(jumps) if jumps else None,
        "target_count_change_count": sum(
            current != previous
            for previous, current in zip(target_counts, target_counts[1:])
        ),
        "primary_id_change_count": sum(
            current != previous
            for previous, current in zip(primary_sequences, primary_sequences[1:])
        ),
        "recommended_without_primary_count": recommended_without_primary_count,
        "primary_without_recommended_count": primary_without_recommended_count,
    }


def _open_video_writer(
    size: tuple[int, int], fps: float
) -> tuple[cv2.VideoWriter, Path]:
    descriptor, name = tempfile.mkstemp(prefix="stage3-5-qwen-viz-", suffix=".mp4")
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(fps)),
        size,
    )
    if not writer.isOpened():
        temporary.unlink(missing_ok=True)
        raise ArtifactValidationError("无法创建 Stage 3.5 Qwen 中心点预览视频")
    return writer, temporary


def visualize_stage3_5_video(
    stage1_dir: str | Path,
    stage3_5_dir: str | Path,
    output_dir: str | Path,
    video_id: str,
    *,
    paths_config: dict[str, Any] | None = None,
    save_images: bool = True,
    write_video: bool = False,
    preview_fps: float | None = None,
    max_side: int = 1600,
    jpeg_quality: int = 92,
    decoder: str = "auto",
    ffmpeg_bin: str = "ffmpeg",
    draw_group_center: bool = False,
    visualization_config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """可视化一个视频的全部 Stage 3.5 Qwen 观察与推荐构图。"""

    if not save_images and not write_video:
        raise ArtifactValidationError("save_images 和 write_video 不能同时关闭")
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage3_5_video = Path(stage3_5_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage3_5_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 3.5 视频没有成功标记: {stage3_5_video}")
    metadata = _read_object(stage1_video / "metadata.json")
    rows, schema_version, observation_artifact = _load_versioned_observations(
        stage3_5_video
    )
    is_v1 = schema_version == _V1_SCHEMA_VERSION
    if not is_v1:
        raw_anchors = _load_raw_crop_anchors(stage3_5_video)
        for row in rows:
            key = (str(row.get("interval_id", "")), int(row.get("sample_index", -1)))
            if key in raw_anchors:
                anchor = raw_anchors[key]
                row["_raw_crop_anchor_present"] = True
                row["_raw_crop_anchor"] = list(anchor) if anchor is not None else None
    enriched_path = stage3_5_video / "enriched_intervals.jsonl"
    enriched = read_jsonl(enriched_path) if enriched_path.is_file() else []
    interval_metadata = {str(row.get("interval_id")): row for row in enriched}
    frame_count = int(metadata.get("frame_count", 0))
    fps = float(metadata.get("fps", 0.0))
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ArtifactValidationError(f"Stage 1 metadata 帧数或 FPS 非法: {video_id}")
    _validate_observations(
        rows, video_id, frame_count, schema_version=schema_version
    )
    source = _source_path(metadata, video_id, paths_config or {})

    video_output = Path(output_dir).resolve() / _safe_component(video_id)
    if video_output.exists():
        if not overwrite:
            raise ArtifactValidationError(f"可视化输出已存在，请使用 --overwrite: {video_output}")
        shutil.rmtree(video_output)
    video_output.mkdir(parents=True, exist_ok=False)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["interval_id"])].append(row)
    interval_results: list[dict[str, Any]] = []
    image_count = 0
    point_count = 0
    try:
        for interval_id, observations in grouped.items():
            observations.sort(key=lambda item: (int(item["frame"]), int(item["sample_index"])))
            frames = [int(item["frame"]) for item in observations]
            if len(frames) != len(set(frames)):
                raise ArtifactValidationError(f"同一区间存在重复可视化帧: {interval_id}")
            samples = sample_interval(
                source,
                frames,
                fps,
                jpeg_quality=max(jpeg_quality, 90),
                max_side=max_side,
                decoder=decoder,
                ffmpeg_bin=ffmpeg_bin,
            )
            interval_dir = video_output / _safe_component(interval_id)
            if save_images:
                interval_dir.mkdir(parents=True, exist_ok=True)
            writer: cv2.VideoWriter | None = None
            temporary_video: Path | None = None
            rendered_size: tuple[int, int] | None = None
            output_video = interval_dir / (
                "qwen_subject_points.mp4" if is_v1 else "qwen_composition.mp4"
            )
            try:
                interval_info = interval_metadata.get(interval_id, {})
                for observation_index, (observation, sample) in enumerate(
                    zip(observations, samples, strict=True)
                ):
                    frame = cv2.imdecode(
                        np.frombuffer(sample.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        raise ArtifactValidationError(
                            f"可视化采样帧 JPEG 回读失败: {interval_id}/{sample.frame}"
                        )
                    previous_observation = (
                        observations[observation_index - 1]
                        if observation_index > 0 else None
                    )
                    if is_v1:
                        rendered = render_stage3_5_v1_observation(
                            frame,
                            observation,
                            interval_metadata=interval_info,
                            previous_observation=previous_observation,
                            visualization_config=visualization_config,
                        )
                    else:
                        rendered = render_stage3_5_observation(
                            frame,
                            observation,
                            metadata=metadata,
                            interval_metadata=interval_info,
                            previous_observation=previous_observation,
                            visualization_config=visualization_config,
                            draw_group_center=draw_group_center,
                        )
                    if save_images:
                        image_path = interval_dir / (
                            f"sample_{int(observation['sample_index']):04d}_"
                            f"frame_{sample.frame:08d}.jpg"
                        )
                        _write_jpeg(image_path, rendered, jpeg_quality)
                        image_count += 1
                    if write_video:
                        current_size = rendered.shape[1], rendered.shape[0]
                        if writer is None:
                            interval_dir.mkdir(parents=True, exist_ok=True)
                            sample_fps_key = (
                                "subject_point_sample_fps"
                                if is_v1 else "subject_observation_sample_fps"
                            )
                            effective_fps = preview_fps or float(
                                interval_info.get(sample_fps_key, 2.0)
                            )
                            writer, temporary_video = _open_video_writer(current_size, effective_fps)
                            rendered_size = current_size
                        if current_size != rendered_size:
                            rendered = cv2.resize(rendered, rendered_size, interpolation=cv2.INTER_AREA)
                        writer.write(rendered)
                    if is_v1:
                        point_count += int(
                            _normalized_point(observation.get("subject_point")) is not None
                        )
                    else:
                        point_count += sum(
                            1
                            for target in observation.get("targets", [])
                            if isinstance(target, dict)
                            and _normalized_point(target.get("subject_point")) is not None
                        )
            except Exception:
                if temporary_video is not None:
                    temporary_video.unlink(missing_ok=True)
                raise
            finally:
                if writer is not None:
                    writer.release()
            if temporary_video is not None:
                output_video.unlink(missing_ok=True)
                shutil.move(str(temporary_video), str(output_video))
            interval_results.append(
                {
                    "interval_id": interval_id,
                    "directory": interval_dir.name,
                    "sample_count": len(observations),
                    "first_frame": frames[0] if frames else None,
                    "last_frame": frames[-1] if frames else None,
                    "preview_video": output_video.name if write_video else None,
                    "stability": (
                        _interval_v1_statistics(observations)
                        if is_v1 else _interval_statistics(observations)
                    ),
                }
            )
        manifest = {
            "visualization": (
                "stage3_5_qwen_subject_points"
                if is_v1 else "stage3_5_qwen_composition"
            ),
            "schema_version": schema_version,
            "observation_artifact": observation_artifact,
            "video_id": video_id,
            "source_video": str(source),
            "source_fps": fps,
            "observation_count": len(rows),
            "point_count": point_count,
            "image_count": image_count,
            "interval_count": len(interval_results),
            "save_images": save_images,
            "write_video": write_video,
            "decoder": decoder,
            "max_side": max_side,
            "intervals": interval_results,
        }
        if not is_v1:
            manifest["raw_crop_anchor_count"] = sum(
                _normalized_point(row.get("_raw_crop_anchor")) is not None for row in rows
            )
        write_json(video_output / "manifest.json", manifest)
        return manifest
    except Exception:
        # 不把半成品误认为一次完成的可视化；源阶段产物永远不会被修改。
        shutil.rmtree(video_output, ignore_errors=True)
        raise


def _successful_video_ids(stage3_5_dir: str | Path) -> list[str]:
    videos_dir = Path(stage3_5_dir).resolve() / "videos"
    if not videos_dir.is_dir():
        raise ArtifactValidationError(f"Stage 3.5 videos 目录不存在: {videos_dir}")
    return sorted(
        path.name
        for path in videos_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "_SUCCESS.json").is_file()
    )


def _visualization_summary(
    results: Sequence[dict[str, Any]], failures: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    visualizations = sorted({str(item["visualization"]) for item in results})
    schema_versions = sorted({str(item["schema_version"]) for item in results})
    return {
        "visualization": (
            visualizations[0] if len(visualizations) == 1 else "stage3_5_versioned"
        ),
        "schema_versions": schema_versions,
        "success_count": len(results),
        "failure_count": len(failures),
        "results": list(results),
        "failures": list(failures),
    }


def run_stage3_5_visualization(
    stage1_dir: str | Path,
    stage3_5_dir: str | Path,
    output_dir: str | Path,
    *,
    video_ids: Iterable[str] | None = None,
    limit: int | None = None,
    strict: bool = False,
    **options: Any,
) -> dict[str, Any]:
    """批量生成 Stage 3.5 Qwen 中心点可视化并写入汇总。"""

    available = _successful_video_ids(stage3_5_dir)
    requested = list(dict.fromkeys(str(value) for value in video_ids)) if video_ids else available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ArtifactValidationError(f"Stage 3.5 不存在成功视频: {', '.join(missing)}")
    if limit is not None:
        if limit < 0:
            raise ArtifactValidationError("limit 不能为负数")
        requested = requested[:limit]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for video_id in requested:
        try:
            results.append(
                visualize_stage3_5_video(
                    stage1_dir, stage3_5_dir, output, video_id, **options
                )
            )
        except Exception as error:
            failure = {
                "video_id": video_id,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            if strict:
                write_json(
                    output / "summary.json",
                    _visualization_summary(results, failures),
                )
                raise
    summary = _visualization_summary(results, failures)
    write_json(output / "summary.json", summary)
    return summary
