"""Stage 3.5 Qwen 多目标中心点的离线可视化。

本模块只读取 Stage 1 和 Stage 3.5 已完成产物，不参与模型推理，也不会修改任何
上游文件。它复用 Stage 3.5 的精确取帧实现，在对应原始视频帧上绘制：

* 每个目标的归一化中心点、``target_id``、Grounding 短语与置信度；
* 所有有效点的包围范围和几何组合中心；
* 区间、原始帧号、时间戳、组模式和观察状态。

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
from typing import Any, Iterable

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


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    text = _ascii(text)
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = min(max(0, origin[0]), max(0, image.shape[1] - text_w - 8))
    y = min(max(text_h + 6, origin[1]), image.shape[0] - baseline - 2)
    cv2.rectangle(
        image,
        (x, y - text_h - 6),
        (x + text_w + 8, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return x, y - text_h - 6, x + text_w + 8, y + baseline + 2


def _label_bounds(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    """计算与 :func:`_label` 完全一致的限界矩形，但不执行绘制。"""

    text = _ascii(text)
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = min(max(0, origin[0]), max(0, image.shape[1] - text_w - 8))
    y = min(max(text_h + 6, origin[1]), image.shape[0] - baseline - 2)
    return x, y - text_h - 6, x + text_w + 8, y + baseline + 2


def _overlaps(
    box: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
    padding: int = 4,
) -> bool:
    return any(
        box[0] < other[2] + padding
        and box[2] + padding > other[0]
        and box[1] < other[3] + padding
        and box[3] + padding > other[1]
        for other in occupied
    )


def _target_label_origin(
    image: np.ndarray,
    text: str,
    point: tuple[int, int],
    radius: int,
    scale: float,
    occupied: list[tuple[int, int, int, int]],
    fallback_index: int,
) -> tuple[int, int]:
    """在目标点四周寻找不与已有文字重叠的位置，必要时进入双列图例区。"""

    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), _ = cv2.getTextSize(
        _ascii(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = point
    candidates = (
        (x + radius + 7, y - radius - 5),
        (x + radius + 7, y + radius + text_h + 10),
        (x - radius - text_w - 15, y - radius - 5),
        (x - radius - text_w - 15, y + radius + text_h + 10),
        (x + radius + 7, y + text_h // 2),
        (x - radius - text_w - 15, y + text_h // 2),
    )
    for candidate in candidates:
        if not _overlaps(_label_bounds(image, text, candidate, scale), occupied):
            return candidate
    # 极密集目标的最终兜底：在画面底部使用左右双列图例，仍保证标签互不覆盖。
    column = fallback_index % 2
    row = fallback_index // 2
    column_x = 8 if column == 0 else image.shape[1] // 2
    baseline_y = image.shape[0] - 10 - row * (text_h + 12)
    return column_x, baseline_y


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


def render_stage3_5_observation(
    frame_bgr: np.ndarray,
    observation: dict[str, Any],
    *,
    draw_group_center: bool = True,
) -> np.ndarray:
    """在一帧 BGR 图像上绘制单条 Stage 3.5 多主体观察。"""

    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ArtifactValidationError("可视化输入帧必须是 HxWx3 BGR 图像")
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    scale = max(0.46, min(0.82, width / 1600.0 * 0.82))
    targets = observation.get("targets", [])
    if not isinstance(targets, list):
        raise ArtifactValidationError("Stage 3.5 observation.targets 必须是数组")

    points: list[tuple[int, int]] = []
    primary_ids = {str(value) for value in observation.get("primary_target_ids", [])}
    label_rows: list[tuple[str, tuple[int, int, int], tuple[int, int], int]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            continue
        normalized = _normalized_point(target.get("subject_point"))
        if normalized is None:
            continue
        x = min(width - 1, max(0, int(round(normalized[0] * (width - 1)))))
        y = min(height - 1, max(0, int(round(normalized[1] * (height - 1)))))
        target_id = str(target.get("target_id") or f"target_{index}")
        is_primary = target_id in primary_ids or target.get("role") == "primary"
        color = _target_color(target_id)
        visibility = str(target.get("visibility", "visible"))
        radius = max(7, int(round(min(width, height) * 0.009)))
        if visibility == "visible":
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), radius + 4, (255, 255, 255), 2, cv2.LINE_AA)
            if is_primary:
                cv2.circle(canvas, (x, y), radius + 8, (0, 215, 255), 3, cv2.LINE_AA)
        elif visibility == "occluded":
            cv2.circle(canvas, (x, y), radius + 2, color, 3, cv2.LINE_AA)
            cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, radius * 2, 2)
        else:
            cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, radius * 2, 3)
        confidence = float(target.get("confidence", 0.0) or 0.0)
        importance = float(target.get("importance", 0.5) or 0.0)
        phrase = str(target.get("grounding_phrase", ""))
        focus = _normalized_point(target.get("focus_point"))
        if focus is not None:
            focus_xy = (
                min(width - 1, max(0, int(round(focus[0] * (width - 1))))),
                min(height - 1, max(0, int(round(focus[1] * (height - 1))))),
            )
            cv2.drawMarker(canvas, focus_xy, (0, 215, 255) if is_primary else color, cv2.MARKER_STAR, radius * 2, 2)
        label_rows.append((
            f"{'PRIMARY' if is_primary else 'support'} | {target_id} | {phrase} | imp={importance:.2f} conf={confidence:.2f} | {visibility}",
            color,
            (x, y),
            radius,
        ))
        points.append((x, y))

    if points and draw_group_center:
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        padding = max(8, int(round(min(width, height) * 0.012)))
        cv2.rectangle(
            canvas,
            (max(0, x1 - padding), max(0, y1 - padding)),
            (min(width - 1, x2 + padding), min(height - 1, y2 + padding)),
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
        cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)

    interval_id = _ascii(observation.get("interval_id", ""), 64)
    phrases = observation.get("grounding_phrases", [])
    phrase_text = ", ".join(_ascii(value, 36) for value in phrases) if isinstance(phrases, list) else ""
    header = (
        f"interval={interval_id} frame={int(observation.get('frame', -1))} "
        f"sample={int(observation.get('sample_index', -1))} "
        f"time={float(observation.get('timestamp_sec', 0.0)):.3f}s "
        f"mode={_ascii(observation.get('group_mode', 'unknown'), 16)} "
        f"composition={_ascii(observation.get('composition_mode', 'unknown'), 16)} "
        f"status={_ascii(observation.get('status', 'unknown'), 16)}"
    )
    occupied = [_label(canvas, header, (8, 30), (255, 255, 255), scale)]
    if phrase_text:
        occupied.append(
            _label(canvas, f"phrases: {phrase_text}", (8, 62), (190, 255, 255), scale * 0.9)
        )
    for index, (text, color, point, radius) in enumerate(label_rows):
        origin = _target_label_origin(
            canvas, text, point, radius, scale, occupied, index
        )
        occupied.append(_label(canvas, text, origin, color, scale))
    if not points:
        _label(
            canvas,
            "NO VALID QWEN SUBJECT POINT",
            (max(8, width // 2 - 180), max(48, height // 2)),
            (80, 80, 255),
            max(0.65, scale),
        )
    return canvas


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
    rows: list[dict[str, Any]], video_id: str, frame_count: int
) -> None:
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
        targets = row.get("targets")
        if not isinstance(targets, list):
            raise ArtifactValidationError(f"观察 targets 非数组: {key}")
        for target in targets:
            if not isinstance(target, dict):
                raise ArtifactValidationError(f"观察 target 非对象: {key}")
            point = target.get("subject_point")
            if point is not None and _normalized_point(point) is None:
                raise ArtifactValidationError(f"观察中心点非法: {key}/{target.get('target_id')}")
            focus = target.get("focus_point", point)
            if focus is not None and _normalized_point(focus) is None:
                raise ArtifactValidationError(f"观察构图点非法: {key}/{target.get('target_id')}")


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
    draw_group_center: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """可视化一个视频的全部 Stage 3.5 Qwen 观察。"""

    if not save_images and not write_video:
        raise ArtifactValidationError("save_images 和 write_video 不能同时关闭")
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage3_5_video = Path(stage3_5_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage3_5_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 3.5 视频没有成功标记: {stage3_5_video}")
    metadata = _read_object(stage1_video / "metadata.json")
    observations_path = stage3_5_video / "subject_observations.jsonl"
    if not observations_path.is_file():
        legacy_hint = "（检测到旧版 subject_points.jsonl，请先重新运行 Stage 3.5 v2）" if (
            stage3_5_video / "subject_points.jsonl"
        ).is_file() else ""
        raise ArtifactValidationError(
            f"Stage 3.5 缺少 subject_observations.jsonl: {stage3_5_video}{legacy_hint}"
        )
    rows = read_jsonl(observations_path)
    enriched_path = stage3_5_video / "enriched_intervals.jsonl"
    enriched = read_jsonl(enriched_path) if enriched_path.is_file() else []
    interval_metadata = {str(row.get("interval_id")): row for row in enriched}
    frame_count = int(metadata.get("frame_count", 0))
    fps = float(metadata.get("fps", 0.0))
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ArtifactValidationError(f"Stage 1 metadata 帧数或 FPS 非法: {video_id}")
    _validate_observations(rows, video_id, frame_count)
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
            output_video = interval_dir / "qwen_subject_points.mp4"
            try:
                for observation, sample in zip(observations, samples, strict=True):
                    frame = cv2.imdecode(
                        np.frombuffer(sample.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        raise ArtifactValidationError(
                            f"可视化采样帧 JPEG 回读失败: {interval_id}/{sample.frame}"
                        )
                    rendered = render_stage3_5_observation(
                        frame, observation, draw_group_center=draw_group_center
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
                            interval_info = interval_metadata.get(interval_id, {})
                            effective_fps = preview_fps or float(
                                interval_info.get("subject_observation_sample_fps", 2.0)
                            )
                            writer, temporary_video = _open_video_writer(current_size, effective_fps)
                            rendered_size = current_size
                        if current_size != rendered_size:
                            rendered = cv2.resize(rendered, rendered_size, interpolation=cv2.INTER_AREA)
                        writer.write(rendered)
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
                }
            )
        manifest = {
            "visualization": "stage3_5_qwen_subject_points",
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
                write_json(output / "summary.json", {
                    "visualization": "stage3_5_qwen_subject_points",
                    "success_count": len(results),
                    "failure_count": len(failures),
                    "results": results,
                    "failures": failures,
                })
                raise
    summary = {
        "visualization": "stage3_5_qwen_subject_points",
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    write_json(output / "summary.json", summary)
    return summary
