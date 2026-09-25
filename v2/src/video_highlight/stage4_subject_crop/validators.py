"""Stage 4 配置、Stage 1/3.5 输入和逐帧构图输出校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl
from video_highlight.common.exceptions import ArtifactValidationError


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"缺少文件: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 根节点不是对象: {path}")
    return value


def list_stage3_5_video_ids(stage3_5_dir: str | Path) -> list[str]:
    videos = Path(stage3_5_dir).resolve() / "videos"
    if not videos.is_dir():
        raise ArtifactValidationError(f"Stage 3.5 videos 目录不存在: {videos}")
    return sorted((row.name for row in videos.iterdir() if row.is_dir() and not row.name.startswith(".") and (row / "_SUCCESS.json").is_file()),
                  key=lambda name: int(name) if name.isdigit() else name,) # 保证有序排列


def load_video_inputs(
    stage1_dir: str | Path, stage3_5_dir: str | Path, video_id: str,project_paths_config:dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    加载 Stage 4 输入并校验。优先读取 v2 ``subject_observations.jsonl``，同时兼容
    v1 ``subject_points.jsonl``。

    Returns:
        metadata: stage1获取的视频的元信息
        scenes: stage1获取的镜头信息
        intervals: stage3_5获取的个高光区间的所有采样帧的中心主体预测信息
    """
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage3_5_video = Path(stage3_5_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage3_5_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 3.5 视频没有成功标记: {stage3_5_video}")
    metadata = _read_object(stage1_video / "metadata.json")
    # 当切换环境后，视频路径发生改变，此时默认使用配置文件路径，默认mp4
    if not Path(metadata["source_path"]).is_file():
        metadata["source_path"] = str(Path(project_paths_config["video_root"]) / (video_id + ".mp4"))
    scenes = read_jsonl(stage1_video / "scenes.jsonl")
    intervals = read_jsonl(stage3_5_video / "enriched_intervals.jsonl")
    observation_path = stage3_5_video / "subject_observations.jsonl"
    legacy_path = stage3_5_video / "subject_points.jsonl"
    if observation_path.is_file():
        observation_rows = read_jsonl(observation_path)
        legacy = False
    elif legacy_path.is_file():
        observation_rows = read_jsonl(legacy_path)
        legacy = True
    else:
        raise ArtifactValidationError(f"Stage 3.5 缺少主体观察文件: {observation_path}")
    observations_by_interval: dict[str, list[dict[str, Any]]] = {}
    for observation in observation_rows:
        interval_id = str(observation.get("interval_id", ""))
        if str(observation.get("video_id")) != video_id:
            raise ArtifactValidationError(f"Stage 3.5 主体观察 video_id 不一致: {interval_id}")
        if legacy:
            value = observation.get("subject_point")
            if value is not None and (
                not isinstance(value, list) or len(value) != 2
                or not all(0.0 <= float(axis) <= 1.0 for axis in value)
            ):
                raise ArtifactValidationError(f"Stage 3.5 主体点非法: {interval_id}/{observation.get('frame')}")
        else:
            if observation.get("group_mode") not in {"single", "multiple"}:
                raise ArtifactValidationError(f"Stage 3.5 group_mode 非法: {interval_id}/{observation.get('frame')}")
            targets = observation.get("targets")
            composition_mode = observation.get("composition_mode", "single_focus")
            if composition_mode not in {"single_focus", "group_focus"}:
                raise ArtifactValidationError(f"Stage 3.5 composition_mode 非法: {interval_id}/{observation.get('frame')}")
            primary_target_ids = observation.get("primary_target_ids", [])
            if not isinstance(primary_target_ids, list):
                raise ArtifactValidationError(f"Stage 3.5 primary_target_ids 非数组: {interval_id}")
            if not isinstance(targets, list):
                raise ArtifactValidationError(f"Stage 3.5 targets 非数组: {interval_id}/{observation.get('frame')}")
            recommended_center = observation.get("recommended_crop_center")
            if recommended_center is not None and (
                not isinstance(recommended_center, list) or len(recommended_center) != 2
                or not all(math.isfinite(float(axis)) and 0.0 <= float(axis) <= 1.0 for axis in recommended_center)
            ):
                raise ArtifactValidationError(
                    f"Stage 3.5 Qwen 推荐构图中心非法: {interval_id}/{observation.get('frame')}"
                )
            recommended_confidence = float(observation.get("recommended_crop_confidence", 0.0))
            if not math.isfinite(recommended_confidence) or not 0.0 <= recommended_confidence <= 1.0:
                raise ArtifactValidationError(
                    f"Stage 3.5 Qwen 推荐构图置信度非法: {interval_id}/{observation.get('frame')}"
                )
            seen_target_ids: set[str] = set()
            for target in targets:
                if not isinstance(target, dict):
                    raise ArtifactValidationError(f"Stage 3.5 target 非对象: {interval_id}")
                target_id = str(target.get("target_id", ""))
                if not target_id or target_id in seen_target_ids:
                    raise ArtifactValidationError(f"Stage 3.5 target_id 为空或重复: {interval_id}/{target_id}")
                seen_target_ids.add(target_id)
                value = target.get("subject_point")
                if value is not None and (
                    not isinstance(value, list) or len(value) != 2
                    or not all(0.0 <= float(axis) <= 1.0 for axis in value)
                ):
                    raise ArtifactValidationError(f"Stage 3.5 target 点非法: {interval_id}/{target_id}")
                focus = target.get("focus_point", value)
                if focus is not None and (
                    not isinstance(focus, list) or len(focus) != 2
                    or not all(0.0 <= float(axis) <= 1.0 for axis in focus)
                ):
                    raise ArtifactValidationError(f"Stage 3.5 focus_point 非法: {interval_id}/{target_id}")
                if target.get("role", "supporting") not in {"primary", "supporting"}:
                    raise ArtifactValidationError(f"Stage 3.5 target role 非法: {interval_id}/{target_id}")
                importance = float(target.get("importance", 0.5))
                if not math.isfinite(importance) or not 0.0 <= importance <= 1.0:
                    raise ArtifactValidationError(f"Stage 3.5 target importance 非法: {interval_id}/{target_id}")
            if any(str(value) not in seen_target_ids for value in primary_target_ids):
                raise ArtifactValidationError(f"Stage 3.5 primary_target_ids 引用未知目标: {interval_id}")
        observations_by_interval.setdefault(interval_id, []).append(observation)
    frame_count = int(metadata["frame_count"])
    previous_end = -1
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        if str(interval.get("video_id")) != video_id or not (0 <= start < end <= frame_count):
            raise ArtifactValidationError(f"Stage 3.5 区间非法: {interval.get('interval_id')}")
        if start < previous_end:
            raise ArtifactValidationError("Stage 3.5 区间重叠或未排序")
        previous_end = end
        interval_id = str(interval["interval_id"])
        rows = sorted(observations_by_interval.pop(interval_id, []), key=lambda row: int(row["frame"]))
        if legacy:
            interval["subject_points"] = rows
            expected_count = int(interval.get("subject_point_sample_count", len(rows)))
        else:
            interval["subject_observations"] = rows
            expected_count = int(interval.get("subject_observation_sample_count", len(rows)))
        if len(rows) != expected_count:
            raise ArtifactValidationError(f"Stage 3.5 主体观察数量与区间摘要不一致: {interval_id}")
        sample_indices = [int(row["sample_index"]) for row in rows]
        if sample_indices != list(range(len(sample_indices))):
            raise ArtifactValidationError(f"Stage 3.5 sample_index 不连续或顺序异常: {interval_id}")
        for observation in rows:
            if not start <= int(observation["frame"]) < end:
                raise ArtifactValidationError(f"Stage 3.5 主体观察不属于区间: {interval_id}/{observation['frame']}")
    if observations_by_interval:
        raise ArtifactValidationError(f"Stage 3.5 主体观察引用未知区间: {sorted(observations_by_interval)}")
    return metadata, scenes, intervals


def validate_config(config: dict[str, Any]) -> None:
    for key in ("runtime", "tracking", "crop_candidates", "composition", "optimizer", "smoothing"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 4 配置缺少对象字段: {key}")
    if str(config["runtime"].get("interval_error_policy", "center")) not in {"center", "error"}:
        raise ArtifactValidationError("runtime.interval_error_policy 只能是 center 或 error")
    fixed_maximum = config["crop_candidates"].get("fixed_maximum", False)
    if not isinstance(fixed_maximum, bool):
        raise ArtifactValidationError("crop_candidates.fixed_maximum 必须是布尔值")
    if int(config["tracking"].get("mask_grid_max_side", 160)) < 16:
        raise ArtifactValidationError("tracking.mask_grid_max_side 必须至少为 16")
    if int(config["tracking"].get("primary_switch_confirm_anchors", 2)) <= 0:
        raise ArtifactValidationError("tracking.primary_switch_confirm_anchors 必须大于 0")
    if int(config["tracking"].get("recovery_confirm_frames", 3)) <= 0:
        raise ArtifactValidationError("tracking.recovery_confirm_frames 必须大于 0")
    if int(config["crop_candidates"].get("mask_top_k_per_scale", 8)) <= 0:
        raise ArtifactValidationError("crop_candidates.mask_top_k_per_scale 必须大于 0")
    candidate_nms = float(config["crop_candidates"].get("mask_candidate_nms_ratio", 0.18))
    if not math.isfinite(candidate_nms) or not 0.0 <= candidate_nms <= 1.0:
        raise ArtifactValidationError("crop_candidates.mask_candidate_nms_ratio 必须在 [0,1] 内")
    qwen_minimum = float(config["crop_candidates"].get("qwen_recommended_min_confidence", 0.20))
    if not math.isfinite(qwen_minimum) or not 0.0 <= qwen_minimum <= 1.0:
        raise ArtifactValidationError("crop_candidates.qwen_recommended_min_confidence 必须在 [0,1] 内")
    composition = config["composition"]
    qwen_weight = float(composition.get("qwen_recommended_center_weight", 0.05))
    if not math.isfinite(qwen_weight) or qwen_weight < 0.0:
        raise ArtifactValidationError("composition.qwen_recommended_center_weight 必须是非负有限数")
    boundary_band = float(composition.get("boundary_band_ratio", 0.03))
    if not math.isfinite(boundary_band) or not 0.0 <= boundary_band <= 0.5:
        raise ArtifactValidationError("composition.boundary_band_ratio 必须在 [0,0.5] 内")
    mask_weights = composition.get("mask_weights", {})
    if not isinstance(mask_weights, dict):
        raise ArtifactValidationError("composition.mask_weights 必须是对象")
    for key, value in mask_weights.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ArtifactValidationError(f"composition.mask_weights.{key} 必须是非负有限数")
    primary_mask_weights = composition.get("primary_mask_weights", {})
    if not isinstance(primary_mask_weights, dict):
        raise ArtifactValidationError("composition.primary_mask_weights 必须是对象")
    for key, value in primary_mask_weights.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ArtifactValidationError(f"composition.primary_mask_weights.{key} 必须是非负有限数")
    for key, default in (("primary_safe_zone_x", [0.35, 0.65]), ("primary_safe_zone_y", [0.30, 0.70])):
        zone = composition.get(key, default)
        if not isinstance(zone, list) or len(zone) != 2 or not 0 <= float(zone[0]) <= float(zone[1]) <= 1:
            raise ArtifactValidationError(f"composition.{key} 必须是 [0,1] 内递增的两个数")
    bypass_interpolated = config["smoothing"].get("bypass_for_interpolated_qwen", True)
    if not isinstance(bypass_interpolated, bool):
        raise ArtifactValidationError("smoothing.bypass_for_interpolated_qwen 必须是布尔值")
    for key, default in (
        ("post_smoothing_min_mask_coverage", 0.70),
        ("post_smoothing_max_coverage_loss", 0.05),
    ):
        value = float(config["smoothing"].get(key, default))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ArtifactValidationError(f"smoothing.{key} 必须在 [0,1] 内")
    distance = float(config["tracking"].get("anchor_max_center_distance_ratio", 0.20))
    if not 0.0 <= distance <= 1.0:
        raise ArtifactValidationError("tracking.anchor_max_center_distance_ratio 必须在 [0,1] 内")
    grounding = config["tracking"].get("grounding", {})
    if not isinstance(grounding, dict):
        raise ArtifactValidationError("tracking.grounding 必须是对象")
    for key, default in (
        ("phrase_memory_anchors", 3),
        ("object_keepalive_anchors", 2),
    ):
        if int(grounding.get(key, default)) < 0:
            raise ArtifactValidationError(f"tracking.grounding.{key} 不能为负数")
    if int(grounding.get("max_grounding_phrases", 12)) <= 0:
        raise ArtifactValidationError("tracking.grounding.max_grounding_phrases 必须大于 0")
    for key, default in (
        ("retention_association_threshold", 0.30),
        ("retention_detection_threshold", 0.25),
        ("carry_score_decay", 0.75),
    ):
        value = float(grounding.get(key, default))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ArtifactValidationError(f"tracking.grounding.{key} 必须在 [0,1] 内")
    if bool(grounding.get("enabled", False)):
        if not str(grounding.get("model", "")).strip():
            raise ArtifactValidationError("启用 Grounding DINO 时 tracking.grounding.model 不能为空")
        for key in ("box_threshold", "text_threshold", "selection_threshold", "nms_iou", "association_threshold"):
            value = float(grounding.get(key, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ArtifactValidationError(f"tracking.grounding.{key} 必须在 [0,1] 内")
        if int(grounding.get("max_objects", 8)) <= 0:
            raise ArtifactValidationError("tracking.grounding.max_objects 必须大于 0")
        if str(grounding.get("error_policy", "qwen")) not in {"qwen", "error"}:
            raise ArtifactValidationError("tracking.grounding.error_policy 只能是 qwen 或 error")
    visualization = config.get("visualization", {})
    if not isinstance(visualization, dict):
        raise ArtifactValidationError("visualization 必须是对象")
    for key, default in (
        ("enabled", False),
        ("always_save_anchor_frames", True),
        ("always_save_fallback_frames", True),
        ("save_images", True),
        ("write_video", False),
        ("annotate_composition", True),
        ("draw_subject_union", False),
        ("draw_prompt_union", False),
    ):
        if not isinstance(visualization.get(key, default), bool):
            raise ArtifactValidationError(f"visualization.{key} 必须是布尔值")
    if int(visualization.get("max_candidate_centers", 5)) <= 0:
        raise ArtifactValidationError("visualization.max_candidate_centers 必须大于 0")
    if int(visualization.get("max_raw_grounding_boxes", 12)) < 0:
        raise ArtifactValidationError("visualization.max_raw_grounding_boxes 不能为负数")
    if int(visualization.get("max_drawn_candidate_centers", 3)) < 0:
        raise ArtifactValidationError("visualization.max_drawn_candidate_centers 不能为负数")
    if int(visualization.get("sidebar_width", 420)) < 240:
        raise ArtifactValidationError("visualization.sidebar_width 必须至少为 240")
    if int(visualization.get("decision_panel_height", 170)) < 80:
        raise ArtifactValidationError("visualization.decision_panel_height 必须至少为 80")
    for key, default in (
        ("tag_font_scale", 0.52),
        ("sidebar_font_scale", 0.56),
        ("decision_font_scale", 0.52),
    ):
        value = float(visualization.get(key, default))
        if not math.isfinite(value) or value <= 0.0:
            raise ArtifactValidationError(f"visualization.{key} 必须是正有限数")
    for key, default in (
        ("tag_font_thickness", 1),
        ("sidebar_font_thickness", 1),
        ("sidebar_line_height", 26),
        ("decision_font_thickness", 1),
        ("decision_line_height", 26),
        ("candidate_marker_radius", 6),
        ("candidate_marker_thickness", 2),
        ("dp_marker_size", 6),
        ("dp_marker_thickness", 3),
        ("final_marker_size", 6),
        ("final_marker_thickness", 4),
        ("mask_marker_size", 6),
        ("mask_marker_thickness", 3),
        ("qwen_marker_radius", 8),
        ("qwen_outer_radius", 13),
        ("qwen_outer_thickness", 2),
        ("qwen_recommended_marker_size", 6),
        ("qwen_recommended_marker_thickness", 1),
        ("focus_marker_size", 6),
        ("focus_marker_thickness", 3),
        ("grounding_marker_size", 6),
        ("grounding_marker_thickness", 2),
        ("fallback_marker_size", 6),
        ("fallback_marker_thickness", 2),
        ("raw_detection_box_thickness", 1),
        ("selected_detection_box_thickness", 3),
        ("final_crop_box_thickness", 4),
        ("fallback_box_thickness", 2),
    ):
        if int(visualization.get(key, default)) <= 0:
            raise ArtifactValidationError(f"visualization.{key} 必须大于 0")
    if int(visualization.get("qwen_outer_radius", 13)) < int(visualization.get("qwen_marker_radius", 8)):
        raise ArtifactValidationError("visualization.qwen_outer_radius 不能小于 qwen_marker_radius")
    sample_fps = float(visualization.get("sample_fps", 2.0))
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ArtifactValidationError("visualization.sample_fps 必须是正有限数")
    mask_alpha = float(visualization.get("mask_alpha", 0.35))
    if not math.isfinite(mask_alpha) or not 0.0 <= mask_alpha <= 1.0:
        raise ArtifactValidationError("visualization.mask_alpha 必须在 [0,1] 内")


def validate_crops(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    width = int(metadata.get("display_width", metadata["width"]))
    height = int(metadata.get("display_height", metadata["height"]))
    frame_count = int(metadata["frame_count"])
    ratio = metadata.get("targetRatioWH", [16, 9])
    target_w, target_h = float(ratio[0]), float(ratio[1])
    previous_frame = -1
    for row in rows:
        frame = row.get("frame")
        box = row.get("bboxes")
        if not isinstance(frame, int) or not (0 <= frame < frame_count) or frame <= previous_frame:
            raise ArtifactValidationError(f"Stage 4 frame 非法或重复: {frame}")
        previous_frame = frame
        if not isinstance(box, list) or len(box) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in box):
            raise ArtifactValidationError(f"frame={frame} 的 bboxes 必须是三个有限数值")
        x, y, crop_w = map(float, box)
        crop_h = crop_w * target_h / target_w
        if not (x >= 0 and y >= 0 and crop_w > 0 and x + crop_w <= width + 1e-6 and y + crop_h <= height + 1e-6):
            raise ArtifactValidationError(f"frame={frame} 构图框越界: {box}")


def validate_stage4_artifacts(video_dir: str | Path) -> dict[str, int]:
    root = Path(video_dir)
    required = ("crops.jsonl", "tracks.jsonl", "diagnostics.jsonl", "grounding_anchors.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Stage 4 缺少产物: {', '.join(missing)}")
    return {"artifact_files": len(required)}
