"""Stage 3.5 主流水线：最终区间即时采样、Qwen 多主体观察与原子持久化。

Stage 3 是时间边界的唯一来源；Stage 1 在这里仅提供源视频路径、FPS 和总帧数。
本阶段明确不读取 Stage 1 的 ``coarse_frames`` 或 ``sample_map.jsonl``，因为
Stage 3 可能已把边界细化到任意原始帧，复用旧粗采样会造成时间轴错位。

``predict`` 模式按默认 2 FPS 对每个最终高光区间重新解码，将该区间的全部
采样 JPEG 作为一个有序多图请求发给 vLLM。``passthrough`` 模式只计算相同的
采样帧计划，为每个计划帧写空 ``targets``；它不会打开视频，也不会
构建或调用模型客户端，因此可用于验证后续数据链路。
"""

from __future__ import annotations

import shutil
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.common.hashing import mapping_sha256
from video_highlight.common.manifest import failure_record, success_record, utc_now_iso
from video_highlight.common.runtime import Timer
from video_highlight.contracts.schema_versions import STAGE3_5_SCHEMA_VERSION

from .frame_sampler import SampledFrame, plan_sample_frames, sample_interval
from .prompt_builder import DEFAULT_SYSTEM_PROMPT, OUTPUT_SCHEMA_OPENAI, build_prompt, build_user_content
from .qwen_adapter import SubjectObservationBackend, build_backend
from .response_parser import parse_predictions
from .validators import list_stage3_video_ids, load_video_inputs, validate_artifacts, validate_config, validate_observations


def _observation_row(video_id: str, interval: dict[str, Any], sample_index: int, frame: int, fps: float, prediction: dict[str, Any], mode: str) -> dict[str, Any]:
    """把模型局部 sample_index 和上游时间轴合成稳定的多主体观察记录。"""

    return {
        "schema_version": STAGE3_5_SCHEMA_VERSION,
        "video_id": video_id,
        "interval_id": str(interval["interval_id"]),
        "sample_index": sample_index,
        "frame": frame,
        "timestamp_sec": frame / fps,
        "subject": interval.get("subject"),
        "group_mode": prediction.get("group_mode", "multiple"),
        "composition_mode": prediction.get("composition_mode", "single_focus"),
        "recommended_crop_center": prediction.get("recommended_crop_center"),
        "recommended_crop_confidence": float(prediction.get("recommended_crop_confidence", 0.0)),
        "primary_target_ids": list(prediction.get("primary_target_ids", [])),
        "grounding_phrases": list(prediction.get("grounding_phrases", [])),
        "targets": list(prediction.get("targets", [])),
        "coordinate_space": "normalized_xy",
        "status": "skipped" if mode == "passthrough" else "predicted",
        "reason": str(prediction.get("reason", "")),
    }


def _enriched_interval(interval: dict[str, Any], sample_count: int, visible_observation_count: int,
                       visible_target_count: int, sample_fps: float, mode: str) -> dict[str, Any]:
    """复制 Stage 3 区间并追加 v2 多主体观察索引，不把观察数组嵌进区间。"""

    row = {key: value for key, value in interval.items() if key != "subject_point"}
    row["source_schema_version"] = row.get("schema_version")
    row["schema_version"] = STAGE3_5_SCHEMA_VERSION
    row["subject_observation_file"] = "subject_observations.jsonl"
    row["subject_observation_sample_fps"] = sample_fps
    row["subject_observation_sample_count"] = sample_count
    row["visible_subject_observation_count"] = visible_observation_count
    row["visible_subject_target_count"] = visible_target_count
    row["subject_observation_status"] = "skipped" if mode == "passthrough" else "predicted"
    return row


def _predict_interval(video_id: str, interval: dict[str, Any], metadata: dict[str, Any],
                      frames: list[SampledFrame], backend: SubjectObservationBackend,
                      config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any],
                      list[dict[str, Any]], list[int],list[dict]]:
    """构造一次多图请求并解析；解析失败可按配置重试整个区间请求。config配置use_batch为false时1，降级为单帧请求"""

    use_batch = config["runtime"].get("use_batch", False) # 是否使用批次输入
    system_prompt = str(config.get("prompt", {}).get("system", DEFAULT_SYSTEM_PROMPT))
    prompt = build_prompt(video_id, interval, frames, use_batch=use_batch, metadata=metadata)
    content = build_user_content(prompt, frames)
    retries = max(0, int(config.get("parsing", {}).get("retries", 1)))
    raw_records: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response = backend.analyze(system_prompt, content, OUTPUT_SCHEMA_OPENAI, len(frames))
        raw_records.append({
            "schema_version": STAGE3_5_SCHEMA_VERSION,
            "video_id": video_id,
            "interval_id": interval["interval_id"],
            "attempt": attempt + 1,
            **response.to_dict(),
        })
        try:
            predictions, missing, error_predictions = parse_predictions(response.text, frames, use_batch)
            request = {
                "schema_version": STAGE3_5_SCHEMA_VERSION,
                "response_id":response.response_id,
                "video_id": video_id,
                "interval_id": interval["interval_id"],
                "subject": interval.get("subject"),
                "sample_count": len(frames),
                "sample_fps": float(config["sampling"].get("fps", 2.0)),
                "source_path": str(metadata["source_path"]),
                "transport": "ordered_jpeg_data_urls",
                "encoded_jpeg_bytes": sum(len(row.jpeg_bytes) for row in frames),
                "timeline": [{"sample_index": row.sample_index, "frame": row.frame, "timestamp_sec": row.timestamp_sec} for row in frames],
                "base64_persisted": False,
                "raw_prompt": {"system_prompt": system_prompt, "user_prompt_short": prompt},
            }
            return predictions, request, raw_records, missing,error_predictions
        except Exception as error:
            last_error = error
    assert last_error is not None
    raise last_error


def process_video(stage1_dir: Path, stage3_dir: Path, video_id: str, videos_output_dir: Path, backend: SubjectObservationBackend | None, config: dict[str, Any], project_paths_config:dict[str, Any],resume: bool, overwrite: bool) -> dict[str, Any]:
    """处理一个视频，并以目录重命名作为原子提交点。"""

    final_dir = videos_output_dir / video_id
    if resume and (final_dir / "_SUCCESS.json").is_file():
        return {"video_id": video_id, "status": "skipped", "reason": "already_successful"}
    work_dir = videos_output_dir / f".{video_id}.inprogress"
    if overwrite:
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    elif final_dir.exists() or work_dir.exists():
        raise ArtifactValidationError(f"Stage 3.5 输出已存在，请使用 --resume 或 --overwrite: {final_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 粗采样文件从不进入返回值；这里只得到元数据和 Stage 3 最终区间。
    metadata, intervals = load_video_inputs(stage1_dir, stage3_dir, video_id,project_paths_config)
    mode = str(config["runtime"].get("mode", "predict"))
    fps = float(metadata["fps"])
    sample_fps = float(config["sampling"].get("fps", 2.0))
    observation_rows: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    with (Timer() as timer):
        for interval in intervals:
            start, end = int(interval["start_frame"]), int(interval["end_frame"])
            planned_frames = plan_sample_frames(start, end, fps, sample_fps)
            missing_predictions: list[int] = []
            error_predictions: list[dict[str, Any]] = []
            if mode == "passthrough":
                # 跳过模式不能调用 sample_interval：即使 source_path 不存在也应能产出契约。
                predictions = [{
                    "group_mode": "multiple",
                    "composition_mode": "single_focus",
                    "recommended_crop_center": None,
                    "recommended_crop_confidence": 0.0,
                    "primary_target_ids": [],
                    "grounding_phrases": [],
                    "targets": [],
                    "reason": "stage3_5_processing_skipped",
                } for _ in planned_frames]
            else:
                if backend is None:
                    raise RuntimeError("predict 模式缺少多主体观察模型 backend")
                # 每个最终区间独立打开源视频并即时取帧；JPEG 只存在于当前内存对象。
                # TODO 后续可以所有区间一次性读完，对于短视频多区间可以加快速度
                frames = sample_interval(
                    metadata["source_path"], planned_frames, fps,
                    jpeg_quality=int(config["sampling"].get("jpeg_quality", 85)),
                    max_side=int(config["sampling"].get("max_side", 1024)),
                    decoder=str(config["sampling"].get("decoder", "auto")),
                    ffmpeg_bin=str(config["sampling"].get("ffmpeg_bin", "ffmpeg")),
                )
                # 一次性最多输入1帧，降低上下文压力，最主要是降低错误概率，只要模型输出中心坐标即可
                # 目前保留批次输入的能力，以便后续加速
                step = int(config["runtime"].get("batch_size", 1))
                chunks_frames = [frames[i:i + step] for i in range(0, len(frames), step)] # 注：Python的切片操作在结束索引超过列表长度时，会自动截断到列表末尾，不会抛出 IndexError
                predictions: list[dict[str, Any]] = []
                missing_predictions: list[int] = []
                for frames_in_chunk in chunks_frames:
                    predictions_chunk, request_chunk, raw_response_chunk, missing_predictions_chunk,error_predictions_chunk = \
                        _predict_interval(video_id, interval, metadata, frames_in_chunk, backend, config)
                    # 合并单个chunk的predictions、missing_predictions、error_predictions
                    for prediction in predictions_chunk:
                        predictions.append(prediction)
                    for missing_prediction in missing_predictions_chunk:
                        missing_predictions.append(missing_prediction)
                    for error_prediction in error_predictions_chunk:
                        error_predictions.append(error_prediction)
                    requests.append(request_chunk)
                    raw_responses.extend(raw_response_chunk)
            interval_observations = [
                _observation_row(video_id, interval, index, frame, fps, predictions[index], mode)
                for index, frame in enumerate(planned_frames)
            ]
            observation_rows.extend(interval_observations)
            visible_observation_count = sum(
                any(target.get("subject_point") is not None for target in row["targets"])
                for row in interval_observations
            )
            visible_target_count = sum(
                target.get("subject_point") is not None
                for row in interval_observations
                for target in row["targets"]
            )
            enriched.append(_enriched_interval(
                interval, len(planned_frames), visible_observation_count,
                visible_target_count, sample_fps, mode
            ))
            diagnostics.append({
                "schema_version": STAGE3_5_SCHEMA_VERSION,
                "video_id": video_id,
                "interval_id": interval["interval_id"],
                "mode": mode,
                "status": "skipped_all_stage3_5_processing" if mode == "passthrough" else "completed",
                "planned_sample_count": len(planned_frames),
                "visible_observation_count": visible_observation_count,
                "visible_target_count": visible_target_count,
                "model_omitted_sample_indices": missing_predictions,
                "error_predictions": error_predictions,
                "sample_decoder": "skipped" if mode == "passthrough" else (
                    frames[0].decoder_backend if frames else "none"
                ),
            })
            # 区间完成即刷新临时目录；若批处理中断，现有内容仍不会被下游当成成功产物。
            write_jsonl(work_dir / "subject_observations.jsonl", observation_rows)
            write_jsonl(work_dir / "enriched_intervals.jsonl", enriched)
            write_jsonl(work_dir / "requests.jsonl", requests)
            write_jsonl(work_dir / "raw_responses.jsonl", raw_responses)
            write_jsonl(work_dir / "diagnostics.jsonl", diagnostics)
        # 空高光视频也必须产生五个空 JSONL，保证下游无需特殊判断文件是否存在。
        if not intervals:
            for name in ("subject_observations.jsonl", "enriched_intervals.jsonl", "requests.jsonl", "raw_responses.jsonl", "diagnostics.jsonl"):
                write_jsonl(work_dir / name, [])
        validate_observations(observation_rows, intervals, metadata, sample_fps)
        validation = validate_artifacts(work_dir)
    success = success_record(
        video_id,
        elapsed_sec=timer.elapsed_sec,
        mode=mode,
        input_interval_count=len(intervals),
        output_sample_count=len(observation_rows),
        located_observation_count=sum(
            any(target.get("subject_point") is not None for target in row["targets"])
            for row in observation_rows
        ),
        located_target_count=sum(
            target.get("subject_point") is not None
            for row in observation_rows
            for target in row["targets"]
        ),
        sample_fps=sample_fps,
        validation=validation,
    )
    write_json(work_dir / "_SUCCESS.json", success)
    work_dir.rename(final_dir)
    return success


def run_stage3_5(stage1_dir: str | Path, stage3_dir: str | Path, output_dir: str | Path, config: dict[str, Any], project_paths_config:dict[str, Any] | None = None,
                 video_ids: set[str] | None = None, limit: int | None = None, resume: bool = False,
                 overwrite: bool = False, strict: bool = False, logger: Any = None) -> dict[str, Any]:
    """批量入口；模型客户端只构建一次，并在所有视频之间复用连接。"""

    validate_config(config)
    project_paths_config = project_paths_config or {}
    stage1_root, stage3_root, output_root = Path(stage1_dir).resolve(), Path(stage3_dir).resolve(), Path(output_dir).resolve()
    videos_output = output_root / "videos"
    videos_output.mkdir(parents=True, exist_ok=True)
    available = list_stage3_video_ids(stage3_root)
    selected = [value for value in available if not video_ids or value in video_ids]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ArtifactValidationError("筛选后没有可处理的 Stage 3 视频")
    mode = str(config["runtime"].get("mode", "predict"))
    # 旁路时绝不初始化 OpenAI 客户端，更不会触发 healthcheck 网络访问。
    backend = None if mode == "passthrough" else build_backend(config)
    if backend is not None and bool(config["api"].get("healthcheck_on_start", True)):
        backend.healthcheck()
    run_info = {
        "schema_version": STAGE3_5_SCHEMA_VERSION,
        "started_at": utc_now_iso(),
        "stage1_dir": str(stage1_root),
        "stage3_dir": str(stage3_root),
        "mode": mode,
        "backend": None if mode == "passthrough" else config["runtime"].get("backend", "openai"),
        "config_sha256": mapping_sha256(config),
        "selected_video_count": len(selected),
    }
    write_json(output_root / "resolved_config.json", deepcopy(config))
    write_json(output_root / "run_manifest.json", run_info)
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, video_id in enumerate(selected, start=1):
        if logger:
            logger.info("[%d/%d] Stage 3.5 处理 video_id=%s mode=%s", position, len(selected), video_id, mode)
        try:
            record = process_video(stage1_root, stage3_root, video_id, videos_output, backend, config, project_paths_config,resume, overwrite)
        except Exception as error:
            record = failure_record(video_id, error)
            record["traceback"] = traceback.format_exc()
            failures.append(record)
            if logger:
                logger.exception("Stage 3.5 video_id=%s 处理失败", video_id)
            if strict:
                manifest.append(record)
                write_jsonl(output_root / "manifest.jsonl", manifest)
                write_jsonl(output_root / "failures.jsonl", failures)
                raise
        manifest.append(record)
        write_jsonl(output_root / "manifest.jsonl", manifest)
        write_jsonl(output_root / "failures.jsonl", failures)
    summary = {
        **run_info,
        "finished_at": utc_now_iso(),
        "success_count": sum(row["status"] == "success" for row in manifest),
        "skipped_count": sum(row["status"] == "skipped" for row in manifest),
        "failure_count": len(failures),
    }
    write_json(output_root / "run_manifest.json", summary)
    if not failures:
        write_json(output_root / "_SUCCESS.json", summary)
    return summary
