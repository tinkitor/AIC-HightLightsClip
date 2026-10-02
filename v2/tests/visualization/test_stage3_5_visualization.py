"""Stage 3.5 Qwen 中心点可视化测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.stage3_5_subject_point.frame_sampler import SampledFrame
from video_highlight.visualization.stage3_5 import (
    _interval_statistics,
    _load_raw_crop_anchors,
    _validate_observations,
    render_stage3_5_observation,
    render_stage3_5_v1_observation,
    visualize_stage3_5_video,
)


def observation(frame: int = 2) -> dict:
    return {
        "schema_version": "stage3.5.v4",
        "video_id": "0",
        "interval_id": "0_interval_0000",
        "sample_index": 0,
        "frame": frame,
        "timestamp_sec": frame / 10.0,
        "group_mode": "multiple",
        "composition_mode": "group_focus",
        "recommended_crop_center": [0.5, 0.5],
        "recommended_crop_confidence": 0.85,
        "_raw_crop_anchor_present": True,
        "_raw_crop_anchor": [0.45, 0.55],
        "primary_target_ids": ["dog", "owner"],
        "grounding_phrases": ["dog", "person"],
        "status": "predicted",
        "subject": "dog and owner",
        "reason": "同时保留狗和主人。",
        "targets": [
            {
                "target_id": "dog",
                "description": "奔跑的狗",
                "grounding_phrase": "dog",
                "role": "primary",
                "importance": 1.0,
                "confidence": 0.9,
                "visibility": "visible",
            },
            {
                "target_id": "owner",
                "description": "狗主人",
                "grounding_phrase": "person",
                "role": "primary",
                "importance": 0.85,
                "confidence": 0.8,
                "visibility": "visible",
            },
        ],
    }


def v1_observation(frame: int = 2) -> dict:
    return {
        "schema_version": "stage3.5.v1",
        "video_id": "0",
        "interval_id": "0_interval_0000",
        "sample_index": 0,
        "frame": frame,
        "timestamp_sec": frame / 10.0,
        "subject": "dog",
        "subject_point": [0.25, 0.5],
        "coordinate_space": "normalized_xy",
        "confidence": 0.9,
        "visibility": "visible",
        "status": "predicted",
        "reason": "主体清晰可见",
    }


class RenderingTests(unittest.TestCase):
    def test_v1_draws_only_single_subject_point_and_sidebar(self) -> None:
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        source = frame.copy()
        rendered = render_stage3_5_v1_observation(
            frame,
            v1_observation(),
            interval_metadata={"reason": "Stage 3 高光原因"},
        )
        self.assertTrue(np.array_equal(frame, source))
        self.assertEqual(rendered.shape[0], 240)
        self.assertGreater(rendered.shape[1], 400)
        self.assertGreater(int(rendered[120, 100].sum()), 0)

    def test_draws_only_composition_center_for_v4_and_sidebar_without_mutating_source(self) -> None:
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        source = frame.copy()
        rendered = render_stage3_5_observation(
            frame,
            observation(),
            metadata={"width": 400, "height": 240, "targetRatioWH": [16, 9]},
            interval_metadata={"reason": "Stage 3 高光原因"},
        )
        self.assertTrue(np.array_equal(frame, source))
        self.assertGreater(int(rendered.sum()), 0)
        self.assertEqual(rendered.shape[0], 240)
        self.assertGreater(rendered.shape[1], 400)
        # v4 主画面只绘制唯一构图中心，不再绘制 target 坐标。
        self.assertGreater(int(rendered[120, 200].sum()), 0)

    def test_invalid_point_is_rejected_by_video_validation(self) -> None:
        row = observation()
        row["targets"][0]["subject_point"] = [1.2, 0.5]
        with self.assertRaises(ArtifactValidationError):
            _validate_observations([row], "0", 20)

    def test_invalid_recommended_center_is_rejected(self) -> None:
        row = observation()
        row["recommended_crop_center"] = [0.5, 1.2]
        with self.assertRaises(ArtifactValidationError):
            _validate_observations([row], "0", 20)

    def test_v1_invalid_subject_point_is_rejected(self) -> None:
        row = v1_observation()
        row["subject_point"] = [1.2, 0.5]
        with self.assertRaises(ArtifactValidationError):
            _validate_observations([row], "0", 20)

    def test_interval_statistics_reports_center_and_primary_contract_mismatch(self) -> None:
        first = observation(frame=2)
        second = observation(frame=3)
        second["recommended_crop_center"] = [0.5, 0.8]
        second["primary_target_ids"] = []
        stats = _interval_statistics([first, second])
        self.assertAlmostEqual(stats["recommended_center_max_jump"], 0.3)
        self.assertEqual(stats["recommended_without_primary_count"], 1)
        self.assertEqual(stats["primary_id_change_count"], 1)


class PipelineTests(unittest.TestCase):
    def test_loads_raw_crop_anchor_by_response_id_and_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / "requests.jsonl", [{
                "response_id": "response-1",
                "interval_id": "interval-1",
                "timeline": [{"sample_index": 3, "frame": 30}],
            }])
            write_jsonl(root / "raw_responses.jsonl", [{
                "response_id": "response-1",
                "text": json.dumps({
                    "targets": [],
                    "reason": "test",
                    "composition_center": [1, 750],
                }),
            }])
            anchors = _load_raw_crop_anchors(root)
            self.assertEqual(anchors[("interval-1", 3)], (0.001, 0.75))

    def test_writes_unicode_safe_image_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_5_video = root / "stage3_5/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_5_video.mkdir(parents=True)
            source = root / "源视频.mp4"
            source.write_bytes(b"placeholder")
            write_json(stage1_video / "metadata.json", {
                "video_id": "0", "source_path": str(source), "fps": 10.0,
                "frame_count": 20, "width": 320, "height": 180,
                "targetRatioWH": [9, 16],
            })
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "subject_observations.jsonl", [observation()])
            write_jsonl(stage3_5_video / "requests.jsonl", [{
                "response_id": "response-1",
                "interval_id": "0_interval_0000",
                "timeline": [{"sample_index": 0, "frame": 2}],
            }])
            write_jsonl(stage3_5_video / "raw_responses.jsonl", [{
                "response_id": "response-1",
                "text": json.dumps({
                    "targets": [],
                    "reason": "test",
                    "composition_center": [450, 550],
                }),
            }])
            write_jsonl(stage3_5_video / "enriched_intervals.jsonl", [{
                "video_id": "0", "interval_id": "0_interval_0000",
                "subject_observation_sample_fps": 2.0,
            }])
            raw = np.zeros((180, 320, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", raw)
            self.assertTrue(ok)
            sampled = [SampledFrame(0, 2, 0.2, encoded.tobytes(), 320, 180, "mock")]
            with patch(
                "video_highlight.visualization.stage3_5.sample_interval",
                return_value=sampled,
            ):
                manifest = visualize_stage3_5_video(
                    root / "stage1", root / "stage3_5", root / "可视化", "0"
                )
            image = root / "可视化/0/0_interval_0000/sample_0000_frame_00000002.jpg"
            self.assertTrue(image.is_file())
            self.assertEqual(manifest["point_count"], 0)
            self.assertEqual(manifest["raw_crop_anchor_count"], 1)
            self.assertEqual(
                manifest["intervals"][0]["stability"]["recommended_center_exact_middle_count"],
                1,
            )
            persisted = json.loads((root / "可视化/0/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["observation_count"], 1)
            self.assertEqual(persisted["visualization"], "stage3_5_qwen_composition")

    def test_v1_subject_points_artifact_uses_version_specific_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_5_video = root / "stage3_5/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_5_video.mkdir(parents=True)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            write_json(stage1_video / "metadata.json", {
                "video_id": "0", "source_path": str(source), "fps": 10.0,
                "frame_count": 20, "width": 320, "height": 180,
            })
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "subject_points.jsonl", [v1_observation()])
            write_jsonl(stage3_5_video / "enriched_intervals.jsonl", [{
                "schema_version": "stage3.5.v1",
                "video_id": "0", "interval_id": "0_interval_0000",
                "subject_point_sample_fps": 2.0,
            }])
            raw = np.zeros((180, 320, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", raw)
            self.assertTrue(ok)
            sampled = [SampledFrame(0, 2, 0.2, encoded.tobytes(), 320, 180, "mock")]
            with patch(
                "video_highlight.visualization.stage3_5.sample_interval",
                return_value=sampled,
            ):
                manifest = visualize_stage3_5_video(
                    root / "stage1", root / "stage3_5", root / "visualization", "0"
                )
            self.assertEqual(manifest["schema_version"], "stage3.5.v1")
            self.assertEqual(manifest["observation_artifact"], "subject_points.jsonl")
            self.assertEqual(manifest["visualization"], "stage3_5_qwen_subject_points")
            self.assertEqual(manifest["point_count"], 1)
            self.assertEqual(
                manifest["intervals"][0]["stability"]["subject_point_count"], 1
            )

    def test_v1_empty_subject_points_is_a_successful_empty_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_5_video = root / "stage3_5/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_5_video.mkdir(parents=True)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            write_json(stage1_video / "metadata.json", {
                "video_id": "0", "source_path": str(source), "fps": 10.0,
                "frame_count": 20, "width": 320, "height": 180,
            })
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "subject_points.jsonl", [])
            manifest = visualize_stage3_5_video(
                root / "stage1", root / "stage3_5", root / "visualization", "0"
            )
            self.assertEqual(manifest["schema_version"], "stage3.5.v1")
            self.assertEqual(manifest["observation_count"], 0)
            self.assertEqual(manifest["interval_count"], 0)
            self.assertEqual(manifest["point_count"], 0)


if __name__ == "__main__":
    unittest.main()
