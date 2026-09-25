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
    _validate_observations,
    render_stage3_5_observation,
    visualize_stage3_5_video,
)


def observation(frame: int = 2) -> dict:
    return {
        "schema_version": "stage3.5.v2",
        "video_id": "0",
        "interval_id": "0_interval_0000",
        "sample_index": 0,
        "frame": frame,
        "timestamp_sec": frame / 10.0,
        "group_mode": "multiple",
        "grounding_phrases": ["dog", "person"],
        "status": "predicted",
        "targets": [
            {
                "target_id": "dog",
                "grounding_phrase": "dog",
                "subject_point": [0.25, 0.5],
                "confidence": 0.9,
                "visibility": "visible",
            },
            {
                "target_id": "owner",
                "grounding_phrase": "person",
                "subject_point": [0.75, 0.5],
                "confidence": 0.8,
                "visibility": "visible",
            },
        ],
    }


class RenderingTests(unittest.TestCase):
    def test_draws_individual_points_and_group_center_without_mutating_source(self) -> None:
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        source = frame.copy()
        rendered = render_stage3_5_observation(frame, observation())
        self.assertTrue(np.array_equal(frame, source))
        self.assertGreater(int(rendered.sum()), 0)
        # 两个点中心分别约为 x=100/299，组合中心约为 x=200。
        self.assertGreater(int(rendered[120, 100].sum()), 0)
        self.assertGreater(int(rendered[120, 299].sum()), 0)
        self.assertGreater(int(rendered[120, 200].sum()), 0)

    def test_invalid_point_is_rejected_by_video_validation(self) -> None:
        row = observation()
        row["targets"][0]["subject_point"] = [1.2, 0.5]
        with self.assertRaises(ArtifactValidationError):
            _validate_observations([row], "0", 20)


class PipelineTests(unittest.TestCase):
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
            })
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "subject_observations.jsonl", [observation()])
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
            self.assertEqual(manifest["point_count"], 2)
            persisted = json.loads((root / "可视化/0/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["observation_count"], 1)


if __name__ == "__main__":
    unittest.main()
