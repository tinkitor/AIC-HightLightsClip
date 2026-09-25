"""Stage 3.5 采样、解析和跳过模式测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.stage3_5_subject_point.frame_sampler import SampledFrame, _split_mjpeg_stream, plan_sample_frames
from video_highlight.stage3_5_subject_point.pipeline import run_stage3_5
from video_highlight.stage3_5_subject_point.prompt_builder import build_prompt
from video_highlight.stage3_5_subject_point.response_parser import parse_predictions


def passthrough_config() -> dict:
    return {
        "api": {},
        "generation": {},
        "sampling": {"fps": 2.0, "jpeg_quality": 85, "max_side": 1024},
        "parsing": {"retries": 0},
        "runtime": {"mode": "passthrough", "backend": "openai"},
    }


class SamplingTests(unittest.TestCase):
    def test_plan_is_anchored_to_refined_start_and_end_is_exclusive(self) -> None:
        self.assertEqual(plan_sample_frames(7, 38, 30.0, 2.0), [7, 22, 37])

    def test_sample_rate_above_source_deduplicates_frames(self) -> None:
        self.assertEqual(plan_sample_frames(2, 6, 2.0, 10.0), [2, 3, 4, 5])

    def test_mjpeg_pipe_is_split_without_persistent_frame_files(self) -> None:
        first = b"\xff\xd8first\xff\xd9"
        second = b"\xff\xd8second\xff\xd9"
        self.assertEqual(_split_mjpeg_stream(first + second), [first, second])


class ParserTests(unittest.TestCase):
    def test_recommended_crop_center_and_confidence_are_preserved(self) -> None:
        frame = SampledFrame(0, 0, 0.0, b"jpeg", 1000, 500)
        text = json.dumps({"predictions": [{
            "sample_index": 0,
            "recommended_crop_center": [0.62, 0.41],
            "recommended_crop_confidence": 0.83,
            "targets": [],
        }]})
        rows, missing, errors = parse_predictions(text, [frame])
        self.assertEqual(missing, [])
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["recommended_crop_center"], [0.62, 0.41])
        self.assertAlmostEqual(rows[0]["recommended_crop_confidence"], 0.83)

    def test_prompt_describes_target_ratio_and_legal_crop_center(self) -> None:
        frame = SampledFrame(0, 0, 0.0, b"jpeg", 1920, 1080)
        prompt = build_prompt(
            "v", {"interval_id": "i", "subject": "runner", "reason": "overtake"}, [frame],
            metadata={"width": 1920, "height": 1080, "targetRatioWH": [9, 16]},
        )
        self.assertIn("target_ratio_wh=[9,16]", prompt)
        self.assertIn("recommended_crop_center", prompt)
        self.assertIn("highlight_reason=overtake", prompt)

    def test_explicit_primary_role_and_focus_point_are_preserved(self) -> None:
        frame = SampledFrame(0, 0, 0.0, b"jpeg", 1000, 500)
        text = json.dumps({"predictions": [{
            "sample_index": 0,
            "group_mode": "multiple",
            "composition_mode": "single_focus",
            "primary_target_ids": ["hero"],
            "grounding_phrases": ["woman", "crowd"],
            "targets": [
                {"target_id": "hero", "description": "woman", "grounding_phrase": "woman", "subject_point": [0.2, 0.5], "focus_point": [0.2, 0.3], "role": "primary", "importance": 1.0, "confidence": 0.9, "visibility": "visible"},
                {"target_id": "crowd", "description": "crowd", "grounding_phrase": "crowd", "subject_point": [0.8, 0.5], "focus_point": [0.8, 0.5], "role": "supporting", "importance": 0.2, "confidence": 0.8, "visibility": "visible"},
            ],
        }]})
        rows, _, errors = parse_predictions(text, [frame])
        self.assertEqual(rows[0]["primary_target_ids"], ["hero"])
        self.assertEqual(rows[0]["targets"][0]["focus_point"], [0.2, 0.3])
        self.assertEqual(rows[0]["targets"][1]["role"], "supporting")
        self.assertEqual(errors, [])

    def test_missing_sample_is_explicitly_filled(self) -> None:
        frames = [
            SampledFrame(index, index * 5, index * 0.5, b"jpeg", 640, 360)
            for index in range(2)
        ]
        text = json.dumps({"predictions": [{
            "sample_index": 0,
            "group_mode": "multiple",
            "grounding_phrases": ["dog", "person"],
            "targets": [
                {"target_id": "dog", "description": "dog", "grounding_phrase": "dog", "subject_point": [0.25, 0.75], "confidence": 0.9, "visibility": "visible"},
                {"target_id": "owner", "description": "owner", "grounding_phrase": "person", "subject_point": [0.75, 0.70], "confidence": 0.8, "visibility": "visible"},
            ],
            "reason": "ok",
        }]})
        rows, missing, errors = parse_predictions(text, frames)
        self.assertEqual(missing, [1])
        self.assertEqual(rows[0]["targets"][0]["subject_point"], [0.25, 0.75])
        self.assertEqual(len(rows[0]["targets"]), 2)
        self.assertEqual(rows[0]["grounding_phrases"], ["dog", "person"])
        self.assertEqual(rows[1]["targets"], [])
        self.assertEqual(errors, [])

    def test_legacy_single_point_response_is_adapted(self) -> None:
        frame = SampledFrame(4, 20, 2.0, b"jpeg", 100, 100)
        text = json.dumps({"predictions": [{
            "sample_index": 4,
            "subject_point": [0.0, 1.0],
            "confidence": 0.7,
            "visibility": "visible",
            "reason": "legacy",
        }]})
        rows, missing, _ = parse_predictions(text, [frame])
        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["targets"][0]["subject_point"], [0.0, 1.0])


class PassthroughPipelineTests(unittest.TestCase):
    def test_skip_processing_neither_opens_source_video_nor_calls_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_video = root / "stage3/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_video.mkdir(parents=True)
            # 故意给不存在的视频路径：若旁路错误地打开视频，本测试会立即失败。
            write_json(stage1_video / "metadata.json", {"video_id": "0", "source_path": "missing.mp4", "fps": 10.0, "frame_count": 30, "width": 640, "height": 360})
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_video / "refined_intervals.jsonl", [{"schema_version": "stage3.v2", "video_id": "0", "interval_id": "0_interval_0000", "start_frame": 3, "end_frame": 15, "subject": "dog"}])
            write_json(stage3_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage3_5(root / "stage1", root / "stage3", root / "stage3_5", passthrough_config(), strict=True)
            rows = [json.loads(line) for line in (root / "stage3_5/videos/0/subject_observations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual([row["frame"] for row in rows], [3, 8, 13])
            self.assertTrue(all(row["targets"] == [] and row["status"] == "skipped" for row in rows))
            self.assertTrue(all(row["recommended_crop_center"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
