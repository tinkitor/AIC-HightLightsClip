"""Stage 4 构图几何、轨迹平滑和独立流水线测试。"""

from __future__ import annotations

import json
import cv2
import numpy as np
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.stage4_subject_crop.boundary_limiter import finalize_bbox, legal_crop_from_state, maximum_crop_width
from video_highlight.stage4_subject_crop.crop_candidates import generate_crop_candidates
from video_highlight.stage4_subject_crop.composition_scorer import composition_cost
from video_highlight.stage4_subject_crop.grounding_selector import (
    GroundedDetection,
    associate_object_ids,
    select_detections,
    stabilize_object_assignments,
)
from video_highlight.stage4_subject_crop.pipeline import run_stage4
from video_highlight.stage4_subject_crop.pipeline import _plan_single_scene_span, _planning_spans
from video_highlight.stage4_subject_crop.mask_geometry import build_mask_evidence, mask_crop_metrics
from video_highlight.stage4_subject_crop.propagation_visualizer import PropagationVisualizer, annotate_final_composition
from video_highlight.stage4_subject_crop.sam2_adapter import (
    SAM2RecoveryGate,
    SAM2SubjectTracker,
    associate_targets_to_objects,
    anchor_windows,
    update_grounding_phrase_memory,
    update_primary_object_state,
)
from video_highlight.stage4_subject_crop.subject_tracker import CenterSubjectTracker, TrackPoint
from video_highlight.stage4_subject_crop.trajectory_smoother import smooth_trajectory


def center_config() -> dict:
    return {
        "runtime": {"interval_error_policy": "center"},
        "tracking": {"backend": "center", "initial_width_ratio": 0.25, "initial_height_ratio": 0.4},
        "crop_candidates": {"scales": [0.7, 1.0], "offsets": [0.0], "subject_margins": [0.2, 0.2, 0.2, 0.2]},
        "composition": {"weights": {"uncovered": 0.55, "centering": 0.25, "zoom": 0.2}},
        "optimizer": {"center_weight": 0.2, "scale_weight": 0.12},
        "smoothing": {"center_alpha": 0.25, "width_alpha": 0.15, "max_center_step_ratio": 0.04},
    }


class GeometryTests(unittest.TestCase):
    def test_visualization_sampling_and_forced_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            visualizer = PropagationVisualizer(
                {
                    "enabled": True,
                    "sample_fps": 2.0,
                    "always_save_anchor_frames": True,
                    "always_save_fallback_frames": True,
                    "save_images": True,
                    "write_video": False,
                },
                Path(temp),
                source_fps=30.0,
                span_start=100,
            )
            self.assertTrue(visualizer.should_save(100, is_anchor=False, is_fallback=False))
            self.assertFalse(visualizer.should_save(114, is_anchor=False, is_fallback=False))
            self.assertTrue(visualizer.should_save(115, is_anchor=False, is_fallback=False))
            self.assertTrue(visualizer.should_save(107, is_anchor=True, is_fallback=False))
            self.assertTrue(visualizer.should_save(108, is_anchor=False, is_fallback=True))

    def test_disabled_visualization_does_not_create_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "disabled"
            visualizer = PropagationVisualizer(
                {"enabled": False, "sample_fps": 2.0}, target, 30.0, 0
            )
            self.assertFalse(visualizer.active)
            self.assertFalse(target.exists())

    def test_visualization_writes_to_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "中文目录"
            source = root / "source.jpg"
            root.mkdir(parents=True)
            ok, encoded = cv2.imencode(".jpg", np.zeros((80, 120, 3), dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(str(source))
            output = root / "输出"
            visualizer = PropagationVisualizer(
                {"enabled": True, "sample_fps": 2.0, "save_images": True, "write_video": False},
                output,
                30.0,
                0,
            )
            from video_highlight.stage4_subject_crop.subject_tracker import TrackPoint
            visualizer.save(0, source, TrackPoint(0, [10.0, 10.0, 50.0, 60.0], 0.8, "sam2_anchor"), is_anchor=True)
            self.assertTrue((output / "frame_000000.jpg").is_file())

    def test_visualization_draws_all_evidence_then_final_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpg"
            cv2.imencode(".jpg", np.zeros((200, 400, 3), dtype=np.uint8))[1].tofile(str(source))
            output = root / "i/scene_000"
            visualizer = PropagationVisualizer(
                {"enabled": True, "sample_fps": 2.0, "save_images": True, "write_video": False},
                output, 30.0, 0,
            )
            mask1 = np.zeros((200, 400), dtype=np.uint8)
            mask2 = np.zeros_like(mask1)
            mask1[50:150, 40:120] = 1
            mask2[60:160, 250:340] = 1
            point = TrackPoint(0, [40, 50, 340, 160], 0.9, "sam2_anchor", 2, (1, 2), None, (1,))
            visualizer.save(
                0, source, point,
                qwen_points=[(0.2, 0.5), (0.75, 0.5)],
                grounding_objects=[
                    {"object_id": 1, "box_xyxy": [35, 45, 125, 155], "score": 0.9, "phrase": "person"},
                    {"object_id": 2, "box_xyxy": [245, 55, 345, 165], "score": 0.8, "phrase": "person"},
                ],
                object_masks={1: mask1, 2: mask2},
                is_anchor=True,
            )
            tracking_render = cv2.imdecode(
                np.fromfile(str(output / "frame_000000.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertGreater(tracking_render.shape[1], 400)
            with patch(
                "video_highlight.stage4_subject_crop.propagation_visualizer.cv2.putText",
                wraps=cv2.putText,
            ) as put_text:
                annotate_final_composition(
                    root,
                    "i",
                    [{"frame": 0, "bboxes": [150, 20, 100]}],
                    [{
                        "frame": 0,
                        "frame_size_wh": [400, 200],
                        "recommended_crop_center_xy": [350.0, 30.0],
                        "recommended_crop_confidence": 0.873,
                        "primary_object_ids": [1],
                        "mask_centers": {"1": [80, 100], "2": [295, 110]},
                        "composition_decision": {
                            "top_local_candidates": [{"rank": 1, "center_xy": [200, 100], "local_cost": 0.2}],
                            # 与最终输出框中心重合，用于覆盖“绿色点遮住 DP”的回归场景。
                            "optimized_bbox_xywh": [150, 20, 100, 100],
                            "final_local_cost": 0.25,
                        },
                        "mask_crop_metrics": {"iou": 0.4, "primary_coverage": 0.9, "min_primary_coverage": 0.9, "supporting_coverage": 0.3, "primary_boundary_cut": 0.0},
                    }],
                    (1, 1),
                    {"enabled": True, "annotate_composition": True, "write_video": False},
                    30.0,
                )
            rendered_text = [str(call.args[1]) for call in put_text.call_args_list]
            self.assertIn("R=(350,30) conf=0.873", rendered_text)
            rendered = cv2.imdecode(np.fromfile(str(output / "frame_000000.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(rendered.shape[0], 200)
            self.assertGreater(int(rendered[:, :, 1].max()), 220)
            self.assertGreater(int(rendered[:, :, 2].max()), 220)
            # 重合中心处是绿色输出点，外圈应保留橙色 DP 同心环。
            dp_ring = rendered[55:86, 185:216]
            orange_ring = (
                (dp_ring[:, :, 2] > 180)
                & (dp_ring[:, :, 1] > 80)
                & (dp_ring[:, :, 1] < 230)
                & (dp_ring[:, :, 0] < 100)
            )
            self.assertTrue(bool(np.any(orange_ring)))
            # 原画右侧信息栏应另有 DP 坐标文本，而不只依赖主画面圆点。
            sidebar = rendered[:200, 400:]
            orange_sidebar = (
                (sidebar[:, :, 2] > 180)
                & (sidebar[:, :, 1] > 80)
                & (sidebar[:, :, 1] < 230)
                & (sidebar[:, :, 0] < 100)
            )
            self.assertTrue(bool(np.any(orange_sidebar)))
            # Qwen 推荐构图中心在主画面显示紫色 R 点，右栏坐标/置信度同样使用紫色。
            recommended_patch = rendered[20:41, 340:361]
            purple_marker = (
                (recommended_patch[:, :, 0] > 100)
                & (recommended_patch[:, :, 1] < 100)
                & (recommended_patch[:, :, 2] > 100)
            )
            self.assertTrue(bool(np.any(purple_marker)))
            purple_sidebar = (
                (sidebar[:, :, 0] > 100)
                & (sidebar[:, :, 1] < 100)
                & (sidebar[:, :, 2] > 100)
            )
            self.assertTrue(bool(np.any(purple_sidebar)))

    def test_visualization_hides_qwen_and_grounding_box_centers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpg"
            cv2.imencode(".jpg", np.zeros((180, 300, 3), dtype=np.uint8))[1].tofile(str(source))
            output = root / "output"
            visualizer = PropagationVisualizer(
                {"enabled": True, "sample_fps": 2.0, "save_images": True, "write_video": False},
                output, 30.0, 0,
            )
            point = TrackPoint(0, [10, 10, 290, 170], 0.9, "sam2_anchor", 7, (7,), None, (7,))
            visualizer.save(
                0, source, point,
                qwen_points=[(0.8, 0.5)],
                grounding_objects=[
                    {"kind": "raw", "object_id": -1, "box_xyxy": [20, 20, 100, 100], "score": 0.8, "phrase": "person"},
                    {"kind": "selected", "object_id": 7, "box_xyxy": [120, 20, 200, 100], "score": 0.9, "phrase": "person"},
                ],
                is_anchor=True,
            )
            rendered = cv2.imdecode(
                np.fromfile(str(output / "frame_000000.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            # 框线和左上角标签仍存在，但两个框中心及 Qwen subject_point 所在处不应有圆点。
            for x, y in ((60, 60), (160, 60), (239, 90)):
                patch = rendered[y - 3:y + 4, x - 3:x + 4]
                self.assertLess(int(patch.max()), 40)

    def test_portrait_crop_is_legal_after_integer_rounding(self) -> None:
        crop = legal_crop_from_state(5_000, -100, 5_000, (1920, 1080), (9, 16))
        x, y, width = finalize_bbox(crop, (1920, 1080), (9, 16))
        self.assertGreater(width, 0)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 1920)
        self.assertLessEqual(y + width * 16 / 9, 1080 + 1e-6)

    def test_smoothing_preserves_length(self) -> None:
        crops = [(0.0, 0.0, 400.0, 225.0), (300.0, 100.0, 500.0, 281.25)]
        result = smooth_trajectory(crops, (1280, 720), (16, 9), {"center_alpha": 0.2, "width_alpha": 0.2, "max_center_step_ratio": 0.02})
        self.assertEqual(len(result), 2)

    def test_planning_spans_split_at_stage1_scene_cut(self) -> None:
        interval = {"start_frame": 2, "end_frame": 12}
        scenes = [
            {"start_frame": 0, "end_frame": 7},
            {"start_frame": 7, "end_frame": 20},
        ]
        self.assertEqual(_planning_spans(interval, scenes), [(2, 7), (7, 12)])

    def test_fixed_maximum_crop_has_only_one_max_width_candidate(self) -> None:
        candidates = generate_crop_candidates(
            [100.0, 100.0, 200.0, 300.0],
            (1920, 1080),
            (9, 16),
            (500.0, 0.0),
            {"fixed_maximum": True, "scales": [0.55, 0.7], "offsets": [-0.12, 0.12]},
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][2], maximum_crop_width((1920, 1080), (9, 16)))

    def test_qwen_recommended_center_is_added_to_normal_candidates(self) -> None:
        candidates = generate_crop_candidates(
            [400.0, 100.0, 600.0, 400.0],
            (1000, 500),
            (9, 16),
            (0.0, 0.0),
            {"fixed_maximum": True, "offsets": [0.0]},
            recommended_crop_center=(180.0, 250.0),
        )
        self.assertGreater(len(candidates), 1)
        self.assertTrue(any(abs(crop[0] + crop[2] * 0.5 - 180.0) < 1e-6 for crop in candidates))

    def test_weak_qwen_score_prefers_nearer_candidate(self) -> None:
        left = legal_crop_from_state(200.0, 250.0, 281.25, (1000, 500), (9, 16))
        right = legal_crop_from_state(800.0, 250.0, 281.25, (1000, 500), (9, 16))
        config = {
            "weights": {"uncovered": 0.0, "centering": 0.0, "zoom": 0.0},
            "qwen_recommended_center_weight": 0.05,
        }
        left_cost = composition_cost(
            left, [0.0, 0.0, 1000.0, 500.0], (1000, 500), config,
            recommended_crop_center=(200.0, 250.0), recommended_crop_confidence=1.0,
        )
        right_cost = composition_cost(
            right, [0.0, 0.0, 1000.0, 500.0], (1000, 500), config,
            recommended_crop_center=(200.0, 250.0), recommended_crop_confidence=1.0,
        )
        self.assertEqual(left_cost, 0.0)
        self.assertGreater(right_cost, left_cost)

    def test_pure_center_uses_only_recommended_maximum_candidate_even_when_variable(self) -> None:
        points = [
            TrackPoint(
                frame, [400.0, 100.0, 600.0, 400.0], 0.8, "qwen_linear",
                recommended_crop_center=(200.0 + frame * 300.0, 250.0),
                recommended_crop_confidence=0.9,
            )
            for frame in range(2)
        ]
        crops, tracks = _plan_single_scene_span(
            {"video_id": "x", "interval_id": "i", "start_frame": 0, "end_frame": 2},
            points, (1000, 500), (9, 16),
            {
                "crop_candidates": {
                    "fixed_maximum": False,
                    "scales": [0.5, 1.0],
                    "offsets": [-0.12, 0.0, 0.12],
                },
                "composition": {"qwen_recommended_center_weight": 0.05},
                "optimizer": {"center_weight": 0.2, "scale_weight": 0.12},
                "smoothing": {"center_alpha": 0.1, "width_alpha": 0.1, "max_center_step_ratio": 0.01},
            },
        )
        self.assertTrue(all(row["composition_decision"]["candidate_count"] == 1 for row in tracks))
        self.assertTrue(all(row["pure_center_recommended_only"] for row in tracks))
        self.assertTrue(all(row["bboxes"][2] == 281 for row in crops))
        optimized_centers = [
            row["composition_decision"]["optimized_bbox_xywh"][0]
            + row["composition_decision"]["optimized_bbox_xywh"][2] * 0.5
            for row in tracks
        ]
        self.assertAlmostEqual(optimized_centers[0], 200.0)
        self.assertAlmostEqual(optimized_centers[1], 500.0)

    def test_fixed_maximum_uses_mask_to_add_better_center_candidates(self) -> None:
        mask = np.zeros((500, 1000), dtype=np.uint8)
        mask[100:400, 40:190] = 1
        evidence = build_mask_evidence(mask, {1: mask}, 100)
        candidates = generate_crop_candidates(
            [40.0, 100.0, 900.0, 400.0],
            (1000, 500),
            (9, 16),
            (0.0, 0.0),
            {"fixed_maximum": True, "offsets": [0.0], "mask_top_k_per_scale": 4},
            evidence,
        )
        self.assertGreater(len(candidates), 1)
        metrics = [mask_crop_metrics(crop, evidence) for crop in candidates]
        best = candidates[max(range(len(candidates)), key=lambda index: metrics[index].iou)]
        self.assertLess(best[0] + best[2] * 0.5, 300.0)
        self.assertGreater(max(row.coverage for row in metrics), 0.95)

    def test_primary_focus_is_candidate_with_fixed_and_variable_scales(self) -> None:
        for fixed in (True, False):
            candidates = generate_crop_candidates(
                [350.0, 100.0, 650.0, 400.0],
                (1000, 500),
                (9, 16),
                (0.0, 0.0),
                {"fixed_maximum": fixed, "scales": [0.6, 1.0], "offsets": [0.0]},
                primary_focus_points=((120.0, 250.0),),
            )
            self.assertTrue(any(abs(crop[0] + crop[2] * 0.5 - 140.625) < 1 for crop in candidates))

    def test_primary_scoring_beats_larger_supporting_mask(self) -> None:
        primary = np.zeros((500, 1000), dtype=np.uint8)
        supporting = np.zeros_like(primary)
        primary[180:320, 60:160] = 1
        supporting[80:420, 650:950] = 1
        evidence = build_mask_evidence(primary | supporting, {1: primary, 2: supporting}, 100)
        left_crop = legal_crop_from_state(110.0, 250.0, 281.25, (1000, 500), (9, 16))
        right_crop = legal_crop_from_state(800.0, 250.0, 281.25, (1000, 500), (9, 16))
        config = {"primary_mask_weights": {"primary_coverage": 0.5, "primary_center": 0.4, "supporting_coverage": 0.1}}
        left_cost = composition_cost(left_crop, [60, 80, 950, 420], (1000, 500), config, evidence, (1,), ((110, 250),))
        right_cost = composition_cost(right_crop, [60, 80, 950, 420], (1000, 500), config, evidence, (1,), ((110, 250),))
        self.assertLess(left_cost, right_cost)
        self.assertGreater(mask_crop_metrics(left_crop, evidence, primary_object_ids=(1,)).primary_coverage, 0.95)

    def test_variable_scale_candidates_are_not_filtered_by_wide_mask_box(self) -> None:
        mask = np.zeros((500, 1000), dtype=np.uint8)
        mask[180:320, 60:180] = 1
        evidence = build_mask_evidence(mask, {1: mask}, 100)
        candidates = generate_crop_candidates(
            [40.0, 100.0, 960.0, 400.0],
            (1000, 500),
            (9, 16),
            (0.0, 0.0),
            {
                "fixed_maximum": False,
                "scales": [0.5, 1.0],
                "offsets": [0.0],
                "mask_top_k_per_scale": 4,
            },
            evidence,
        )
        widths = {round(crop[2], 3) for crop in candidates}
        self.assertEqual(len(widths), 2)
        config = {
            "mask_weights": {
                "iou": 0.4,
                "coverage": 0.3,
                "object_coverage": 0.2,
                "min_object_coverage": 0.1,
            }
        }
        best = min(candidates, key=lambda crop: composition_cost(
            crop, [40.0, 100.0, 960.0, 400.0], (1000, 500), config, evidence
        ))
        self.assertAlmostEqual(best[2], maximum_crop_width((1000, 500), (9, 16)) * 0.5)

    def test_variable_scale_mask_planning_runs_end_to_end(self) -> None:
        mask = np.zeros((500, 1000), dtype=np.uint8)
        mask[180:320, 60:180] = 1
        evidence = build_mask_evidence(mask, {1: mask}, 100)
        points = [
            TrackPoint(frame, [40.0, 100.0, 960.0, 400.0], 0.9, "sam2_propagated", 1, (1,), evidence)
            for frame in range(2)
        ]
        crops, tracks = _plan_single_scene_span(
            {"video_id": "x", "interval_id": "i", "start_frame": 0, "end_frame": 2},
            points,
            (1000, 500),
            (9, 16),
            {
                "crop_candidates": {
                    "fixed_maximum": False,
                    "scales": [0.5, 1.0],
                    "offsets": [0.0],
                    "subject_margins": [0.2, 0.2, 0.2, 0.2],
                    "mask_top_k_per_scale": 4,
                },
                "composition": {
                    "mask_weights": {
                        "iou": 0.4,
                        "coverage": 0.3,
                        "object_coverage": 0.2,
                        "min_object_coverage": 0.1,
                    }
                },
                "optimizer": {"center_weight": 0.2, "scale_weight": 0.12},
                "smoothing": {
                    "center_alpha": 1.0,
                    "width_alpha": 1.0,
                    "max_center_step_ratio": 1.0,
                },
            },
        )
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0]["bboxes"][2], 140)
        self.assertIn("mask_crop_metrics", tracks[0])
        self.assertIn("bbox_center_mask_metrics", tracks[0])
        self.assertIn("composition_decision", tracks[0])

    def test_center_backend_interpolates_qwen_points_inside_each_scene(self) -> None:
        tracker = CenterSubjectTracker({"initial_width_ratio": 0.2, "initial_height_ratio": 0.2})
        interval = {
            "start_frame": 0,
            "end_frame": 12,
            "subject_points": [
                {"frame": 0, "subject_point": [0.2, 0.5]},
                {"frame": 4, "subject_point": [0.6, 0.5]},
                {"frame": 10, "subject_point": [0.8, 0.5]},
            ],
        }
        scenes = [{"start_frame": 0, "end_frame": 6}, {"start_frame": 6, "end_frame": 12}]
        points = tracker.track(Path(), interval, (1000, 500), scenes)
        centers = [(row.subject_box[0] + row.subject_box[2]) * 0.5 for row in points]
        self.assertAlmostEqual(centers[2], 400.0)
        # 不允许从第一镜头的末点跨硬切插值到第二镜头的首点。
        self.assertAlmostEqual(centers[5], 600.0)
        self.assertAlmostEqual(centers[6], 800.0)
        self.assertEqual(points[2].source, "qwen_linear")

    def test_center_backend_interpolates_recommended_crop_center_independently(self) -> None:
        tracker = CenterSubjectTracker({"initial_width_ratio": 0.2, "initial_height_ratio": 0.2})
        interval = {
            "start_frame": 0,
            "end_frame": 5,
            "subject_observations": [
                {
                    "frame": 0,
                    "targets": [],
                    "recommended_crop_center": [0.2, 0.4],
                    "recommended_crop_confidence": 0.8,
                },
                {
                    "frame": 4,
                    "targets": [],
                    "recommended_crop_center": [0.8, 0.6],
                    "recommended_crop_confidence": 1.0,
                },
            ],
        }
        points = tracker.track(Path(), interval, (1000, 500), [{"start_frame": 0, "end_frame": 5}])
        self.assertEqual(points[2].source, "center_fallback")
        self.assertEqual(points[2].recommended_crop_center, (500.0, 250.0))
        self.assertAlmostEqual(points[2].recommended_crop_confidence, 0.9)

    def test_sam2_windows_restart_at_every_qwen_anchor(self) -> None:
        interval = {
            "start_frame": 10,
            "end_frame": 30,
            "subject_points": [
                {"frame": 12, "subject_point": [0.2, 0.4]},
                {"frame": 20, "subject_point": [0.7, 0.4]},
                {"frame": 25, "subject_point": None},
            ],
        }
        windows = anchor_windows(interval, 10, 30)
        self.assertEqual([(row.start, row.end) for row in windows], [(10, 12), (12, 20), (20, 30)])
        self.assertEqual(windows[0].point, (0.2, 0.4))
        self.assertEqual(windows[2].point, (0.7, 0.4))

    def test_center_backend_unions_multiple_subject_points(self) -> None:
        tracker = CenterSubjectTracker({"initial_width_ratio": 0.1, "initial_height_ratio": 0.2})
        interval = {
            "start_frame": 0,
            "end_frame": 2,
            "subject_observations": [{
                "frame": 0,
                "group_mode": "multiple",
                "grounding_phrases": ["person"],
                "targets": [
                    {"target_id": "left", "subject_point": [0.2, 0.5], "confidence": 0.9},
                    {"target_id": "right", "subject_point": [0.8, 0.5], "confidence": 0.8},
                ],
            }],
        }
        rows = tracker.track(Path(), interval, (1000, 500), [{"start_frame": 0, "end_frame": 2}])
        self.assertEqual(rows[0].object_count, 2)
        self.assertLess(rows[0].subject_box[0], 200)
        self.assertGreater(rows[0].subject_box[2], 800)


class GroundingSelectionTests(unittest.TestCase):
    def test_qwen_primary_target_maps_to_stable_object_id(self) -> None:
        assignments = [
            (7, GroundedDetection((50, 100, 200, 400), 0.8, "woman")),
            (9, GroundedDetection((600, 40, 980, 480), 0.95, "crowd")),
        ]
        observation = {
            "primary_target_ids": ["hero"],
            "targets": [
                {"target_id": "hero", "grounding_phrase": "woman", "subject_point": [0.12, 0.5], "focus_point": [0.12, 0.35], "role": "primary", "importance": 1.0, "confidence": 0.9},
                {"target_id": "crowd", "grounding_phrase": "crowd", "subject_point": [0.8, 0.5], "role": "supporting", "importance": 0.3, "confidence": 0.9},
            ],
        }
        mapped = associate_targets_to_objects(observation, assignments, (1000, 500))
        self.assertTrue(mapped[7]["is_primary"])
        self.assertFalse(mapped[9]["is_primary"])

    def test_primary_switch_requires_two_consecutive_anchors(self) -> None:
        current, challenger, streak, switched = update_primary_object_state((1,), (), 0, (2,), {1, 2}, 2)
        self.assertEqual(current, (1,))
        self.assertEqual((challenger, streak, switched), ((2,), 1, False))
        current, challenger, streak, switched = update_primary_object_state(current, challenger, streak, (2,), {1, 2}, 2)
        self.assertEqual(current, (2,))
        self.assertTrue(switched)

    def test_phrase_memory_survives_one_qwen_omission_then_expires(self) -> None:
        memory: dict[str, int] = {}
        first = update_grounding_phrase_memory(["man", "dog", "car"], memory, 0, 2)
        second = update_grounding_phrase_memory(["man"], memory, 1, 2)
        fourth = update_grounding_phrase_memory(["man"], memory, 3, 2)
        self.assertEqual(first, ["man", "dog", "car"])
        self.assertEqual(second, ["man", "car", "dog"])
        self.assertEqual(fourth, ["man"])

    def test_multiple_qwen_points_claim_distinct_detections(self) -> None:
        detections = [
            GroundedDetection((100, 100, 300, 500), 0.8, "person"),
            GroundedDetection((700, 100, 900, 500), 0.8, "person"),
            GroundedDetection((400, 100, 600, 500), 0.95, "person"),
        ]
        selected, scored = select_detections(
            detections,
            [(0.2, 0.5), (0.8, 0.5)],
            {},
            "multiple",
            (1000, 600),
            {"selection_threshold": 0.3, "multiple_min_point_score": 0.7, "max_objects": 4},
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual({row.box_xyxy for row in selected}, {detections[0].box_xyxy, detections[1].box_xyxy})
        self.assertEqual(len(scored), 3)

    def test_object_ids_are_reused_by_spatial_association(self) -> None:
        detections = [
            GroundedDetection((105, 100, 305, 500), 0.8, "person"),
            GroundedDetection((705, 100, 905, 500), 0.8, "person"),
        ]
        assigned, next_id = associate_object_ids(
            detections,
            {3: [100, 100, 300, 500], 7: [700, 100, 900, 500]},
            8,
            (1000, 600),
            {"association_threshold": 0.3},
        )
        self.assertEqual([object_id for object_id, _ in assigned], [3, 7])
        self.assertEqual(next_id, 8)

    def test_single_qwen_observation_retains_grounded_previous_objects(self) -> None:
        detections = [
            GroundedDetection((105, 100, 305, 500), 0.8, "man"),
            GroundedDetection((405, 100, 605, 500), 0.8, "dog"),
            GroundedDetection((705, 100, 905, 500), 0.8, "car"),
        ]
        selected, _ = select_detections(
            detections,
            [(0.2, 0.5)],
            {
                1: [100, 100, 300, 500],
                2: [400, 100, 600, 500],
                3: [700, 100, 900, 500],
            },
            "single",
            (1000, 600),
            {
                "selection_threshold": 0.45,
                "association_threshold": 0.3,
                "retention_association_threshold": 0.3,
                "retention_detection_threshold": 0.25,
                "max_objects": 8,
            },
        )
        self.assertEqual({row.phrase for row in selected}, {"man", "dog", "car"})

    def test_missing_objects_are_carried_for_two_anchors_then_expire(self) -> None:
        previous = {
            1: [100, 100, 300, 500],
            2: [400, 100, 600, 500],
            3: [700, 100, 900, 500],
        }
        current = [GroundedDetection((105, 100, 305, 500), 0.9, "man")]
        config = {
            "association_threshold": 0.3,
            "object_keepalive_anchors": 2,
            "carry_score_decay": 0.75,
            "max_objects": 8,
        }
        first, next_id, missing, carried, expired = stabilize_object_assignments(
            current, previous, {}, 4, (1000, 600), config
        )
        self.assertEqual({object_id for object_id, _ in first}, {1, 2, 3})
        self.assertEqual(carried, [2, 3])
        self.assertEqual(expired, [])
        self.assertEqual(missing, {1: 0, 2: 1, 3: 1})

        second, next_id, missing, carried, expired = stabilize_object_assignments(
            current, previous, missing, next_id, (1000, 600), config
        )
        self.assertEqual({object_id for object_id, _ in second}, {1, 2, 3})
        self.assertEqual(missing, {1: 0, 2: 2, 3: 2})
        self.assertEqual(expired, [])

        third, _, missing, carried, expired = stabilize_object_assignments(
            current, previous, missing, next_id, (1000, 600), config
        )
        self.assertEqual([object_id for object_id, _ in third], [1])
        self.assertEqual(missing, {1: 0})
        self.assertEqual(carried, [])
        self.assertEqual(expired, [2, 3])

    def test_sam2_unions_only_active_object_masks(self) -> None:
        class FakeTensor:
            def __init__(self, value: np.ndarray) -> None:
                self.value = value

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        tracker = SAM2SubjectTracker.__new__(SAM2SubjectTracker)
        first = np.full((1, 10, 20), -1.0, dtype=np.float32)
        second = np.full((1, 10, 20), -1.0, dtype=np.float32)
        stale = np.full((1, 10, 20), -1.0, dtype=np.float32)
        first[:, 2:5, 1:4] = 1.0
        second[:, 4:8, 12:18] = 1.0
        stale[:, 0:2, 8:10] = 1.0
        union, boxes = tracker._active_masks(
            [1, 2, 99],
            [FakeTensor(first), FakeTensor(second), FakeTensor(stale)],
            {1, 2},
            (20, 10),
        )
        point = tracker._track_point(5, union, "sam2_grounding_dino_propagated", (1, 2))
        self.assertEqual(set(boxes), {1, 2})
        self.assertEqual(point.subject_box, [1.0, 2.0, 18.0, 8.0])
        self.assertEqual(point.object_count, 2)
        self.assertEqual(int(union[0, 8]), 0)

    def test_anchor_validation_uses_nearest_object_not_group_union_center(self) -> None:
        from video_highlight.stage4_subject_crop.subject_tracker import TrackPoint

        tracker = SAM2SubjectTracker.__new__(SAM2SubjectTracker)
        tracker.config = {
            "anchor_min_point_coverage": 0.5,
            "anchor_max_center_distance_ratio": 0.05,
            "grounding_min_coverage": 0.1,
        }
        prediction = TrackPoint(
            5, [50.0, 40.0, 950.0, 160.0], 0.8, "sam2_anchor", 2, (1, 2)
        )
        mask = np.zeros((200, 1000), dtype=np.uint8)
        valid = tracker._anchor_is_valid(
            prediction,
            mask,
            [(0.10, 0.50)],
            [(50.0, 40.0, 150.0, 160.0), (850.0, 40.0, 950.0, 160.0)],
            {1: [50.0, 40.0, 150.0, 160.0], 2: [850.0, 40.0, 950.0, 160.0]},
            (1000, 200),
        )
        self.assertTrue(valid)


class SAM2RecoveryGateTests(unittest.TestCase):
    def test_valid_masks_pass_immediately_before_any_missing_frame(self) -> None:
        gate = SAM2RecoveryGate(3)
        self.assertEqual([gate.observe(True) for _ in range(4)], [True, True, True, True])

    def test_recovery_requires_consecutive_valid_frames_and_missing_resets_streak(self) -> None:
        gate = SAM2RecoveryGate(3)
        observations = [True, False, True, False, True, True, True, True]
        accepted = [gate.observe(valid) for valid in observations]
        self.assertEqual(accepted, [True, False, False, False, False, False, True, True])


class CenterPipelineTests(unittest.TestCase):
    def test_stage4_consumes_stage1_and_stage3_5_without_stage3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_5_video = root / "stage3_5/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_5_video.mkdir(parents=True)
            write_json(stage1_video / "metadata.json", {"video_id": "0", "source_path": "not-used.mp4", "width": 720, "height": 1280, "display_width": 720, "display_height": 1280, "frame_count": 20, "fps": 30.0, "targetRatioWH": [16, 9]})
            write_jsonl(stage1_video / "scenes.jsonl", [{"scene_id": 0, "start_frame": 0, "end_frame": 20}])
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "enriched_intervals.jsonl", [{"video_id": "0", "interval_id": "0_interval_0000", "start_frame": 2, "end_frame": 6, "subject": "dog"}])
            write_jsonl(stage3_5_video / "subject_points.jsonl", [{"video_id": "0", "interval_id": "0_interval_0000", "sample_index": 0, "frame": 2, "subject_point": [0.3, 0.4]}])
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage4(
                root / "stage1",
                root / "stage3_5",
                root / "stage4",
                center_config(),
                {"video_root": str(root)},
                strict=True,
            )
            rows = [json.loads(line) for line in (root / "stage4/videos/0/crops.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual([row["frame"] for row in rows], [2, 3, 4, 5])
            self.assertTrue(all(len(row["bboxes"]) == 3 for row in rows))
            self.assertFalse((root / "stage2").exists())
            self.assertFalse((root / "stage3").exists())

    def test_stage4_consumes_v2_multi_target_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_5_video = root / "stage3_5/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_5_video.mkdir(parents=True)
            write_json(stage1_video / "metadata.json", {"video_id": "0", "source_path": "not-used.mp4", "width": 1000, "height": 500, "frame_count": 10, "fps": 10.0, "targetRatioWH": [16, 9]})
            write_jsonl(stage1_video / "scenes.jsonl", [{"scene_id": 0, "start_frame": 0, "end_frame": 10}])
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "enriched_intervals.jsonl", [{
                "schema_version": "stage3.5.v2", "video_id": "0", "interval_id": "0_interval_0000",
                "start_frame": 2, "end_frame": 5, "subject": "two people",
                "subject_observation_sample_count": 1,
            }])
            write_jsonl(stage3_5_video / "subject_observations.jsonl", [{
                "schema_version": "stage3.5.v2", "video_id": "0", "interval_id": "0_interval_0000",
                "sample_index": 0, "frame": 2, "group_mode": "multiple",
                "grounding_phrases": ["person"],
                "targets": [
                    {"target_id": "left", "description": "left", "grounding_phrase": "person", "subject_point": [0.2, 0.5], "confidence": 0.9, "visibility": "visible"},
                    {"target_id": "right", "description": "right", "grounding_phrase": "person", "subject_point": [0.8, 0.5], "confidence": 0.8, "visibility": "visible"},
                ],
            }])
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage4(root / "stage1", root / "stage3_5", root / "stage4", center_config(), {"video_root": str(root)}, strict=True)
            tracks = [json.loads(line) for line in (root / "stage4/videos/0/tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertTrue(all(row["object_count"] == 2 for row in tracks))
            self.assertTrue((root / "stage4/videos/0/grounding_anchors.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
