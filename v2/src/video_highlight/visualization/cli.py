"""阶段产物可视化命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping

from .stage3_5 import run_stage3_5_visualization


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线可视化流水线各阶段产物")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stage3_5 = subparsers.add_parser("stage3_5", aliases=["stage3.5"], help="绘制 Qwen 多主体中心点")
    stage3_5.add_argument("--stage1-dir", type=Path, required=True)
    stage3_5.add_argument("--stage3-5-dir", type=Path, required=True)
    stage3_5.add_argument("--output-dir", type=Path, required=True)
    stage3_5.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    stage3_5.add_argument("--video-id", action="append", dest="video_ids")
    stage3_5.add_argument("--limit", type=int)
    stage3_5.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto")
    stage3_5.add_argument("--ffmpeg-bin", default="ffmpeg")
    stage3_5.add_argument("--max-side", type=int, default=1600, help="输出图片最大边；0 表示原尺寸")
    stage3_5.add_argument("--jpeg-quality", type=int, default=92)
    stage3_5.add_argument("--write-video", action="store_true", help="每个区间额外生成 MP4 预览")
    stage3_5.add_argument("--preview-fps", type=float)
    stage3_5.add_argument("--no-images", action="store_true", help="不保留逐采样帧 JPG，需配合 --write-video")
    stage3_5.add_argument("--no-group-center", action="store_true", help="不绘制多点包围范围与组合中心")
    stage3_5.add_argument("--overwrite", action="store_true")
    stage3_5.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    paths = load_mapping(args.paths_config)
    summary = run_stage3_5_visualization(
        args.stage1_dir,
        args.stage3_5_dir,
        args.output_dir,
        video_ids=args.video_ids,
        limit=args.limit,
        strict=args.strict,
        paths_config=paths,
        save_images=not args.no_images,
        write_video=args.write_video,
        preview_fps=args.preview_fps,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        decoder=args.decoder,
        ffmpeg_bin=args.ffmpeg_bin,
        draw_group_center=not args.no_group_center,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failure_count"] else 0
