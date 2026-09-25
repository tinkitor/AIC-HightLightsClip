"""无需安装项目包即可启动 Stage 3.5 Qwen 中心点可视化。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.visualization.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["stage3_5", *sys.argv[1:]]))
