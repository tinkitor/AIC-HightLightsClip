"""构造单帧主体观察提示、图像内容和 OpenAI JSON Schema。"""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any

from .frame_sampler import SampledFrame


_INTEGER_POINT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "原图坐标系中的 [x,y] 千分制整数点；左上为 [0,0]，右下为 [1000,1000]。",
    "items": {
        "type": "integer",
        "minimum": 0,
        "maximum": 1000,
        "description": "x 或 y 的千分制整数坐标。",
    },
    "minItems": 2,
    "maxItems": 2,
}


OUTPUT_SCHEMA_OPENAI: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "subject_observation",
        "strict": True,
        "schema": {
            "type": "object",
            "description": "当前单帧的主体检测语义和唯一推荐构图中心。",
            "properties": {
                "targets": {
                    "type": "array",
                    "description": "当前帧中需要检测和跟踪的可见实体；仅保留主主体和直接参与核心事件的实体。",
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "description": "一个可见、完整且适合 Grounding DINO 检测的实体。",
                        "properties": {
                            "grounding_phrase": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 40,
                                "description": "简短、具体、全小写的英文实体名词短语。",
                            },
                            "focus_phrase": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 40,
                                "description": "该实体内部最值得作为构图视觉中心的可检测英文部位短语；没有合适部位时复制 grounding_phrase。",
                            },
                            "primary": {
                                "type": "boolean",
                                "description": "该实体是否直接对应 required_highlight_subject。",
                            },
                        },
                        "required": ["grounding_phrase", "focus_phrase", "primary"],
                        "additionalProperties": False,
                    },
                },
                "reason": {
                    "type": "string",
                    "maxLength": 120,
                    "description": "一句话说明主主体识别和 composition_center 选择依据。",
                },
                "composition_center": {
                    **_INTEGER_POINT_SCHEMA,
                    "type": ["array", "null"],
                    "description": "尚未考虑画面边界的唯一理想裁剪窗口中心；主主体不可见或无法可靠判断时为 null。",
                },
            },
            "required": ["targets", "reason", "composition_center"],
            "additionalProperties": False,
        },
    },
}


def output_schema_for_sample_indices(sample_indices: list[int]) -> dict[str, Any]:
    """校验单帧调用约束并返回独立的响应 schema。"""

    indices = [int(value) for value in sample_indices]
    if len(indices) != 1 or indices[0] < 0:
        raise ValueError("简化响应 schema 每次只允许一个非负 sample_index")
    return deepcopy(OUTPUT_SCHEMA_OPENAI)


DEFAULT_SYSTEM_PROMPT = """你是单帧主体语义与竖屏/横屏重构图分析器。
composition_center 是唯一允许输出的坐标，必须使用原图坐标系内 0 到 1000 的整数 [x,y]；左上为 [0,0]，右下为 [1000,1000]。

严格遵守：
1. required_highlight_subject 是唯一主主体来源。只有直接对应它的可见实体才可设 primary=true；不可见时不得把其他物体升级为主主体。
2. targets 只包含需要独立检测和跟踪的完整可见实体，最多 4 个。道路、地板、门窗、墙、家具、植物等通常属于背景；仅当它本身就是 required_highlight_subject，或直接参与 highlight_context 的核心事件时才可加入。
3. 不得把手、脚等身体部位从可识别的完整人物或动物中拆成独立 target。targets 只描述实体语义，不得包含任何坐标。
4. grounding_phrase 使用简短、具体、全小写英文实体名词，不写动作句、方位词或宽泛场景词。
5. focus_phrase 只写该 target 内部最值得作为构图视觉中心、且适合开放词汇检测的英文名词短语，例如 face、upper body、dog head；不得写坐标、方位词、动作句或另一个实体。没有可靠可检测部位时复制 grounding_phrase。
6. composition_center 表示理想裁剪窗口中心，不是主体几何中心，也不要求落在主体内部。结合 target_crop_ratio，先保证所有 primary 完整，再为动作方向、视线和必要互动对象保留合理空间。
7. 不得用 [500,500] 表示不确定；只有画面主体关系确实适合居中时才输出它。没有可靠可见 primary，或无法可靠判断构图中心时，必须输出 composition_center=null；没有可靠可见 primary 时同时输出 targets=[]。
8. reason 只写一句简短说明。只输出符合 schema 的一个 JSON 对象，完成后立即停止。"""


def _target_ratio(metadata: dict[str, Any] | None) -> str:
    ratio = (metadata or {}).get("targetRatioWH", [16, 9])
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 2:
        ratio = [16, 9]
    try:
        width, height = float(ratio[0]), float(ratio[1])
    except (TypeError, ValueError):
        width, height = 16.0, 9.0
    if width <= 0 or height <= 0:
        width, height = 16.0, 9.0
    return f"{width:g}:{height:g}"


def build_prompt(
    video_id: str,
    interval: dict[str, Any],
    frames: list[SampledFrame],
    use_batch: bool = False,
    metadata: dict[str, Any] | None = None,
) -> str:
    """构造一个单帧任务；边界合法化等确定性工作由代码完成。"""

    del video_id
    if use_batch or len(frames) != 1:
        raise ValueError("简化 Stage 3.5 提示词仅支持单帧调用")
    subject = str(interval.get("subject") or "画面中的主要高光主体").strip()
    context = str(interval.get("reason") or "").strip()
    return (
        f"required_highlight_subject={subject}\n"
        f"highlight_context={context}\n"
        f"target_crop_ratio={_target_ratio(metadata)}\n"
        "只分析随附的这一帧。先找与 required_highlight_subject 直接对应的可见主主体，"
        "再决定少量必要 targets 和唯一的理想 composition_center。边界裁剪由程序处理。"
    )


def build_user_content(prompt: str, frames: list[SampledFrame]) -> list[dict[str, Any]]:
    """图像只以进程内 data URL 发送；调用完成后不会持久化 Base64。"""

    if len(frames) != 1:
        raise ValueError("简化 Stage 3.5 请求每次必须且只能包含一帧")
    data = base64.b64encode(frames[0].jpeg_bytes).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
    ]
