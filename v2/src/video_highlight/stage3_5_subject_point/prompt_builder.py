"""构造逐帧多主体观察提示、图像内容和 OpenAI JSON Schema。"""

from __future__ import annotations

import base64
import math
from typing import Any

from .frame_sampler import SampledFrame


OUTPUT_SCHEMA_OPENAI: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "subject_observation_predictions",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer", "minimum": 0},
                            "group_mode": {"type": "string", "enum": ["single", "multiple"]},
                            "composition_mode": {"type": "string", "enum": ["single_focus", "group_focus"]},
                            "recommended_crop_center": {
                                "type": ["array", "null"],
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "recommended_crop_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "primary_target_ids": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1, "maxLength": 40},
                                "maxItems": 4,
                            },
                            "grounding_phrases": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1, "maxLength": 80},
                                "maxItems": 8,
                            },
                            "targets": {
                                "type": "array",
                                "maxItems": 12,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_id": {"type": "string", "minLength": 1, "maxLength": 40},
                                        "description": {"type": "string", "maxLength": 80},
                                        "grounding_phrase": {"type": "string", "minLength": 1, "maxLength": 80},
                                        "subject_point": {
                                            "type": ["array", "null"],
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        "focus_point": {
                                            "type": ["array", "null"],
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        "role": {"type": "string", "enum": ["primary", "supporting"]},
                                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        "visibility": {"type": "string", "enum": ["visible", "occluded", "not_found"]},
                                    },
                                    "required": [
                                        "target_id", "description", "grounding_phrase", "role", "importance",
                                        "subject_point", "focus_point", "confidence", "visibility"
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "sample_index", "group_mode", "composition_mode", "primary_target_ids",
                            "recommended_crop_center", "recommended_crop_confidence",
                            "grounding_phrases", "targets", "reason"
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["predictions"],
            "additionalProperties": False,
        },
    },
}


DEFAULT_SYSTEM_PROMPT = """你是视频逐帧多主体定位与构图角色分析器。对每张图识别为了完整呈现指定高光主体而必须保留的人物、动物或物体。
必须根据 highlight_subject 优先判断叙事主主体，而不是按目标面积判断。primary_target_ids 只包含真正决定高光语义的目标；其他必要目标
标记为 supporting。单核心使用 composition_mode=single_focus，确实需要多个目标共同表达事件时使用 group_focus。recommended_crop_center
是目标画幅裁剪框在原图中的最终推荐中心，不是任何单个目标的中心；必须综合主次主体、动作/视线方向、互动关系和必要环境，且不要机械地取各目标点平均。
应为运动或视线方向保留空间；无法可靠构图时返回 null 并将 recommended_crop_confidence 设为 0。subject_point 是用
于目标检测关联的视觉中心；focus_point 是希望接近裁剪中心的构图焦点，人物优先脸部或上半身，其他对象使用语义关键部位。不可见或无法可靠定
位时点为 null。importance 表示构图重要性，confidence 只表示定位可信度，两者不得混淆。grounding_phrase 与 grounding_phrases 必须
是简短、具体、全小写的英文名词短语，适合开放词汇目标检测。target_id 应根据身份或外观生成简短稳定标识，同一区间中尽量复用。禁止为了填满
数组而加入背景目标。每个 sample_index 必须且只能返回一次，只返回满足 JSON Schema 的对象。"""


def _crop_context(metadata: dict[str, Any] | None, frames: list[SampledFrame]) -> str:
    """把目标画幅及最大合法框写入提示词，使“裁剪中心”具有确定几何含义。"""

    metadata = metadata or {}
    fallback_width = frames[0].width if frames else 1
    fallback_height = frames[0].height if frames else 1
    width = int(metadata.get("display_width", metadata.get("width", fallback_width)))
    height = int(metadata.get("display_height", metadata.get("height", fallback_height)))
    ratio = metadata.get("targetRatioWH", [16, 9])
    if not isinstance(ratio, list) or len(ratio) != 2:
        ratio = [16, 9]
    target_w, target_h = float(ratio[0]), float(ratio[1])
    if target_w <= 0 or target_h <= 0:
        target_w, target_h = 16.0, 9.0
    crop_width = min(float(width), math.floor(height * target_w / target_h + 1e-9))
    crop_height = crop_width * target_h / target_w
    min_x = crop_width * 0.5 / max(1.0, width)
    max_x = 1.0 - min_x
    min_y = crop_height * 0.5 / max(1.0, height)
    max_y = 1.0 - min_y
    return (
        f"source_frame_wh=[{width},{height}]\n"
        f"target_ratio_wh=[{target_w:g},{target_h:g}]\n"
        "crop_mode=fixed_maximum\n"
        f"maximum_crop_wh=[{crop_width:.3f},{crop_height:.3f}]\n"
        f"legal_crop_center_x=[{min_x:.6f},{max_x:.6f}]\n"
        f"legal_crop_center_y=[{min_y:.6f},{max_y:.6f}]\n"
    )


def build_prompt(
    video_id: str,
    interval: dict[str, Any],
    frames: list[SampledFrame],
    use_batch: bool = False,
    metadata: dict[str, Any] | None = None,
) -> str:
    """根据采样帧信息构造提示词，use_sequence默认为False，此时提示词只会让模型每次只单独判断一帧"""
    subject = interval.get("subject") or "画面中的主要高光主体"
    category = interval.get("category") or "unknown"
    highlight_reason = interval.get("reason") or ""
    interval_id = interval["interval_id"]
    crop_context = _crop_context(metadata, frames)

    if not use_batch:
        # 单帧独立判断模式：通常 frames 只包含当前这一帧
        if not frames:
            raise ValueError("frames 不能为空")
        row = frames[0]
        return (
            f"video_id={video_id}\n"
            f"interval_id={interval_id}\n"
            f"subject={subject}\n"
            f"highlight_category={category}\n"
            f"highlight_reason={highlight_reason}\n"
            f"{crop_context}"
            f"当前帧：sample_index={row.sample_index}, "
            f"original_frame={row.frame}, "
            f"timestamp_sec={row.timestamp_sec:.6f}\n"
            "请列出为了完整呈现该高光主体必须保留的所有目标，并在其中找出 primary_target_ids，"
            "注意primary_target_ids 只包含真正决定高光语义的目标,其他保留目标均标记为 supporting。"
            "不要在预设主体不可见时擅自改成无关主体。"
            "每个可见目标返回归一化中心，并生成可供 Grounding DINO 使用的英文名词短语。"
            "另请基于给定最大合法目标画幅，返回整帧唯一的 recommended_crop_center；它是裁剪框中心而非主体中心。"
            "返回示例："
            '{"predictions":[{"sample_index":34,"group_mode":"multiple","composition_mode":"single_focus",'
            '"primary_target_ids":["player_red"],"recommended_crop_center":[0.62,0.50],'
            '"recommended_crop_confidence":0.90,'
            '"grounding_phrases":["basketball player","person"],"targets":['
            '{"target_id":"player_red","description":"红衣球员","grounding_phrase":"basketball player",'
            '"role":"primary","importance":1.0,"subject_point":[0.68,0.52],'
            '"focus_point":[0.68,0.38],"confidence":0.99,"visibility":"visible"}],'
            '"reason":"主体清晰可见"}]}'
        )
    timeline = "\n".join(
        f"- sample_index={row.sample_index}, original_frame={row.frame}, timestamp_sec={row.timestamp_sec:.6f}"
        for row in frames
    )
    subject = interval.get("subject") or "画面中的主要高光主体"
    return (
        f"video_id={video_id}\ninterval_id={interval['interval_id']}\n"
        f"subject={subject}\nhighlight_category={category}\nhighlight_reason={highlight_reason}\n"
        f"{crop_context}"
        f"采样时间线（后续图像严格按此顺序排列）：\n{timeline}\n"
        "请独立判断每一帧，列出为了完整呈现指定高光主体必须保留的所有目标，并明确主次角色。"
        "同一对象跨帧尽量复用 target_id；可见目标必须返回中心，无法可靠定位时返回 null，禁止改选无关主体。"
        "每帧还必须独立给出 recommended_crop_center，表示给定目标画幅下裁剪框本身的推荐中心。"
    )


def build_user_content(prompt: str, frames: list[SampledFrame]) -> list[dict[str, Any]]:
    """图像只以进程内 data URL 发送；调用完成后不会持久化 Base64。"""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for row in frames:
        content.append({"type": "text", "text": f"sample_index={row.sample_index}"})
        data = base64.b64encode(row.jpeg_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return content
