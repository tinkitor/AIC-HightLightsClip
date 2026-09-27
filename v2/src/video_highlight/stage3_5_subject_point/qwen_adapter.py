"""Stage 3.5 的 vLLM/OpenAI 兼容客户端；不包含任何模型部署代码。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from video_highlight.common.exceptions import ExternalToolError


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    response_id: str | None
    model: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "response_id": self.response_id,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


class SubjectObservationBackend(Protocol):
    def healthcheck(self) -> None: ...
    def analyze(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        response_format: dict[str, Any],
        sample_count: int,
    ) -> ModelResponse: ...


class OpenAIQwenBackend:
    def __init__(self, api: dict[str, Any], generation: dict[str, Any]) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ExternalToolError("缺少 openai 包，请在 video-clip Conda 环境中安装") from error
        self.model = str(api["model"])
        self.generation = generation
        self.client = OpenAI(
            base_url=str(api["base_url"]).rstrip("/") + "/",
            api_key=str(api.get("api_key", "EMPTY")),
            timeout=float(api.get("timeout_sec", 300.0)),
            max_retries=int(api.get("max_retries", 2)),
        )

    def healthcheck(self) -> None:
        try:
            model_ids = {item.id for item in self.client.models.list().data}
        except Exception as error:
            raise ExternalToolError(f"无法访问 vLLM OpenAI API: {error}") from error
        if self.model not in model_ids:
            raise ExternalToolError(f"vLLM 未暴露模型 {self.model!r}，当前模型: {sorted(model_ids)}")

    def analyze(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        response_format: dict[str, Any],
        sample_count: int,
    ) -> ModelResponse:
        del sample_count
        parameters: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": float(self.generation.get("temperature", 0.0)),
            "top_p": float(self.generation.get("top_p", 0.9)),
            "max_tokens": int(self.generation.get("max_tokens", 4096)),
            "response_format": response_format,
        }
        if self.generation.get("extra_body"):
            parameters["extra_body"] = self.generation["extra_body"]
        try:
            response = self.client.chat.completions.create(**parameters)
        except Exception as error:
            raise ExternalToolError(f"Qwen API 调用失败: {error}") from error
        if not response.choices:
            raise ExternalToolError("Qwen API 返回空 choices")
        choice = response.choices[0]
        content = choice.message.content
        text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict)) if isinstance(content, list) else str(content or "")
        finish_reason = getattr(choice, "finish_reason", None)
        if not text.strip():
            raise ExternalToolError(f"Qwen API 返回空 content（finish_reason={finish_reason!r}）")
        if finish_reason == "length":
            raise ExternalToolError(f"Qwen API 输出达到 max_tokens 而被截断,raw_response={choice!r}")
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        return ModelResponse(text, getattr(response, "id", None), getattr(response, "model", None), finish_reason, usage)


class MockQwenBackend:
    """离线链路测试：按精简单帧协议返回一个位于画面中心的目标。"""

    def healthcheck(self) -> None:
        return None

    def analyze(self, system_prompt: str, user_content: list[dict[str, Any]], response_format: dict[str, Any], sample_count: int) -> ModelResponse:
        del system_prompt, user_content, response_format
        if sample_count != 1:
            raise ExternalToolError("MockQwenBackend 仅支持精简单帧请求")
        result = {
            "crop_anchor": [500, 500],
            "targets": [
                {"grounding_phrase": "person", "point": [500, 500], "primary": True}
            ],
            "reason": "mock center",
        }
        return ModelResponse(json.dumps(result), "mock", "mock", "stop", None)


def build_backend(config: dict[str, Any]) -> SubjectObservationBackend:
    name = str(config.get("runtime", {}).get("backend", "openai")).lower()
    if name == "openai":
        return OpenAIQwenBackend(config["api"], config.get("generation", {}))
    if name == "mock":
        return MockQwenBackend()
    raise ExternalToolError(f"未知 Stage 3.5 backend: {name}")
