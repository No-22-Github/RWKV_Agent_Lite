"""Base LLM provider interface."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

# Default timeout moved to config/schema.py: ProvidersConfig.upstream_timeout

_tokenizer_cache: dict[str, Any] = {}
TextDeltaHandler = Callable[[str], Awaitable[None]]


def _get_tokenizer(model: str = "gpt-4o") -> Any:
    """Get a cached tokenizer without importing tiktoken on the hot import path."""
    if model in _tokenizer_cache:
        return _tokenizer_cache[model]

    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised via safe wrapper
        raise RuntimeError("tiktoken is not installed") from exc

    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    _tokenizer_cache[model] = tokenizer
    return tokenizer


def _count_message_content_tokens(content: Any, tokenizer: Any) -> int:
    """Estimate tokens for one message content payload."""
    if isinstance(content, str):
        return len(tokenizer.encode(content))

    total = 0
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                total += len(tokenizer.encode(item.get("text", "")))
            elif item_type in {"image_url", "input_image"}:
                total += 85
            else:
                total += len(tokenizer.encode(json.dumps(item, ensure_ascii=False)))
        return total

    if content is not None:
        total += len(tokenizer.encode(json.dumps(content, ensure_ascii=False)))
    return total


def estimate_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
) -> int:
    """Estimate the request token count using tiktoken."""
    tokenizer = _get_tokenizer(model)
    total = 0

    for msg in messages:
        total += 4
        total += len(tokenizer.encode(str(msg.get("role", ""))))
        total += _count_message_content_tokens(msg.get("content"), tokenizer)

        for key in ("tool_calls", "function_call", "name", "tool_call_id"):
            value = msg.get(key)
            if not value:
                continue
            if isinstance(value, str):
                total += len(tokenizer.encode(value))
            else:
                total += len(tokenizer.encode(json.dumps(value, ensure_ascii=False)))

    if tools:
        total += len(tokenizer.encode(json.dumps(tools, ensure_ascii=False)))

    return total


def maybe_estimate_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    mode: str = "off",
    *,
    upstream: str | None = None,
) -> int | None:
    """Best-effort token estimation that never blocks a real LLM request."""
    if mode == "off":
        return None

    try:
        return estimate_tokens(messages, tools, model)
    except Exception as exc:
        log = logger.warning if mode == "on" else logger.debug
        log(
            "LLM_REQ token estimation unavailable upstream={} model={} mode={} error={}",
            upstream or "unknown",
            model,
            mode,
            str(exc),
        )
        return None


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace empty text content that causes provider 400 errors.

        Empty content can appear when MCP tools return nothing. Most providers
        reject empty-string content or empty text blocks in list content.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = (
                    None
                    if (msg.get("role") == "assistant" and msg.get("tool_calls"))
                    else "(empty)"
                )
                result.append(clean)
                continue

            if isinstance(content, list):
                filtered = [
                    item
                    for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            result.append(msg)
        return result

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        on_text_delta: TextDeltaHandler | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            on_text_delta: Optional callback invoked with streamed text deltas.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
