"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import json
import inspect
import time
from typing import Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from rvoone.providers.base import (
    LLMProvider,
    LLMResponse,
    TextDeltaHandler,
    ToolCallRequest,
    maybe_estimate_tokens,
)


def _preview_text(value: str | None, limit: int = 120) -> str:
    """Build a short single-line preview for logs."""
    if not value:
        return ""
    text = value.replace("\n", "\\n")
    return text[:limit] + "..." if len(text) > limit else text


def _extract_last_user_preview(messages: list[dict[str, Any]]) -> str:
    """Get the latest user text payload for request logs."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return _preview_text(content)
        if isinstance(content, list):
            for item in reversed(content):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    return _preview_text(item.get("text", ""))
            return "[non-text user content]"
    return ""


def _tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    """Extract tool names for concise observability."""
    if not tools:
        return []
    names: list[str] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _usage_counts(usage: dict[str, int]) -> tuple[int | None, int | None, int | None]:
    """Normalize prompt/completion token names for logs."""
    input_tokens = usage.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


async def _emit_text_delta(callback: TextDeltaHandler | None, delta: str) -> None:
    """Forward a streamed text delta if one was provided."""
    if callback and delta:
        await callback(delta)


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    """Parse streamed tool-call arguments into a dict payload."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json_repair.loads(value)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    return {"raw": value}


async def _maybe_await_close(stream: Any) -> None:
    """Close OpenAI async streams when the SDK exposes a close hook."""
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _stream_finish_reason(choice: Any, current: str) -> str:
    """Prefer an explicit finish_reason from the latest streamed chunk."""
    finish_reason = getattr(choice, "finish_reason", None)
    return finish_reason or current


class CustomProvider(LLMProvider):
    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
        timeout: int = 60,
        request_dump: bool = False,
        stream_mode: str = "auto",
        token_estimation: str = "off",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.request_dump = request_dump
        self.stream_mode = stream_mode
        self.token_estimation = token_estimation
        # Match the Codex provider's single-request budget instead of the OpenAI SDK's long default.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
        )

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
        resolved_model = model or self.default_model
        clean_messages = self._sanitize_empty_content(messages)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": clean_messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        upstream = self.api_base or "default-openai-compatible"
        estimated_tokens = maybe_estimate_tokens(
            clean_messages,
            tools,
            resolved_model,
            self.token_estimation,
            upstream=upstream,
        )
        if estimated_tokens is None:
            logger.info(
                "LLM_REQ upstream={} model={} messages={} tools={}",
                upstream,
                resolved_model,
                len(clean_messages),
                len(tools) if tools else 0,
            )
        else:
            logger.info(
                "LLM_REQ upstream={} model={} messages={} tools={} estimated_tokens={}",
                upstream,
                resolved_model,
                len(clean_messages),
                len(tools) if tools else 0,
                estimated_tokens,
            )
        logger.debug(
            "LLM_REQ detail upstream={} model={} max_tokens={} temperature={} tool_names={} last_user={}",
            upstream,
            resolved_model,
            kwargs["max_tokens"],
            kwargs["temperature"],
            _tool_names(tools),
            _extract_last_user_preview(clean_messages),
        )
        request_payload = dict(kwargs)
        if self.stream_mode == "on":
            request_payload["stream"] = True
        elif self.stream_mode == "auto" and on_text_delta:
            request_payload["stream"] = True
        if self.request_dump:
            logger.debug(
                "LLM_REQ_BODY upstream={} model={} payload=\n{}",
                upstream,
                resolved_model,
                json.dumps(request_payload, ensure_ascii=False, indent=2),
            )
        start_time = time.perf_counter()
        try:
            if self.stream_mode == "on" or (self.stream_mode == "auto" and on_text_delta):
                kwargs["stream"] = True
                parsed = await self._parse_stream(
                    await self._client.chat.completions.create(**kwargs),
                    on_text_delta=on_text_delta,
                )
            else:
                parsed = self._parse(await self._client.chat.completions.create(**kwargs))
            duration = time.perf_counter() - start_time
            input_tokens, output_tokens, total_tokens = _usage_counts(parsed.usage)
            logger.info(
                "LLM_RESP upstream={} model={} duration={:.2f}s finish_reason={} input_tokens={} output_tokens={} total_tokens={}",
                upstream,
                resolved_model,
                duration,
                parsed.finish_reason,
                input_tokens,
                output_tokens,
                total_tokens,
            )
            logger.debug(
                "LLM_RESP detail upstream={} model={} tool_calls={} content={}",
                upstream,
                resolved_model,
                [tc.name for tc in parsed.tool_calls],
                _preview_text(parsed.content),
            )
            return parsed
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                "Upstream error !! {} model={} duration={:.2f}s error={}",
                upstream,
                resolved_model,
                duration,
                str(e),
            )
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(
                id=tc.id,
                name=tc.function.name,
                arguments=json_repair.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments,
            )
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
            if u
            else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    async def _parse_stream(
        self,
        stream: Any,
        *,
        on_text_delta: TextDeltaHandler | None = None,
    ) -> LLMResponse:
        """Aggregate a streamed OpenAI-compatible chat completion into one response."""
        content_parts: list[str] = []
        tool_call_buffers: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}
        reasoning_parts: list[str] = []

        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage:
                        usage = {
                            "prompt_tokens": getattr(chunk_usage, "prompt_tokens", None),
                            "completion_tokens": getattr(chunk_usage, "completion_tokens", None),
                            "total_tokens": getattr(chunk_usage, "total_tokens", None),
                        }
                        usage = {key: value for key, value in usage.items() if value is not None}
                    continue

                choice = choices[0]
                finish_reason = _stream_finish_reason(choice, finish_reason)
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                content_delta = getattr(delta, "content", None)
                if isinstance(content_delta, str) and content_delta:
                    content_parts.append(content_delta)
                    await _emit_text_delta(on_text_delta, content_delta)

                reasoning_delta = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    reasoning_parts.append(reasoning_delta)

                for tool_delta in getattr(delta, "tool_calls", None) or []:
                    index = getattr(tool_delta, "index", 0) or 0
                    buf = tool_call_buffers.setdefault(
                        index, {"id": None, "name": None, "arguments": ""}
                    )
                    tool_id = getattr(tool_delta, "id", None)
                    if tool_id:
                        buf["id"] = tool_id
                    fn = getattr(tool_delta, "function", None)
                    if fn is None:
                        continue
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        buf["name"] = fn_name
                    fn_args = getattr(fn, "arguments", None)
                    if isinstance(fn_args, str) and fn_args:
                        buf["arguments"] += fn_args
        finally:
            await _maybe_await_close(stream)

        tool_calls = [
            ToolCallRequest(
                id=str(buf.get("id") or f"call_{index}"),
                name=str(buf.get("name") or ""),
                arguments=_coerce_tool_arguments(buf.get("arguments") or "{}"),
            )
            for index, buf in sorted(tool_call_buffers.items())
            if buf.get("name")
        ]
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model
