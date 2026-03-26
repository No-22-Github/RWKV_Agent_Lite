"""Tests for normalized upstream provider timeouts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import typer
from rich.console import Console

from rvoone.cli.runtime import make_provider
from rvoone.config.schema import Config
from rvoone.providers.custom_provider import CustomProvider

DEFAULT_UPSTREAM_TIMEOUT_S = Config().providers.upstream_timeout


def test_custom_provider_uses_normalized_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_async_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )

    monkeypatch.setattr("rvoone.providers.custom_provider.AsyncOpenAI", fake_async_openai)

    CustomProvider(api_key="key", api_base="http://localhost:8000/v1", default_model="model")

    assert captured["timeout"] == DEFAULT_UPSTREAM_TIMEOUT_S
    assert captured["max_retries"] == 0


def test_make_provider_passes_request_dump_to_custom_provider() -> None:
    config = Config()
    config.agents.defaults.model = "custom/test-model"
    config.agents.defaults.provider = "custom"
    config.providers.custom.api_base = "http://localhost:8000/v1"
    config.providers.custom.request_dump = True
    config.providers.custom.stream_mode = "off"

    provider = make_provider(config, Console())

    assert isinstance(provider, CustomProvider)
    assert provider.request_dump is True
    assert provider.stream_mode == "off"


@pytest.mark.asyncio
async def test_custom_provider_uses_plain_openai_client_and_chat_payload(monkeypatch) -> None:
    client_kwargs: dict[str, object] = {}
    request_kwargs: dict[str, object] = {}

    class _FakeCreate:
        async def __call__(self, **kwargs):
            request_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    def fake_async_openai(**kwargs):
        client_kwargs.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        )

    monkeypatch.setattr("rvoone.providers.custom_provider.AsyncOpenAI", fake_async_openai)

    provider = CustomProvider(
        api_key="key",
        api_base="http://localhost:8000/v1",
        default_model="model",
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}], model="model")

    assert "default_headers" not in client_kwargs
    assert request_kwargs["model"] == "model"
    assert request_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_custom_provider_collapses_repeated_newlines_in_text_messages(monkeypatch) -> None:
    request_kwargs: dict[str, object] = {}

    class _FakeCreate:
        async def __call__(self, **kwargs):
            request_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    monkeypatch.setattr(
        "rvoone.providers.custom_provider.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        ),
    )

    provider = CustomProvider(
        api_key="key",
        api_base="http://localhost:8000/v1",
        default_model="model",
    )

    await provider.chat(
        messages=[
            {"role": "system", "content": "a\n\nb\n\n\nc"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello\r\n\r\nworld"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                ],
            },
        ],
        model="model",
    )

    assert request_kwargs["messages"] == [
        {"role": "system", "content": "a\n------\nb\n------\nc"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello\n------\nworld"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        },
    ]


@pytest.mark.asyncio
async def test_custom_provider_token_estimation_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(
        "rvoone.providers.base._get_tokenizer",
        lambda model="gpt-4o": (_ for _ in ()).throw(RuntimeError("tokenizer unavailable")),
    )

    class _FakeCreate:
        async def __call__(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    monkeypatch.setattr(
        "rvoone.providers.custom_provider.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        ),
    )

    provider = CustomProvider(
        api_key="key",
        api_base="http://localhost:8000/v1",
        default_model="model",
        token_estimation="on",
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}], model="model")

    assert response.content == "ok"


@pytest.mark.asyncio
async def test_custom_provider_logs_full_request_payload_when_enabled(monkeypatch) -> None:
    debug_calls: list[tuple[object, ...]] = []

    class _FakeCreate:
        async def __call__(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    monkeypatch.setattr(
        "rvoone.providers.custom_provider.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        ),
    )
    monkeypatch.setattr(
        "rvoone.providers.custom_provider.logger.debug",
        lambda *args: debug_calls.append(args),
    )

    provider = CustomProvider(
        api_key="key",
        api_base="http://localhost:8000/v1",
        default_model="model",
        request_dump=True,
        stream_mode="off",
    )

    await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="model",
        tools=[{"type": "function", "function": {"name": "probe", "parameters": {}}}],
    )

    payload_logs = [
        call
        for call in debug_calls
        if call and call[0] == "LLM_REQ_BODY upstream={} model={} payload=\n{}"
    ]

    assert payload_logs
    _, upstream, model, payload = payload_logs[-1]
    payload_text = str(payload)
    assert upstream == "http://localhost:8000/v1"
    assert model == "model"
    assert '"messages"' in payload_text
    assert '"tools"' in payload_text
    assert '"tool_choice": "auto"' in payload_text


@pytest.mark.asyncio
async def test_custom_provider_stream_mode_forces_streaming(monkeypatch) -> None:
    request_kwargs: dict[str, object] = {}

    class _FakeStream:
        def __init__(self, chunks: list[object]):
            self._chunks = chunks
            self._index = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

        async def close(self) -> None:
            self.closed = True

    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="Hi", tool_calls=None, reasoning_content=None
                        ),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        ]
    )

    class _FakeCreate:
        async def __call__(self, **kwargs):
            request_kwargs.update(kwargs)
            return stream

    monkeypatch.setattr(
        "rvoone.providers.custom_provider.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        ),
    )

    provider = CustomProvider(
        api_key="key",
        api_base="http://localhost:8000/v1",
        default_model="model",
        stream_mode="on",
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}], model="model")

    assert request_kwargs["stream"] is True
    assert response.content == "Hi"
    assert response.usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.mark.asyncio
async def test_custom_provider_stream_mode_disables_streaming(monkeypatch) -> None:
    request_kwargs: dict[str, object] = {}

    class _FakeCreate:
        async def __call__(self, **kwargs):
            request_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    monkeypatch.setattr(
        "rvoone.providers.custom_provider.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        ),
    )

    provider = CustomProvider(
        api_key="key",
        api_base="http://localhost:8000/v1",
        default_model="model",
        stream_mode="off",
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}], model="model")

    assert "stream" not in request_kwargs
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_custom_provider_streams_text_deltas_when_callback_provided(monkeypatch) -> None:
    request_kwargs: dict[str, object] = {}

    class _FakeStream:
        def __init__(self, chunks: list[object]):
            self._chunks = chunks
            self._index = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

        async def close(self) -> None:
            self.closed = True

    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="Hel", tool_calls=None, reasoning_content=None
                        ),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            content="lo", tool_calls=None, reasoning_content=None
                        ),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            ),
        ]
    )

    class _FakeCreate:
        async def __call__(self, **kwargs):
            request_kwargs.update(kwargs)
            return stream

    monkeypatch.setattr(
        "rvoone.providers.custom_provider.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeCreate()))
        ),
    )

    provider = CustomProvider(
        api_key="key", api_base="http://localhost:8000/v1", default_model="model"
    )
    deltas: list[str] = []

    async def _collect(delta: str) -> None:
        deltas.append(delta)

    response = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="model",
        on_text_delta=_collect,
    )

    assert request_kwargs["stream"] is True
    assert deltas == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert stream.closed is True


def test_make_provider_passes_configured_timeout_to_custom_provider() -> None:
    config = Config()
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_base = "http://localhost:8000/v1"
    config.providers.upstream_timeout = 17

    provider = make_provider(config, Console(record=True))

    assert isinstance(provider, CustomProvider)
    assert provider._client.timeout == 17


def test_make_provider_keeps_custom_provider_minimal() -> None:
    config = Config()
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_base = "http://localhost:8000/v1"

    provider = make_provider(config, Console(record=True))

    assert isinstance(provider, CustomProvider)
    assert not hasattr(provider, "extra_body")


def test_make_provider_passes_token_estimation_to_custom_provider() -> None:
    config = Config()
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_base = "http://localhost:8000/v1"
    config.providers.custom.token_estimation = "auto"

    provider = make_provider(config, Console(record=True))

    assert isinstance(provider, CustomProvider)
    assert provider.token_estimation == "auto"


def test_make_provider_uses_explicit_custom_source_prefix() -> None:
    config = Config()
    config.agents.defaults.model = "siliconflow/deepseek-ai/DeepSeek-V3"
    config.providers.custom_sources["siliconflow"] = Config().providers.custom.model_copy(
        update={
            "api_key": "source-key",
            "api_base": "https://example.com/v1",
        }
    )

    provider = make_provider(config, Console(record=True))

    assert isinstance(provider, CustomProvider)
    assert provider.default_model == "deepseek-ai/DeepSeek-V3"
    assert provider.api_base == "https://example.com/v1"


def test_make_provider_requires_explicit_prefix_for_custom_source() -> None:
    config = Config()
    config.agents.defaults.model = "qwen/qwen3-coder"
    config.providers.custom_sources["first"] = Config().providers.custom.model_copy(
        update={"api_base": "https://first.example/v1", "api_key": "k1"}
    )
    config.providers.custom_sources["second"] = Config().providers.custom.model_copy(
        update={"api_base": "https://second.example/v1", "api_key": "k2"}
    )

    with pytest.raises(typer.Exit):
        make_provider(config, Console(record=True))
