"""Tests for the model gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.providers.base import LLMResponse
from rvoone.providers.gateway import ModelGateway


@pytest.mark.asyncio
async def test_model_gateway_delegates_chat_to_active_provider() -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))

    gateway = ModelGateway(provider, "gpt-4o-mini")
    response = await gateway.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    provider.chat.assert_awaited_once_with(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="gpt-4o-mini",
        max_tokens=4096,
        temperature=0.2,
        reasoning_effort=None,
        on_text_delta=None,
    )
    assert response.content == "ok"


def test_model_gateway_switches_provider_via_factory() -> None:
    original = MagicMock()
    replacement = MagicMock()
    factory = MagicMock(return_value=replacement)

    gateway = ModelGateway(
        original, "gpt-4o-mini", provider_name="custom", provider_factory=factory
    )
    active = gateway.switch_model("gpt-4.1-mini", "custom")

    factory.assert_called_once_with(model="gpt-4.1-mini", provider_name="custom")
    assert active is replacement
    assert gateway.provider is replacement
    assert gateway.model == "gpt-4.1-mini"
    assert gateway.provider_name == "custom"
