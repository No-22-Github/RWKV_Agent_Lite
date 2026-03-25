"""Unified model gateway for active provider access."""

from __future__ import annotations

from typing import Any, Callable

from rvoone.providers.base import LLMProvider, LLMResponse, TextDeltaHandler


class ModelGateway:
    """Own the currently active model/provider pair behind a stable interface."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        provider_name: str | None = None,
        provider_factory: Callable[..., LLMProvider] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.provider_name = provider_name
        self.provider_factory = provider_factory

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        on_text_delta: TextDeltaHandler | None = None,
    ) -> LLMResponse:
        """Send one chat request through the currently active provider."""
        return await self.provider.chat(
            messages=messages,
            tools=tools,
            model=model or self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            on_text_delta=on_text_delta,
        )

    def switch_model(self, model: str, provider_name: str | None = None) -> LLMProvider:
        """Switch active model/provider and return the selected provider."""
        if self.provider_factory:
            self.provider = self.provider_factory(model=model, provider_name=provider_name)
        self.model = model
        self.provider_name = provider_name
        return self.provider
