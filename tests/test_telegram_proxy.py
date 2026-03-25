"""Tests for Telegram proxy setup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.bus.queue import MessageBus
from rvoone.channels.telegram import TelegramChannel
from rvoone.config.schema import TelegramConfig


class _FakeBuilder:
    def __init__(self, app):
        self._app = app
        self.token = MagicMock(return_value=self)
        self.request = MagicMock(return_value=self)
        self.get_updates_request = MagicMock(return_value=self)
        self.build = MagicMock(return_value=app)


@pytest.mark.asyncio
async def test_telegram_start_passes_proxy_via_httpx_request(monkeypatch) -> None:
    created_requests: list[dict] = []

    def fake_httpx_request(**kwargs):
        created_requests.append(kwargs)
        return SimpleNamespace()

    app = MagicMock()
    app.add_error_handler = MagicMock()
    app.add_handler = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.bot = SimpleNamespace(
        get_me=AsyncMock(return_value=SimpleNamespace(username="testbot")),
        set_my_commands=AsyncMock(),
    )

    async def fake_start_polling(**kwargs):
        channel._running = False

    app.updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling), stop=AsyncMock()
    )
    builder = _FakeBuilder(app)

    monkeypatch.setattr("rvoone.channels.telegram.HTTPXRequest", fake_httpx_request)
    monkeypatch.setattr("rvoone.channels.telegram.Application.builder", lambda: builder)

    channel = TelegramChannel(
        TelegramConfig(token="test-token", proxy="socks5://127.0.0.1:1080"),
        MessageBus(),
    )

    await channel.start()

    assert len(created_requests) == 2
    assert all(req["proxy"] == "socks5://127.0.0.1:1080" for req in created_requests)
    builder.request.assert_called_once()
    builder.get_updates_request.assert_called_once()
    app.updater.start_polling.assert_awaited_once_with(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )
