"""Tests for centralized slash-command routing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rvoone.bus.events import InboundMessage, OutboundMessage
from rvoone.commands.router import CommandRouter


@pytest.mark.asyncio
async def test_command_router_handles_help_and_start() -> None:
    router = CommandRouter(
        handle_new=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="new")
        ),
        handle_model=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="model")
        ),
        handle_status=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="status")
        ),
        handle_stop=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="stop")
        ),
    )

    start = await router.route(
        InboundMessage(channel="cli", sender_id="u", chat_id="1", content="/start")
    )
    help_msg = await router.route(
        InboundMessage(channel="cli", sender_id="u", chat_id="1", content="/help")
    )

    assert start is not None
    assert "Hi! I'm rvoone" in start.content
    assert help_msg is not None
    assert "/new" in help_msg.content
    assert "/model" in help_msg.content
    assert "/status" in help_msg.content


@pytest.mark.asyncio
async def test_command_router_normalizes_telegram_mentions() -> None:
    handle_new = AsyncMock(
        return_value=OutboundMessage(channel="telegram", chat_id="1", content="new")
    )
    router = CommandRouter(
        handle_new=handle_new,
        handle_model=AsyncMock(
            return_value=OutboundMessage(channel="telegram", chat_id="1", content="model")
        ),
        handle_status=AsyncMock(
            return_value=OutboundMessage(channel="telegram", chat_id="1", content="status")
        ),
        handle_stop=AsyncMock(
            return_value=OutboundMessage(channel="telegram", chat_id="1", content="stop")
        ),
    )

    response = await router.route(
        InboundMessage(channel="telegram", sender_id="u", chat_id="1", content="/new@demo_bot")
    )

    assert response is not None
    handle_new.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_router_dispatches_status() -> None:
    handle_status = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="1", content="Session status:")
    )
    router = CommandRouter(
        handle_new=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="new")
        ),
        handle_model=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="model")
        ),
        handle_status=handle_status,
        handle_stop=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="stop")
        ),
    )

    response = await router.route(
        InboundMessage(channel="cli", sender_id="u", chat_id="1", content="/status")
    )

    assert response is not None
    handle_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_router_dispatches_model() -> None:
    handle_model = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="1", content="Model selection")
    )
    router = CommandRouter(
        handle_new=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="new")
        ),
        handle_model=handle_model,
        handle_status=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="status")
        ),
        handle_stop=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="stop")
        ),
    )

    response = await router.route(
        InboundMessage(channel="cli", sender_id="u", chat_id="1", content="/model")
    )

    assert response is not None
    handle_model.assert_awaited_once()
