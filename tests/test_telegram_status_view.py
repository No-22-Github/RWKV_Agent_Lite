"""Tests for Telegram /status inline keyboard views."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.bus.events import OutboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.channels.telegram import TelegramChannel
from rvoone.config.schema import TelegramConfig


@pytest.mark.asyncio
async def test_telegram_status_view_create_and_update() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.answer_callback_query = AsyncMock()
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Session status\n\nPage 1/3",
            metadata={
                "_interactive_control": "create",
                "_interactive_view": "status",
                "_interactive_page": 1,
                "_interactive_pages": 3,
            },
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Session status\n\nPage 2/3",
            metadata={
                "_interactive_control": "update",
                "_interactive_view": "status",
                "_interactive_page": 2,
                "_interactive_pages": 3,
                "_interactive_message_id": 456,
                "_interactive_callback_query_id": "cbq-1",
            },
        )
    )

    assert bot.send_message.await_count == 1
    assert bot.edit_message_text.await_count == 1
    send_kwargs = bot.send_message.await_args.kwargs
    edit_kwargs = bot.edit_message_text.await_args.kwargs
    assert send_kwargs["chat_id"] == 123
    assert send_kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "view:status:3"
    assert edit_kwargs["message_id"] == 456
    assert edit_kwargs["reply_markup"].inline_keyboard[0][1].callback_data == "view:status:2"
    assert edit_kwargs["reply_markup"].inline_keyboard[0][2].callback_data == "view:status:3"
    bot.answer_callback_query.assert_awaited_once_with("cbq-1")


@pytest.mark.asyncio
async def test_telegram_status_callback_forwards_status_command() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._handle_message = AsyncMock()
    query = SimpleNamespace(
        data="view:status:2",
        id="cbq-2",
        message=SimpleNamespace(chat_id=123, message_id=456),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=42, username="tester"),
    )

    await channel._on_callback_query(update, SimpleNamespace())

    channel._handle_message.assert_awaited_once_with(
        sender_id="42|tester",
        chat_id="123",
        content="/status 2",
        metadata={
            "_interactive_message_id": 456,
            "_interactive_callback_query_id": "cbq-2",
        },
    )


def test_parse_interactive_callback() -> None:
    assert TelegramChannel._parse_interactive_callback("view:status:2") == ("status", 2)
    assert TelegramChannel._parse_interactive_callback("status:2") is None
    assert TelegramChannel._parse_interactive_callback("view:status:bad") is None


def test_parse_model_callback() -> None:
    assert TelegramChannel._parse_model_callback("model:set:gpt-4o-mini") == "gpt-4o-mini"
    assert TelegramChannel._parse_model_callback("view:status:2") is None


@pytest.mark.asyncio
async def test_telegram_model_callback_forwards_model_command() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._handle_message = AsyncMock()
    query = SimpleNamespace(
        data="model:set:gpt-4o-mini",
        id="cbq-model",
        message=SimpleNamespace(chat_id=123, message_id=456),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=42, username="tester"),
    )

    await channel._on_callback_query(update, SimpleNamespace())

    channel._handle_message.assert_awaited_once_with(
        sender_id="42|tester",
        chat_id="123",
        content="/model gpt-4o-mini",
        metadata={
            "_interactive_message_id": 456,
            "_interactive_callback_query_id": "cbq-model",
        },
    )
