"""Tests for Telegram tool-status control messages."""

from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.bus.events import OutboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.channels.telegram import TelegramChannel
from rvoone.config.schema import TelegramConfig


@pytest.mark.asyncio
async def test_telegram_tool_status_create_update_delete() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=321))
    bot.pin_chat_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.delete_message = AsyncMock()
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "create",
                "_status_key": "turn-1",
                "_status_text": 'Tool call: exec({"command":"uname -a"})',
                "_status_pin": True,
            },
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "update",
                "_status_key": "turn-1",
                "_status_text": 'Tool call: web_search({"query":"kernel"})',
            },
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "delete",
                "_status_key": "turn-1",
                "_status_delete_delay_s": 1.5,
            },
        )
    )

    bot.send_message.assert_awaited_once_with(
        chat_id=123,
        text='Tool call: exec({"command":"uname -a"})',
        reply_parameters=None,
    )
    bot.pin_chat_message.assert_awaited_once_with(
        chat_id=123,
        message_id=321,
        disable_notification=True,
    )
    bot.edit_message_text.assert_awaited_once_with(
        chat_id=123,
        message_id=321,
        text='Tool call: web_search({"query":"kernel"})',
    )
    bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=321)
    assert channel._status_messages == {}


@pytest.mark.asyncio
async def test_telegram_tool_status_update_without_cache_is_ignored() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.pin_chat_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.delete_message = AsyncMock()
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "update",
                "_status_key": "missing",
                "_status_text": 'Tool call: exec({"command":"uname -a"})',
            },
        )
    )

    bot.send_message.assert_not_called()
    bot.pin_chat_message.assert_not_called()
    bot.edit_message_text.assert_not_called()
    bot.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_tool_status_pin_failure_does_not_abort_status_message() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=321))
    bot.pin_chat_message = AsyncMock(side_effect=RuntimeError("no permission"))
    bot.edit_message_text = AsyncMock()
    bot.delete_message = AsyncMock()
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "create",
                "_status_key": "turn-1",
                "_status_text": 'Tool call: exec({"command":"uname -a"})',
                "_status_pin": True,
            },
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "update",
                "_status_key": "turn-1",
                "_status_text": 'Tool call: exec({"command":"echo ok"})',
            },
        )
    )

    bot.send_message.assert_awaited_once()
    bot.pin_chat_message.assert_awaited_once()
    bot.edit_message_text.assert_awaited_once_with(
        chat_id=123,
        message_id=321,
        text='Tool call: exec({"command":"echo ok"})',
    )


@pytest.mark.asyncio
async def test_telegram_tool_status_create_without_pin_stays_unpinned() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=321))
    bot.pin_chat_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.delete_message = AsyncMock()
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_status_control": "create",
                "_status_key": "turn-1",
                "_status_text": 'Tool call: exec({"command":"uname -a"})',
            },
        )
    )

    bot.send_message.assert_awaited_once_with(
        chat_id=123,
        text='Tool call: exec({"command":"uname -a"})',
        reply_parameters=None,
    )
    bot.pin_chat_message.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_tool_status_delete_waits_before_removing() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=321))
    bot.pin_chat_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.delete_message = AsyncMock()
    channel._app = SimpleNamespace(bot=bot)
    channel._status_messages["123:turn-1"] = 321

    with patch("rvoone.channels.telegram.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        await channel.send(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="",
                metadata={
                    "_status_control": "delete",
                    "_status_key": "turn-1",
                    "_status_delete_delay_s": 1.5,
                },
            )
        )

    sleep_mock.assert_awaited_once_with(1.5)
    bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=321)
