"""Tests for Telegram reply-draft streaming."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import RetryAfter

from rvoone.agent.loop import AgentLoop
from rvoone.bus.events import InboundMessage, OutboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.channels.telegram import TelegramChannel
from rvoone.config.schema import TelegramConfig
from rvoone.providers.base import LLMResponse


def _make_loop(tmp_path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus, provider=provider, workspace=tmp_path, model="test-model", memory_window=10
    )


async def _drain_background_tasks() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_private_telegram_reply_stream_emits_draft_updates(tmp_path) -> None:
    loop = _make_loop(tmp_path)

    async def _fake_chat(*args, **kwargs):
        on_text_delta = kwargs.get("on_text_delta")
        if on_text_delta:
            await on_text_delta("Hello")
            await on_text_delta(" world.")
        return LLMResponse(content="Hello world.", tool_calls=[])

    loop.provider.chat = AsyncMock(side_effect=_fake_chat)
    loop.tools.get_definitions = MagicMock(return_value=[])

    msg = InboundMessage(
        channel="telegram",
        sender_id="user1",
        chat_id="123",
        content="Hi",
        metadata={"is_group": False},
    )
    await loop._dispatch(msg)

    outbound: list[OutboundMessage] = []
    while loop.bus.outbound_size:
        outbound.append(await loop.bus.consume_outbound())

    drafts = [item for item in outbound if item.metadata.get("_draft_control") == "update"]
    assert drafts
    assert drafts[-1].metadata["_draft_text"] == "Hello world."
    final = next(item for item in outbound if item.content == "Hello world.")
    assert final.metadata["_draft_control"] == "complete"
    assert final.metadata["_draft_id"] >= 1
    assert not any(
        item.metadata.get("_progress") and item.content == "Hello world." for item in outbound
    )


@pytest.mark.asyncio
async def test_group_telegram_reply_drafts_are_disabled(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    seen_callbacks: list[bool] = []

    async def _fake_chat(*args, **kwargs):
        seen_callbacks.append(kwargs.get("on_text_delta") is not None)
        return LLMResponse(content="Hello group.", tool_calls=[])

    loop.provider.chat = AsyncMock(side_effect=_fake_chat)
    loop.tools.get_definitions = MagicMock(return_value=[])

    msg = InboundMessage(
        channel="telegram",
        sender_id="user1",
        chat_id="123",
        content="Hi",
        metadata={"is_group": True},
    )
    await loop._dispatch(msg)

    outbound: list[OutboundMessage] = []
    while loop.bus.outbound_size:
        outbound.append(await loop.bus.consume_outbound())

    assert seen_callbacks == [False]
    assert not any(item.metadata.get("_draft_control") for item in outbound)
    assert any(item.content == "Hello group." for item in outbound)


@pytest.mark.asyncio
async def test_telegram_channel_uses_send_message_draft() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._draft_send_interval_s = 0.0
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_draft_control": "update",
                "_draft_id": 7,
                "_draft_text": "Hello world.",
            },
        )
    )
    await _drain_background_tasks()

    bot.send_message_draft.assert_awaited_once_with(
        chat_id=123,
        draft_id=7,
        text="Hello world.",
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_telegram_channel_disables_reply_drafts_after_failure() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._draft_send_interval_s = 0.0
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(side_effect=RuntimeError("forum topic mode disabled"))
    channel._app = SimpleNamespace(bot=bot)

    msg = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="",
        metadata={
            "_draft_control": "update",
            "_draft_id": 7,
            "_draft_text": "Hello world.",
        },
    )

    await channel.send(msg)
    await channel.send(msg)
    await _drain_background_tasks()

    assert bot.send_message_draft.await_count == 1
    assert channel._drafts_globally_disabled is True


@pytest.mark.asyncio
async def test_telegram_channel_retries_after_retry_after_rate_limit(monkeypatch) -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._draft_send_interval_s = 0.0
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(side_effect=[RetryAfter(3), True])
    channel._app = SimpleNamespace(bot=bot)

    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("rvoone.channels.telegram.asyncio.sleep", _fake_sleep)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_draft_control": "update",
                "_draft_id": 7,
                "_draft_text": "Hello",
            },
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_draft_control": "update",
                "_draft_id": 7,
                "_draft_text": "Hello world.",
            },
        )
    )
    for _ in range(20):
        if bot.send_message_draft.await_count >= 2:
            break
        await real_sleep(0)

    assert bot.send_message_draft.await_count == 2
    assert bot.send_message_draft.await_args_list[-1].kwargs["text"] == "Hello world."
    assert any(delay >= 3 for delay in sleep_calls)
    assert channel._drafts_globally_disabled is False


@pytest.mark.asyncio
async def test_telegram_channel_complete_stops_pending_retry(monkeypatch) -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._draft_send_interval_s = 0.0
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(side_effect=RetryAfter(3))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    channel._app = SimpleNamespace(bot=bot)

    real_sleep = asyncio.sleep
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def _fake_sleep(delay: float) -> None:
        if delay >= 3:
            sleep_started.set()
            await release_sleep.wait()
            return
        await real_sleep(0)

    monkeypatch.setattr("rvoone.channels.telegram.asyncio.sleep", _fake_sleep)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_draft_control": "update",
                "_draft_id": 7,
                "_draft_text": "Hello",
            },
        )
    )
    await asyncio.wait_for(sleep_started.wait(), timeout=1.0)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Final",
            metadata={
                "_draft_control": "complete",
                "_draft_id": 7,
            },
        )
    )
    release_sleep.set()
    for _ in range(10):
        await real_sleep(0)

    assert bot.send_message_draft.await_count == 1
    bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_channel_complete_clears_visible_draft_before_final_send() -> None:
    channel = TelegramChannel(TelegramConfig(token="test-token"), MessageBus())
    channel._draft_send_interval_s = 0.0
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    channel._app = SimpleNamespace(bot=bot)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            metadata={
                "_draft_control": "update",
                "_draft_id": 7,
                "_draft_text": "Hello world.",
            },
        )
    )
    await _drain_background_tasks()

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Final",
            metadata={
                "_draft_control": "complete",
                "_draft_id": 7,
            },
        )
    )

    assert bot.send_message_draft.await_count == 2
    assert bot.send_message_draft.await_args_list[-1].kwargs["text"] == channel._DRAFT_CLEAR_TEXT
    bot.send_message.assert_awaited_once_with(
        chat_id=123,
        text="Final",
        parse_mode="HTML",
        reply_parameters=None,
    )
