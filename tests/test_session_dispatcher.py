"""Direct tests for session dispatcher behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rvoone.application.dispatcher import SessionDispatcher
from rvoone.bus.events import InboundMessage, OutboundMessage


def _owner() -> SimpleNamespace:
    owner = SimpleNamespace()
    owner.state = SimpleNamespace(
        session_lock=lambda _key: asyncio.Lock(),
        active_tasks={},
        processing_tasks=set(),
    )
    owner._update_session_runtime = lambda *_args, **_kwargs: None
    owner._typing_target = lambda msg: (msg.channel, msg.chat_id)
    owner.presenter = SimpleNamespace(
        tool_status_enabled=lambda _channel: False,
        tool_status_pin_enabled=lambda _channel: False,
        next_tool_status_key=lambda: "k1",
        publish_tool_status_delete=AsyncMock(),
        publish_typing_stop=AsyncMock(),
    )
    owner._process_message = AsyncMock(
        return_value=OutboundMessage(channel="test", chat_id="c1", content="ok")
    )
    owner.bus = SimpleNamespace(publish_outbound=AsyncMock())
    owner._clear_session_runtime = lambda _key: None
    return owner


@pytest.mark.asyncio
async def test_dispatcher_publishes_process_result() -> None:
    owner = _owner()
    dispatcher = SessionDispatcher(owner)

    await dispatcher.dispatch(
        InboundMessage(channel="test", sender_id="u", chat_id="c1", content="hi")
    )

    owner._process_message.assert_awaited_once()
    owner.bus.publish_outbound.assert_awaited_once()
