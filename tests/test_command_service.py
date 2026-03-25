"""Direct tests for command service behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.application.commands import CommandService
from rvoone.bus.events import InboundMessage


def _owner() -> SimpleNamespace:
    owner = SimpleNamespace()
    owner.state = SimpleNamespace(
        active_tasks={},
        processing_tasks=set(),
        runtime={},
    )
    owner.subagents = SimpleNamespace(
        cancel_by_session=AsyncMock(return_value=0), _session_tasks={}
    )
    owner.sessions = SimpleNamespace(get_or_create=MagicMock())
    owner.dispatcher = SimpleNamespace(session_lock=lambda _key: asyncio.Lock())
    owner.bus = SimpleNamespace(pending_events=lambda _key: 0)
    owner._mcp_connected = False
    owner._mcp_servers = {}
    owner.tools = []
    owner.enable_event_handling = True
    owner._running = True
    owner.presenter = SimpleNamespace(
        apply_interactive_view=lambda metadata, **kwargs: {**metadata, **kwargs}
    )
    owner.conversations = SimpleNamespace(archive_session=AsyncMock())
    owner.model = "gpt-4o-mini"
    owner.main_provider_name = "custom"
    owner.subagent_model = None
    owner.subagent_provider = None
    owner.configured_providers = {"custom": {"available_models": ["gpt-4o-mini"]}}
    owner.provider_factory = None
    owner.model_gateway = SimpleNamespace(model="gpt-4o-mini", provider_name="custom")
    owner.provider = MagicMock()
    return owner


@pytest.mark.asyncio
async def test_handle_stop_cancels_active_tasks() -> None:
    owner = _owner()
    cancelled = asyncio.Event()

    async def _slow() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(_slow())
    await asyncio.sleep(0)
    owner.state.active_tasks["test:c1"] = [task]

    service = CommandService(owner)
    response = await service.handle_stop(
        InboundMessage(channel="test", sender_id="u", chat_id="c1", content="/stop")
    )

    assert cancelled.is_set()
    assert "Stopped 1 task" in response.content


def test_available_model_options_deduplicates_models() -> None:
    owner = _owner()
    owner.subagent_model = "gpt-4o-mini"
    owner.configured_providers = {"custom": {"available_models": ["gpt-4o-mini", "gpt-4.1-mini"]}}

    service = CommandService(owner)
    assert service.available_model_options() == [
        ("gpt-4o-mini", "custom"),
        ("gpt-4.1-mini", "custom"),
    ]
