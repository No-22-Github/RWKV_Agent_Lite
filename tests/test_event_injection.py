"""Tests for session-scoped event injection."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.agent.context import ContextBuilder
from rvoone.agent.loop import AgentLoop
from rvoone.bus.events import InboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.providers.base import LLMResponse, ToolCallRequest


def _make_loop(
    tmp_path: Path, *, enable_event_handling: bool = False
) -> tuple[AgentLoop, MessageBus]:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        memory_window=10,
        enable_event_handling=enable_event_handling,
    )
    return loop, bus


@pytest.mark.asyncio
async def test_message_bus_accumulates_events() -> None:
    bus = MessageBus()

    await bus.publish_event("cli:direct", "first")
    await bus.publish_event("cli:direct", "second")

    assert await bus.check_events("cli:direct") == "- first\n- second"
    assert await bus.check_events("cli:direct") is None


def test_system_prompt_includes_event_handling(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt(enable_event_handling=True)

    assert "Event Handling" in prompt
    assert "<SYS_EVENT>" in prompt
    assert "ALWAYS takes priority" in prompt


@pytest.mark.asyncio
async def test_run_agent_loop_injects_events_and_cancels_pending_tools(tmp_path: Path) -> None:
    loop, bus = _make_loop(tmp_path, enable_event_handling=True)

    tool_call = ToolCallRequest(id="call-1", name="exec", arguments={"command": "echo one"})
    first = LLMResponse(content="Searching", tool_calls=[tool_call])
    finish = ToolCallRequest(id="call-2", name="exec", arguments={"command": "echo done"})
    second = LLMResponse(content="Switching tasks", tool_calls=[])
    third = LLMResponse(content="Switching tasks", tool_calls=[finish])

    calls = iter([first, second, third])
    loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="Switching tasks")

    await bus.publish_event("cli:direct", "Actually, do something else")
    final_content, tools_used, messages = await loop.runtime.run_agent_loop(
        initial_messages=[{"role": "user", "content": "original"}],
        session_key="cli:direct",
    )

    assert final_content == "Switching tasks"
    assert tools_used == []
    loop.tools.execute.assert_not_awaited()
    assert any(isinstance(m.get("content"), str) and "<SYS_EVENT" in m["content"] for m in messages)
    assert any(
        m.get("content") == "CANCELLED: User interrupted"
        for m in messages
        if m.get("role") == "tool"
    )


@pytest.mark.asyncio
async def test_run_keeps_running_after_interrupt_event(tmp_path: Path) -> None:
    loop, bus = _make_loop(tmp_path, enable_event_handling=True)
    first_msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="first")
    second_msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="second")

    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_dispatch(msg: InboundMessage) -> None:
        loop.state.processing_tasks.add(msg.session_key)
        started.set()
        try:
            await release.wait()
        finally:
            loop.state.processing_tasks.discard(msg.session_key)

    inbound_items = [first_msg, second_msg]

    async def fake_consume() -> InboundMessage:
        if inbound_items:
            return inbound_items.pop(0)
        await asyncio.sleep(60)
        raise AssertionError("consume should not return again")

    loop._dispatch = fake_dispatch  # type: ignore[method-assign]
    bus.consume_inbound = AsyncMock(side_effect=fake_consume)

    run_task = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert not run_task.done()
        assert await bus.check_events(first_msg.session_key) == "- second"
    finally:
        release.set()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
