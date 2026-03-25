"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from rvoone.agent.loop import AgentLoop
    from rvoone.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = Path(tempfile.mkdtemp(prefix="rvoone-task-cancel-"))

    with (
        patch("rvoone.agent.loop.ContextBuilder"),
        patch("rvoone.agent.loop.SessionManager"),
        patch("rvoone.agent.loop.SubagentManager") as mock_sub_mgr,
    ):
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self):
        from rvoone.bus.events import InboundMessage

        loop, _ = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        out = await loop._handle_stop(msg)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        from rvoone.bus.events import InboundMessage

        loop, _ = _make_loop()
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop.state.active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        out = await loop._handle_stop(msg)

        assert cancelled.is_set()
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self):
        from rvoone.bus.events import InboundMessage

        loop, _ = _make_loop()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop.state.active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        out = await loop._handle_stop(msg)

        assert all(e.is_set() for e in events)
        assert "2 task" in out.content


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self):
        from rvoone.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self):
        from rvoone.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        async def mock_process(msg, **kwargs):
            order.append(f"start-{msg.content}")
            await asyncio.sleep(0.05)
            order.append(f"end-{msg.content}")
            return OutboundMessage(channel="test", chat_id="c1", content=msg.content)

        setattr(loop, "_process_message", AsyncMock(side_effect=mock_process))
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]

    @pytest.mark.asyncio
    async def test_dispatch_allows_parallel_processing_across_sessions(self):
        from rvoone.bus.events import InboundMessage, OutboundMessage

        loop, _ = _make_loop()
        started = 0
        max_active = 0
        active = 0
        release = asyncio.Event()

        async def mock_process(msg, **kwargs):
            nonlocal started, active, max_active
            started += 1
            active += 1
            max_active = max(max_active, active)
            if started == 2:
                release.set()
            await release.wait()
            active -= 1
            return OutboundMessage(channel="test", chat_id=msg.chat_id, content=msg.content)

        setattr(loop, "_process_message", AsyncMock(side_effect=mock_process))
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c2", content="b")

        await asyncio.gather(
            asyncio.create_task(loop._dispatch(msg1)),
            asyncio.create_task(loop._dispatch(msg2)),
        )

        assert max_active == 2


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_reports_current_session_state(self):
        from rvoone.bus.events import InboundMessage

        loop, _ = _make_loop()
        session = MagicMock()
        session.messages = ["u", "a", "u2"]
        session.updated_at = None
        loop.sessions = MagicMock(get_or_create=MagicMock(return_value=session))

        task = asyncio.create_task(asyncio.sleep(60))
        await asyncio.sleep(0)
        loop.state.active_tasks["test:c1"] = [task]

        page1 = await loop.commands.handle_status(
            InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/status 1")
        )
        page2 = await loop.commands.handle_status(
            InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/status 2")
        )
        page3 = await loop.commands.handle_status(
            InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/status 3")
        )

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "Messages: 3" in page1.content
        assert "Page 1/3" in page1.content
        assert "Active tasks: 1" in page2.content
        assert "Page 2/3" in page2.content
        assert "Registered tools:" in page3.content
        assert "Page 3/3" in page3.content

    @pytest.mark.asyncio
    async def test_model_command_switches_active_model(self):
        from rvoone.bus.events import InboundMessage

        loop, _ = _make_loop()
        new_provider = MagicMock()
        loop.provider_factory = MagicMock(return_value=new_provider)
        loop.main_provider_name = "custom"
        loop.configured_providers = {
            "custom": {"available_models": ["gpt-4o-mini", "gpt-4.1-mini"]},
        }
        loop.subagents.default_model = None
        loop.subagents.model = "gpt-4o-mini"

        response = await loop.commands.handle_model(
            InboundMessage(
                channel="cli", sender_id="u1", chat_id="c1", content="/model gpt-4.1-mini"
            )
        )

        assert loop.model == "gpt-4.1-mini"
        assert loop.provider is new_provider
        assert loop.subagents.model == "gpt-4.1-mini"
        assert "Switched to: gpt-4.1-mini" in response.content


class TestSubagentCancellation:
    @pytest.mark.asyncio
    async def test_cancel_by_session(self):
        from rvoone.agent.subagent import SubagentManager
        from rvoone.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        count = await mgr.cancel_by_session("test:c1")
        assert count == 1
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancel_by_session_no_tasks(self):
        from rvoone.agent.subagent import SubagentManager
        from rvoone.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)
        assert await mgr.cancel_by_session("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_run_subagent_uses_overridden_model_and_provider(self, tmp_path):
        from rvoone.agent.subagent import SubagentManager
        from rvoone.bus.queue import MessageBus
        from rvoone.providers.base import LLMResponse

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "main-model"
        child_provider = MagicMock()
        child_provider.chat = AsyncMock(return_value=LLMResponse(content="done"))
        factory = MagicMock(return_value=child_provider)
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,
            bus=bus,
            default_model="sub-default",
            default_provider="custom",
            provider_factory=factory,
        )

        await mgr._run_subagent(
            "sub-1",
            "task body",
            "label",
            {"channel": "cli", "chat_id": "direct"},
            model="gpt-4o-mini",
            provider="custom",
        )

        factory.assert_called_once_with(
            model="gpt-4o-mini",
            provider_name="custom",
        )
        child_provider.chat.assert_awaited_once()
        assert child_provider.chat.await_args.kwargs["model"] == "gpt-4o-mini"
