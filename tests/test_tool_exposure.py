from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.agent.loop import AgentLoop
from rvoone.agent.toolset import build_tool_registry
from rvoone.application.conversation import ConversationService
from rvoone.bus.events import InboundMessage
from rvoone.providers.base import LLMResponse


class DummySpawnManager:
    pass


class DummyCronService:
    pass


def _build_registry():
    return build_tool_registry(
        workspace=Path("/tmp/workspace"),
        restrict_to_workspace=True,
        exec_timeout=30,
        exec_path_append="",
        include_completion_tools=True,
        include_spawn_tool=True,
        spawn_manager=DummySpawnManager(),
        include_models_tool=True,
        include_exposure_tools=True,
        list_tool_categories_callback=lambda session_key: {"session_key": session_key},
        enable_tool_categories_callback=lambda session_key, categories: {
            "session_key": session_key,
            "enabled": categories,
        },
        models_tool_kwargs={
            "main_model": "gpt-4o-mini",
            "main_provider": "custom",
            "subagent_model": None,
            "subagent_provider": "auto",
            "configured_providers": {},
        },
        cron_service=DummyCronService(),
    )


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    bus = MagicMock()
    bus.check_events = AsyncMock(return_value=None)
    return AgentLoop(
        bus=bus, provider=provider, workspace=tmp_path, model="test-model", memory_window=10
    )


def test_registry_filters_visible_tool_definitions() -> None:
    registry = _build_registry()

    core_defs = registry.get_definitions(registry.get_visible_tool_names({"core"}))
    core_names = {entry["function"]["name"] for entry in core_defs}

    assert "exec" in core_names
    assert "web_fetch" in core_names
    assert "write_file" not in core_names
    assert "web_search" not in core_names


@pytest.mark.asyncio
async def test_registry_rejects_hidden_tool_execution() -> None:
    registry = _build_registry()
    visible = set()

    result = await registry.execute("exec", {"command": "echo hi"}, allowed_names=visible)

    assert "not currently enabled" in result


def test_loop_lists_and_enables_categories_per_session(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    before = loop._list_tool_categories_for_session("cli:direct")
    enable_result = loop._enable_tool_categories_for_session(
        "cli:direct", ["editing", "web", "unknown"]
    )
    after = loop._list_tool_categories_for_session("cli:direct")

    assert before["enabled_categories"] == ["core"]
    assert enable_result["newly_enabled"] == []
    assert enable_result["unknown_categories"] == ["editing", "unknown", "web"]
    assert after["enabled_categories"] == ["core"]


def test_loop_enable_tool_categories_accepts_core_without_error(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    result = loop._enable_tool_categories_for_session("cli:direct", ["core"])

    assert result["enabled_categories"] == ["core"]
    assert result["newly_enabled"] == ["core"]
    assert result["unknown_categories"] == []


@pytest.mark.asyncio
async def test_run_agent_loop_uses_core_tool_subset_before_enabling(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    calls = iter([LLMResponse(content="Done", tool_calls=[])])
    loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    loop._set_tool_context("cli", "direct")

    await loop.runtime.run_agent_loop(
        initial_messages=[{"role": "user", "content": "Start"}],
        session_key="cli:direct",
    )

    assert loop.provider.chat.await_args is not None
    tool_names = {
        entry["function"]["name"] for entry in loop.provider.chat.await_args.kwargs["tools"]
    }
    assert "exec" in tool_names
    assert "web_fetch" in tool_names
    assert "write_file" not in tool_names


@pytest.mark.asyncio
async def test_archive_session_resets_enabled_tool_categories(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    service = ConversationService(loop)
    session = loop.sessions.get_or_create("cli:direct")
    session.messages = []
    loop.state.enable_tool_categories("cli:direct", {"editing", "web"})
    loop._publish_command_feedback = AsyncMock()
    loop._generate_new_session_greeting = AsyncMock(return_value=None)

    response = await service.archive_session(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/new")
    )

    assert response.content == "New session started."
    assert loop.state.ensure_enabled_tool_categories("cli:direct", defaults={"core"}) == {"core"}


@pytest.mark.asyncio
async def test_session_scoped_tool_visibility_matches_defaults(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    loop._set_tool_context("cli", "direct")
    heartbeat_visible = loop._get_visible_tool_names("heartbeat")

    loop._set_tool_context("cli", "direct")
    direct_visible = loop._get_visible_tool_names("cli:direct")

    assert heartbeat_visible == direct_visible
