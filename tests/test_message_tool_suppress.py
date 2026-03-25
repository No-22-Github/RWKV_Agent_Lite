from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rvoone.agent.loop import AgentLoop
from rvoone.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MagicMock()
    bus.check_events = AsyncMock(return_value=None)
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus, provider=provider, workspace=tmp_path, model="test-model", memory_window=10
    )


@pytest.mark.asyncio
async def test_run_agent_loop_allows_plain_text_exit(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    loop.provider.chat = AsyncMock(return_value=LLMResponse(content="Direct answer", tool_calls=[]))
    loop.tools.get_definitions = MagicMock(return_value=[])

    final_content, tools_used, _ = await loop.runtime.run_agent_loop(
        initial_messages=[{"role": "user", "content": "Start"}],
        session_key="cli:direct",
    )

    assert final_content == "Direct answer"
    assert tools_used == []


@pytest.mark.asyncio
async def test_run_agent_loop_executes_tools_then_returns_text(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    tool = ToolCallRequest(id="call1", name="exec", arguments={"command": "echo 1"})
    calls = iter(
        [
            LLMResponse(content="Starting", tool_calls=[tool]),
            LLMResponse(content="Done", tool_calls=[]),
        ]
    )
    loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")

    final_content, tools_used, _ = await loop.runtime.run_agent_loop(
        initial_messages=[{"role": "user", "content": "Start"}],
        session_key="cli:direct",
    )

    assert final_content == "Done"
    assert tools_used == ["exec"]


def test_tool_registry_caches_definitions_until_registry_changes(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    definitions1 = loop.tools.get_definitions()
    definitions2 = loop.tools.get_definitions()

    assert definitions1 is definitions2
