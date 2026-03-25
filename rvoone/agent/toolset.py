"""Shared builders for agent tool registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rvoone.agent.tools.registry import ToolRegistry
from rvoone.agent.tools.shell import ExecTool
from rvoone.agent.tools.web_fetch import WebFetchTool

CORE_CATEGORY = "core"

DEFAULT_ENABLED_TOOL_CATEGORIES = {CORE_CATEGORY}


def build_tool_registry(
    *,
    workspace: Path,
    restrict_to_workspace: bool,
    exec_timeout: int,
    exec_path_append: str,
    include_completion_tools: bool = False,
    include_spawn_tool: bool = False,
    spawn_manager: Any | None = None,
    include_models_tool: bool = False,
    models_tool_kwargs: dict[str, Any] | None = None,
    cron_service: Any | None = None,
    include_exposure_tools: bool = False,
    list_tool_categories_callback: Callable[[str | None], dict[str, Any]] | None = None,
    enable_tool_categories_callback: Callable[[str | None, list[str]], dict[str, Any]]
    | None = None,
) -> ToolRegistry:
    """Build a tool registry for a main agent or subagent."""
    registry = ToolRegistry()
    allowed_dir = workspace if restrict_to_workspace else None
    registry.register(
        ExecTool(
            working_dir=str(workspace),
            timeout=exec_timeout,
            restrict_to_workspace=restrict_to_workspace,
            path_append=exec_path_append,
        ),
        category=CORE_CATEGORY,
    )
    registry.register(WebFetchTool(), category=CORE_CATEGORY)

    return registry
