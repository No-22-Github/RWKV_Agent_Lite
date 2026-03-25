from pathlib import Path

from rich.console import Console

from rvoone.agent.toolset import DEFAULT_ENABLED_TOOL_CATEGORIES, build_tool_registry
from rvoone.cli.runtime import build_agent_kwargs, make_provider
from rvoone.config.schema import Config
from rvoone.providers.custom_provider import CustomProvider


class DummySpawnManager:
    pass


class DummyCronService:
    pass


def test_build_tool_registry_includes_main_agent_tools() -> None:
    registry = build_tool_registry(
        workspace=Path("/tmp/workspace"),
        restrict_to_workspace=True,
        exec_timeout=30,
        exec_path_append="/tmp/bin",
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

    assert registry.has("exec")
    assert registry.has("web_fetch")


def test_build_tool_registry_omits_optional_tools_by_default() -> None:
    registry = build_tool_registry(
        workspace=Path("/tmp/workspace"),
        restrict_to_workspace=False,
        exec_timeout=30,
        exec_path_append="",
    )

    assert not registry.has("spawn")
    assert not registry.has("cron")


def test_build_tool_registry_tracks_default_enabled_categories() -> None:
    assert DEFAULT_ENABLED_TOOL_CATEGORIES == {"core"}


def test_make_provider_prefers_custom_provider() -> None:
    config = Config()
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_key = "test-key"
    config.providers.custom.api_base = "http://localhost:11434/v1"

    provider = make_provider(config, Console(record=True))

    assert isinstance(provider, CustomProvider)


def test_build_agent_kwargs_extracts_shared_agent_settings() -> None:
    config = Config()
    config.agents.defaults.workspace = "~/workspace"
    config.agents.defaults.model = "gpt-4o-mini"
    config.agents.defaults.provider = "custom"

    kwargs = build_agent_kwargs(config)

    assert kwargs["workspace"] == Path("~/workspace").expanduser()
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["main_provider_name"] == "custom"
    assert kwargs["configured_providers"]["custom"]["configured"] is False
