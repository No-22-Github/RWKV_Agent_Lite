"""Runtime assembly helpers for CLI commands."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console

from rvoone.config.schema import Config


def make_provider(
    config: Config,
    console: Console,
    *,
    model: str | None = None,
    provider_name: str | None = None,
):
    """Create the active LLM provider from config."""
    from rvoone.providers.custom_provider import CustomProvider

    model = model or config.agents.defaults.model
    provider_name = config.get_provider_name(model, provider_name)
    provider_config = config.get_provider(model, provider_name)
    timeout = config.providers.upstream_timeout

    if provider_name == "custom":
        resolved_model = config.strip_model_provider_prefix(model)
        return CustomProvider(
            api_key=provider_config.api_key if provider_config else "no-key",
            api_base=config.get_api_base(model, provider_name) or "http://localhost:8000/v1",
            default_model=resolved_model,
            timeout=timeout,
            request_dump=provider_config.request_dump if provider_config else False,
            stream_mode=provider_config.stream_mode if provider_config else "auto",
            token_estimation=provider_config.token_estimation if provider_config else "off",
        )

    console.print("[red]Error: No supported LLM provider configured.[/red]")
    console.print("Use providers.custom for OpenAI-compatible endpoints.")
    raise typer.Exit(1)


def configured_provider_snapshot(
    config: Config,
) -> dict[str, dict[str, bool | str | list[str] | None]]:
    """Build a compact provider status snapshot for agent introspection."""
    custom_sources = config.providers.custom_sources
    return {
        "custom": {
            "configured": bool(
                config.providers.custom.api_key
                or config.providers.custom.api_base
                or any(cfg.api_key or cfg.api_base for cfg in custom_sources.values())
            ),
            "available_models": config.providers.custom.available_models,
        },
    }


def build_agent_kwargs(config: Config) -> dict[str, Any]:
    """Build the common AgentLoop kwargs derived from config."""
    provider_name = config.get_provider_name(
        config.agents.defaults.model, config.agents.defaults.provider
    )
    return {
        "workspace": config.workspace_path,
        "model": config.agents.defaults.model,
        "temperature": config.agents.defaults.temperature,
        "max_tokens": config.agents.defaults.max_tokens,
        "max_iterations": config.agents.defaults.max_tool_iterations,
        "memory_window": config.agents.defaults.memory_window,
        "reasoning_effort": config.agents.defaults.reasoning_effort,
        "exec_config": config.tools.exec,
        "restrict_to_workspace": config.tools.restrict_to_workspace,
        "channels_config": config.channels,
        "enable_event_handling": config.agents.defaults.enable_event_handling,
        "subagent_model": config.agents.subagent.model,
        "subagent_provider": config.agents.subagent.provider,
        "configured_providers": configured_provider_snapshot(config),
        "main_provider_name": provider_name,
    }
