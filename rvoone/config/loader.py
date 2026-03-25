"""Configuration loading utilities."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rvoone.config.schema import Config
from rvoone.utils.helpers import get_data_path

try:
    import tomli_w
except ImportError:  # pragma: no cover - fallback for environments without tomli-w installed
    tomli_w = None


_NULL_SENTINEL = "__rvoone_NULL__"


@dataclass(frozen=True)
class SplitConfigFile:
    section: str
    filename: str


_SPLIT_CONFIG_FILES = (
    SplitConfigFile("agents", "agent.toml"),
    SplitConfigFile("channels", "chat.toml"),
    SplitConfigFile("providers", "llm.toml"),
    SplitConfigFile("gateway", "server.toml"),
    SplitConfigFile("tools", "tools.toml"),
)


def get_config_path() -> Path:
    """Get the default configuration directory path."""
    return Path.home() / ".rvoone" / "config"


def get_data_dir() -> Path:
    """Get the rvoone data directory."""
    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()
    data: dict[str, Any] = {}

    try:
        if path.exists():
            if path.is_dir():
                data = _load_split_config_dir(path)
            else:
                data = _load_toml_config(path)
    except (tomllib.TOMLDecodeError, ValueError) as e:
        print(f"Warning: Failed to load config from {path}: {e}")
        print("Hint: check TOML syntax near this location.")
        print("Using default configuration.")
        data = {}

    return Config.model_validate(data)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()

    data = _serialize_main_config(config)

    if _should_use_split_dir(path):
        _save_split_config_dir(data, path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_toml_file(path, data)


def save_config_templates(config: Config, config_path: Path | None = None) -> None:
    """Write user-facing split config templates."""
    path = config_path or get_config_path()
    if not _should_use_split_dir(path):
        raise ValueError("Template config output requires a config directory path")

    path.mkdir(parents=True, exist_ok=True)
    (path / "agent.toml").write_text(_render_agents_template(config), encoding="utf-8")
    (path / "chat.toml").write_text(_render_channels_template(config), encoding="utf-8")
    (path / "llm.toml").write_text(_render_providers_template(config), encoding="utf-8")
    (path / "server.toml").write_text(_render_gateway_template(config), encoding="utf-8")
    (path / "tools.toml").write_text(_render_tools_template(config), encoding="utf-8")


def _comment_block(lines: list[str], *comments: str) -> None:
    """Append a formatted comment block."""
    for comment in comments:
        lines.append(f"# {comment}" if comment else "#")


def _join_template_sections(*sections: str) -> str:
    """Join non-empty template sections with a blank line."""
    return "\n\n".join(section.rstrip() for section in sections if section.strip()).rstrip() + "\n"


def _load_toml_config(path: Path) -> dict[str, Any]:
    """Load and normalize TOML config data."""
    text = path.read_text(encoding="utf-8")
    normalized_text = _preprocess_null_literals(text)
    data = tomllib.loads(normalized_text)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a table")

    return _restore_null_sentinel(data)


def _describe_config_error(path: Path, error: tomllib.TOMLDecodeError | ValueError) -> str:
    if isinstance(error, tomllib.TOMLDecodeError):
        lineno = getattr(error, "lineno", None)
        colno = getattr(error, "colno", None)
        message = str(error)
        if lineno is not None and colno is not None:
            return f"{path}: {message} (at line {lineno}, column {colno})"
    return f"{path}: {error}"


def _load_split_config_dir(dir_path: Path) -> dict[str, Any]:
    """Load config from a directory of top-level section TOML files."""
    data: dict[str, Any] = {}

    for split_file in _SPLIT_CONFIG_FILES:
        file_path = _resolve_split_config_path(dir_path, split_file)
        if file_path is None:
            continue
        try:
            section_data = _load_toml_config(file_path)
        except (tomllib.TOMLDecodeError, ValueError) as error:
            raise ValueError(_describe_config_error(file_path, error)) from error
        if section_data:
            data[split_file.section] = section_data

    return data


def _serialize_main_config(config: Config) -> dict[str, Any]:
    """Serialize config for the main TOML file, excluding runtime-only fields."""
    return config.model_dump(by_alias=True)


def _save_split_config_dir(data: dict[str, Any], dir_path: Path) -> None:
    """Save config into section-specific TOML files."""
    dir_path.mkdir(parents=True, exist_ok=True)

    for split_file in _SPLIT_CONFIG_FILES:
        section_data = data.get(split_file.section, {})
        file_path = dir_path / split_file.filename
        _write_toml_file(file_path, section_data)


def _resolve_split_config_path(dir_path: Path, split_file: SplitConfigFile) -> Path | None:
    file_path = dir_path / split_file.filename
    return file_path if file_path.exists() else None


def _write_toml_file(path: Path, data: dict[str, Any]) -> None:
    """Write a TOML file using tomli-w when possible."""
    with open(path, "wb") as f:
        if tomli_w is not None and not _contains_none(data):
            tomli_w.dump(data, f)
        else:
            f.write(_dump_toml(data).encode("utf-8"))


def _render_template_table(
    path: tuple[str, ...],
    table: dict[str, Any],
    *,
    prune_empty: bool = True,
) -> str:
    """Render a single table subtree."""
    filtered = _prune_template_values(table) if prune_empty else table
    lines: list[str] = []
    _append_toml_table(lines, path, filtered)
    return "\n".join(lines).rstrip() + "\n"


def _prune_template_values(value: Any) -> Any:
    """Drop empty optional values from user-facing templates."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            pruned = _prune_template_values(item)
            if pruned in (None, {}, []):
                continue
            result[key] = pruned
        return result
    if isinstance(value, list):
        return value if value else None
    if value == "":
        return None
    return value


def _render_agents_template(config: Config) -> str:
    data = {
        "defaults": config.agents.defaults.model_dump(by_alias=True),
        "subagent": config.agents.subagent.model_dump(by_alias=True),
    }
    intro: list[str] = []
    _comment_block(
        intro,
        "Agent defaults.",
        "Change `workspace`, `model`, and optionally `provider` to get started.",
        "Leave the advanced knobs below as-is unless you already know you need them.",
    )
    return _join_template_sections(
        "\n".join(intro), _render_template_table((), data, prune_empty=False)
    )


def _render_channels_template(config: Config) -> str:
    data = config.channels.model_dump(by_alias=True)
    intro: list[str] = []
    _comment_block(
        intro,
        "Chat delivery and channel settings.",
        "Use the top-level flags for progress and tool feedback in chat surfaces.",
        "Configure `[telegram]` only if you want rvoone to send or receive Telegram messages.",
    )
    return _join_template_sections(
        "\n".join(intro), _render_template_table((), data, prune_empty=False)
    )


def _render_gateway_template(config: Config) -> str:
    data = config.gateway.model_dump(by_alias=True)
    intro: list[str] = []
    _comment_block(
        intro,
        "Server settings.",
        "Use this file to adjust the heartbeat interval or disable it entirely.",
    )
    return _join_template_sections(
        "\n".join(intro), _render_template_table((), data, prune_empty=False)
    )


def _render_tools_template(config: Config) -> str:
    data = config.tools.model_dump(by_alias=True)
    intro: list[str] = []
    _comment_block(
        intro,
        "Optional tool settings.",
        "Leave this file mostly untouched unless you need shell tuning or stricter workspace limits.",
    )
    return _join_template_sections(
        "\n".join(intro), _render_template_table((), data, prune_empty=False)
    )


def _render_provider_block(name: str, data: dict[str, Any], *, prune_empty: bool = True) -> str:
    """Render a provider block."""
    rendered = _render_template_table((name,), data, prune_empty=prune_empty)
    return "" if not rendered.strip() else rendered


def _render_provider_example_block(name: str) -> str:
    """Render a commented provider example without activating it in config."""
    return (
        "# Example source-routed setup.\n"
        "# Use this when you want model names like `example/gpt-4o-mini` or `local/qwen/qwen3-coder`.\n"
        "# The part before the first `/` selects the source.\n"
        "#\n"
        f"# [{name}]\n"
        '# apiBase = "http://localhost:8000/v1"\n'
        '# apiKey = "no-key"\n'
        '# tokenEstimation = "auto"  # off | auto | on\n'
        "# availableModels = []\n"
    )


def _render_providers_template(config: Config) -> str:
    intro: list[str] = []
    _comment_block(
        intro,
        "LLM settings.",
        "Most setups only need one OpenAI-compatible endpoint under `[custom]`.",
        "Use `[customSources.<name>]` only when you intentionally route models by prefix.",
        "OpenAI Codex uses OAuth and does not require an API key here.",
    )

    parts = ["\n".join(intro), f"upstreamTimeout = {config.providers.upstream_timeout}\n"]

    custom_sources = config.providers.custom_sources
    if custom_sources:
        parts.append(
            "\n".join(
                [
                    "# Multi-source routing.",
                    "# Example model values:",
                    '# model = "siliconflow/deepseek-ai/DeepSeek-V3"',
                    '# model = "local/qwen/qwen3-coder-480b-a35b-instruct"',
                ]
            )
        )
        for source_name, source_cfg in custom_sources.items():
            rendered = _render_provider_block(
                f"customSources.{source_name}",
                source_cfg.model_dump(by_alias=True),
                prune_empty=False,
            )
            if rendered:
                parts.append(rendered)
    else:
        parts.append(_render_provider_example_block("customSources.example"))

    parts.append(
        "\n".join(
            [
                "# Single OpenAI-compatible endpoint.",
                "# This is the recommended place to start for Ollama, vLLM, One API, OpenRouter-compatible gateways, etc.",
            ]
        )
        + "\n"
        + _render_provider_block(
            "custom", config.providers.custom.model_dump(by_alias=True), prune_empty=False
        )
    )

    groq_data = {
        "apiKey": config.providers.groq.api_key,
        "availableModels": config.providers.groq.available_models,
    }
    parts.append(
        "# Optional Groq settings used for transcription-related features only.\n"
        + _render_provider_block("groq", groq_data, prune_empty=False)
    )

    return _join_template_sections(*parts)


def _should_use_split_dir(path: Path) -> bool:
    """Return whether a path should be treated as the split config directory."""
    return not path.suffix


def _contains_none(value: Any) -> bool:
    """Return whether a nested structure contains None values."""
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


def _dump_toml(data: dict[str, Any]) -> str:
    """Fallback TOML writer for environments where tomli-w is unavailable."""
    lines: list[str] = []
    _append_toml_table(lines, (), data)
    return "\n".join(lines).rstrip() + "\n"


def _append_toml_table(lines: list[str], path: tuple[str, ...], table: dict[str, Any]) -> None:
    """Append a TOML table and its subtables."""
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, dict[str, Any]]] = []

    for key, value in table.items():
        if isinstance(value, dict):
            if value:
                table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    emit_header = bool(path and (scalar_items or not table_items))

    if emit_header:
        if lines:
            lines.append("")
        lines.append(f"[{'.'.join(path)}]")

    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")

    for key, value in table_items:
        _append_toml_table(lines, path + (key,), value)


def _format_toml_value(value: Any) -> str:
    """Format a TOML scalar or array value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _preprocess_null_literals(text: str) -> str:
    """Translate rvoone-style `null` literals into TOML-compatible sentinel strings."""
    return re.sub(
        r"(?m)^(\s*[A-Za-z0-9_.-]+\s*=\s*)null(\s*(?:#.*)?)$",
        rf'\1"{_NULL_SENTINEL}"\2',
        text,
    )


def _restore_null_sentinel(value: Any) -> Any:
    """Convert sentinel strings back into Python None."""
    if isinstance(value, dict):
        return {key: _restore_null_sentinel(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_null_sentinel(item) for item in value]
    if value == _NULL_SENTINEL:
        return None
    return value
