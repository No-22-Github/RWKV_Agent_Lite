"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TelegramConfig(Base):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""  # Bot token from @BotFather
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs or usernames
    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    reply_to_message: bool = False  # If true, bot replies quote the original message


class ChannelsConfig(Base):
    """Configuration for chat channels."""

    send_progress: bool = True  # stream agent's text progress to the channel
    send_message_drafts: bool = True  # stream Telegram private-chat replies via sendMessageDraft
    send_message_draft_interval_s: float = 2.0  # min seconds between Telegram draft updates
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    send_tool_status: bool = True  # show a single editable Telegram tool-status message per turn
    pin_tool_status: bool = False  # request Telegram to pin the tool-status message
    tool_status_max_chars: int = 180  # max chars for Telegram tool-status message text
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.rvoone/workspace"
    model: str = "gpt-4o-mini"
    provider: str = "auto"  # "custom" or "auto"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
    reasoning_effort: str | None = None  # low / medium / high — enables LLM thinking mode
    enable_event_handling: bool = True  # Inject queued user interrupts during long tool runs


class SubagentDefaults(Base):
    """Default subagent configuration."""

    model: str | None = None
    provider: str = "auto"  # "custom" or "auto"


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    subagent: SubagentDefaults = Field(default_factory=SubagentDefaults)


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    request_dump: bool = False
    stream_mode: Literal["auto", "on", "off"] = "auto"
    token_estimation: Literal["off", "auto", "on"] = (
        "auto"  # Optional request-token estimate logging
    )
    available_models: list[str] = Field(default_factory=list)


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    upstream_timeout: int = 60  # Seconds before LLM API request times out
    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    custom_sources: dict[str, ProviderConfig] = Field(
        default_factory=dict
    )  # Named OpenAI-compatible upstreams selected by source/model prefix
    groq: ProviderConfig = Field(default_factory=ProviderConfig)  # Voice transcription only


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes


class GatewayConfig(Base):
    """Gateway/server configuration."""

    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    timeout: int = 60
    path_append: str = ""


class ToolsConfig(Base):
    """Tools configuration."""

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory


class Config(BaseSettings):
    """Root configuration for rvoone."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _resolve_custom_source(
        self,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[ProviderConfig | None, str | None]:
        """Resolve the custom provider config and optional source name."""
        if provider not in (None, "auto", "custom"):
            return None, None

        active_model = model or self.agents.defaults.model
        if active_model:
            source, _, _ = active_model.partition("/")
            if source and source in self.providers.custom_sources:
                return self.providers.custom_sources[source], source

        if self.providers.custom.api_key or self.providers.custom.api_base:
            return self.providers.custom, None
        return None, None

    def strip_model_provider_prefix(self, model: str) -> str:
        """Remove a configured provider prefix from a model identifier when needed."""
        source, sep, remainder = model.partition("/")
        if sep and source in self.providers.custom_sources:
            return remainder
        return model

    def _match_provider(
        self,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match the active LLM provider. Groq is excluded because it is transcription-only."""

        forced = provider or self.agents.defaults.provider
        if forced != "auto":
            if forced == "custom":
                custom_cfg, _ = self._resolve_custom_source(model, forced)
                if custom_cfg:
                    return custom_cfg, "custom"
                return self.providers.custom, "custom"
            return None, None

        custom_cfg, _ = self._resolve_custom_source(model, forced)
        if custom_cfg:
            return custom_cfg, "custom"
        return None, None

    def get_provider(
        self, model: str | None = None, provider: str | None = None
    ) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base). Falls back to first available."""
        p, _ = self._match_provider(model, provider)
        return p

    def get_provider_name(
        self, model: str | None = None, provider: str | None = None
    ) -> str | None:
        """Get the name of the matched LLM provider ("custom")."""
        _, name = self._match_provider(model, provider)
        return name

    def get_api_base(self, model: str | None = None, provider: str | None = None) -> str | None:
        """Get API base URL for the active LLM provider."""

        p, name = self._match_provider(model, provider)
        if p and p.api_base:
            return p.api_base
        if name == "custom":
            return "http://localhost:8000/v1"
        return None

    model_config = SettingsConfigDict(env_prefix="rvoone_", env_nested_delimiter="__")
