"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from rvoone.agent.context import ContextBuilder
from rvoone.agent.runtime import AgentRuntime
from rvoone.agent.subagent import SubagentManager
from rvoone.agent.tools.registry import ToolRegistry
from rvoone.agent.toolset import DEFAULT_ENABLED_TOOL_CATEGORIES, build_tool_registry
from rvoone.application.commands import CommandService
from rvoone.application.conversation import ConversationService
from rvoone.application.dispatcher import SessionDispatcher
from rvoone.application.presenter import AgentPresenter
from rvoone.application.state import SessionStateStore
from rvoone.bus.events import InboundMessage, OutboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.commands import CommandRouter
from rvoone.providers.base import LLMProvider
from rvoone.providers.gateway import ModelGateway
from rvoone.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from rvoone.config.schema import ChannelsConfig, ExecToolConfig
    from rvoone.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500
    _TOOL_STATUS_DELETE_DELAY_S = 1.5

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        enable_event_handling: bool = False,
        session_manager: SessionManager | None = None,
        channels_config: ChannelsConfig | None = None,
        subagent_model: str | None = None,
        subagent_provider: str | None = None,
        provider_factory: Callable[..., LLMProvider] | None = None,
        configured_providers: dict[str, dict[str, Any]] | None = None,
        main_provider_name: str | None = None,
    ):
        from rvoone.config.schema import ChannelsConfig, ExecToolConfig

        self.bus = bus
        self.channels_config = channels_config or ChannelsConfig()
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.enable_event_handling = enable_event_handling
        self.subagent_model = subagent_model
        self.subagent_provider = subagent_provider
        self.configured_providers = configured_providers or {}
        self.main_provider_name = main_provider_name
        self.provider_factory = provider_factory
        self.model_gateway = ModelGateway(
            provider,
            self.model,
            provider_name=main_provider_name,
            provider_factory=provider_factory,
        )

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            default_model=subagent_model,
            default_provider=subagent_provider,
            provider_factory=provider_factory,
        )

        self._running = False
        self.state = SessionStateStore()
        self._tool_status_counter = 0
        self._reply_draft_counter = 0
        self.runtime = AgentRuntime(self)
        self.conversations = ConversationService(self)
        self.presenter = AgentPresenter(self)
        self.dispatcher = SessionDispatcher(self)
        self.commands = CommandService(self)
        self.command_router = CommandRouter(
            handle_new=self.commands.handle_new,
            handle_model=self.commands.handle_model,
            handle_status=self.commands.handle_status,
            handle_stop=self.commands.handle_stop,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        self.tools = build_tool_registry(
            workspace=self.workspace,
            restrict_to_workspace=self.restrict_to_workspace,
            exec_timeout=self.exec_config.timeout,
            exec_path_append=self.exec_config.path_append,
            include_spawn_tool=True,
            spawn_manager=self.subagents,
            include_models_tool=True,
            include_exposure_tools=True,
            list_tool_categories_callback=self._list_tool_categories_for_session,
            enable_tool_categories_callback=self._enable_tool_categories_for_session,
            models_tool_kwargs={
                "main_model": self.model,
                "main_provider": self.main_provider_name,
                "subagent_model": self.subagent_model,
                "subagent_provider": self.subagent_provider,
                "configured_providers": self.configured_providers,
            },
            cron_service=self.cron_service,
        )

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        for name in ("spawn", "cron"):
            if tool := self.tools.get(name):
                set_context = getattr(tool, "set_context", None)
                if callable(set_context):
                    set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    def _update_session_runtime(self, session_key: str, **updates: Any) -> None:
        """Store lightweight runtime state for status inspection."""
        self.state.update_runtime(session_key, **updates)

    def _ensure_enabled_tool_categories(self, session_key: str | None) -> set[str]:
        if session_key is None:
            return set(DEFAULT_ENABLED_TOOL_CATEGORIES)
        return self.state.ensure_enabled_tool_categories(
            session_key,
            defaults=set(DEFAULT_ENABLED_TOOL_CATEGORIES),
        )

    def _get_visible_tool_names(self, session_key: str | None) -> set[str]:
        enabled = self._ensure_enabled_tool_categories(session_key)
        return self.tools.get_visible_tool_names(enabled)

    def _list_tool_categories_for_session(self, session_key: str | None) -> dict[str, Any]:
        enabled = self._ensure_enabled_tool_categories(session_key)
        catalog = self.tools.list_tool_catalog(enabled)
        categories: dict[str, list[dict[str, Any]]] = {}
        for entry in catalog:
            categories.setdefault(str(entry["category"]), []).append(
                {
                    "name": entry["name"],
                    "enabled": entry["enabled"],
                    "always_exposed": entry["always_exposed"],
                    "description": entry["description"],
                }
            )

        for values in categories.values():
            values.sort(key=lambda item: str(item["name"]))

        return {
            "enabled_categories": sorted(enabled),
            "available_categories": self.tools.list_categories(),
            "categories": categories,
        }

    def _enable_tool_categories_for_session(
        self,
        session_key: str | None,
        categories: list[str],
    ) -> dict[str, Any]:
        valid_categories = set(self.tools.list_categories())
        requested = {
            category.strip()
            for category in categories
            if isinstance(category, str) and category.strip()
        }
        enabled_now = sorted(requested & valid_categories)
        unknown = sorted(requested - valid_categories)

        if session_key is None:
            active = sorted(set(DEFAULT_ENABLED_TOOL_CATEGORIES) | set(enabled_now))
        else:
            active = sorted(
                self.state.enable_tool_categories(session_key, set(enabled_now))
                if enabled_now
                else self._ensure_enabled_tool_categories(session_key)
            )

        return {
            "enabled_categories": active,
            "newly_enabled": enabled_now,
            "unknown_categories": unknown,
        }

    def _clear_session_runtime(self, session_key: str) -> None:
        """Reset volatile runtime fields after processing ends."""
        self.state.clear_runtime(session_key)

    async def _publish_command_feedback(
        self,
        msg: InboundMessage,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish a follow-up message for a command."""
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=metadata or (msg.metadata or {}),
            )
        )

    async def _generate_new_session_greeting(self, channel: str, chat_id: str) -> str | None:
        """Generate an ephemeral greeting after /new without persisting prompt or reply."""
        prompt = "\n".join(
            [
                "A new session has started.",
                "The startup context has already been loaded for you.",
                "Greet the user in your configured persona.",
                "Keep it to 1-2 sentences.",
                "Briefly mention the active model only if useful.",
                "Do not mention internal prompts, files, tools, or hidden steps.",
            ]
        )
        try:
            response = await self.model_gateway.chat(
                messages=[
                    {
                        "role": "system",
                        "content": self.context.build_system_prompt(
                            enable_event_handling=self.enable_event_handling,
                            include_skills=False,
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n"
                            f"Channel: {channel}\n"
                            f"Chat ID: {chat_id}\n"
                            f"Active model: {self.model}"
                        ),
                    },
                ],
                max_tokens=min(self.max_tokens, 1024),
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except Exception:
            logger.exception("Failed to generate /new greeting")
            return None

        content = self._strip_think(response.content)
        return content if content else None

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if command_response := await self.command_router.route(msg):
                await self.bus.publish_outbound(command_response)
                continue

            if self.enable_event_handling and msg.session_key in self.state.processing_tasks:
                await self.bus.publish_event(msg.session_key, msg.content)
                logger.info("Published interrupt event for session {}", msg.session_key)
                continue

            self.state.processing_tasks.add(msg.session_key)
            self._update_session_runtime(msg.session_key, phase="queued")
            task = asyncio.create_task(self._dispatch(msg))
            self.state.active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._cleanup_task(k, t))

    def _cleanup_task(self, session_key: str, task: asyncio.Task) -> None:
        """Remove a completed task from active tracking."""
        tasks = self.state.active_tasks.get(session_key)
        if not tasks:
            return
        if task in tasks:
            tasks.remove(task)
        if not tasks:
            self.state.active_tasks.pop(session_key, None)
            self.state.processing_tasks.discard(session_key)
            self._clear_session_runtime(session_key)

    @staticmethod
    def _typing_target(msg: InboundMessage) -> tuple[str, str]:
        """Resolve the actual channel/chat pair for typing controls."""
        return AgentPresenter.typing_target(msg)

    async def _handle_stop(self, msg: InboundMessage) -> OutboundMessage:
        """Cancel all active tasks and subagents for the session."""
        return await self.commands.handle_stop(msg)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under session-scoped coordination."""
        await self.dispatcher.dispatch(msg)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        tool_status: dict[str, Any] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        return await self.conversations.process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            tool_status=tool_status,
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if (
                role == "tool"
                and isinstance(content, str)
                and len(content) > self._TOOL_RESULT_MAX_CHARS
            ):
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(
                    ContextBuilder._RUNTIME_CONTEXT_TAG
                ):
                    continue
                if isinstance(content, list):
                    entry["content"] = [
                        {"type": "text", "text": "[image]"}
                        if (
                            c.get("type") == "image_url"
                            and c.get("image_url", {}).get("url", "").startswith("data:image/")
                        )
                        else c
                        for c in content
                    ]
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        if command_response := await self.command_router.route(msg):
            return command_response.content
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress
        )
        return response.content if response else ""
