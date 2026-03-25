"""Application-level command handlers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rvoone.bus.events import OutboundMessage

if TYPE_CHECKING:
    from rvoone.agent.loop import AgentLoop
    from rvoone.bus.events import InboundMessage


class CommandService:
    """Handle slash commands without embedding their logic in AgentLoop."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def handle_new(self, msg: "InboundMessage") -> OutboundMessage:
        return await self.owner.conversations.archive_session(msg)

    async def handle_stop(self, msg: "InboundMessage") -> OutboundMessage:
        owner = self.owner
        tasks = owner.state.active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        owner.state.processing_tasks.discard(msg.session_key)
        sub_cancelled = await owner.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    async def handle_status(self, msg: "InboundMessage") -> OutboundMessage:
        owner = self.owner
        session = owner.sessions.get_or_create(msg.session_key)
        active_tasks = sum(
            1 for task in owner.state.active_tasks.get(msg.session_key, []) if not task.done()
        )
        subagents = len(getattr(owner.subagents, "_session_tasks", {}).get(msg.session_key, set()))
        runtime = owner.state.runtime.get(msg.session_key, {})
        phase = str(
            runtime.get("phase")
            or ("processing" if msg.session_key in owner.state.processing_tasks else "idle")
        )
        current_tool = runtime.get("current_tool")
        current_tool_args = runtime.get("current_tool_args")
        updated_at = getattr(session, "updated_at", None)
        updated_text = updated_at.isoformat(sep=" ", timespec="seconds") if updated_at else "n/a"
        page = self.parse_status_page(msg.content)
        pages = [
            (
                "🕊 Session status 🕊\n\n"
                "📋 Session\n"
                f"- Key: {msg.session_key}\n"
                f"- Channel: {msg.channel}\n"
                f"- Chat ID: {msg.chat_id}\n"
                f"- Messages: {len(session.messages)}\n"
                f"- Last updated: {updated_text}"
            ),
            (
                "🕊 Session status 🕊\n\n"
                "⚙️ Execution\n"
                f"- Agent loop running: {'yes' if owner._running else 'no'}\n"
                f"- Active tasks: {active_tasks}\n"
                f"- Session locked: {'yes' if owner.dispatcher.session_lock(msg.session_key).locked() else 'no'}\n"
                f"- Phase: {phase}\n"
                f"- Current tool: {current_tool or 'none'}\n"
                f"- Tool args: {current_tool_args or 'n/a'}\n"
                f"- Subagents: {subagents}"
            ),
            (
                "🕊 Session status 🕊\n\n"
                "🧭 Runtime\n"
                f"- Pending interrupts: {owner.bus.pending_events(msg.session_key)}\n"
                f"- Registered tools: {len(owner.tools)}\n"
                f"- Event handling: {'enabled' if owner.enable_event_handling else 'disabled'}"
            ),
        ]
        page_count = len(pages)
        page = min(max(page, 1), page_count)
        text = f"{pages[page - 1]}\n\nPage {page}/{page_count}"
        metadata = dict(msg.metadata or {})
        if msg.channel == "telegram":
            metadata = owner.presenter.apply_interactive_view(
                metadata,
                view="status",
                page=page,
                total_pages=page_count,
            )
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=text, metadata=metadata
        )

    def available_model_options(self) -> list[tuple[str, str | None]]:
        owner = self.owner
        options: list[tuple[str, str | None]] = []
        seen: set[str] = set()

        def _add(model: str | None, provider: str | None) -> None:
            if not isinstance(model, str) or not model or model in seen:
                return
            seen.add(model)
            options.append((model, provider))

        _add(owner.model, owner.main_provider_name)
        _add(owner.subagent_model, owner.subagent_provider)
        for provider_name, info in owner.configured_providers.items():
            for model in info.get("available_models", []) or []:
                _add(model, provider_name)
        return options

    async def handle_model(self, msg: "InboundMessage") -> OutboundMessage:
        owner = self.owner
        metadata = dict(msg.metadata or {})
        options = self.available_model_options()
        current = owner.model
        parts = msg.content.strip().split(maxsplit=1)

        if len(parts) > 1:
            requested = parts[1].strip()
            provider_hint = next(
                (provider for model, provider in options if model == requested), None
            )
            if not any(model == requested for model, _ in options):
                available = ", ".join(model for model, _ in options) or "(none configured)"
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Unknown model: {requested}\nAvailable: {available}",
                    metadata=metadata,
                )
            if owner.provider_factory:
                owner.model_gateway.provider_factory = owner.provider_factory
                owner.provider = owner.model_gateway.switch_model(requested, provider_hint)
            else:
                owner.model_gateway.model = requested
                owner.model_gateway.provider_name = provider_hint
            owner.model = requested
            owner.main_provider_name = provider_hint
            if not getattr(owner.subagents, "default_model", None):
                owner.subagents.model = requested
            current = requested

        lines = [
            "🧠 Model selection",
            "",
            f"Current: {current}",
            f"Provider: {owner.main_provider_name or 'unresolved'}",
        ]
        if len(parts) > 1:
            lines.append(f"Switched to: {current}")
        lines.extend(["", "Configured models:"])
        lines.extend(
            f"{'✓' if model == current else '•'} {model} [{provider or 'auto'}]"
            for model, provider in options
        )
        if len(parts) == 1 and msg.channel != "telegram":
            lines.extend(["", "Use `/model <model_name>` to switch."])

        if msg.channel == "telegram":
            metadata = owner.presenter.apply_interactive_view(
                metadata,
                view="model",
                buttons=[
                    [
                        {
                            "text": f"{'✓ ' if model == current else ''}{model}",
                            "callback_data": f"model:set:{model}",
                        }
                    ]
                    for model, _ in options
                ],
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="\n".join(lines),
            metadata=metadata,
        )

    @staticmethod
    def parse_status_page(content: str) -> int:
        parts = content.strip().split()
        if len(parts) < 2:
            return 1
        try:
            return int(parts[1])
        except ValueError:
            return 1
