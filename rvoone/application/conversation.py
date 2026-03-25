"""Conversation application service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from rvoone.agent.toolset import DEFAULT_ENABLED_TOOL_CATEGORIES
from rvoone.bus.events import OutboundMessage

if TYPE_CHECKING:
    from rvoone.agent.loop import AgentLoop
    from rvoone.bus.events import InboundMessage


@dataclass(slots=True)
class _ReplyDraftStream:
    """Throttle Telegram draft updates for one in-flight reply."""

    owner: "AgentLoop"
    channel: str
    chat_id: str
    draft_id: int
    message_thread_id: int | None = None
    update_interval_s: float = 0.25
    max_chars: int = 4000
    _content: str = field(default="", init=False)
    _last_sent: str = field(default="", init=False)
    _last_sent_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def handle_delta(self, delta: str) -> None:
        """Append one streamed model delta and publish at a controlled cadence."""
        if not delta:
            return
        async with self._lock:
            self._content += delta
            if self._should_publish(delta):
                await self._publish_locked()

    async def flush(self) -> None:
        """Publish the latest buffered text if anything changed."""
        async with self._lock:
            await self._publish_locked(force=True)

    def _should_publish(self, delta: str) -> bool:
        now = asyncio.get_running_loop().time()
        return (
            now - self._last_sent_at >= self.update_interval_s
            or "\n" in delta
            or any(mark in delta for mark in ".!?")
        )

    def _preview_text(self) -> str:
        return self._content[: self.max_chars]

    async def _publish_locked(self, *, force: bool = False) -> None:
        preview = self._preview_text()
        if not preview or preview == self._last_sent:
            return
        if not force and len(preview) < 8:
            return
        await self.owner.presenter.publish_reply_draft(
            self.channel,
            self.chat_id,
            self.draft_id,
            preview,
            message_thread_id=self.message_thread_id,
        )
        self._last_sent = preview
        self._last_sent_at = asyncio.get_running_loop().time()


class ConversationService:
    """Coordinate sessions, memory, and runtime execution for one message."""

    def __init__(self, owner: "AgentLoop"):
        self.owner = owner

    async def process_message(
        self,
        msg: "InboundMessage",
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        tool_status: dict[str, Any] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        owner = self.owner
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = owner.sessions.get_or_create(key)
            owner._set_tool_context(channel, chat_id)
            history = session.get_history(max_messages=owner.memory_window)
            messages = owner.context.build_messages(
                history=history,
                current_message=msg.content,
                channel=channel,
                chat_id=chat_id,
                enable_event_handling=owner.enable_event_handling,
            )
            final_content, _, all_msgs = await owner.runtime.run_agent_loop(
                messages, session_key=key
            )
            owner._save_turn(session, all_msgs, 1 + len(history))
            owner.sessions.save(session)
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = owner.sessions.get_or_create(key)

        owner._set_tool_context(msg.channel, msg.chat_id)

        history = session.get_history(max_messages=owner.memory_window)
        initial_messages = owner.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            enable_event_handling=owner.enable_event_handling,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await owner.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        reply_draft_stream: _ReplyDraftStream | None = None
        if owner.presenter.reply_drafts_enabled(msg.channel, msg.metadata):
            reply_draft_stream = _ReplyDraftStream(
                owner=owner,
                channel=msg.channel,
                chat_id=msg.chat_id,
                draft_id=owner.presenter.next_reply_draft_id(),
                message_thread_id=msg.metadata.get("message_thread_id"),
            )

        await owner.presenter.publish_typing_keepalive(msg.channel, msg.chat_id)
        typing_keepalive = owner.presenter.start_typing_keepalive(msg.channel, msg.chat_id)

        try:
            final_content, _, all_msgs = await owner.runtime.run_agent_loop(
                initial_messages,
                on_progress=on_progress or _bus_progress,
                on_text_delta=reply_draft_stream.handle_delta if reply_draft_stream else None,
                session_key=key,
                tool_status=tool_status,
            )
        finally:
            if typing_keepalive and not typing_keepalive.done():
                typing_keepalive.cancel()
            if reply_draft_stream:
                await reply_draft_stream.flush()

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        owner._save_turn(session, all_msgs, 1 + len(history))
        owner.sessions.save(session)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        metadata = dict(msg.metadata or {})
        if reply_draft_stream:
            metadata["_draft_control"] = "complete"
            metadata["_draft_id"] = reply_draft_stream.draft_id
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=metadata,
        )

    async def archive_session(self, msg: "InboundMessage") -> OutboundMessage:
        """Archive the current session synchronously and return a confirmation."""
        owner = self.owner
        session = owner.sessions.get_or_create(msg.session_key)
        session.clear()
        owner.sessions.save(session)
        owner.sessions.invalidate(session.key)
        owner.state.reset_enabled_tool_categories(
            session.key,
            defaults=set(DEFAULT_ENABLED_TOOL_CATEGORIES),
        )
        status_text = "New session started."
        await owner._publish_command_feedback(msg, status_text)
        greeting = await owner._generate_new_session_greeting(msg.channel, msg.chat_id)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=greeting or status_text,
        )
