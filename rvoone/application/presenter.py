"""Channel-specific presentation helpers for agent responses."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from rvoone.application.controls import (
    InteractiveViewControl,
    ReplyDraftControl,
    ToolStatusControl,
    TypingControl,
)
from rvoone.bus.events import OutboundMessage

if TYPE_CHECKING:
    from rvoone.agent.loop import AgentLoop
    from rvoone.bus.events import InboundMessage


class AgentPresenter:
    """Build and publish channel-specific UI control messages."""

    def __init__(self, owner: "AgentLoop") -> None:
        self.owner = owner

    async def publish_typing_keepalive(self, channel: str, chat_id: str, ttl: int = 30) -> None:
        """Publish a channel-specific typing keepalive control message."""
        if channel != "telegram":
            return
        await self.owner.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="",
                metadata=TypingControl(action="renew", ttl=ttl).to_metadata(),
            )
        )

    async def publish_typing_stop(self, channel: str, chat_id: str) -> None:
        """Publish a channel-specific typing stop control message."""
        if channel != "telegram":
            return
        await self.owner.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="",
                metadata=TypingControl(action="stop").to_metadata(),
            )
        )

    def next_tool_status_key(self) -> str:
        """Generate a per-process unique key for a tool status message."""
        self.owner._tool_status_counter += 1
        return f"tool-status-{self.owner._tool_status_counter}"

    def next_reply_draft_id(self) -> int:
        """Generate a per-process unique Telegram draft identifier."""
        self.owner._reply_draft_counter += 1
        return self.owner._reply_draft_counter

    def tool_status_enabled(self, channel: str) -> bool:
        """Whether editable tool-status messages should be published."""
        return channel == "telegram" and self.owner.channels_config.send_tool_status

    def reply_drafts_enabled(
        self,
        channel: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Whether Telegram reply drafts should be emitted for this message."""
        if channel != "telegram" or not self.owner.channels_config.send_message_drafts:
            return False
        return not bool((metadata or {}).get("is_group"))

    def tool_status_pin_enabled(self, channel: str) -> bool:
        """Whether Telegram tool-status messages should request pinning."""
        return channel == "telegram" and self.owner.channels_config.pin_tool_status

    def format_tool_status(self, tool_name: str, arguments: Any) -> str:
        """Render a single-line tool status message with truncation."""
        args_str = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        text = f"Tool call: {tool_name}({args_str})"
        limit = max(int(self.owner.channels_config.tool_status_max_chars), 32)
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 1)] + "..."

    async def publish_tool_status_create(
        self,
        channel: str,
        chat_id: str,
        status_key: str,
        text: str,
        *,
        pin_in_chat: bool = False,
    ) -> None:
        """Create the Telegram tool-status message."""
        if not self.tool_status_enabled(channel):
            return
        await self.owner.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="",
                metadata=ToolStatusControl(
                    action="create",
                    status_key=status_key,
                    text=text,
                    pin=pin_in_chat,
                ).to_metadata(),
            )
        )

    async def publish_tool_status_update(
        self,
        channel: str,
        chat_id: str,
        status_key: str,
        text: str,
    ) -> None:
        """Update the Telegram tool-status message."""
        if not self.tool_status_enabled(channel):
            return
        await self.owner.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="",
                metadata=ToolStatusControl(
                    action="update",
                    status_key=status_key,
                    text=text,
                ).to_metadata(),
            )
        )

    async def publish_tool_status_delete(self, channel: str, chat_id: str, status_key: str) -> None:
        """Delete the Telegram tool-status message."""
        if not self.tool_status_enabled(channel):
            return
        await self.owner.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="",
                metadata=ToolStatusControl(
                    action="delete",
                    status_key=status_key,
                    delete_delay_s=self.owner._TOOL_STATUS_DELETE_DELAY_S,
                ).to_metadata(),
            )
        )

    async def publish_reply_draft(
        self,
        channel: str,
        chat_id: str,
        draft_id: int,
        text: str,
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Publish one Telegram draft update for an in-flight reply."""
        if channel != "telegram" or not text:
            return
        await self.owner.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="",
                metadata=ReplyDraftControl(
                    draft_id=draft_id,
                    text=text,
                    message_thread_id=message_thread_id,
                ).to_metadata(),
            )
        )

    def start_typing_keepalive(
        self, channel: str, chat_id: str, ttl: int = 30
    ) -> asyncio.Task | None:
        """Start a background typing keepalive loop for tool execution."""
        if channel != "telegram":
            return None

        async def _loop() -> None:
            try:
                while True:
                    await self.publish_typing_keepalive(channel, chat_id, ttl=ttl)
                    await asyncio.sleep(max(ttl - 5, 1))
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_loop())

    @staticmethod
    def typing_target(msg: "InboundMessage") -> tuple[str, str]:
        """Resolve the actual channel/chat pair for typing controls."""
        if msg.channel != "system":
            return msg.channel, msg.chat_id
        if ":" in msg.chat_id:
            return msg.chat_id.split(":", 1)
        return "cli", msg.chat_id

    @staticmethod
    def apply_interactive_view(
        metadata: dict[str, Any],
        *,
        view: str,
        page: int | None = None,
        total_pages: int | None = None,
        buttons: list[list[dict[str, str]]] | None = None,
    ) -> dict[str, Any]:
        """Attach Telegram interactive view metadata."""
        metadata.update(
            InteractiveViewControl(
                view=view,
                create=not bool(metadata.get("_interactive_message_id")),
                message_id=metadata.get("_interactive_message_id"),
                callback_query_id=metadata.get("_interactive_callback_query_id"),
                page=page,
                total_pages=total_pages,
                buttons=buttons,
            ).to_metadata()
        )
        return metadata
