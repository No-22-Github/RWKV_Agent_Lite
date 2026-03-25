"""Session-scoped dispatch coordination."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from rvoone.bus.events import OutboundMessage

if TYPE_CHECKING:
    from rvoone.agent.loop import AgentLoop
    from rvoone.bus.events import InboundMessage


class SessionDispatcher:
    """Coordinate per-session execution without globally serializing all traffic."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def session_lock(self, session_key: str) -> asyncio.Lock:
        """Return the lock guarding one session."""
        return self.owner.state.session_lock(session_key)

    async def dispatch(self, msg: "InboundMessage") -> None:
        """Process one message under a session-scoped lock."""
        owner = self.owner
        owner.state.processing_tasks.add(msg.session_key)
        owner._update_session_runtime(msg.session_key, phase="processing")
        typing_channel, typing_chat_id = owner._typing_target(msg)
        tool_status = {
            "enabled": owner.presenter.tool_status_enabled(typing_channel),
            "pin_enabled": owner.presenter.tool_status_pin_enabled(typing_channel),
            "channel": typing_channel,
            "chat_id": typing_chat_id,
            "key": owner.presenter.next_tool_status_key(),
            "created": False,
        }
        try:
            async with self.session_lock(msg.session_key):
                try:
                    response = await owner._process_message(msg, tool_status=tool_status)
                    if response is not None:
                        await owner.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await owner.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="",
                                metadata=msg.metadata or {},
                            )
                        )
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", msg.session_key)
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", msg.session_key)
                    await owner.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="Sorry, I encountered an error.",
                        )
                    )
        finally:
            if tool_status.get("created"):
                await owner.presenter.publish_tool_status_delete(
                    tool_status["channel"], tool_status["chat_id"], tool_status["key"]
                )
            await owner.presenter.publish_typing_stop(typing_channel, typing_chat_id)
            owner._clear_session_runtime(msg.session_key)
            if not owner.state.active_tasks.get(msg.session_key):
                owner.state.processing_tasks.discard(msg.session_key)
