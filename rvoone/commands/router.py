"""Centralized slash-command routing."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from rvoone.bus.events import InboundMessage, OutboundMessage

SUPPORTED_COMMANDS: tuple[tuple[str, str], ...] = (
    ("start", "Start the bot"),
    ("new", "Start a new conversation"),
    ("model", "Show or switch the active model"),
    ("status", "Show current session status"),
    ("stop", "Stop the current task"),
    ("help", "Show available commands"),
)

_HELP_TEXT = (
    "🕊 rvoone commands:\n"
    "/new — Start a new conversation\n"
    "/model — Show or switch the active model\n"
    "/status — Show current session status\n"
    "/stop — Stop the current task\n"
    "/help — Show available commands"
)

_START_TEXT = (
    "👋 Hi! I'm rvoone.\n\n"
    "Send me a message and I'll respond!\n"
    "Type /help to see available commands."
)


class CommandRouter:
    """Route supported slash commands before normal agent processing."""

    def __init__(
        self,
        *,
        handle_new: Callable[[InboundMessage], Awaitable[OutboundMessage]],
        handle_model: Callable[[InboundMessage], Awaitable[OutboundMessage]],
        handle_status: Callable[[InboundMessage], Awaitable[OutboundMessage]],
        handle_stop: Callable[[InboundMessage], Awaitable[OutboundMessage]],
    ) -> None:
        self._handle_new = handle_new
        self._handle_model = handle_model
        self._handle_status = handle_status
        self._handle_stop = handle_stop

    @staticmethod
    def _normalize_command(content: str) -> str | None:
        text = content.strip()
        if not text.startswith("/"):
            return None
        head = text.split(None, 1)[0]
        command = head.split("@", 1)[0].lower()
        return command if command in {f"/{name}" for name, _ in SUPPORTED_COMMANDS} else None

    async def route(self, msg: InboundMessage) -> OutboundMessage | None:
        """Return a command response for supported commands, else None."""
        command = self._normalize_command(msg.content)
        if command is None:
            return None
        if command == "/start":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=_START_TEXT)
        if command == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=_HELP_TEXT)
        if command == "/new":
            return await self._handle_new(msg)
        if command == "/model":
            return await self._handle_model(msg)
        if command == "/status":
            return await self._handle_status(msg)
        return await self._handle_stop(msg)
