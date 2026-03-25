"""Async message queue for decoupled channel-agent communication."""

import asyncio

from rvoone.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._event_queues: dict[str, asyncio.Queue[str]] = {}

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()

    async def publish_event(self, session_key: str, content: str) -> None:
        """Publish an event for a specific session."""
        if session_key not in self._event_queues:
            self._event_queues[session_key] = asyncio.Queue()
        await self._event_queues[session_key].put(content)

    async def check_events(self, session_key: str) -> str | None:
        """Drain queued events for a session and return them as one block."""
        queue = self._event_queues.get(session_key)
        if not queue:
            return None

        events: list[str] = []
        while not queue.empty():
            try:
                events.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        return "\n".join(f"- {event}" for event in events) if events else None

    def pending_events(self, session_key: str) -> int:
        """Return the count of queued events for a session."""
        queue = self._event_queues.get(session_key)
        return queue.qsize() if queue else 0
