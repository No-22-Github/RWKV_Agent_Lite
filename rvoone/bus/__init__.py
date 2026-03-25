"""Message bus module for decoupled channel-agent communication."""

from rvoone.bus.events import InboundMessage, OutboundMessage
from rvoone.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
