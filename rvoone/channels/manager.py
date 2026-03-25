"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from rvoone.bus.events import OutboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.channels.base import BaseChannel
from rvoone.config.schema import Config


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    _SEND_TIMEOUT_S = 45.0

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._send_tasks: set[asyncio.Task] = set()

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels based on config."""

        # Telegram channel
        if self.config.channels.telegram.enabled:
            try:
                from rvoone.channels.telegram import TelegramChannel

                self.channels["telegram"] = TelegramChannel(
                    self.config.channels.telegram,
                    self.bus,
                    groq_api_key=self.config.providers.groq.api_key,
                    draft_send_interval_s=self.config.channels.send_message_draft_interval_s,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning("Telegram channel not available: {}", e)

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for task in list(self._send_tasks):
            task.cancel()
        if self._send_tasks:
            await asyncio.gather(*self._send_tasks, return_exceptions=True)
        self._send_tasks.clear()

        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        while True:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self.config.channels.send_tool_hints:
                        continue
                    if (
                        not msg.metadata.get("_tool_hint")
                        and not self.config.channels.send_progress
                    ):
                        continue

                channel = self.channels.get(msg.channel)
                if channel:
                    self._schedule_send(msg.channel, channel, msg)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _schedule_send(self, channel_name: str, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Deliver one outbound message without blocking the dispatcher loop."""
        task = asyncio.create_task(self._send_with_timeout(channel_name, channel, msg))
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)

    async def _send_with_timeout(
        self,
        channel_name: str,
        channel: BaseChannel,
        msg: OutboundMessage,
    ) -> None:
        """Send one outbound message with a hard timeout to avoid wedging delivery."""
        try:
            await asyncio.wait_for(channel.send(msg), timeout=self._SEND_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error(
                "Timed out sending outbound message to {} chat_id={} metadata_keys={}",
                channel_name,
                msg.chat_id,
                sorted(msg.metadata.keys()),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error sending to {}: {}", channel_name, e)

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
