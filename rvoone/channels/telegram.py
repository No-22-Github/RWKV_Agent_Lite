"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from loguru import logger
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyParameters,
    Update,
)
from telegram.error import RetryAfter
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from rvoone.bus.events import OutboundMessage
from rvoone.bus.queue import MessageBus
from rvoone.channels.base import BaseChannel
from rvoone.commands import SUPPORTED_COMMANDS
from rvoone.config.schema import TelegramConfig


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []

    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", save_code_block, text)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []

    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", save_inline_code, text)

    # 3. Headers # Title -> just the title text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r"^>\s*(.*)$", r"\1", text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    return text


def _split_message(content: str, max_len: int = 4000) -> list[str]:
    """Split content into chunks within max_len, preferring line breaks."""
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        pos = cut.rfind("\n")
        if pos == -1:
            pos = cut.rfind(" ")
        if pos == -1:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


@dataclass(slots=True)
class _DraftUpdateState:
    """Mutable state for one Telegram reply draft."""

    chat_id: int
    draft_id: int
    message_thread_id: int | None = None
    latest_text: str = ""
    sent_text: str = ""
    retry_after_s: float = 0.0
    wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.

    Simple and reliable - no webhook/public IP needed.
    """

    name = "telegram"
    _DRAFT_IDLE_TIMEOUT_S = 15.0
    _DRAFT_CLEAR_TEXT = "\u2060"

    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [BotCommand(name, description) for name, description in SUPPORTED_COMMANDS]

    def __init__(
        self,
        config: TelegramConfig,
        bus: MessageBus,
        groq_api_key: str = "",
        draft_send_interval_s: float = 2.0,
    ):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self._draft_send_interval_s = max(float(draft_send_interval_s), 0.0)
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task
        self._typing_deadlines: dict[str, float] = {}  # chat_id -> monotonic deadline
        self._status_messages: dict[
            str, int
        ] = {}  # "{chat_id}:{status_key}" -> telegram message_id
        self._media_group_buffers: dict[str, dict] = {}
        self._media_group_tasks: dict[str, asyncio.Task] = {}
        self._drafts_globally_disabled = False
        self._draft_updates: dict[str, _DraftUpdateState] = {}

    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return

        self._running = True

        # Build the application with larger connection pool to avoid pool-timeout on long runs
        request_kwargs = {
            "connection_pool_size": 16,
            "pool_timeout": 5.0,
            "connect_timeout": 30.0,
            "read_timeout": 30.0,
        }
        if self.config.proxy:
            request_kwargs["proxy"] = self.config.proxy
        req = HTTPXRequest(**request_kwargs)
        updates_req = HTTPXRequest(**request_kwargs)
        builder = (
            Application.builder()
            .token(self.config.token)
            .request(req)
            .get_updates_request(updates_req)
        )
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Forward slash commands to the centralized command layer.
        self._app.add_handler(MessageHandler(filters.COMMAND, self._forward_command))
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))

        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (
                    filters.TEXT
                    | filters.PHOTO
                    | filters.VOICE
                    | filters.AUDIO
                    | filters.Document.ALL
                )
                & ~filters.COMMAND,
                self._on_message,
            )
        )

        logger.info("Starting Telegram bot (polling mode)...")

        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        logger.info("Telegram bot @{} connected", bot_info.username)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            logger.debug("Telegram bot commands registered")
        except Exception as e:
            logger.warning("Failed to register bot commands: {}", e)

        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,  # Ignore old messages on startup
        )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        for task in self._media_group_tasks.values():
            task.cancel()
        self._media_group_tasks.clear()
        self._media_group_buffers.clear()
        for state in self._draft_updates.values():
            if state.task and not state.task.done():
                state.task.cancel()
        self._draft_updates.clear()

        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            logger.error("Invalid chat_id: {}", msg.chat_id)
            return

        metadata = msg.metadata or {}
        typing_control = metadata.get("_typing_control")
        if typing_control:
            if typing_control == "renew":
                self._renew_typing(msg.chat_id, ttl=int(metadata.get("_typing_ttl", 30)))
            elif typing_control == "stop":
                self._stop_typing(msg.chat_id)
            return

        status_control = metadata.get("_status_control")
        if status_control:
            status_key = str(metadata.get("_status_key", ""))
            cache_key = f"{msg.chat_id}:{status_key}"
            try:
                if status_control == "create":
                    sent = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=str(metadata.get("_status_text", "")),
                        reply_parameters=None,
                    )
                    self._status_messages[cache_key] = sent.message_id
                    if metadata.get("_status_pin"):
                        try:
                            await self._app.bot.pin_chat_message(
                                chat_id=chat_id,
                                message_id=sent.message_id,
                                disable_notification=True,
                            )
                        except Exception as pin_error:
                            logger.debug(
                                "Failed to pin Telegram tool-status {}: {}", cache_key, pin_error
                            )
                elif status_control == "update":
                    message_id = self._status_messages.get(cache_key)
                    if message_id is not None:
                        await self._app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text=str(metadata.get("_status_text", "")),
                        )
                elif status_control == "delete":
                    message_id = self._status_messages.pop(cache_key, None)
                    if message_id is not None:
                        delay_s = float(metadata.get("_status_delete_delay_s", 0))
                        if delay_s > 0:
                            await asyncio.sleep(delay_s)
                        await self._app.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception as e:
                self._status_messages.pop(cache_key, None)
                logger.warning(
                    "Failed Telegram tool-status {} for {}: {}", status_control, cache_key, e
                )
            return

        draft_control = metadata.get("_draft_control")
        if draft_control:
            draft_id = int(metadata.get("_draft_id", 0) or 0)
            if draft_control == "complete" and draft_id > 0:
                await self._complete_reply_draft(chat_id=chat_id, draft_id=draft_id)
            draft_text = str(metadata.get("_draft_text", ""))
            message_thread_id = metadata.get("_draft_message_thread_id")
            if draft_control == "update" and draft_id > 0 and draft_text:
                await self._queue_reply_draft_update(
                    chat_id=chat_id,
                    draft_id=draft_id,
                    text=draft_text,
                    message_thread_id=int(message_thread_id)
                    if message_thread_id is not None
                    else None,
                )
                return

        interactive_control = metadata.get("_interactive_control")
        if interactive_control:
            view = str(metadata.get("_interactive_view", ""))
            page = max(int(metadata.get("_interactive_page", 1)), 1)
            total_pages = max(int(metadata.get("_interactive_pages", 1)), 1)
            buttons = metadata.get("_interactive_buttons")
            reply_markup = self._build_interactive_keyboard(
                view, page, total_pages, buttons=buttons
            )
            try:
                if interactive_control == "create":
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=msg.content,
                        reply_markup=reply_markup,
                        reply_parameters=None,
                    )
                elif interactive_control == "update":
                    message_id = metadata.get("_interactive_message_id")
                    if message_id is not None:
                        await self._app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(message_id),
                            text=msg.content,
                            reply_markup=reply_markup,
                        )
                    callback_query_id = metadata.get("_interactive_callback_query_id")
                    if callback_query_id:
                        await self._app.bot.answer_callback_query(callback_query_id)
            except Exception as e:
                logger.warning(
                    "Failed Telegram interactive view {}:{} for {}: {}",
                    view,
                    interactive_control,
                    chat_id,
                    e,
                )
            return

        reply_params = None
        if self.config.reply_to_message:
            reply_to_message_id = metadata.get("message_id")
            if reply_to_message_id:
                reply_params = ReplyParameters(
                    message_id=reply_to_message_id, allow_sending_without_reply=True
                )

        # Send media files
        for media_path in msg.media or []:
            try:
                media_type = self._get_media_type(media_path)
                sender = {
                    "photo": self._app.bot.send_photo,
                    "voice": self._app.bot.send_voice,
                    "audio": self._app.bot.send_audio,
                }.get(media_type, self._app.bot.send_document)
                param = (
                    "photo"
                    if media_type == "photo"
                    else media_type
                    if media_type in ("voice", "audio")
                    else "document"
                )
                with open(media_path, "rb") as f:
                    await sender(chat_id=chat_id, **{param: f}, reply_parameters=reply_params)
            except Exception as e:
                filename = media_path.rsplit("/", 1)[-1]
                logger.error("Failed to send media {}: {}", media_path, e)
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    reply_parameters=reply_params,
                )

        # Send text content
        if msg.content and msg.content != "[empty message]":
            for chunk in _split_message(msg.content):
                try:
                    html = _markdown_to_telegram_html(chunk)
                    await self._app.bot.send_message(
                        chat_id=chat_id, text=html, parse_mode="HTML", reply_parameters=reply_params
                    )
                except Exception as e:
                    logger.warning("HTML parse failed, falling back to plain text: {}", e)
                    try:
                        await self._app.bot.send_message(
                            chat_id=chat_id, text=chunk, reply_parameters=reply_params
                        )
                    except Exception as e2:
                        logger.error("Error sending Telegram message: {}", e2)

    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        await self._handle_message(
            sender_id=self._sender_id(update.effective_user),
            chat_id=str(update.message.chat_id),
            content=update.message.text,
        )

    @staticmethod
    def _build_interactive_keyboard(
        view: str,
        page: int,
        total_pages: int,
        buttons: list[list[dict[str, str]]] | None = None,
    ) -> InlineKeyboardMarkup | None:
        """Build a generic pager keyboard for interactive views."""
        if buttons:
            return InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(button["text"], callback_data=button["callback_data"])
                        for button in row
                    ]
                    for row in buttons
                ]
            )
        if total_pages <= 1:
            return None
        prev_page = total_pages if page <= 1 else page - 1
        next_page = 1 if page >= total_pages else page + 1
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Prev", callback_data=f"view:{view}:{prev_page}"),
                    InlineKeyboardButton("Refresh", callback_data=f"view:{view}:{page}"),
                    InlineKeyboardButton("Next", callback_data=f"view:{view}:{next_page}"),
                ]
            ]
        )

    @staticmethod
    def _parse_interactive_callback(data: str) -> tuple[str, int] | None:
        """Parse callback payloads like `view:status:2`."""
        parts = data.split(":", 2)
        if len(parts) != 3 or parts[0] != "view":
            return None
        try:
            page = max(int(parts[2]), 1)
        except ValueError:
            return None
        return parts[1], page

    @staticmethod
    def _parse_model_callback(data: str) -> str | None:
        """Parse callback payloads like `model:set:gpt-4o-mini`."""
        prefix = "model:set:"
        if not data.startswith(prefix):
            return None
        return data[len(prefix) :] or None

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Telegram inline button callbacks."""
        query = update.callback_query
        if not query:
            return
        if not update.effective_user:
            await query.answer()
            return

        data = query.data or ""
        parsed = self._parse_interactive_callback(data)
        model_name = self._parse_model_callback(data)
        if parsed is None and model_name is None:
            await query.answer()
            return
        if not query.message:
            await query.answer()
            return

        if model_name is not None:
            await self._handle_message(
                sender_id=self._sender_id(update.effective_user),
                chat_id=str(query.message.chat_id),
                content=f"/model {model_name}",
                metadata={
                    "_interactive_message_id": query.message.message_id,
                    "_interactive_callback_query_id": query.id,
                },
            )
            return

        view, page = parsed

        await self._handle_message(
            sender_id=self._sender_id(update.effective_user),
            chat_id=str(query.message.chat_id),
            content=f"/{view} {page}",
            metadata={
                "_interactive_message_id": query.message.message_id,
                "_interactive_callback_query_id": query.id,
            },
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Handle media files
        media_file = None
        media_type = None

        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"

        # Download media if present
        if media_file and self._app:
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(media_type, getattr(media_file, "mime_type", None))

                # Save to workspace/media/
                from pathlib import Path

                media_dir = Path.home() / ".rvoone" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)

                file_path = media_dir / f"{media_file.file_id[:16]}{ext}"
                await file.download_to_drive(str(file_path))

                media_paths.append(str(file_path))

                # Handle voice transcription
                if media_type == "voice" or media_type == "audio":
                    from rvoone.providers.transcription import GroqTranscriptionProvider

                    transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                    transcription = await transcriber.transcribe(file_path)
                    if transcription:
                        logger.info("Transcribed {}: {}...", media_type, transcription[:50])
                        content_parts.append(f"[transcription: {transcription}]")
                    else:
                        content_parts.append(f"[{media_type}: {file_path}]")
                else:
                    content_parts.append(f"[{media_type}: {file_path}]")

                logger.debug("Downloaded {} to {}", media_type, file_path)
            except Exception as e:
                logger.error("Failed to download media: {}", e)
                content_parts.append(f"[{media_type}: download failed]")

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug("Telegram message from {}: {}...", sender_id, content[:50])

        str_chat_id = str(chat_id)

        # Telegram media groups: buffer briefly, forward as one aggregated turn.
        if media_group_id := getattr(message, "media_group_id", None):
            key = f"{str_chat_id}:{media_group_id}"
            if key not in self._media_group_buffers:
                self._media_group_buffers[key] = {
                    "sender_id": sender_id,
                    "chat_id": str_chat_id,
                    "contents": [],
                    "media": [],
                    "metadata": {
                        "message_id": message.message_id,
                        "message_thread_id": getattr(message, "message_thread_id", None),
                        "user_id": user.id,
                        "username": user.username,
                        "first_name": user.first_name,
                        "is_group": message.chat.type != "private",
                    },
                }
            buf = self._media_group_buffers[key]
            if content and content != "[empty message]":
                buf["contents"].append(content)
            buf["media"].extend(media_paths)
            if key not in self._media_group_tasks:
                self._media_group_tasks[key] = asyncio.create_task(self._flush_media_group(key))
            return

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.message_id,
                "message_thread_id": getattr(message, "message_thread_id", None),
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private",
            },
        )

    async def _flush_media_group(self, key: str) -> None:
        """Wait briefly, then forward buffered media-group as one turn."""
        try:
            await asyncio.sleep(0.6)
            if not (buf := self._media_group_buffers.pop(key, None)):
                return
            content = "\n".join(buf["contents"]) or "[empty message]"
            await self._handle_message(
                sender_id=buf["sender_id"],
                chat_id=buf["chat_id"],
                content=content,
                media=list(dict.fromkeys(buf["media"])),
                metadata=buf["metadata"],
            )
        finally:
            self._media_group_tasks.pop(key, None)

    async def _queue_reply_draft_update(
        self,
        *,
        chat_id: int,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
    ) -> None:
        """Coalesce draft updates so one Telegram draft is sent serially."""
        if self._drafts_globally_disabled:
            return

        key = f"{chat_id}:{draft_id}"
        state = self._draft_updates.get(key)
        if state is None:
            state = _DraftUpdateState(
                chat_id=chat_id,
                draft_id=draft_id,
                message_thread_id=message_thread_id,
            )
            self._draft_updates[key] = state
            state.task = asyncio.create_task(self._draft_update_loop(key))

        state.latest_text = text[:4000]
        state.message_thread_id = message_thread_id
        state.wakeup.set()

    async def _draft_update_loop(self, key: str) -> None:
        """Deliver the latest text for one draft with retry-aware throttling."""
        try:
            while self._app and not self._drafts_globally_disabled:
                state = self._draft_updates.get(key)
                if state is None:
                    return

                try:
                    await asyncio.wait_for(state.wakeup.wait(), timeout=self._DRAFT_IDLE_TIMEOUT_S)
                    state.wakeup.clear()
                except asyncio.TimeoutError:
                    if state.latest_text == state.sent_text:
                        return
                    continue

                if not state.latest_text or state.latest_text == state.sent_text:
                    continue

                retry_delay_s = state.retry_after_s
                state.retry_after_s = 0.0
                delay_s = max(
                    retry_delay_s, self._draft_send_interval_s if state.sent_text else 0.0
                )
                if delay_s > 0:
                    await asyncio.sleep(delay_s)

                if state.latest_text == state.sent_text:
                    continue

                sent = await self._send_reply_draft(
                    chat_id=state.chat_id,
                    draft_id=state.draft_id,
                    text=state.latest_text,
                    message_thread_id=state.message_thread_id,
                )
                if sent:
                    state.sent_text = state.latest_text
        except asyncio.CancelledError:
            pass
        finally:
            self._draft_updates.pop(key, None)

    async def _complete_reply_draft(self, *, chat_id: int, draft_id: int) -> None:
        """Stop tracking one draft and best-effort clear it before the final reply."""
        state = self._draft_updates.pop(f"{chat_id}:{draft_id}", None)
        if state and state.task and not state.task.done():
            state.task.cancel()
        if state and state.sent_text:
            await self._clear_reply_draft(
                chat_id=chat_id,
                draft_id=draft_id,
                message_thread_id=state.message_thread_id,
            )

    async def _clear_reply_draft(
        self,
        *,
        chat_id: int,
        draft_id: int,
        message_thread_id: int | None = None,
    ) -> None:
        """Replace the visible draft with an invisible placeholder to shrink duplicate windows."""
        if not self._app or self._drafts_globally_disabled:
            return
        try:
            await self._app.bot.send_message_draft(
                chat_id=chat_id,
                draft_id=draft_id,
                text=self._DRAFT_CLEAR_TEXT,
                message_thread_id=message_thread_id,
            )
        except RetryAfter as e:
            logger.debug(
                "Telegram reply draft clear rate-limited for chat_id={} draft_id={}: {}",
                chat_id,
                draft_id,
                e,
            )
        except Exception as e:
            logger.debug("Failed to clear Telegram reply draft {}:{}: {}", chat_id, draft_id, e)

    async def _send_reply_draft(
        self,
        *,
        chat_id: int,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
    ) -> bool:
        """Send one Telegram draft update when draft streaming is available."""
        if not self._app or self._drafts_globally_disabled:
            return False

        sender = getattr(self._app.bot, "send_message_draft", None)
        if not callable(sender):
            self._drafts_globally_disabled = True
            logger.warning(
                "Telegram reply drafts unsupported by installed python-telegram-bot; disabling"
            )
            return False

        try:
            await sender(
                chat_id=chat_id,
                draft_id=draft_id,
                text=text[:4000],
                message_thread_id=message_thread_id,
            )
            return True
        except RetryAfter as e:
            retry_after = (
                e.retry_after.total_seconds()
                if hasattr(e.retry_after, "total_seconds")
                else float(e.retry_after)
            )
            state = self._draft_updates.get(f"{chat_id}:{draft_id}")
            if state is not None:
                state.retry_after_s = max(float(retry_after), 1.0)
                state.wakeup.set()
            logger.warning(
                "Telegram reply drafts rate-limited; retrying in {:.1f}s for chat_id={} draft_id={}",
                float(retry_after),
                chat_id,
                draft_id,
            )
            return False
        except Exception as e:
            self._drafts_globally_disabled = True
            logger.warning("Telegram reply drafts unavailable, disabling draft streaming: {}", e)
            return False

    def _renew_typing(self, chat_id: str, ttl: int = 30) -> None:
        """Extend or start the typing indicator lease for a chat."""
        loop = asyncio.get_running_loop()
        self._typing_deadlines[chat_id] = loop.time() + max(ttl, 1)
        task = self._typing_tasks.get(chat_id)
        if task is None or task.done():
            self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        self._typing_deadlines.pop(chat_id, None)
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _typing_loop(self, chat_id: str) -> None:
        """Send 'typing' while the current lease remains valid."""
        try:
            while self._app:
                deadline = self._typing_deadlines.get(chat_id)
                if deadline is None:
                    break
                now = asyncio.get_running_loop().time()
                if now >= deadline:
                    break
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(min(4, max(deadline - now, 0.1)))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Typing indicator stopped for {}: {}", chat_id, e)
        finally:
            self._typing_deadlines.pop(chat_id, None)
            self._typing_tasks.pop(chat_id, None)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        logger.error("Telegram error: {}", context.error)

    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "audio/ogg": ".ogg",
                "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")
