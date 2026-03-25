"""Typed builders for channel control metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TypingControl:
    """Typing control message metadata."""

    action: str
    ttl: int | None = None

    def to_metadata(self) -> dict[str, int | str]:
        metadata: dict[str, int | str] = {"_typing_control": self.action}
        if self.ttl is not None:
            metadata["_typing_ttl"] = self.ttl
        return metadata


@dataclass(slots=True)
class ToolStatusControl:
    """Editable tool status metadata."""

    action: str
    status_key: str
    text: str | None = None
    pin: bool | None = None
    delete_delay_s: float | None = None

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "_status_control": self.action,
            "_status_key": self.status_key,
        }
        if self.text is not None:
            metadata["_status_text"] = self.text
        if self.pin is not None:
            metadata["_status_pin"] = self.pin
        if self.delete_delay_s is not None:
            metadata["_status_delete_delay_s"] = self.delete_delay_s
        return metadata


@dataclass(slots=True)
class ReplyDraftControl:
    """Telegram reply-draft metadata."""

    draft_id: int
    text: str
    message_thread_id: int | None = None

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "_draft_control": "update",
            "_draft_id": self.draft_id,
            "_draft_text": self.text,
        }
        if self.message_thread_id is not None:
            metadata["_draft_message_thread_id"] = self.message_thread_id
        return metadata


@dataclass(slots=True)
class InteractiveViewControl:
    """Interactive Telegram view metadata."""

    view: str
    create: bool
    message_id: int | None = None
    callback_query_id: str | None = None
    page: int | None = None
    total_pages: int | None = None
    buttons: list[list[dict[str, str]]] | None = None

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "_interactive_control": "create" if self.create else "update",
            "_interactive_view": self.view,
        }
        if self.message_id is not None:
            metadata["_interactive_message_id"] = self.message_id
        if self.callback_query_id is not None:
            metadata["_interactive_callback_query_id"] = self.callback_query_id
        if self.page is not None:
            metadata["_interactive_page"] = self.page
        if self.total_pages is not None:
            metadata["_interactive_pages"] = self.total_pages
        if self.buttons is not None:
            metadata["_interactive_buttons"] = self.buttons
        return metadata
