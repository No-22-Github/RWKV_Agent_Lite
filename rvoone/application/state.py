"""Session-scoped runtime state storage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionStateStore:
    """Own volatile runtime state for active sessions."""

    active_tasks: dict[str, list[asyncio.Task]] = field(default_factory=dict)
    processing_tasks: set[str] = field(default_factory=set)
    runtime: dict[str, dict[str, Any]] = field(default_factory=dict)
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    enabled_tool_categories: dict[str, set[str]] = field(default_factory=dict)

    def update_runtime(self, session_key: str, **updates: Any) -> None:
        state = self.runtime.setdefault(session_key, {})
        state.update(updates)

    def clear_runtime(self, session_key: str) -> None:
        state = self.runtime.setdefault(session_key, {})
        state["phase"] = "idle"
        state["current_tool"] = None
        state["current_tool_args"] = None

    def ensure_enabled_tool_categories(
        self,
        session_key: str,
        *,
        defaults: set[str],
    ) -> set[str]:
        current = self.enabled_tool_categories.get(session_key)
        if current is None:
            current = set(defaults)
            self.enabled_tool_categories[session_key] = current
        return set(current)

    def enable_tool_categories(self, session_key: str, categories: set[str]) -> set[str]:
        current = self.enabled_tool_categories.setdefault(session_key, set())
        current.update(categories)
        return set(current)

    def reset_enabled_tool_categories(self, session_key: str, *, defaults: set[str]) -> set[str]:
        next_categories = set(defaults)
        self.enabled_tool_categories[session_key] = next_categories
        return set(next_categories)

    def session_lock(self, session_key: str) -> asyncio.Lock:
        lock = self.locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[session_key] = lock
        return lock
