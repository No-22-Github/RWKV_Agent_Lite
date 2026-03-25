"""Tests for session runtime state storage."""

import asyncio

from rvoone.application.state import SessionStateStore


def test_state_store_updates_and_clears_runtime() -> None:
    store = SessionStateStore()

    store.update_runtime("s1", phase="processing", current_tool="exec")
    assert store.runtime["s1"]["phase"] == "processing"
    assert store.runtime["s1"]["current_tool"] == "exec"

    store.clear_runtime("s1")
    assert store.runtime["s1"] == {
        "phase": "idle",
        "current_tool": None,
        "current_tool_args": None,
    }


def test_state_store_reuses_session_lock() -> None:
    store = SessionStateStore()
    lock1 = store.session_lock("s1")
    lock2 = store.session_lock("s1")
    lock3 = store.session_lock("s2")

    assert lock1 is lock2
    assert isinstance(lock3, asyncio.Lock)
    assert lock1 is not lock3
