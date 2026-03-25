"""Tests for channel presentation helpers."""

from __future__ import annotations

from types import SimpleNamespace

from rvoone.application.presenter import AgentPresenter


def test_apply_interactive_view_preserves_callback_metadata() -> None:
    presenter = AgentPresenter(SimpleNamespace())
    metadata = presenter.apply_interactive_view(
        {
            "_interactive_message_id": 456,
            "_interactive_callback_query_id": "cbq-1",
        },
        view="status",
        page=2,
        total_pages=3,
    )

    assert metadata == {
        "_interactive_control": "update",
        "_interactive_view": "status",
        "_interactive_page": 2,
        "_interactive_pages": 3,
        "_interactive_message_id": 456,
        "_interactive_callback_query_id": "cbq-1",
    }
