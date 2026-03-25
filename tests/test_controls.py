"""Tests for typed channel control metadata builders."""

from rvoone.application.controls import InteractiveViewControl, ToolStatusControl, TypingControl


def test_typing_control_metadata() -> None:
    assert TypingControl(action="renew", ttl=30).to_metadata() == {
        "_typing_control": "renew",
        "_typing_ttl": 30,
    }


def test_tool_status_control_metadata() -> None:
    assert ToolStatusControl(
        action="create", status_key="k1", text="hello", pin=True
    ).to_metadata() == {
        "_status_control": "create",
        "_status_key": "k1",
        "_status_text": "hello",
        "_status_pin": True,
    }


def test_interactive_view_control_metadata() -> None:
    assert InteractiveViewControl(
        view="status",
        create=False,
        message_id=123,
        callback_query_id="cbq",
        page=2,
        total_pages=3,
    ).to_metadata() == {
        "_interactive_control": "update",
        "_interactive_view": "status",
        "_interactive_message_id": 123,
        "_interactive_callback_query_id": "cbq",
        "_interactive_page": 2,
        "_interactive_pages": 3,
    }


def test_interactive_view_control_omits_ids_on_create() -> None:
    assert InteractiveViewControl(
        view="status", create=True, page=1, total_pages=3
    ).to_metadata() == {
        "_interactive_control": "create",
        "_interactive_view": "status",
        "_interactive_page": 1,
        "_interactive_pages": 3,
    }
