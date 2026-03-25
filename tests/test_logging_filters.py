"""Tests for stdlib logging filters and bridging."""

from __future__ import annotations

import logging

from loguru import logger

from rvoone.logging_filters import configure_stdlib_logging, install_stdlib_filters


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="root",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_install_stdlib_filters_is_noop_without_provider_filters() -> None:
    root = logging.getLogger()
    old_filters = list(root.filters)
    try:
        for f in list(root.filters):
            root.removeFilter(f)
        if hasattr(root, "_rvoone_filters_installed"):
            delattr(root, "_rvoone_filters_installed")

        install_stdlib_filters()
        assert root.filters == []
    finally:
        for f in list(root.filters):
            root.removeFilter(f)
        for f in old_filters:
            root.addFilter(f)
        if hasattr(root, "_rvoone_filters_installed"):
            delattr(root, "_rvoone_filters_installed")


def test_install_stdlib_filters_keeps_other_warnings() -> None:
    root = logging.getLogger()
    old_filters = list(root.filters)
    try:
        for f in list(root.filters):
            root.removeFilter(f)
        if hasattr(root, "_rvoone_filters_installed"):
            delattr(root, "_rvoone_filters_installed")

        install_stdlib_filters()
        assert root.filters == []
    finally:
        for f in list(root.filters):
            root.removeFilter(f)
        for f in old_filters:
            root.addFilter(f)
        if hasattr(root, "_rvoone_filters_installed"):
            delattr(root, "_rvoone_filters_installed")


def test_configure_stdlib_logging_suppresses_httpx_info_when_disabled(caplog) -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    httpx_logger = logging.getLogger("httpx")
    old_httpx_handlers = list(httpx_logger.handlers)
    old_httpx_propagate = httpx_logger.propagate
    old_httpx_level = httpx_logger.level
    try:
        configure_stdlib_logging(enabled=False)
        with caplog.at_level(logging.INFO, logger="httpx"):
            httpx_logger.info("hidden request log")
        assert "hidden request log" not in caplog.text
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        httpx_logger.handlers[:] = old_httpx_handlers
        httpx_logger.propagate = old_httpx_propagate
        httpx_logger.setLevel(old_httpx_level)


def test_configure_stdlib_logging_forwards_to_loguru_when_enabled() -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    sample_logger = logging.getLogger("sample.runtime")
    old_sample_handlers = list(sample_logger.handlers)
    old_sample_propagate = sample_logger.propagate
    old_sample_level = sample_logger.level
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        configure_stdlib_logging(enabled=True)
        sample_logger.info("forwarded request log")
        assert any(message.strip() == "forwarded request log" for message in messages)
    finally:
        logger.remove(sink_id)
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        sample_logger.handlers[:] = old_sample_handlers
        sample_logger.propagate = old_sample_propagate
        sample_logger.setLevel(old_sample_level)


def test_configure_stdlib_logging_hides_httpx_info_unless_debug() -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    httpx_logger = logging.getLogger("httpx")
    old_httpx_handlers = list(httpx_logger.handlers)
    old_httpx_propagate = httpx_logger.propagate
    old_httpx_level = httpx_logger.level
    try:
        configure_stdlib_logging(enabled=True, debug=False)
        assert not httpx_logger.isEnabledFor(logging.INFO)

        configure_stdlib_logging(enabled=True, debug=True)
        assert httpx_logger.isEnabledFor(logging.INFO)
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        httpx_logger.handlers[:] = old_httpx_handlers
        httpx_logger.propagate = old_httpx_propagate
        httpx_logger.setLevel(old_httpx_level)
