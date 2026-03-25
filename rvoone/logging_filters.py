"""Stdlib logging filters used by rvoone runtime commands."""

from __future__ import annotations

import logging

from loguru import logger

_FILTER_FLAG = "_rvoone_filters_installed"
_HANDLER_FLAG = "_rvoone_loguru_bridge"


class _ForwardStdlibToLoguru(logging.Handler):
    """Forward stdlib log records through rvoone's loguru sink."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            logger.bind(stdlib_logger=record.name).opt(
                exception=record.exc_info,
                depth=6,
            ).log(record.levelname, message)
        except Exception:
            self.handleError(record)


def install_stdlib_filters() -> None:
    """Install one-time stdlib logging filters for noisy third-party warnings."""
    root = logging.getLogger()
    if getattr(root, _FILTER_FLAG, False):
        return
    setattr(root, _FILTER_FLAG, True)


def configure_stdlib_logging(enabled: bool, debug: bool = False) -> None:
    """Route stdlib logs through rvoone logging when enabled, else suppress noisy INFO logs."""
    root = logging.getLogger()
    install_stdlib_filters()

    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            root.removeHandler(handler)

    if enabled:
        handler = _ForwardStdlibToLoguru()
        setattr(handler, _HANDLER_FLAG, True)
        root.addHandler(handler)
        root.setLevel(logging.DEBUG if debug else logging.INFO)
        for name in ("httpx", "httpcore"):
            lib_logger = logging.getLogger(name)
            lib_logger.handlers.clear()
            lib_logger.propagate = True
            lib_logger.setLevel(logging.INFO if debug else logging.WARNING)
        return

    for name in ("httpx", "httpcore"):
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = False
        lib_logger.setLevel(logging.WARNING)
