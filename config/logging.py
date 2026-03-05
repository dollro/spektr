"""Structured logging configuration for Spektr.

Provides JSON and text formatters with support for extra fields
like duration_ms, file_name, tool, query, and result_count.
"""

from __future__ import annotations

import json
import logging
import os


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, object] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(
                record.exc_info,
            )
        for key in (
            "duration_ms",
            "file_name",
            "mime_type",
            "page_count",
            "entity_count",
            "tool",
            "query",
            "result_count",
        ):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data)


_TEXT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_initialized = False


def setup_logging() -> None:
    """Configure root logger from settings or environment."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    try:
        from config.settings import settings

        level = settings.log_level.upper()
        fmt = settings.log_format.lower()
    except Exception:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        fmt = os.environ.get("LOG_FORMAT", "json").lower()

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root.handlers.clear()
    root.addHandler(handler)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
