"""Structured logging configuration for Spektr.

Provides JSON and text formatters with support for extra fields
like duration_ms, file_name, tool, query, and result_count.
"""

from __future__ import annotations

import json
import logging
import os


def _current_trace_ids() -> tuple[str, str] | None:
    """Return (trace_id, span_id) as hex strings from OTEL context, if any."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return (format(ctx.trace_id, "032x"), format(ctx.span_id, "016x"))
    except ImportError:
        pass
    except Exception:
        pass
    return None


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
        # Correlate with distributed traces when OTEL context is active
        trace_ids = _current_trace_ids()
        if trace_ids is not None:
            log_data["trace_id"], log_data["span_id"] = trace_ids
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
