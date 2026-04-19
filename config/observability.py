"""Distributed tracing via Logfire / OpenTelemetry.

Single call to ``setup_observability()`` from any process entrypoint
wires up pydantic-ai, FastAPI, and httpx instrumentation so end-to-end
traces (ingest → embed → Qdrant → MCP tool → LLM) are stitched by
trace_id. The trace_id is also injected into our JSON log records so
logs and traces can be correlated without a paid backend.

Local-only mode (default): spans are emitted to stdout/OTLP but NOT
sent to Logfire Cloud. Set ``LOGFIRE_TOKEN`` and
``OBSERVABILITY_LOCAL_ONLY=false`` to ship to the hosted service.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_initialized = False


def setup_observability() -> None:
    """Configure Logfire + OTEL instrumentation. Idempotent."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    try:
        import logfire
    except ImportError:
        logger.debug("logfire not installed; skipping observability setup")
        _initialized = True
        return

    from config.settings import settings

    send_to_logfire = bool(
        settings.logfire_token and not settings.observability_local_only,
    )
    logfire.configure(
        service_name=settings.service_name,
        token=settings.logfire_token or None,
        send_to_logfire=send_to_logfire,
        console=False,  # keep stdout clean — our JSONFormatter owns it
    )

    # Instrument only what's actually installed
    try:
        logfire.instrument_pydantic_ai()
    except Exception:
        logger.debug("pydantic-ai instrumentation not available", exc_info=True)

    try:
        logfire.instrument_httpx()
    except Exception:
        logger.debug("httpx instrumentation not available", exc_info=True)

    _initialized = True
    logger.info(
        "Observability ready (service=%s, local_only=%s)",
        settings.service_name,
        settings.observability_local_only,
    )


def instrument_fastapi(app: object) -> None:
    """Add FastAPI instrumentation. Call after ``setup_observability()``.

    Kept separate because it takes the app instance and each entrypoint
    (live_ingest, agent.api) creates its own.
    """
    try:
        import logfire

        logfire.instrument_fastapi(app)  # type: ignore[arg-type]
    except Exception:
        logger.debug("FastAPI instrumentation failed", exc_info=True)
