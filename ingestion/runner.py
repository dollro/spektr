"""Process entrypoint for the bulk ingestion pipeline (Path A).

Provisions infrastructure, then runs the CocoIndex app in one of three modes:

===========================  ================================================
one-shot                     ``python -m ingestion.pipeline``
live, local/sharepoint       ``--live`` → ``app.update(live=True)``; the
                             ``localfs`` watcher is a real push trigger
live, s3                     ``--live`` → the SQS-triggered loop in
                             ``ingestion/sqs_trigger.py``, because CocoIndex v1
                             has no S3 push trigger
===========================  ================================================

``app.update()`` does **not** raise when individual files fail — component
failures are logged and swallowed by the framework — so every mode reports
``stats().total.num_errors`` explicitly and the exit code reflects it.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from config.logging import get_logger
from config.settings import settings
from ingestion._utils import run_async
from ingestion.app import build_app, describe_watched_source, use_s3_source
from ingestion.graph_engine import close_graph_engine
from ingestion.neo4j_setup import create_neo4j_schema, get_driver
from ingestion.qdrant_setup import ensure_collections
from ingestion.qdrant_target import create_qdrant_client

logger = get_logger(__name__)


def _provision() -> None:
    """Create Qdrant collections and the Neo4j schema (both idempotent).

    ``ensure_collections`` stays the sole authority over the Qdrant collections:
    the CocoIndex targets are mounted with ``managed_by=USER`` precisely so the
    engine never creates, replaces or drops them.
    """
    client = create_qdrant_client()
    try:
        ensure_collections(client)
    finally:
        client.close()

    async def _setup_neo4j() -> None:
        driver = get_driver()
        try:
            await create_neo4j_schema(driver)
        finally:
            await driver.close()

    run_async(_setup_neo4j())


def _report(handle: Any) -> int:
    """Log an update handle's stats and return its error count."""
    stats = handle.stats()
    if stats is None:
        return 0
    total = stats.total
    logger.info(
        "Update finished: %d added, %d reprocessed, %d unchanged, %d deleted, %d errors",
        total.num_adds,
        total.num_reprocesses,
        total.num_unchanged,
        total.num_deletes,
        total.num_errors,
        extra={
            "num_adds": total.num_adds,
            "num_reprocesses": total.num_reprocesses,
            "num_unchanged": total.num_unchanged,
            "num_deletes": total.num_deletes,
            "num_errors": total.num_errors,
        },
    )
    return int(total.num_errors)


async def run_update(app: Any, *, full_reprocess: bool = False) -> int:
    """Run one catch-up update and return the number of errored components."""
    handle = app.update(full_reprocess=full_reprocess)
    await handle.result()
    return _report(handle)


async def _run(live: bool, full_reprocess: bool) -> int:
    app = build_app()

    if not live:
        return await run_update(app, full_reprocess=full_reprocess)

    logger.info(
        "Entering live mode — watching %s for changes. Press Ctrl-C to stop.",
        describe_watched_source(),
    )
    if settings.document_source == "sharepoint":
        logger.info(
            "DOCUMENT_SOURCE=sharepoint: ensure `task sharepoint-sync` "
            "is also running so the local mirror gets populated."
        )

    if use_s3_source():
        # CocoIndex v1's amazon_s3 source has no live mode; drive catch-up runs
        # from SQS events plus a periodic sweep instead.
        from ingestion.sqs_trigger import run_sqs_triggered

        return await run_sqs_triggered(app, full_reprocess=full_reprocess)

    # Blocks until interrupted: the localfs source keeps streaming changes.
    handle = app.update(live=True, full_reprocess=full_reprocess)
    await handle.result()
    return _report(handle)


def run_pipeline(live: bool = False, full_reprocess: bool = False) -> int:
    """Initialize and run the ingestion pipeline. Returns a process exit code."""
    from config.observability import setup_observability

    setup_observability()
    t0 = time.monotonic()
    logger.info("Pipeline starting (live=%s, full_reprocess=%s)", live, full_reprocess)

    _provision()

    try:
        num_errors = run_async(_run(live, full_reprocess))
    finally:
        if settings.graph_enabled:
            run_async(close_graph_engine())

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "Pipeline completed in %dms (%d errored files)",
        duration_ms,
        num_errors,
        extra={"duration_ms": duration_ms, "num_errors": num_errors},
    )
    return 1 if num_errors else 0


def main() -> None:
    from config.logging import setup_logging

    setup_logging()
    argv = sys.argv[1:]
    sys.exit(
        run_pipeline(
            live="--live" in argv,
            full_reprocess="--full-reprocess" in argv,
        )
    )


if __name__ == "__main__":
    main()
