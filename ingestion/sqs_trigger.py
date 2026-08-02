"""SQS-as-trigger daemon for the S3 document source.

CocoIndex v1 dropped v0's ``AmazonS3(sqs_queue_url=...)`` push trigger: the v1
``amazon_s3`` connector is scan-only, with no ``live=`` mode. Incremental
reconciliation itself is intact — a catch-up run still reprocesses only changed
objects and still deletes the Qdrant points of removed ones. What was lost is
only the *discovery latency*.

This module restores it without duplicating any stored bytes and without adding
a broker. SQS is used purely as a **trigger**: on an event we debounce, then run
one ordinary ``app.update()``. Nothing is downloaded except objects that
actually changed, and those are read straight from S3 by the pipeline.

Three triggers, all calling the same update:

======================  ====================================================
SQS event (debounced)   the normal path — seconds of latency
interval timer          safety net for missed or expired events (default 24h)
daemon startup          recovers changes made while the daemon was down; SQS
                        retention caps out at 14 days, so older events are not
                        recoverable by replay at all
======================  ====================================================

A LIST sweep is metadata-only: ~$0.005 per 1,000 requests at 1,000 keys each,
so scanning a 10k-object bucket costs about $0.00005 per run.

Two deliberate safety properties:

* messages are deleted **only after** an update that reported zero errors, so a
  crash or a failed file replays the event instead of dropping it;
* ``app.update()`` never raises for per-file failures, so the error count is
  read explicitly from ``stats().total.num_errors``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

_MAX_MESSAGES_PER_RECEIVE = 10
_LONG_POLL_SECONDS = 20
_DELETE_BATCH_SIZE = 10
_MAX_DRAIN_MESSAGES = 1000


async def _receive(sqs: Any, queue_url: str, *, wait: int) -> list[dict[str, Any]]:
    """Long-poll the queue. Returns [] on error (the interval sweep covers us)."""
    try:
        resp = await sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=_MAX_MESSAGES_PER_RECEIVE,
            WaitTimeSeconds=wait,
        )
    except Exception:
        logger.exception("SQS receive failed; falling back to the interval sweep")
        return []
    messages: list[dict[str, Any]] = resp.get("Messages", [])
    return messages


async def _drain(sqs: Any, queue_url: str) -> list[dict[str, Any]]:
    """Pull whatever else is immediately available after the debounce window.

    Bounded: on a continuously-fed queue an unbounded drain would never hand
    control back to the update. Anything left over is picked up by the next
    poll, and a single catch-up scan covers it anyway.
    """
    drained: list[dict[str, Any]] = []
    while len(drained) < _MAX_DRAIN_MESSAGES:
        batch = await _receive(sqs, queue_url, wait=0)
        if not batch:
            break
        drained.extend(batch)
    return drained


async def _delete_messages(
    sqs: Any, queue_url: str, messages: list[dict[str, Any]]
) -> None:
    """Delete handled messages in batches of ten (the SQS API limit)."""
    for start in range(0, len(messages), _DELETE_BATCH_SIZE):
        chunk = messages[start : start + _DELETE_BATCH_SIZE]
        entries = [
            {"Id": str(i), "ReceiptHandle": m["ReceiptHandle"]}
            for i, m in enumerate(chunk)
            if "ReceiptHandle" in m
        ]
        if not entries:
            continue
        try:
            await sqs.delete_message_batch(QueueUrl=queue_url, Entries=entries)
        except Exception:
            logger.exception(
                "Failed to delete %d SQS message(s); they will be redelivered",
                len(entries),
            )


def _open_sqs_client() -> Any:
    """Return an async context manager yielding an aiobotocore SQS client.

    Factored out so the trigger loop can be tested without botocore.
    """
    from aiobotocore.session import get_session  # type: ignore[import-untyped]

    return get_session().create_client(
        "sqs",
        region_name=settings.aws_region or None,
        endpoint_url=settings.aws_endpoint_url or None,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


async def run_sqs_triggered(app: Any, *, full_reprocess: bool = False) -> int:
    """Run catch-up updates driven by SQS events plus a periodic sweep.

    Blocks until cancelled. Returns the error count of the last update, so an
    interrupted daemon still surfaces a non-zero exit code after a bad run.
    """
    from ingestion.runner import run_update

    interval = settings.s3_full_scan_interval_hours * 3600.0
    queue_url = settings.s3_sqs_queue_url

    # Startup sweep: recovers anything that changed while we were down.
    num_errors = await run_update(app, full_reprocess=full_reprocess)
    loop = asyncio.get_running_loop()
    last_full = loop.time()

    if not queue_url:
        logger.warning(
            "S3_SQS_QUEUE_URL is not set — falling back to interval-only sweeps "
            "every %.1fh. Change latency equals the interval.",
            settings.s3_full_scan_interval_hours,
        )
        while True:
            await asyncio.sleep(interval)
            num_errors = await run_update(app)

    async with _open_sqs_client() as sqs:
        logger.info(
            "SQS trigger active on %s (debounce %.1fs, sweep every %.1fh)",
            queue_url,
            settings.s3_sqs_debounce_seconds,
            settings.s3_full_scan_interval_hours,
        )
        while True:
            messages = await _receive(sqs, queue_url, wait=_LONG_POLL_SECONDS)
            due = (loop.time() - last_full) >= interval
            if not messages and not due:
                continue

            if messages:
                # Coalesce a burst of events into a single update.
                await asyncio.sleep(settings.s3_sqs_debounce_seconds)
                messages.extend(await _drain(sqs, queue_url))
                logger.info("SQS trigger: %d event(s) coalesced", len(messages))
            else:
                logger.info("Interval sweep due; running catch-up scan")

            num_errors = await run_update(app)
            last_full = loop.time()

            if messages:
                if num_errors:
                    logger.error(
                        "Update reported %d errored file(s); keeping %d SQS "
                        "message(s) for redelivery",
                        num_errors,
                        len(messages),
                    )
                else:
                    await _delete_messages(sqs, queue_url, messages)

    return num_errors  # pragma: no cover - the loop only exits via cancellation
