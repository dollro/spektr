"""Per-file processing for the bulk ingestion path (CocoIndex v1, Path A).

This module owns everything that happens to *one* source file: MIME
classification, page extraction, chunking, embedding, Qdrant point declaration
and graph ingestion. The CocoIndex app that mounts it lives in
``ingestion/app.py``; the process entrypoint lives in ``ingestion/runner.py``.

Two things carried over from v0 unchanged, deliberately:

* **The poison-pill contract.** Under v1 a raising function writes no memo
  entry, so the file is re-processed on the next run — exactly the v0 behaviour
  where a raise left the tracking row unwritten. Swallowing after
  ``PIPELINE_MAX_RETRIES`` writes the memo entry and the file is not retried.
* **Graph writes are side effects**, not declared target states, because
  Graphiti is episodic and CocoIndex reconciles declared state. Cleanup on
  source deletion is handled by ``ingestion/graph_target.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import time
from datetime import UTC, datetime
from typing import Any

from config.logging import get_logger
from config.settings import settings
from ingestion._failure_tracker import get_tracker
from ingestion.embedder import Embedder
from ingestion.file_processor import (
    TextChunk,
    docling_chunk,
    file_to_pages,
)
from ingestion.graph_engine import GraphEngine, get_graph_engine
from ingestion.page_processor import _build_page_tasks
from ingestion.schema_inducer import SchemaInducer

logger = get_logger(__name__)

# globset patterns: bare *.pdf matches a single path segment, so it would
# miss files in subdirectories. Use **/*.ext to cover any depth — required
# for the SharePoint syncer (which mirrors folder structure) and harmless
# for flat layouts.
SUPPORTED_PATTERNS = [
    "**/*.pdf",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.gif",
    "**/*.bmp",
    "**/*.webp",
    "**/*.md",
    "**/*.txt",
    "**/*.csv",
    "**/*.json",
    "**/*.xml",
    "**/*.html",
    "**/*.yaml",
    "**/*.yml",
]


def _guess_mime(filename: str) -> str:
    """Guess MIME type from filename."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


async def _ingest_to_graph(
    source_file: str,
    chunks: list[TextChunk],
    engine: GraphEngine,
) -> None:
    """Ingest chunks via the active graph engine."""
    await engine.ingest(chunks=chunks, source_key=source_file)


_schema_inducer: SchemaInducer | None = None


def _get_schema_inducer() -> SchemaInducer:
    global _schema_inducer  # noqa: PLW0603
    if _schema_inducer is None:
        _schema_inducer = SchemaInducer()
    return _schema_inducer


async def _ingest_to_graph_with_schema(
    source_file: str,
    chunks: list[TextChunk],
    engine: GraphEngine,
) -> None:
    """Ingest chunks via graph engine, optionally with induced schema."""
    schema = None

    if settings.schema_induction_enabled and settings.graph_engine == "gliner" and chunks:
        # Use first chunks' text as sample for schema induction
        sample = " ".join((c.contextualized_text or c.text) for c in chunks[:3])
        inducer = _get_schema_inducer()
        induced = await inducer.induce(sample)
        if induced.entity_types or induced.relationship_types:
            schema = inducer.merge_with_base(induced)

    await engine.ingest(chunks=chunks, source_key=source_file, schema=schema)


async def _process_pages(
    *,
    filename: str,
    pages: list[Any],
    dl_chunks: list[TextChunk] | None,
    mime: str,
    now: str,
    dense: Any,
    multivec: Any,
    embedder: Embedder,
    graph_engine: GraphEngine | None,
) -> None:
    """Run every page of one file: text pages concurrently, images serially."""
    sem = asyncio.Semaphore(2)
    all_chunks: list[TextChunk] = []

    async def _bounded(coro: Any) -> Any:
        async with sem:
            return await coro

    text_tasks: list[Any] = []
    image_tasks: list[Any] = []
    try:
        for page in pages:
            pt = _build_page_tasks(
                page,
                filename,
                mime,
                now,
                dense,
                multivec,
                embedder,
                graph_engine,
                docling_chunks=dl_chunks,
                chunk_collector=all_chunks if graph_engine else None,
            )
            text_tasks.extend(pt.text)
            image_tasks.extend(pt.image)

        # Text: concurrent (lightweight, small TPM footprint)
        if text_tasks:
            await asyncio.gather(*[_bounded(t) for t in text_tasks])

        # Bulk graph ingestion after all text pages are processed
        if graph_engine and all_chunks:
            await _ingest_to_graph_with_schema(filename, all_chunks, graph_engine)

        # Images: sequential (heavy, TPM-sensitive)
        for task in image_tasks:
            await task
    except BaseException:
        # Close unawaited coroutines to suppress RuntimeWarnings
        for coro in text_tasks + image_tasks:
            coro.close()
        raise


async def process_file_impl(
    content: bytes,
    filename: str,
    *,
    dense: Any,
    multivec: Any = None,
    embedder: Embedder,
) -> str | None:
    """Process a single file: classify, embed, declare points, write graph.

    Returns the content fingerprint when graph data was written for this file,
    else ``None``. The caller declares that fingerprint as a target state so
    deleting the source file later cleans up its episodes/entities — declaring
    is done by the caller because it requires an active component context.

    Undecorated on purpose: ``@coco.fn`` requires that same context, so the
    CocoIndex-facing wrapper in ``ingestion/app.py`` is a thin shim around this
    function and unit tests can call it directly.
    """
    t0 = time.monotonic()
    try:
        result = file_to_pages(filename, content)
        pages = result.pages
    except Exception:
        logger.exception(
            "Failed to process file: %s",
            filename,
            extra={"file_name": filename},
        )
        return None

    if not pages:
        logger.warning(
            "No pages extracted from %s, skipping.",
            filename,
            extra={"file_name": filename},
        )
        return None

    mime = _guess_mime(filename)
    now = datetime.now(tz=UTC).isoformat()
    graph_engine_inst: GraphEngine | None = None

    # Compute Docling chunks once for the whole document
    dl_chunks = docling_chunk(result.docling_document) if result.docling_document else None
    if dl_chunks:
        logger.info(
            "Using Docling HybridChunker: %d chunks for %s",
            len(dl_chunks),
            filename,
            extra={"file_name": filename, "chunk_count": len(dl_chunks)},
        )

    logger.info(
        "Processing file: %s",
        filename,
        extra={
            "file_name": filename,
            "mime_type": mime,
            "page_count": len(pages),
        },
    )

    try:
        has_text = any(p.text.strip() for p in pages)
        if has_text and settings.graph_enabled:
            graph_engine_inst = get_graph_engine()

        async with asyncio.timeout(settings.pipeline_timeout):
            await _process_pages(
                filename=filename,
                pages=pages,
                dl_chunks=dl_chunks,
                mime=mime,
                now=now,
                dense=dense,
                multivec=multivec,
                embedder=embedder,
                graph_engine=graph_engine_inst,
            )
    except (TimeoutError, Exception) as exc:
        tracker = get_tracker()
        count = tracker.record_failure(filename, error=repr(exc))
        if isinstance(exc, TimeoutError):
            logger.error(
                "File processing timed out after %ds: %s (attempt %d/%d)",
                settings.pipeline_timeout,
                filename,
                count,
                settings.pipeline_max_retries,
                extra={"file_name": filename, "fail_count": count},
            )
        else:
            logger.exception(
                "Pipeline failed for file: %s (attempt %d/%d)",
                filename,
                count,
                settings.pipeline_max_retries,
                extra={"file_name": filename, "fail_count": count},
            )
        if count < settings.pipeline_max_retries:
            # Re-raise: CocoIndex writes no memo entry for a call that raised,
            # so the next run re-processes this file.
            raise
        logger.critical(
            "POISON PILL: %s failed %d times, giving up. Returning normally "
            "writes a memoization entry, so CocoIndex will not retry it. To "
            "force a retry: clear state/ingestion_failures.db and re-run with "
            "`task ingest -- --full-reprocess`. Last error: %r",
            filename,
            count,
            exc,
            extra={"file_name": filename, "fail_count": count, "poisoned": True},
        )
    else:
        get_tracker().reset(filename)

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "Finished file: %s in %dms",
        filename,
        duration_ms,
        extra={
            "file_name": filename,
            "mime_type": mime,
            "page_count": len(pages),
            "duration_ms": duration_ms,
        },
    )

    # Graph data was written (possibly partially, if this file poisoned), so it
    # must stay reclaimable when the source file disappears.
    if graph_engine_inst is not None:
        return hashlib.sha256(content).hexdigest()
    return None


if __name__ == "__main__":
    # Backwards-compatible entrypoint: `python -m ingestion.pipeline [--live]`
    # is what the Taskfile and both compose files invoke.
    from ingestion.runner import main

    main()
