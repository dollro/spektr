"""The CocoIndex v1 application for bulk ingestion (Path A).

Replaces v0's ``@cocoindex.flow_def`` / ``FlowBuilder`` / collector / export
graph with plain async Python: an ``App`` binds a ``@coco.fn`` main function
which mounts a processing component per source file and declares target states
from inside it.

Source selection by ``DOCUMENT_SOURCE`` is unchanged in meaning:

===============  ============================================================
``local``        ``localfs.walk_dir(..., live=…)`` — a real filesystem watcher
``sharepoint``   the same watcher, over the mirror the syncer populates
``s3``           ``amazon_s3.list_objects(...)`` — scan only. v1 dropped the
                 SQS push trigger, so live mode is driven externally by
                 ``ingestion/sqs_trigger.py``.
===============  ============================================================
"""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator
from typing import Any

import cocoindex as coco
from cocoindex.connectors import amazon_s3, localfs
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from config.logging import get_logger
from config.settings import settings
from ingestion.embedder import Embedder, create_embedder
from ingestion.graph_target import declare_graph_source
from ingestion.pipeline import SUPPORTED_PATTERNS, process_file_impl
from ingestion.qdrant_target import (
    QDRANT_DB,
    create_qdrant_client,
    mount_dense_target,
    mount_multivec_target,
)

logger = get_logger(__name__)

APP_NAME = "RagIngestion"

EMBEDDER: coco.ContextKey[Embedder] = coco.ContextKey("spektr/embedder")
S3_CLIENT: coco.ContextKey[Any] = coco.ContextKey("spektr/s3")


def use_s3_source() -> bool:
    """True when the pipeline reads directly from S3."""
    return settings.document_source == "s3"


def describe_watched_source() -> str:
    """Human-readable label for the source being watched."""
    if settings.document_source == "s3":
        queue = "SQS-triggered" if settings.s3_sqs_queue_url else "interval-only"
        return f"s3://{settings.s3_bucket_name}/{settings.s3_prefix} ({queue})"
    if settings.document_source == "sharepoint":
        return f"local mirror at {settings.local_documents_path} (fed by sharepoint-sync)"
    return f"local filesystem at {settings.local_documents_path}"


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    """Provide shared resources and point CocoIndex at its LMDB state dir."""
    state_dir = pathlib.Path(settings.cocoindex_db_path)
    state_dir.parent.mkdir(parents=True, exist_ok=True)
    builder.settings.db_path = state_dir

    builder.provide(QDRANT_DB, create_qdrant_client())

    # One embedder for the whole run. Under v0 each file got its own instance
    # because every file ran in a fresh event loop; v1 is async end to end.
    embedder = create_embedder()
    builder.provide(EMBEDDER, embedder)

    if use_s3_source():
        from aiobotocore.session import get_session  # type: ignore[import-untyped]

        session = get_session()
        client_cm = session.create_client(
            "s3",
            region_name=settings.aws_region or None,
            endpoint_url=settings.aws_endpoint_url or None,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        await builder.provide_async_with(S3_CLIENT, client_cm)

    try:
        yield
    finally:
        if hasattr(embedder, "tokens_used"):
            logger.info(
                "Embedder token usage for this run: %.0f estimated tokens",
                embedder.tokens_used,
                extra={"estimated_tokens": embedder.tokens_used},
            )
        await embedder.close()


def source_key(file: FileLike[Any]) -> str:
    """The logical key for a source file — what lands in ``source_file``.

    Must stay identical to v0's ``filename`` field: a path relative to the
    source root (``arxiv.pdf``, ``specs/api.md``), never an absolute path.
    Every Qdrant payload, the delete path, ``list_documents`` and the eval
    fixtures are keyed on it.

    ``amazon_s3`` already yields prefix-stripped relative paths; ``localfs``
    yields the walked path, which carries the configured root.
    """
    path = file.file_path.path
    if use_s3_source():
        return path.as_posix()
    try:
        return path.relative_to(pathlib.PurePath(settings.local_documents_path)).as_posix()
    except ValueError:
        return path.as_posix()


@coco.fn(memo=True)
async def process_file(file: FileLike[Any], dense: Any, multivec: Any) -> None:
    """One processing component per source file.

    Memoized: unchanged files (and unchanged code) skip re-execution entirely,
    and their previously declared points stay reconciled as-is.
    """
    filename = source_key(file)
    content = await file.read()
    embedder = coco.use_context(EMBEDDER)

    fingerprint = await process_file_impl(
        content,
        filename,
        dense=dense,
        multivec=multivec,
        embedder=embedder,
    )
    if fingerprint is not None:
        declare_graph_source(filename, fingerprint)


def _source_items() -> Any:
    """Return ``(key, FileLike)`` items for the configured document source."""
    matcher = PatternFilePathMatcher(included_patterns=SUPPORTED_PATTERNS)
    if use_s3_source():
        return amazon_s3.list_objects(
            coco.use_context(S3_CLIENT),
            settings.s3_bucket_name,
            prefix=settings.s3_prefix,
            path_matcher=matcher,
        ).items()
    # live=True makes this a LiveMapView: it snapshots first and then watches,
    # so the same call serves both catch-up and live runs.
    return localfs.walk_dir(
        pathlib.Path(settings.local_documents_path),
        recursive=True,
        live=True,
        path_matcher=matcher,
    ).items()


@coco.fn
async def app_main() -> None:
    """Mount the targets, then one processing component per source file."""
    dense = await mount_dense_target()
    multivec = await mount_multivec_target() if settings.multivec_enabled else None

    await coco.mount_each(process_file, _source_items(), dense, multivec)


def build_app() -> coco.App[Any, Any]:
    """Construct the ingestion app.

    ``max_inflight_components`` bounds how many files are processed at once.
    CocoIndex's default is 1024, which would blow straight past the embedding
    providers' rate limits.
    """
    return coco.App(
        coco.AppConfig(
            name=APP_NAME,
            max_inflight_components=settings.pipeline_max_concurrent_files,
        ),
        app_main,
    )
