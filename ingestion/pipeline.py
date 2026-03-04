"""CocoIndex ingestion pipeline for RAG document processing.

Reads files from local filesystem or S3, processes them through
MIME classification, chunking, embedding, and entity extraction,
then exports to Qdrant (dense + multivec) and Neo4j.
"""

import asyncio
import io
import mimetypes
import os
import time
import uuid
from datetime import UTC, datetime

import cocoindex
from PIL import Image
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.logging import get_logger
from config.settings import settings
from ingestion._utils import run_async
from ingestion.file_processor import (
    TextChunk,
    file_to_pages,
    semantic_chunk,
)
from ingestion.graph_writer import GraphitiWriter
from ingestion.jina_cocoindex_ops import (
    jina_embed_image,
    jina_embed_image_multivec,
    jina_embed_text,
)
from ingestion.neo4j_setup import create_neo4j_schema, get_driver
from ingestion.qdrant_setup import ensure_collections

logger = get_logger(__name__)

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    """Lazily initialize a shared QdrantClient instance."""
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


_SUPPORTED_PATTERNS = [
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.webp",
    "*.md",
    "*.txt",
    "*.csv",
    "*.json",
    "*.xml",
    "*.html",
    "*.yaml",
    "*.yml",
]


def _guess_mime(filename: str) -> str:
    """Guess MIME type from filename."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _make_chunk_id(source_file: str, page: int, chunk_idx: int) -> str:
    """Deterministic chunk ID for idempotent upserts."""
    return f"{source_file}::p{page}::c{chunk_idx}"


def _make_point_id(key: str) -> str:
    """Deterministic UUID from a string key."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _build_page_tasks(
    page,  # type: ignore[no-untyped-def]
    source_file: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    graphiti_writer: GraphitiWriter | None,
) -> list:  # type: ignore[type-arg]
    """Return a list of async coroutines for processing one page."""
    tasks = []

    if page.content_type == "text":
        tasks.append(
            _async_process_text_page(
                source_file,
                page.text,
                page.page_number,
                mime,
                now,
                qdrant,
                graphiti_writer,
            )
        )
    elif page.content_type == "pdf":
        if page.text.strip():
            tasks.append(
                _async_process_text_page(
                    source_file,
                    page.text,
                    page.page_number,
                    mime,
                    now,
                    qdrant,
                    graphiti_writer,
                )
            )
        tasks.append(
            _async_process_visual_page(
                source_file,
                page.image_bytes,
                page.page_number,
                "pdf_page",
                mime,
                now,
                qdrant,
            )
        )
    else:
        tasks.append(
            _async_process_visual_page(
                source_file,
                page.image_bytes,
                page.page_number,
                "image",
                mime,
                now,
                qdrant,
            )
        )

    return tasks


async def _async_process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    graphiti_writer: GraphitiWriter | None,
) -> None:
    """Async wrapper for _process_text_page."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _process_text_page,
        source_file,
        text,
        page_number,
        mime,
        now,
        qdrant,
        graphiti_writer,
    )


async def _async_process_visual_page(
    source_file: str,
    image_bytes: bytes,
    page_number: int,
    content_type: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
) -> None:
    """Async wrapper for _process_visual_page."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _process_visual_page,
        source_file,
        image_bytes,
        page_number,
        content_type,
        mime,
        now,
        qdrant,
    )


def _process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    graphiti_writer: GraphitiWriter | None,
) -> None:
    """Process a text page: chunk, embed, store dense, ingest to Graphiti."""
    chunks = semantic_chunk(text, page_number=page_number)
    if not chunks:
        return

    # Embed all chunks
    dense_points: list[models.PointStruct] = []
    for chunk in chunks:
        chunk_id = _make_chunk_id(
            source_file,
            page_number,
            chunk.chunk_index,
        )
        try:
            vector = jina_embed_text(chunk.text)
        except Exception:
            logger.exception(
                "Text embedding failed for %s chunk %d",
                source_file,
                chunk.chunk_index,
            )
            continue

        point_id = _make_point_id(chunk_id)
        dense_points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "source_file": source_file,
                    "content_type": "text_chunk",
                    "page_number": page_number,
                    "chunk_index": chunk.chunk_index,
                    "char_count": len(chunk.text),
                    "text_content": chunk.text,
                    "metadata": {
                        "mime_type": mime,
                        "ingested_at": now,
                        "s3_key": source_file,
                    },
                },
            ),
        )

    if dense_points:
        qdrant.upsert(
            collection_name=DENSE_COLLECTION,
            points=dense_points,
        )

    # Graphiti episode ingestion (handles entity extraction internally)
    if graphiti_writer is not None:
        _ingest_to_graphiti(source_file, chunks, graphiti_writer)


def _process_visual_page(
    source_file: str,
    image_bytes: bytes,
    page_number: int,
    content_type: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
) -> None:
    """Process an image/PDF page: dense + multivec embedding."""
    page_key = f"{source_file}::p{page_number}"

    # Dense embedding
    try:
        vector = jina_embed_image(image_bytes)
        qdrant.upsert(
            collection_name=DENSE_COLLECTION,
            points=[
                models.PointStruct(
                    id=_make_point_id(page_key),
                    vector=vector,
                    payload={
                        "source_file": source_file,
                        "content_type": content_type,
                        "page_number": page_number,
                        "chunk_index": 0,
                        "text_content": "",
                        "metadata": {
                            "mime_type": mime,
                            "ingested_at": now,
                            "s3_key": source_file,
                        },
                    },
                ),
            ],
        )
    except Exception:
        logger.exception(
            "Dense image embedding failed for %s page %d",
            source_file,
            page_number,
        )

    # ColBERT multi-vector
    try:
        vectors = jina_embed_image_multivec(image_bytes)
        img = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = img.size
        qdrant.upsert(
            collection_name=MULTIVEC_COLLECTION,
            points=[
                models.PointStruct(
                    id=_make_point_id(f"mv::{page_key}"),
                    vector={"colbert": vectors},
                    payload={
                        "source_file": source_file,
                        "content_type": content_type,
                        "page_number": page_number,
                        "image_width": img_width,
                        "image_height": img_height,
                        "metadata": {
                            "mime_type": mime,
                            "ingested_at": now,
                            "s3_key": source_file,
                        },
                    },
                ),
            ],
        )
    except Exception:
        logger.exception(
            "Multivec embedding failed for %s page %d",
            source_file,
            page_number,
        )


def _ingest_to_graphiti(
    source_file: str,
    chunks: list[TextChunk],
    graphiti_writer: GraphitiWriter,
) -> None:
    """Ingest chunks as Graphiti episodes (entity extraction is automatic)."""

    async def _do_ingest() -> None:
        ref_time = datetime.now(tz=UTC)
        for chunk in chunks:
            try:
                await graphiti_writer.ingest_chunk(
                    chunk_text=chunk.text,
                    source_key=source_file,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    reference_time=ref_time,
                )
            except Exception:
                logger.exception(
                    "Graphiti ingestion failed for %s chunk %d",
                    source_file,
                    chunk.chunk_index,
                )

    run_async(_do_ingest())


@cocoindex.op.function()
def ingest_file(content: bytes, filename: str) -> str:
    """Process a single file: classify, embed, store to Qdrant + Neo4j.

    Returns filename as passthrough for CocoIndex lineage tracking.
    """
    t0 = time.monotonic()
    try:
        pages = file_to_pages(filename, content)
    except Exception:
        logger.exception(
            "Failed to process file: %s",
            filename,
            extra={"file_name": filename},
        )
        return filename

    if not pages:
        logger.warning(
            "No pages extracted from %s, skipping.",
            filename,
            extra={"file_name": filename},
        )
        return filename

    mime = _guess_mime(filename)
    now = datetime.now(tz=UTC).isoformat()
    qdrant = _get_qdrant_client()
    graphiti_writer: GraphitiWriter | None = None

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
        if has_text:
            graphiti_writer = GraphitiWriter()

        async def _process_all_pages() -> None:
            tasks = []
            for page in pages:
                tasks.extend(
                    _build_page_tasks(
                        page,
                        filename,
                        mime,
                        now,
                        qdrant,
                        graphiti_writer,
                    )
                )
            if tasks:
                await asyncio.gather(*tasks)

        run_async(_process_all_pages())
    except Exception:
        logger.exception(
            "Pipeline failed for file: %s",
            filename,
            extra={"file_name": filename},
        )
    finally:
        if graphiti_writer is not None:
            run_async(graphiti_writer.close())

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
    return filename


def _use_s3_source() -> bool:
    """Check if S3 source is configured."""
    return settings.document_source == "s3"


def handle_s3_delete(s3_key: str) -> None:
    """Handle S3 object deletion: remove Qdrant points and invalidate graph.

    Called when an s3:ObjectRemoved:* event is received from SQS.
    Deletes all Qdrant points matching the source_file and invalidates
    related Graphiti episodes.
    """
    logger.info(
        "Handling S3 delete for %s",
        s3_key,
        extra={"file_name": s3_key},
    )
    qdrant = _get_qdrant_client()

    try:
        # Delete dense collection points by source_file
        qdrant.delete(
            collection_name=DENSE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=s3_key),
                        ),
                    ],
                ),
            ),
        )
        # Delete multivec collection points by source_file
        qdrant.delete(
            collection_name=MULTIVEC_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=s3_key),
                        ),
                    ],
                ),
            ),
        )
        logger.info(
            "Deleted Qdrant points for %s",
            s3_key,
            extra={"file_name": s3_key},
        )
    except Exception:
        logger.exception(
            "Failed to delete Qdrant points for %s",
            s3_key,
            extra={"file_name": s3_key},
        )

    # Invalidate Graphiti episodes by source_description
    # Graphiti doesn't expose bulk invalidation by source, so we
    # search for matching episodes and mark them individually.
    try:

        async def _invalidate_graph() -> None:
            from ingestion.graphiti_client import (
                close_graphiti,
                get_graphiti,
            )

            client = await get_graphiti()
            edges = await client.search(s3_key)
            invalidated = 0
            for edge in edges:
                if edge.source_description == s3_key:
                    edge.expired_at = datetime.now(tz=UTC)
                    invalidated += 1
            logger.info(
                "Invalidated %d Graphiti edges for %s",
                invalidated,
                s3_key,
                extra={"file_name": s3_key},
            )
            await close_graphiti()

        run_async(_invalidate_graph())
    except Exception:
        logger.exception(
            "Failed to invalidate Graphiti data for %s",
            s3_key,
            extra={"file_name": s3_key},
        )


@cocoindex.flow_def(name="RagIngestion")
def rag_ingestion_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
) -> None:
    """Define the RAG ingestion pipeline.

    Uses CocoIndex for source management and incremental state.
    All embedding, storage, and entity extraction happen in the
    ingest_file custom op which writes directly to Qdrant + Neo4j.
    """
    if _use_s3_source():
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.AmazonS3(
                bucket_name=settings.s3_bucket_name,
                sqs_queue_url=settings.s3_sqs_queue_url,
                binary=True,
                included_patterns=_SUPPORTED_PATTERNS,
            ),
        )
        filename_field = "key"
    else:
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=settings.local_documents_path,
                binary=True,
                included_patterns=_SUPPORTED_PATTERNS,
            ),
        )
        filename_field = "filename"

    collector = data_scope.add_collector()

    with data_scope["files"].row() as doc:
        doc["result"] = doc["content"].transform(
            ingest_file,
            filename=doc[filename_field],
        )

        collector.collect(
            filename=doc[filename_field],
            result=doc["result"],
        )

    collector.export(
        "ingestion_log",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename"],
    )


def run_pipeline() -> None:
    """Initialize and run the ingestion pipeline."""
    t0 = time.monotonic()
    logger.info("Pipeline starting")
    os.environ.setdefault("COCOINDEX_DATABASE_URL", settings.database_url)
    cocoindex.init()

    # Provision infrastructure
    ensure_collections(_get_qdrant_client())

    async def _setup_neo4j() -> None:
        driver = get_driver()
        try:
            await create_neo4j_schema(driver)
        finally:
            await driver.close()

    run_async(_setup_neo4j())

    # Setup and run pipeline
    cocoindex.setup_all_flows()
    run_async(
        cocoindex.update_all_flows_async(
            cocoindex.FlowLiveUpdaterOptions(live_mode=False, print_stats=True),
        )
    )

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "Pipeline completed in %dms",
        duration_ms,
        extra={"duration_ms": duration_ms},
    )


if __name__ == "__main__":
    from config.logging import setup_logging

    setup_logging()
    run_pipeline()
