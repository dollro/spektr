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
from ingestion.embedder import Embedder, create_embedder
from ingestion.file_processor import (
    TextChunk,
    docling_chunk,
    file_to_pages,
    semantic_chunk,
)
from ingestion.graph_writer import GraphitiWriter
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


class _PageTasks:
    """Text and image coroutines for a single page."""

    __slots__ = ("text", "image")

    def __init__(self) -> None:
        self.text: list = []  # type: ignore[type-arg]
        self.image: list = []  # type: ignore[type-arg]


def _build_page_tasks(
    page,  # type: ignore[no-untyped-def]
    source_file: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graphiti_writer: GraphitiWriter | None,
    docling_chunks: list[TextChunk] | None = None,
    chunk_collector: list[TextChunk] | None = None,
) -> _PageTasks:
    """Return text and image coroutines for processing one page."""
    tasks = _PageTasks()

    if page.content_type == "text":
        tasks.text.append(
            _process_text_page(
                source_file,
                page.text,
                page.page_number,
                mime,
                now,
                qdrant,
                embedder,
                graphiti_writer,
                docling_chunks=docling_chunks,
                chunk_collector=chunk_collector,
            )
        )
    elif page.content_type == "pdf":
        if page.text.strip():
            tasks.text.append(
                _process_text_page(
                    source_file,
                    page.text,
                    page.page_number,
                    mime,
                    now,
                    qdrant,
                    embedder,
                    graphiti_writer,
                    docling_chunks=docling_chunks,
                    chunk_collector=chunk_collector,
                )
            )

        should_embed_image = (
            settings.image_embed_strategy == "all"
            or (
                settings.image_embed_strategy == "smart"
                and page.has_visual_content
            )
        )
        if should_embed_image:
            tasks.image.append(
                _process_visual_page(
                    source_file,
                    page.image_bytes,
                    page.page_number,
                    "pdf_page",
                    mime,
                    now,
                    qdrant,
                    embedder,
                    graphiti_writer,
                )
            )
    else:
        tasks.image.append(
            _process_visual_page(
                source_file,
                page.image_bytes,
                page.page_number,
                "image",
                mime,
                now,
                qdrant,
                embedder,
                graphiti_writer,
            )
        )

    return tasks


async def _process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graphiti_writer: GraphitiWriter | None,
    docling_chunks: list[TextChunk] | None = None,
    chunk_collector: list[TextChunk] | None = None,
) -> None:
    """Process a text page: chunk, embed, store dense, ingest to Graphiti.

    When docling_chunks are provided, uses them (filtered to this page)
    with late_chunking=True for contextual embeddings.
    Falls back to semantic_chunk() without late chunking otherwise.
    """
    use_late_chunking = False
    if docling_chunks is not None:
        chunks = [
            c for c in docling_chunks if c.page_number == page_number
        ]
        use_late_chunking = bool(chunks)

    if not docling_chunks or not use_late_chunking:
        chunks = semantic_chunk(text, page_number=page_number)

    if not chunks:
        return

    # Batch-embed all chunks in a single API call
    try:
        all_texts = [chunk.text for chunk in chunks]
        vectors = await embedder.embed_text(
            all_texts, late_chunking=use_late_chunking,
        )
    except Exception:
        logger.exception(
            "Text embedding failed for %s page %d (%d chunks)",
            source_file,
            page_number,
            len(chunks),
        )
        return

    dense_points: list[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        chunk_id = _make_chunk_id(
            source_file,
            page_number,
            chunk.chunk_index,
        )
        point_id = _make_point_id(chunk_id)
        payload = {
            "source_file": source_file,
            "content_type": "text_chunk",
            "page_number": page_number,
            "chunk_index": chunk.chunk_index,
            "char_count": len(chunk.text),
            "text_content": chunk.text,
            "metadata": {
                "mime_type": mime,
                "ingested_at": now,
                "source_key": source_file,
            },
        }
        if chunk.contextualized_text is not None:
            payload["contextualized_text"] = chunk.contextualized_text

        dense_points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ),
        )

    if dense_points:
        qdrant.upsert(
            collection_name=DENSE_COLLECTION,
            points=dense_points,
        )

    # Collect chunks for bulk Graphiti ingestion (called after all pages)
    if chunk_collector is not None:
        chunk_collector.extend(chunks)


async def _process_visual_page(
    source_file: str,
    image_bytes: bytes,
    page_number: int,
    content_type: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graphiti_writer: GraphitiWriter | None = None,
) -> None:
    """Process an image/PDF page: dense + multivec embedding."""
    page_key = f"{source_file}::p{page_number}"

    # Dense embedding
    try:
        vector = await embedder.embed_image(image_bytes)
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
                            "source_key": source_file,
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

    # ColBERT multi-vector (requires jina-colbert-v2)
    if settings.multivec_enabled:
        try:
            vectors = await embedder.embed_multi_vector(image_bytes)
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
                                "source_key": source_file,
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

    # VLM caption -> Graphiti (when enabled)
    if settings.vlm_generation_enabled and graphiti_writer is not None:
        await _caption_and_ingest_visual(
            source_file,
            image_bytes,
            page_number,
            graphiti_writer,
        )


_vlm_client_anthropic: object | None = None
_vlm_client_openai: object | None = None


def _get_vlm_client() -> object:
    """Return a lazily-initialized VLM API client (singleton)."""
    provider = settings.llm_api_type.lower()
    if provider == "anthropic":
        global _vlm_client_anthropic  # noqa: PLW0603
        if _vlm_client_anthropic is None:
            import anthropic

            _vlm_client_anthropic = anthropic.AsyncAnthropic(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
            )
        return _vlm_client_anthropic
    else:
        global _vlm_client_openai  # noqa: PLW0603
        if _vlm_client_openai is None:
            import openai

            _vlm_client_openai = openai.AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
            )
        return _vlm_client_openai


async def _caption_visual_page(image_bytes: bytes) -> str:
    """Generate a text description of a visual page using VLM."""
    import base64 as _b64

    provider = settings.llm_api_type.lower()
    _b64_str = _b64.b64encode(image_bytes).decode()
    client = _get_vlm_client()

    prompt = (
        "Describe the content of this document page in detail. "
        "Extract all entities (people, organizations, products, dates, "
        "numbers), relationships, and key facts. Be factual and concise."
    )

    if provider == "anthropic":
        resp = await client.messages.create(
            model=settings.llm_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _b64_str,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return resp.content[0].text
    else:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_b64_str}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""


async def _caption_and_ingest_visual(
    source_file: str,
    image_bytes: bytes,
    page_number: int,
    graphiti_writer: GraphitiWriter,
) -> None:
    """Caption a visual page and ingest the text to Graphiti."""
    try:
        caption = await _caption_visual_page(image_bytes)
        if not caption or not caption.strip():
            return

        await graphiti_writer.ingest_chunk(
            chunk_text=caption,
            source_key=source_file,
            page_number=page_number,
            chunk_index=0,
        )
        logger.info(
            "VLM caption ingested for %s page %d (%d chars)",
            source_file,
            page_number,
            len(caption),
        )
    except Exception:
        logger.exception(
            "VLM caption/ingestion failed for %s page %d",
            source_file,
            page_number,
        )


async def _ingest_to_graphiti(
    source_file: str,
    chunks: list[TextChunk],
    graphiti_writer: GraphitiWriter,
) -> None:
    """Ingest chunks as Graphiti episodes using bulk API.

    Uses add_episode_bulk for speed. GraphitiWriter.ingest_bulk
    handles fallback to sequential on failure.
    """
    ref_time = datetime.now(tz=UTC)
    await graphiti_writer.ingest_bulk(
        chunks=chunks,
        source_key=source_file,
        reference_time=ref_time,
    )


@cocoindex.op.function()
def ingest_file(content: bytes, filename: str) -> str:
    """Process a single file: classify, embed, store to Qdrant + Neo4j.

    Returns filename as passthrough for CocoIndex lineage tracking.
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

    # Compute Docling chunks once for the whole document
    dl_chunks = (
        docling_chunk(result.docling_document)
        if result.docling_document
        else None
    )
    if dl_chunks:
        logger.info(
            "Using Docling HybridChunker: %d chunks for %s",
            len(dl_chunks),
            filename,
            extra={
                "file_name": filename,
                "chunk_count": len(dl_chunks),
            },
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
            graphiti_writer = GraphitiWriter()

        async def _process_all_pages() -> None:
            embedder = create_embedder()
            sem = asyncio.Semaphore(2)
            all_chunks: list[TextChunk] = []

            async def _bounded(coro):  # type: ignore[no-untyped-def]
                async with sem:
                    return await coro

            try:
                text_tasks: list = []  # type: ignore[type-arg]
                image_tasks: list = []  # type: ignore[type-arg]
                for page in pages:
                    pt = _build_page_tasks(
                        page,
                        filename,
                        mime,
                        now,
                        qdrant,
                        embedder,
                        graphiti_writer,
                        docling_chunks=dl_chunks,
                        chunk_collector=all_chunks if graphiti_writer else None,
                    )
                    text_tasks.extend(pt.text)
                    image_tasks.extend(pt.image)

                # Text: concurrent (lightweight, small TPM footprint)
                if text_tasks:
                    await asyncio.gather(*[_bounded(t) for t in text_tasks])

                # Bulk Graphiti ingestion after all text pages are processed
                if graphiti_writer and all_chunks:
                    await _ingest_to_graphiti(
                        filename, all_chunks, graphiti_writer,
                    )

                # Images: sequential (heavy, TPM-sensitive)
                for task in image_tasks:
                    await task
            finally:
                if hasattr(embedder, "tokens_used"):
                    logger.info(
                        "Token usage for %s: %.0f estimated tokens",
                        filename,
                        embedder.tokens_used,
                        extra={
                            "file_name": filename,
                            "estimated_tokens": embedder.tokens_used,
                        },
                    )
                await embedder.close()

        run_async(
            _process_all_pages(),
            timeout=settings.pipeline_timeout,
        )
    except TimeoutError:
        logger.error(
            "File processing timed out after %ds: %s",
            settings.pipeline_timeout,
            filename,
            extra={"file_name": filename},
        )
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


def handle_file_delete(source_key: str) -> None:
    """Handle file deletion: remove Qdrant points and invalidate graph.

    Works for both S3 and local file sources.
    Deletes all Qdrant points matching the source_file and invalidates
    related Graphiti episodes.
    """
    logger.info(
        "Handling file delete for %s",
        source_key,
        extra={"file_name": source_key},
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
                            match=models.MatchValue(value=source_key),
                        ),
                    ],
                ),
            ),
        )
        # Delete multivec collection points by source_file
        if settings.multivec_enabled:
            qdrant.delete(
                collection_name=MULTIVEC_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=source_key),
                            ),
                        ],
                    ),
                ),
            )
        logger.info(
            "Deleted Qdrant points for %s",
            source_key,
            extra={"file_name": source_key},
        )
    except Exception:
        logger.exception(
            "Failed to delete Qdrant points for %s",
            source_key,
            extra={"file_name": source_key},
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
            edges = await client.search(source_key)
            invalidated = 0
            for edge in edges:
                if edge.source_description == source_key:
                    edge.expired_at = datetime.now(tz=UTC)
                    await edge.save(client.driver)
                    invalidated += 1
            logger.info(
                "Invalidated %d Graphiti edges for %s",
                invalidated,
                source_key,
                extra={"file_name": source_key},
            )
            await close_graphiti()

        run_async(_invalidate_graph())
    except Exception:
        logger.exception(
            "Failed to invalidate Graphiti data for %s",
            source_key,
            extra={"file_name": source_key},
        )


# Backward-compatible alias
handle_s3_delete = handle_file_delete


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
