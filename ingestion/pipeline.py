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

from config.constants import (
    DENSE_COLLECTION,
    DENSE_VECTOR_NAME,
    MULTIVEC_COLLECTION,
    SPARSE_VECTOR_NAME,
)
from config.logging import get_logger
from config.settings import settings
from ingestion._failure_tracker import get_tracker
from ingestion._utils import run_async
from ingestion.embedder import Embedder, create_embedder
from ingestion.file_processor import (
    TextChunk,
    docling_chunk,
    file_to_pages,
    semantic_chunk,
)
from ingestion.graph_engine import GraphEngine, close_graph_engine, get_graph_engine
from ingestion.neo4j_setup import create_neo4j_schema, get_driver
from ingestion.qdrant_setup import ensure_collections
from ingestion.schema_inducer import SchemaInducer
from ingestion.sparse_embedder import encode_documents
from ingestion.target_connector import RagTarget

logger = get_logger(__name__)

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    """Lazily initialize a shared QdrantClient instance."""
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


# globset patterns: bare *.pdf matches a single path segment, so it would
# miss files in subdirectories. Use **/*.ext to cover any depth — required
# for the SharePoint syncer (which mirrors folder structure) and harmless
# for flat layouts.
_SUPPORTED_PATTERNS = [
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
    graph_engine: GraphEngine | None,
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
                graph_engine,
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
                    graph_engine,
                    docling_chunks=docling_chunks,
                    chunk_collector=chunk_collector,
                )
            )

        should_embed_image = settings.image_embed_strategy == "all" or (
            settings.image_embed_strategy == "smart" and page.has_visual_content
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
                    graph_engine,
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
                graph_engine,
            )
        )

    return tasks


def _build_chunk_point(
    *,
    source_file: str,
    page_number: int,
    chunk_index: int,
    text: str,
    contextualized_text: str | None,
    vector: list[float],
    mime: str,
    now: str,
    embedder_model: str,
    embedder_dim: int,
) -> models.PointStruct:
    """Build one dense-collection point with named dense + sparse vectors."""
    chunk_id = _make_chunk_id(source_file, page_number, chunk_index)
    payload = {
        "source_file": source_file,
        "content_type": "text_chunk",
        "page_number": page_number,
        "chunk_index": chunk_index,
        "char_count": len(text),
        "text_content": text,
        "embedder_model": embedder_model,
        "embedder_dim": embedder_dim,
        "metadata": {
            "mime_type": mime,
            "ingested_at": now,
            "source_key": source_file,
        },
    }
    if contextualized_text is not None:
        payload["contextualized_text"] = contextualized_text

    sparse = encode_documents([text])[0]
    return models.PointStruct(
        id=_make_point_id(chunk_id),
        vector={DENSE_VECTOR_NAME: vector, SPARSE_VECTOR_NAME: sparse},
        payload=payload,
    )


async def _process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graph_engine: GraphEngine | None,
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
        chunks = [c for c in docling_chunks if c.page_number == page_number]
        use_late_chunking = bool(chunks)

    if not docling_chunks or not use_late_chunking:
        chunks = semantic_chunk(text, page_number=page_number)

    if not chunks:
        return

    # Batch-embed all chunks in a single API call
    try:
        all_texts = [chunk.text for chunk in chunks]
        vectors = await embedder.embed_text(
            all_texts,
            late_chunking=use_late_chunking,
        )
    except Exception:
        logger.exception(
            "Text embedding failed for %s page %d (%d chunks)",
            source_file,
            page_number,
            len(chunks),
        )
        return

    dense_points: list[models.PointStruct] = [
        _build_chunk_point(
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            contextualized_text=chunk.contextualized_text,
            vector=vector,
            mime=mime,
            now=now,
            embedder_model=embedder.model_name,
            embedder_dim=embedder.dim,
        )
        for chunk, vector in zip(chunks, vectors)
    ]

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
    graph_engine: GraphEngine | None = None,
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
                    vector={DENSE_VECTOR_NAME: vector},
                    payload={
                        "source_file": source_file,
                        "content_type": content_type,
                        "page_number": page_number,
                        "chunk_index": 0,
                        "text_content": "",
                        "embedder_model": embedder.model_name,
                        "embedder_dim": embedder.dim,
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
                            "embedder_model": embedder.model_name,
                            "embedder_dim": embedder.dim,
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
    if settings.vlm_generation_enabled and graph_engine is not None:
        await _caption_and_ingest_visual(
            source_file,
            image_bytes,
            page_number,
            graph_engine,
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
    graph_engine: GraphEngine,
) -> None:
    """Caption a visual page and ingest the text to graph."""
    try:
        caption = await _caption_visual_page(image_bytes)
        if not caption or not caption.strip():
            return
        chunk = TextChunk(text=caption, chunk_index=0, page_number=page_number)
        await graph_engine.ingest([chunk], source_file)
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
    graph_engine_inst: GraphEngine | None = None

    # Compute Docling chunks once for the whole document
    dl_chunks = docling_chunk(result.docling_document) if result.docling_document else None
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
            graph_engine_inst = get_graph_engine()

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
                        graph_engine_inst,
                        docling_chunks=dl_chunks,
                        chunk_collector=all_chunks if graph_engine_inst else None,
                    )
                    text_tasks.extend(pt.text)
                    image_tasks.extend(pt.image)

                # Text: concurrent (lightweight, small TPM footprint)
                if text_tasks:
                    await asyncio.gather(*[_bounded(t) for t in text_tasks])

                # Bulk graph ingestion after all text pages are processed
                if graph_engine_inst and all_chunks:
                    await _ingest_to_graph_with_schema(
                        filename,
                        all_chunks,
                        graph_engine_inst,
                    )

                # Images: sequential (heavy, TPM-sensitive)
                for task in image_tasks:
                    await task
            except BaseException:
                # Close unawaited coroutines to suppress RuntimeWarnings
                for coro in text_tasks + image_tasks:
                    coro.close()
                raise
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
            raise
        logger.critical(
            "POISON PILL: %s failed %d times, giving up. "
            "CocoIndex will mark processed; clear state/ingestion_failures.db "
            "and delete the tracking row to retry. Last error: %r",
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
    return filename


def _use_s3_source() -> bool:
    """Check if S3 source is configured."""
    return settings.document_source == "s3"


def _describe_watched_source() -> str:
    """Human-readable label for the source CocoIndex is currently watching."""
    if settings.document_source == "s3":
        return "SQS"
    if settings.document_source == "sharepoint":
        return f"local mirror at {settings.local_documents_path} (fed by sharepoint-sync)"
    return f"local filesystem at {settings.local_documents_path}"


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
    else:
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=settings.local_documents_path,
                binary=True,
                included_patterns=_SUPPORTED_PATTERNS,
            ),
        )
    # Both LocalFile and AmazonS3 sources expose "filename" as the row key.
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
        "rag_target",
        RagTarget(qdrant_url=settings.qdrant_url),
        primary_key_fields=["filename"],
    )


def run_pipeline(live: bool = False) -> None:
    """Initialize and run the ingestion pipeline.

    When ``live`` is True, CocoIndex stays running and watches the
    configured source (SQS for S3, filesystem events for local) until
    the process is interrupted. Use Ctrl-C to stop.
    """
    from config.observability import setup_observability

    setup_observability()
    t0 = time.monotonic()
    logger.info("Pipeline starting (live=%s)", live)
    os.environ.setdefault("COCOINDEX_DATABASE_URL", settings.database_url)

    # Export AWS credentials to os.environ so CocoIndex's Rust-based S3
    # client can see them (pydantic-settings reads .env into Settings but
    # does NOT populate process env). Only needed when using the S3 source.
    if _use_s3_source():
        if settings.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
        if settings.aws_region:
            os.environ["AWS_REGION"] = settings.aws_region
            os.environ["AWS_DEFAULT_REGION"] = settings.aws_region
        if settings.aws_endpoint_url:
            os.environ["AWS_ENDPOINT_URL"] = settings.aws_endpoint_url

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
    if live:
        watched = _describe_watched_source()
        logger.info(
            "Entering live mode — watching %s for changes. Press Ctrl-C to stop.",
            watched,
        )
        if settings.document_source == "sharepoint":
            logger.info(
                "DOCUMENT_SOURCE=sharepoint: ensure `task sharepoint-sync` "
                "is also running so the local mirror gets populated."
            )
    run_async(
        cocoindex.update_all_flows_async(
            cocoindex.FlowLiveUpdaterOptions(
                live_mode=live,
                print_stats=True,
            ),
        )
    )

    # Clean up graph engine singleton (if created)
    if settings.graph_enabled:
        run_async(close_graph_engine())

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "Pipeline completed in %dms",
        duration_ms,
        extra={"duration_ms": duration_ms},
    )


if __name__ == "__main__":
    import sys

    from config.logging import setup_logging

    setup_logging()
    live = "--live" in sys.argv[1:]
    run_pipeline(live=live)
