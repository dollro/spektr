"""CocoIndex ingestion pipeline for RAG document processing.

Reads files from local filesystem or S3, processes them through
MIME classification, chunking, embedding, and entity extraction,
then exports to Qdrant (dense + multivec) and Neo4j.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import uuid
from datetime import UTC, datetime

import cocoindex
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.settings import settings
from ingestion.entity_extractor import extract_entities, get_llm_client
from ingestion.file_processor import (
    TextChunk,
    file_to_pages,
    semantic_chunk,
)
from ingestion.graph_writer import GraphWriter
from ingestion.jina_cocoindex_ops import (
    jina_embed_image,
    jina_embed_image_multivec,
    jina_embed_text,
)
from ingestion.neo4j_setup import create_neo4j_schema, get_driver
from ingestion.qdrant_setup import ensure_collections

logger = logging.getLogger(__name__)

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


def _run_async(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine from a sync context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    graph_writer: GraphWriter | None,
) -> None:
    """Process a text page: chunk, embed, store dense, extract entities."""
    chunks = semantic_chunk(text)
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

    # Entity extraction + graph writes
    if graph_writer is not None:
        _write_entities(source_file, chunks, graph_writer)


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


def _write_entities(
    source_file: str,
    chunks: list[TextChunk],
    graph_writer: GraphWriter,
) -> None:
    """Extract entities from chunks and write to Neo4j."""
    llm_client = get_llm_client()

    async def _do_write() -> None:
        for chunk in chunks:
            chunk_id = _make_chunk_id(
                source_file,
                chunk.page_number,
                chunk.chunk_index,
            )
            await graph_writer.upsert_chunk(
                chunk_id=chunk_id,
                text_preview=chunk.text[:200],
                page_number=chunk.page_number,
                s3_key=source_file,
            )
            try:
                result = await extract_entities(chunk.text, llm_client)
                await graph_writer.write_extraction_result(
                    s3_key=source_file,
                    chunk_id=chunk_id,
                    extraction_result=result,
                )
            except Exception:
                logger.exception(
                    "Entity extraction failed for chunk %s",
                    chunk_id,
                )

    _run_async(_do_write())


@cocoindex.op.function()
def ingest_file(content: bytes, filename: str) -> str:
    """Process a single file: classify, embed, store to Qdrant + Neo4j.

    Returns filename as passthrough for CocoIndex lineage tracking.
    """
    try:
        pages = file_to_pages(filename, content)
    except Exception:
        logger.exception("Failed to process file: %s", filename)
        return filename

    if not pages:
        logger.warning("No pages extracted from %s, skipping.", filename)
        return filename

    mime = _guess_mime(filename)
    now = datetime.now(tz=UTC).isoformat()
    qdrant = QdrantClient(url=settings.qdrant_url)
    graph_writer: GraphWriter | None = None

    try:
        # Upsert document node in Neo4j
        has_text = any(p.content_type == "text" for p in pages)
        if has_text:
            graph_writer = GraphWriter()
            _run_async(
                graph_writer.upsert_document(
                    s3_key=filename,
                    filename=filename,
                    mime_type=mime,
                    page_count=len(pages),
                    source_bucket=settings.s3_bucket_name,
                ),
            )

        for page in pages:
            if page.content_type == "text":
                _process_text_page(
                    source_file=filename,
                    text=page.text,
                    page_number=page.page_number,
                    mime=mime,
                    now=now,
                    qdrant=qdrant,
                    graph_writer=graph_writer,
                )
            else:
                _process_visual_page(
                    source_file=filename,
                    image_bytes=page.image_bytes,
                    page_number=page.page_number,
                    content_type=("pdf_page" if page.content_type == "pdf" else "image"),
                    mime=mime,
                    now=now,
                    qdrant=qdrant,
                )
    except Exception:
        logger.exception("Pipeline failed for file: %s", filename)
    finally:
        qdrant.close()
        if graph_writer is not None:
            _run_async(graph_writer.close())

    return filename


def _use_s3_source() -> bool:
    """Check if S3 source is configured."""
    return bool(settings.s3_bucket_name and settings.s3_sqs_queue_url)


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
                path="documents",
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
    cocoindex.init()

    # Provision infrastructure
    qdrant = QdrantClient(url=settings.qdrant_url)
    ensure_collections(qdrant)
    qdrant.close()

    driver = get_driver()
    _run_async(create_neo4j_schema(driver))
    _run_async(driver.close())

    # Open and run pipeline
    flow = cocoindex.open_flow("RagIngestion", rag_ingestion_flow)
    flow.setup()
    cocoindex.update_all_flows()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_pipeline()
