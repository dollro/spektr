"""Per-page processing: chunk, embed, and declare Qdrant points.

Split out of ``pipeline.py`` during the CocoIndex v1 migration.

The behavioural change from v0 is where points go. Previously each page called
``qdrant.upsert(...)`` directly, and a custom target connector deleted points by
payload filter when a source file disappeared. Now points are *declared* on a
native CocoIndex collection target:

* CocoIndex batches the upserts across the whole reconcile flush;
* deletion is per point id, derived automatically when the file stops being
  declared — no filter-delete, and points CocoIndex never declared (Path B's
  live sessions) are untouchable by it;
* nothing is written until the whole component's processing has succeeded, so a
  mid-file failure can no longer leave half a document in Qdrant.
"""

from __future__ import annotations

import io
import uuid
from typing import Any

from PIL import Image
from qdrant_client import models

from config.constants import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from config.logging import get_logger
from config.settings import settings
from ingestion.embedder import Embedder
from ingestion.file_processor import TextChunk, semantic_chunk
from ingestion.graph_engine import GraphEngine
from ingestion.sparse_embedder import encode_documents
from ingestion.vlm_caption import caption_and_ingest_visual

logger = get_logger(__name__)


def make_chunk_id(source_file: str, page: int, chunk_idx: int) -> str:
    """Deterministic chunk ID for idempotent upserts."""
    return f"{source_file}::p{page}::c{chunk_idx}"


def make_point_id(key: str) -> str:
    """Deterministic UUID from a string key.

    Qdrant (and CocoIndex's own point-id validation) accepts a u64 or a UUID;
    the uuid5 keeps ids stable across re-ingests of the same chunk.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class _PageTasks:
    """Text and image coroutines for a single page."""

    __slots__ = ("text", "image")

    def __init__(self) -> None:
        self.text: list[Any] = []
        self.image: list[Any] = []


def _build_page_tasks(
    page: Any,
    source_file: str,
    mime: str,
    now: str,
    dense: Any,
    multivec: Any,
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
                dense,
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
                    dense,
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
                    dense,
                    multivec,
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
                dense,
                multivec,
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
    sparse_vector: models.SparseVector,
    mime: str,
    now: str,
    embedder_model: str,
    embedder_dim: int,
) -> models.PointStruct:
    """Build one dense-collection point with named dense + sparse vectors.

    ``sparse_vector`` is passed in rather than computed here so the miniCOIL
    encoder is called once per page with the full chunk list instead of once per
    chunk.
    """
    chunk_id = make_chunk_id(source_file, page_number, chunk_index)
    payload: dict[str, Any] = {
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

    return models.PointStruct(
        id=make_point_id(chunk_id),
        vector={DENSE_VECTOR_NAME: vector, SPARSE_VECTOR_NAME: sparse_vector},
        payload=payload,
    )


async def _process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    dense: Any,
    embedder: Embedder,
    graph_engine: GraphEngine | None,
    docling_chunks: list[TextChunk] | None = None,
    chunk_collector: list[TextChunk] | None = None,
) -> None:
    """Process a text page: chunk, embed, declare dense points, collect chunks.

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

    # Batch-embed all chunks in a single API call.
    #
    # A failure here MUST propagate. Under v0 these points were upserted
    # directly, so swallowing the error merely skipped an upsert and any
    # previously-written points survived. Under v1 the points are *declared*:
    # a page we fail to declare is reconciled to non-existence, i.e. its
    # existing points are deleted — and returning normally would memoize the
    # file so it never retries. Raising leaves the file un-memoized, keeps its
    # previous target states intact, and feeds the poison-pill counter.
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
        raise

    # One miniCOIL encode for the whole page, not one per chunk.
    sparse_vectors = encode_documents(all_texts)

    for chunk, vector, sparse_vector in zip(chunks, vectors, sparse_vectors, strict=True):
        dense.declare_point(
            _build_chunk_point(
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                contextualized_text=chunk.contextualized_text,
                vector=vector,
                sparse_vector=sparse_vector,
                mime=mime,
                now=now,
                embedder_model=embedder.model_name,
                embedder_dim=embedder.dim,
            )
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
    dense: Any,
    multivec: Any,
    embedder: Embedder,
    graph_engine: GraphEngine | None = None,
) -> None:
    """Process an image/PDF page: dense + optional multivec embedding."""
    page_key = f"{source_file}::p{page_number}"

    # Dense embedding. Failures propagate for the same reason as in
    # _process_text_page: a point we do not declare is a point CocoIndex
    # deletes.
    try:
        vector = await embedder.embed_image(image_bytes)
        dense.declare_point(
            models.PointStruct(
                id=make_point_id(page_key),
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
        )
    except Exception:
        logger.exception(
            "Dense image embedding failed for %s page %d",
            source_file,
            page_number,
        )
        raise

    # ColBERT multi-vector (requires jina-colbert-v2)
    if settings.multivec_enabled and multivec is not None:
        try:
            vectors = await embedder.embed_multi_vector(image_bytes)
            img = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = img.size
            multivec.declare_point(
                models.PointStruct(
                    id=make_point_id(f"mv::{page_key}"),
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
            )
        except Exception:
            logger.exception(
                "Multivec embedding failed for %s page %d",
                source_file,
                page_number,
            )
            raise

    # VLM caption -> graph (when enabled)
    if settings.vlm_generation_enabled and graph_engine is not None:
        await caption_and_ingest_visual(
            source_file,
            image_bytes,
            page_number,
            graph_engine,
        )
