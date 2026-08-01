"""Dense and sparse retrieval channels over documents_dense.

Each channel returns its own ranked list. Fusion is a separate stage — these
functions never compare scores across channels, because cosine similarity and
miniCOIL scores are not on a comparable scale.
"""

from __future__ import annotations

import asyncio
import logging

from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from config.settings import settings
from ingestion.embedder import Embedder, create_embedder
from ingestion.sparse_embedder import encode_query
from retrieval.models import Candidate

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None
_embedder: Embedder | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def _get_embedder() -> Embedder:
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


def build_filter(
    content_type: str | None,
    source_file: str | None,
) -> models.Filter | None:
    """Assemble a plain must-filter for the no-session case."""
    conditions: list[models.FieldCondition] = []
    if content_type is not None:
        conditions.append(
            models.FieldCondition(
                key="content_type", match=models.MatchValue(value=content_type)
            )
        )
    if source_file is not None:
        conditions.append(
            models.FieldCondition(
                key="source_file", match=models.MatchValue(value=source_file)
            )
        )
    return models.Filter(must=conditions) if conditions else None  # type: ignore[arg-type]


def build_kb_filter(
    content_type: str | None,
    source_file: str | None,
) -> models.Filter:
    """Bulk knowledge-base filter: excludes live session data.

    Uses `must_not: is_live == True` rather than the old _dual_query's
    `is_live == False OR is_null(is_live)` should-clause
    (server/tools/vector_search.py). That old shape does NOT work: Qdrant's
    IsNullCondition matches an explicit JSON `null` value, not a missing
    key, and bulk KB points (Path A) never write `is_live` at all — so its
    null branch matches zero points. Verified empirically against a live
    Qdrant 1.17 instance: the should-clause returned 0 of 68 real KB
    points; `must_not` returns all 68. `must_not` naturally treats a
    missing field as "not equal to True", which list_documents.py and
    list_document_chunks.py already rely on elsewhere in this codebase —
    this brings build_kb_filter in line with the pattern proven to work,
    rather than reproducing a filter shape that was never actually
    exercised against un-mocked Qdrant.

    Unlike the old code, content_type/source_file apply uniformly here
    (there's no should-branch for them to be scoped out of).
    """
    must: list[models.Condition] = []
    if content_type is not None:
        must.append(
            models.FieldCondition(
                key="content_type", match=models.MatchValue(value=content_type)
            )
        )
    if source_file is not None:
        must.append(
            models.FieldCondition(
                key="source_file", match=models.MatchValue(value=source_file)
            )
        )
    return models.Filter(
        must=must,
        must_not=[models.FieldCondition(key="is_live", match=models.MatchValue(value=True))],
    )


def build_live_filter(session_id: str) -> models.Filter:
    """Live-session filter: only points tagged with this session_id.

    session_id is written at the TOP LEVEL of the payload by
    ingestion/live_ingest.py, not nested under metadata.
    """
    return models.Filter(
        must=[
            models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
        ],
    )


def _to_candidates(points: list, channel: str) -> list[Candidate]:  # type: ignore[type-arg]
    """Convert Qdrant scored points into Candidates.

    Live points (is_live=True, set by ingestion/live_ingest.py) get
    source_type: "live" synthesised into their metadata at read time,
    mirroring server/tools/vector_search.py's old _dual_query. Live points
    never carry this in their stored payload metadata, so callers that need
    to tell live from bulk KB hits (e.g. shape_response) rely on this.
    """
    out: list[Candidate] = []
    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        if payload.get("is_live"):
            metadata = {**metadata, "source_type": "live"}
        out.append(
            Candidate(
                id=str(point.id),
                text=payload.get("text_content", payload.get("text", "")),
                source_file=payload.get("source_file", ""),
                page_number=payload.get("page_number", 0),
                chunk_index=payload.get("chunk_index", 0),
                score=point.score,
                channel=channel,
                metadata=metadata,
            )
        )
    return out


async def dense_channel(
    query: str,
    limit: int,
    query_filter: models.Filter | None,
) -> list[Candidate]:
    """Semantic similarity search over the named 'dense' vector."""
    if not query or not query.strip():
        return []

    vector = await _get_embedder().embed_text_query(query)
    response = await asyncio.to_thread(
        _get_qdrant_client().query_points,
        collection_name=DENSE_COLLECTION,
        query=vector,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _to_candidates(response.points, "dense")


async def sparse_channel(
    query: str,
    limit: int,
    query_filter: models.Filter | None,
) -> list[Candidate]:
    """Lexical search over the named 'sparse' miniCOIL vector."""
    if not query or not query.strip():
        return []

    vector = await asyncio.to_thread(encode_query, query)
    response = await asyncio.to_thread(
        _get_qdrant_client().query_points,
        collection_name=DENSE_COLLECTION,
        query=vector,
        using=SPARSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _to_candidates(response.points, "sparse")
