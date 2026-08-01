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
    session_id: str | None,
) -> models.Filter | None:
    """Assemble a Qdrant filter from optional constraints."""
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
    if session_id is not None:
        # session_id is written at the TOP LEVEL of the payload by
        # ingestion/live_ingest.py, not nested under metadata. The existing
        # _dual_query in server/tools/vector_search.py filters on the same key.
        conditions.append(
            models.FieldCondition(
                key="session_id", match=models.MatchValue(value=session_id)
            )
        )
    return models.Filter(must=conditions) if conditions else None  # type: ignore[arg-type]


def _to_candidates(points: list, channel: str) -> list[Candidate]:  # type: ignore[type-arg]
    """Convert Qdrant scored points into Candidates."""
    out: list[Candidate] = []
    for point in points:
        payload = point.payload or {}
        out.append(
            Candidate(
                id=str(point.id),
                text=payload.get("text_content", payload.get("text", "")),
                source_file=payload.get("source_file", ""),
                page_number=payload.get("page_number", 0),
                chunk_index=payload.get("chunk_index", 0),
                score=point.score,
                channel=channel,
                metadata=payload.get("metadata", {}),
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
