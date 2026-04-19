"""Dense vector search tool for MCP server.

Embeds a query and searches the Qdrant dense collection.
When session_id is provided, runs dual queries for transcript + KB.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION
from config.settings import settings
from ingestion.embedder import Embedder, create_embedder
from server.models import SearchResult

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


async def vector_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
    *,
    _skip_rerank: bool = False,
) -> list[dict]:  # type: ignore[type-arg]
    """Search documents by semantic similarity.

    When session_id is provided, runs two parallel queries:
    1. Transcript chunks from the active session
    2. Bulk KB results (excluding live data)

    Args:
        query: Natural language search query.
        limit: Maximum number of results (default 10).
        content_type: Optional MIME type filter.
        source_file: Optional source file name filter.
        session_id: Optional session ID for live meeting context.
        _skip_rerank: Internal flag — skip reranking.
    """
    if not query or not query.strip():
        return []
    limit = max(1, min(limit, 100))

    try:
        query_vector = await _get_embedder().embed_text_query(query)
        qdrant = _get_qdrant_client()

        if session_id is not None:
            return _dual_query(
                qdrant, query_vector, session_id, limit, content_type, source_file
            )

        # Standard single-query path (no session)
        conditions: list[models.FieldCondition] = []
        if content_type is not None:
            conditions.append(
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value=content_type),
                )
            )
        if source_file is not None:
            conditions.append(
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file),
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

        response = qdrant.query_points(
            collection_name=DENSE_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results: list[dict] = []  # type: ignore[type-arg]
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    score=point.score,
                    text=payload.get("text_content", payload.get("text", "")),
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 0),
                    content_type=payload.get("content_type", ""),
                    metadata=payload.get("metadata", {}),
                ).model_dump()
            )

        if settings.rerank_enabled and results and not _skip_rerank:
            from server.tools.reranker import rerank

            results = await rerank(query, results, top_k=limit)

        return results
    except Exception as exc:
        logger.exception("vector_search failed")
        return [
            {
                "error": f"vector_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]


def _dual_query(
    qdrant: QdrantClient,
    query_vector: list[float],
    session_id: str,
    limit: int,
    content_type: str | None,
    source_file: str | None,
) -> list[dict]:  # type: ignore[type-arg]
    """Run dual queries: transcript (by session) + KB (excluding live)."""
    transcript_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            ),
        ],
    )

    kb_conditions: list[models.Condition] = [
        models.FieldCondition(
            key="is_live",
            match=models.MatchValue(value=False),
        ),
    ]
    if content_type is not None:
        kb_conditions.append(
            models.FieldCondition(
                key="content_type",
                match=models.MatchValue(value=content_type),
            )
        )
    if source_file is not None:
        kb_conditions.append(
            models.FieldCondition(
                key="source_file",
                match=models.MatchValue(value=source_file),
            )
        )

    kb_filter = models.Filter(
        should=[
            models.Filter(must=kb_conditions),
            # Also match points without is_live field (bulk KB)
            models.Filter(
                must=[
                    models.IsNullCondition(
                        is_null=models.PayloadField(key="is_live"),
                    ),
                ],
            ),
        ],
    )

    transcript_resp = qdrant.query_points(
        collection_name=DENSE_COLLECTION,
        query=query_vector,
        query_filter=transcript_filter,
        limit=limit,
        with_payload=True,
    )
    kb_resp = qdrant.query_points(
        collection_name=DENSE_COLLECTION,
        query=query_vector,
        query_filter=kb_filter,
        limit=limit,
        with_payload=True,
    )

    results: list[dict] = []  # type: ignore[type-arg]

    # Transcript results sorted by timestamp
    transcript_points = sorted(
        transcript_resp.points,
        key=lambda p: (p.payload or {}).get("timestamp", ""),
    )
    for point in transcript_points:
        payload = point.payload or {}
        results.append(
            SearchResult(
                score=point.score,
                text=payload.get("text_content", ""),
                source_file=payload.get("source_file", ""),
                page_number=payload.get("page_number", 0),
                content_type=payload.get("content_type", ""),
                metadata={
                    **payload.get("metadata", {}),
                    "source_type": "transcript",
                    "speaker": payload.get("speaker"),
                    "timestamp": payload.get("timestamp"),
                },
            ).model_dump()
        )

    # KB results sorted by score
    for point in kb_resp.points:
        payload = point.payload or {}
        results.append(
            SearchResult(
                score=point.score,
                text=payload.get("text_content", payload.get("text", "")) or "",
                source_file=payload.get("source_file", ""),
                page_number=payload.get("page_number", 0),
                content_type=payload.get("content_type", ""),
                metadata=payload.get("metadata", {}),
            ).model_dump()
        )

    return results
