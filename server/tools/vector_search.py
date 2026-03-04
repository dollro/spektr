"""Dense vector search tool for MCP server.

Embeds a query via Jina v4 and searches the Qdrant dense collection.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION
from config.settings import settings
from ingestion.embedder import JinaV4Embedder
from server.models import SearchResult

logger = logging.getLogger(__name__)

_qdrant_client = QdrantClient(url=settings.qdrant_url)
_embedder = JinaV4Embedder()


async def vector_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    *,
    _skip_rerank: bool = False,
) -> list[dict]:  # type: ignore[type-arg]
    """Search documents by semantic similarity.

    Embeds the query with Jina v4 and performs nearest-neighbor
    search against the Qdrant dense collection. Supports filtering
    by content type and source file name.

    Args:
        query: Natural language search query.
        limit: Maximum number of results (default 10).
        content_type: Optional MIME type filter.
        source_file: Optional source file name filter.
        _skip_rerank: Internal flag — skip reranking so callers
            like hybrid_search can apply a single rerank pass.

    Returns:
        List of search results with score, text, source_file,
        page_number, content_type, and metadata.
    """
    if not query or not query.strip():
        return []
    limit = max(1, min(limit, 100))

    try:
        query_vector = await _embedder.embed_text_query(query)

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

        query_filter = models.Filter(must=conditions) if conditions else None

        response = _qdrant_client.query_points(
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
                    text=payload.get("text", ""),
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
