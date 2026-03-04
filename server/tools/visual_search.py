"""ColBERT multi-vector visual search tool for MCP server.

Embeds a query via Jina v4 ColBERT and searches the Qdrant
multi-vector collection for visually similar document pages.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from config.constants import MULTIVEC_COLLECTION
from config.settings import settings
from ingestion.embedder import JinaV4Embedder
from server.models import VisualSearchResult

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None
_embedder: JinaV4Embedder | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def _get_embedder() -> JinaV4Embedder:
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = JinaV4Embedder()
    return _embedder


async def visual_search(
    query: str,
    limit: int = 5,
) -> list[dict]:  # type: ignore[type-arg]
    """Search documents by visual similarity using ColBERT.

    Finds document pages whose visual layout matches the query,
    useful for charts, diagrams, tables, and formatted content.

    Args:
        query: Natural language search query.
        limit: Maximum number of results (default 5).

    Returns:
        List of visual search results with score,
        source_file, page_number, content_type, s3_key,
        and metadata.
    """
    if not query or not query.strip():
        return []
    limit = max(1, min(limit, 100))

    try:
        query_vectors = await _get_embedder().embed_query_multi_vector(query)

        response = _get_qdrant_client().query_points(
            collection_name=MULTIVEC_COLLECTION,
            query=query_vectors,
            using="colbert",
            limit=limit,
            with_payload=True,
        )

        results: list[dict] = []  # type: ignore[type-arg]
        for point in response.points:
            payload = point.payload or {}
            results.append(
                VisualSearchResult(
                    score=point.score,
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 0),
                    content_type=payload.get("content_type", ""),
                    s3_key=payload.get("s3_key", ""),
                    metadata=payload.get("metadata", {}),
                ).model_dump()
            )

        if settings.vlm_generation_enabled and results:
            from server.tools.vlm_generator import (
                generate_visual_answer,
            )

            vlm_answer = await generate_visual_answer(query, results)
            if vlm_answer is not None:
                results.insert(
                    0,
                    {
                        "type": "vlm_answer",
                        "answer": vlm_answer,
                        "query": query,
                    },
                )

        return results
    except Exception as exc:
        logger.exception("visual_search failed")
        return [
            {
                "error": f"visual_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]
