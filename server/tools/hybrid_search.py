"""Combined vector + knowledge graph search tool.

Runs dense vector search and graph search in parallel,
returning results from both for comprehensive retrieval.
"""

from __future__ import annotations

import asyncio
import logging

from config.settings import settings
from server.tools.graph_search import graph_search
from server.tools.vector_search import vector_search

logger = logging.getLogger(__name__)


async def hybrid_search(
    query: str,
    limit: int = 10,
) -> dict:  # type: ignore[type-arg]
    """Combined vector and knowledge graph search.

    Runs semantic vector search and graph entity search in
    parallel, returning results from both sources. Handles
    partial failures gracefully.

    Args:
        query: Natural language search query.
        limit: Max results per backend (default 10).

    Returns:
        Dict with vector_results, graph_results, query,
        strategy, and errors list.
    """
    if not query or not query.strip():
        return {
            "vector_results": [],
            "graph_results": [],
            "query": query,
            "strategy": "parallel",
        }
    limit = max(1, min(limit, 100))

    vector_task = asyncio.create_task(
        vector_search(query, limit=limit, _skip_rerank=True)
    )
    graph_task = asyncio.create_task(graph_search(query, limit=limit))

    vector_results: list[dict] = []  # type: ignore[type-arg]
    graph_results: list[dict] = []  # type: ignore[type-arg]
    errors: list[str] = []

    try:
        vector_results = await vector_task
    except Exception as exc:
        logger.exception("Vector search failed in hybrid_search")
        vector_results = [{"error": "Vector search unavailable"}]
        errors.append(f"vector_search: {exc}")

    try:
        graph_results = await graph_task
    except Exception as exc:
        logger.exception("Graph search failed in hybrid_search")
        graph_results = [{"error": "Graph search unavailable"}]
        errors.append(f"graph_search: {exc}")

    if settings.rerank_enabled and vector_results:
        has_error = any("error" in r for r in vector_results)
        if not has_error:
            from server.tools.reranker import rerank

            try:
                vector_results = await rerank(query, vector_results, top_k=limit)
            except Exception as exc:
                logger.warning("Rerank failed in hybrid: %s", exc)

    # Deduplicate: remove graph facts whose source already appears
    # in vector results to avoid redundant information
    has_vector_error = any("error" in r for r in vector_results)
    has_graph_error = any("error" in r for r in graph_results)
    if not has_vector_error and not has_graph_error:
        vector_sources = {
            r.get("source_file")
            for r in vector_results
            if r.get("source_file")
        }
        graph_results = [
            g for g in graph_results
            if g.get("source") not in vector_sources
        ]

    result: dict = {  # type: ignore[type-arg]
        "vector_results": vector_results,
        "graph_results": graph_results,
        "query": query,
        "strategy": "parallel",
    }
    if errors:
        result["errors"] = errors
    return result
