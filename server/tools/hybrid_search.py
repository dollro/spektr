"""Combined vector + knowledge graph search tool.

Runs dense vector search and graph search in parallel,
returning results from both for comprehensive retrieval.
When session_id is provided, separates transcript results
from KB results.
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
    session_id: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """Combined vector and knowledge graph search.

    Runs semantic vector search and graph entity search in
    parallel, returning results from both sources.

    When session_id is provided, transcript results are separated
    from KB results for clearer presentation to the LLM.

    Args:
        query: Natural language search query.
        limit: Max results per backend (default 10).
        session_id: Optional session ID for live meeting context.
    """
    if not query or not query.strip():
        return {
            "vector_results": [],
            "graph_results": [],
            "transcript_results": [],
            "query": query,
            "session_id": session_id,
            "strategy": "parallel",
        }
    limit = max(1, min(limit, 100))

    vector_task = asyncio.create_task(
        vector_search(query, limit=limit, session_id=session_id, _skip_rerank=True)
    )
    graph_task = asyncio.create_task(graph_search(query, limit=limit, session_id=session_id))

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

    # Separate transcript from KB results when session is active
    transcript_results: list[dict] = []  # type: ignore[type-arg]
    kb_results: list[dict] = []  # type: ignore[type-arg]

    has_vector_error = any("error" in r for r in vector_results)
    if session_id and not has_vector_error:
        for r in vector_results:
            meta = r.get("metadata", {})
            if meta.get("source_type") == "transcript":
                transcript_results.append(r)
            else:
                kb_results.append(r)
    else:
        kb_results = vector_results

    if settings.rerank_enabled and kb_results:
        has_error = any("error" in r for r in kb_results)
        if not has_error:
            from server.tools.reranker import rerank

            try:
                kb_results = await rerank(query, kb_results, top_k=limit)
            except Exception as exc:
                logger.warning("Rerank failed in hybrid: %s", exc)

    # Deduplicate graph facts
    has_graph_error = any("error" in r for r in graph_results)
    if not has_vector_error and not has_graph_error:
        vector_sources = {r.get("source_file") for r in kb_results if r.get("source_file")}
        graph_results = [g for g in graph_results if g.get("source") not in vector_sources]

    result: dict = {  # type: ignore[type-arg]
        "vector_results": kb_results,
        "transcript_results": transcript_results,
        "graph_results": graph_results,
        "query": query,
        "session_id": session_id,
        "strategy": "parallel",
    }
    if errors:
        result["errors"] = errors
    return result
