"""LLM-augmented fused search.

Same core as multi_search — dense + sparse -> RRF -> rerank — wrapped in two
extra stages: query decomposition before retrieval, and a relevance-gated
single retry after reranking. Returns the identical schema to multi_search.

BREAKING CHANGE: this tool previously returned
{vector_results, graph_results, live_results}. It now returns a single ranked
`results` list plus `graph_facts`. See docs/server/search-tools.md.
"""

from __future__ import annotations

import asyncio
import logging

from retrieval.pipeline import smart_pipeline
from server.tools.graph_search import graph_search
from server.tools.multi_search import _empty, run_graph, shape_response

logger = logging.getLogger(__name__)


async def hybrid_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """Fused search with query decomposition and a relevance-gated retry.

    Splits multi-part questions into sub-queries, retrieves and fuses across
    all of them, reranks against the original query, and widens the candidate
    pool once if the best result is weak.

    Costs one cheap LLM call for decomposition. Use multi_search when that
    cost or latency is unwelcome — the schemas are identical.

    Args:
        query: Natural language search query.
        limit: Max results (default 10, capped at 100).
        content_type: Optional MIME/content-type filter.
        source_file: Optional source file filter.
        session_id: Optional live-session ID.
    """
    if not query or not query.strip():
        return _empty(query, session_id)
    limit = max(1, min(limit, 100))

    pipeline_task = asyncio.create_task(
        smart_pipeline(
            query=query,
            limit=limit,
            content_type=content_type,
            source_file=source_file,
            session_id=session_id,
        )
    )
    graph_task = asyncio.create_task(run_graph(graph_search, query, limit, session_id))

    output = await pipeline_task
    graph_facts, graph_failed = await graph_task

    degraded = list(output.degraded)
    if graph_failed:
        degraded.append("graph")

    return shape_response(
        output, graph_facts, query, session_id, degraded, include_llm_fields=True
    )
