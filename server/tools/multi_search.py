"""Deterministic fused search — dense + sparse -> RRF -> rerank.

No LLM calls anywhere in this path. Use this when latency and cost matter
more than recall on hard multi-part questions; use hybrid_search otherwise.
Both tools return the identical schema, so callers can swap freely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from retrieval.pipeline import PipelineOutput, fast_pipeline
from server.tools.graph_search import graph_search

logger = logging.getLogger(__name__)


def _empty(query: str, session_id: str | None) -> dict:  # type: ignore[type-arg]
    return {
        "results": [],
        "graph_facts": [],
        "live_results": [],
        "query": query,
        "session_id": session_id,
    }


def shape_response(
    output: PipelineOutput,
    graph_facts: list[dict],  # type: ignore[type-arg]
    query: str,
    session_id: str | None,
    degraded: list[str],
    *,
    include_llm_fields: bool,
) -> dict:  # type: ignore[type-arg]
    """Build the shared response dict from a pipeline result.

    Shared by multi_search and hybrid_search so the two schemas cannot drift.
    """
    live: list[dict] = []  # type: ignore[type-arg]
    kb: list[dict] = []  # type: ignore[type-arg]
    for item in output.results:
        target = live if session_id and item.metadata.get("source_type") == "live" else kb
        target.append(item.model_dump())

    response: dict = {  # type: ignore[type-arg]
        "results": kb,
        "graph_facts": graph_facts,
        "live_results": live,
        "query": query,
        "session_id": session_id,
    }
    if include_llm_fields:
        response["sub_queries"] = output.sub_queries
        response["retried"] = output.retried
    if degraded:
        response["degraded"] = degraded
    # Total retrieval failure is louder than partial degradation — callers
    # that ignore `degraded` must still not mistake this for "no matches".
    if "dense" in degraded and "sparse" in degraded:
        response["error"] = "All retrieval channels unavailable"
    return response


GraphSearchFn = Callable[..., Awaitable[list]]  # type: ignore[type-arg]


async def run_graph(
    search_fn: GraphSearchFn, query: str, limit: int, session_id: str | None
) -> tuple[list, bool]:  # type: ignore[type-arg]
    """Query the graph, reporting failure rather than raising.

    Takes the search callable as a parameter — rather than reaching for the
    `graph_search` name bound in this module — so that callers (e.g.
    hybrid_search) which patch their own module-level import actually take
    effect. A shared closure over this module's binding would silently
    ignore patches applied to the caller's module.
    """
    try:
        return await search_fn(query, limit=limit, session_id=session_id), False
    except Exception:
        logger.exception("graph_search failed")
        return [], True


async def multi_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """Fused dense + sparse search with reranking. No LLM calls.

    Runs lexical and semantic retrieval concurrently, merges them with
    reciprocal rank fusion, and reranks the result. Graph facts are returned
    separately as supporting context, not fused into the ranking.

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
        fast_pipeline(
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
        output, graph_facts, query, session_id, degraded, include_llm_fields=False
    )
