"""Knowledge graph search tool for MCP server.

Engine-agnostic: dispatches to whichever GraphEngine is configured
via the GRAPH_ENGINE setting. When session_id is provided, also
queries Graphiti for temporal session context.
"""

from __future__ import annotations

import logging

from ingestion.graph_engine import get_graph_engine
from server.models import GraphFact

logger = logging.getLogger(__name__)


async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
    session_id: str | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """Search the knowledge graph for entities and relationships.

    When session_id is provided, queries both:
    1. Graphiti (temporal facts from the live session, filtered by group_id)
    2. The configured graph engine (GLiNER2 entities from bulk KB)

    Args:
        query: Search text.
        search_type: 'entity' (default). Reserved for future modes.
        limit: Maximum results (default 10).
        session_id: Optional session ID for live session context.

    Returns a single-element list carrying an `error` key on failure. Fused
    callers must use `search_graph` instead, which raises — see its docstring.
    """
    try:
        return await search_graph(query, search_type, limit, session_id)
    except ValueError:
        # An unsupported `search_type` is the caller's bug, not a degraded
        # backend, and it raised here before this split. Reporting it as a
        # soft error dict would hide a typo behind an empty-looking result.
        raise
    except Exception as exc:
        logger.exception("graph_search failed")
        return [
            {
                "error": f"graph_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]


async def search_graph(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
    session_id: str | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """Graph search that RAISES on failure. The composable entry point.

    Split out of `graph_search` because the two callers need opposite things
    and one function cannot serve both.

    `multi_search`/`hybrid_search` reach the graph through
    `multi_search.run_graph`, which marks the channel degraded by catching an
    exception. While this function's body was inlined into `graph_search` and
    wrapped in a bare `except` that RETURNED an error dict, nothing ever
    raised — so `"graph"` could never reach `degraded`, and the error dict was
    handed back inside `graph_facts` dressed as a fact. That silently violated
    the pipeline contract in CLAUDE.md: a stage failure appends its name to
    `degraded` and the tool still returns results.

    The MCP-facing `graph_search` keeps the caught, dict-shaped response,
    because a direct tool call has no `degraded` channel to report into.
    """
    if not query or not query.strip():
        return []
    if search_type != "entity":
        raise ValueError(
            f"search_type='{search_type}' is not yet implemented. Use 'entity' instead."
        )
    limit = max(1, min(limit, 100))

    results: list[dict] = []  # type: ignore[type-arg]

    if session_id is not None:
        # Query Graphiti for temporal session facts. Kept non-fatal on
        # purpose: a session with no episodes yet is normal, and the bulk KB
        # results below are still worth returning.
        try:
            from ingestion.graphiti_client import get_graphiti

            client = await get_graphiti()
            edges = await client.search(query, group_ids=[session_id])
            for edge in edges[:limit]:
                results.append(
                    GraphFact(
                        fact=edge.fact,
                        source=edge.name,
                        created_at=str(edge.created_at),
                        expired_at=(str(edge.expired_at) if edge.expired_at else None),
                    ).model_dump()
                )
        except Exception:
            logger.exception("Graphiti search failed for session %s", session_id)

    # Query the configured graph engine (GLiNER2 or Graphiti). A failure here
    # is the whole channel failing, so it propagates.
    engine = get_graph_engine()
    engine_results = await engine.search(query, limit=limit)
    results.extend(r.model_dump() for r in engine_results)

    return results
