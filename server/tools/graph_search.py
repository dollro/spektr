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
    """
    if not query or not query.strip():
        return []
    if search_type != "entity":
        raise ValueError(
            f"search_type='{search_type}' is not yet implemented. Use 'entity' instead."
        )
    limit = max(1, min(limit, 100))

    try:
        results: list[dict] = []  # type: ignore[type-arg]

        if session_id is not None:
            # Query Graphiti for temporal session facts
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

        # Query the configured graph engine (GLiNER2 or Graphiti)
        engine = get_graph_engine()
        engine_results = await engine.search(query, limit=limit)
        results.extend(r.model_dump() for r in engine_results)

        return results
    except Exception as exc:
        logger.exception("graph_search failed")
        return [
            {
                "error": f"graph_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]
