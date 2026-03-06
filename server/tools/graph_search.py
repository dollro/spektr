"""Knowledge graph search tool for MCP server.

Engine-agnostic: dispatches to whichever GraphEngine is configured
via the GRAPH_ENGINE setting.
"""

from __future__ import annotations

import logging

from ingestion.graph_engine import get_graph_engine

logger = logging.getLogger(__name__)


async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
) -> list[dict]:  # type: ignore[type-arg]
    """Search the knowledge graph for entities and relationships.

    Uses the configured graph engine's search method.

    Args:
        query: Search text.
        search_type: 'entity' (default). Reserved for future modes.
        limit: Maximum results (default 10).
    """
    if not query or not query.strip():
        return []
    if search_type != "entity":
        raise ValueError(
            f"search_type='{search_type}' is not yet implemented. Use 'entity' instead."
        )
    limit = max(1, min(limit, 100))

    try:
        engine = get_graph_engine()
        results = await engine.search(query, limit=limit)
        return [r.model_dump() for r in results]
    except Exception as exc:
        logger.exception("graph_search failed")
        return [
            {
                "error": f"graph_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]
