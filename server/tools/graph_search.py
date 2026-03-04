"""Knowledge graph search tool for MCP server.

Searches the Neo4j knowledge graph via Graphiti for entities,
relationships, and temporal metadata.
"""

from __future__ import annotations

import logging

from ingestion.graphiti_client import get_graphiti
from server.models import GraphFact

logger = logging.getLogger(__name__)


async def _search_entities(
    query: str,
    limit: int,
) -> list[dict]:  # type: ignore[type-arg]
    """Search Graphiti for relevant facts and entities."""
    client = await get_graphiti()
    edges = await client.search(query)

    results: list[dict] = []  # type: ignore[type-arg]
    for edge in edges[:limit]:
        results.append(
            GraphFact(
                fact=edge.fact,
                source=edge.source_description,
                created_at=str(edge.created_at),
                expired_at=(
                    str(edge.expired_at) if edge.expired_at else None
                ),
            ).model_dump()
        )
    return results


async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
) -> list[dict]:  # type: ignore[type-arg]
    """Search the knowledge graph for entities and relationships.

    Uses Graphiti's semantic search to find relevant facts,
    entities, and their temporal metadata.

    Args:
        query: Search text.
        search_type: 'entity' (default). Reserved for future modes.
        limit: Maximum results (default 10).
    """
    if not query or not query.strip():
        return []
    if search_type != "entity":
        raise ValueError(
            f"search_type='{search_type}' is not yet implemented."
            " Use 'entity' instead."
        )
    limit = max(1, min(limit, 100))

    try:
        return await _search_entities(query, limit)
    except Exception as exc:
        logger.exception("graph_search failed")
        return [
            {
                "error": f"graph_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]
