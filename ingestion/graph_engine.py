"""Pluggable graph engine: protocol, factory, and implementations.

Switch between engines via the GRAPH_ENGINE setting:
  - "graphiti" — LLM-based extraction via Graphiti (slow, rich)
  - "gliner"  — local GLiNER2 model on CPU (fast, zero API cost)
"""

from __future__ import annotations

import logging
from typing import Protocol

from ingestion.file_processor import TextChunk
from server.models import GraphFact

logger = logging.getLogger(__name__)


class GraphEngine(Protocol):
    """Unified interface for knowledge graph ingestion and search."""

    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None: ...

    async def search(self, query: str, limit: int = 10) -> list[GraphFact]: ...

    async def close(self) -> None: ...


class GraphitiEngine:
    """Graph engine backed by Graphiti (LLM-based extraction)."""

    def __init__(self) -> None:
        from ingestion.graph_writer import GraphitiWriter

        self._writer = GraphitiWriter()

    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None:
        await self._writer.ingest_bulk(chunks=chunks, source_key=source_key)

    async def search(self, query: str, limit: int = 10) -> list[GraphFact]:
        from ingestion.graphiti_client import get_graphiti

        client = await get_graphiti()
        edges = await client.search(query)
        return [
            GraphFact(
                fact=edge.fact,
                source=edge.source_description,
                created_at=str(edge.created_at),
                expired_at=(str(edge.expired_at) if edge.expired_at else None),
            )
            for edge in edges[:limit]
        ]

    async def close(self) -> None:
        await self._writer.close()


class GLiNEREngine:
    """Graph engine backed by GLiNER2 (local CPU extraction).

    Placeholder — full implementation in Task 5.
    """

    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None:
        raise NotImplementedError("GLiNEREngine.ingest not yet implemented")

    async def search(self, query: str, limit: int = 10) -> list[GraphFact]:
        raise NotImplementedError("GLiNEREngine.search not yet implemented")

    async def close(self) -> None:
        pass


def get_graph_engine() -> GraphEngine:
    """Factory: create graph engine based on settings."""
    from config.settings import settings

    engine = settings.graph_engine.lower()
    if engine == "graphiti":
        return GraphitiEngine()
    if engine == "gliner":
        return GLiNEREngine()
    msg = f"Unknown graph engine: {engine!r}. Use 'graphiti' or 'gliner'."
    raise ValueError(msg)
