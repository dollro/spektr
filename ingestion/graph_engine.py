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

    Extracts entities and relationships using a 205MB local model,
    then writes directly to Neo4j via Cypher MERGE statements.
    Zero LLM API calls.
    """

    def __init__(self) -> None:
        from gliner2 import GLiNER2
        from neo4j import AsyncGraphDatabase

        from config.constants import ENTITY_TYPES, RELATIONSHIP_TYPES
        from config.settings import settings

        self._extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        # Map SCREAMING_CASE to lowercase for GLiNER2 schema
        entity_map = {t.lower(): t for t in ENTITY_TYPES}
        relation_map = {t.lower(): t for t in RELATIONSHIP_TYPES}
        self._entity_map = entity_map
        self._relation_map = relation_map

        self._schema = (
            self._extractor.create_schema()
            .entities(list(entity_map.keys()))
            .relations(list(relation_map.keys()))
        )

    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None:
        if not chunks:
            return

        for chunk in chunks:
            text = chunk.contextualized_text or chunk.text
            result = self._extractor.extract(text, self._schema)

            entities = result.get("entities", {})
            relations = result.get("relation_extraction", {})

            async with self._driver.session() as session:
                # Upsert entities
                for entity_type_lower, names in entities.items():
                    entity_type = self._entity_map.get(
                        entity_type_lower, entity_type_lower.upper()
                    )
                    for name in names:
                        normalized = name.strip().title()
                        if not normalized:
                            continue
                        await session.run(
                            "MERGE (e:Entity {name: $name, type: $type}) "
                            "ON CREATE SET e.first_seen = datetime() "
                            "SET e.last_seen = datetime(), "
                            "e.source = $source",
                            name=normalized,
                            type=entity_type,
                            source=source_key,
                        )

                # Upsert relationships
                for rel_type_lower, pairs in relations.items():
                    rel_type = self._relation_map.get(rel_type_lower, rel_type_lower.upper())
                    for head, tail in pairs:
                        head_norm = head.strip().title()
                        tail_norm = tail.strip().title()
                        if not head_norm or not tail_norm:
                            continue
                        await session.run(
                            "MATCH (s:Entity {name: $source}) "
                            "MATCH (t:Entity {name: $target}) "
                            "CALL apoc.merge.relationship("
                            "s, $relation, $props, {}, t, {}"
                            ") YIELD rel RETURN rel",
                            source=head_norm,
                            target=tail_norm,
                            relation=rel_type,
                            props={
                                "source": source_key,
                                "confidence": 1.0,
                            },
                        )

        logger.info(
            "GLiNER extracted entities from %d chunks for %s",
            len(chunks),
            source_key,
        )

    async def search(self, query: str, limit: int = 10) -> list[GraphFact]:
        raise NotImplementedError("GLiNEREngine.search not yet implemented")

    async def close(self) -> None:
        await self._driver.close()


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
