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

    @staticmethod
    def _merge_chunks(chunks: list[TextChunk], min_chars: int = 200) -> list[str]:
        """Merge small consecutive chunks into page-level texts.

        GLiNER needs substantial context; tiny fragments yield no entities.
        Groups by page_number, concatenates, then splits any text exceeding
        max_chars on page boundaries.
        """
        from itertools import groupby

        pages: dict[int, str] = {}
        for chunk in chunks:
            text = (chunk.contextualized_text or chunk.text).strip()
            if not text:
                continue
            page = chunk.page_number or 0
            pages[page] = f"{pages.get(page, '')} {text}".strip()

        return [text for text in pages.values() if len(text) >= min_chars]

    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None:
        if not chunks:
            return

        merged_texts = self._merge_chunks(chunks)
        if not merged_texts:
            return

        async with self._driver.session() as session:
            for text in merged_texts:
                result = self._extractor.extract(text, self._schema)

                entities = result.get("entities", {})
                relations = result.get("relation_extraction", {})

                # Upsert entities (with description from source text)
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
                            "ON CREATE SET e.first_seen = datetime(), "
                            "e.description = $description "
                            "SET e.last_seen = datetime(), "
                            "e.source = $source",
                            name=normalized,
                            type=entity_type,
                            description=text[:500],
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
        """Search Neo4j via full-text index, traverse relationships."""
        cypher = (
            "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) "
            "YIELD node AS e, score "
            "WITH e, score ORDER BY score DESC LIMIT $limit "
            "OPTIONAL MATCH (e)-[r]->(t:Entity) "
            "RETURN e.name AS entity_name, e.type AS entity_type, "
            "type(r) AS rel_type, t.name AS target_name, "
            "r.confidence AS confidence, score"
        )
        results: list[GraphFact] = []
        seen: set[str] = set()

        async with self._driver.session() as session:
            result = await session.run(cypher, query=query, limit=limit)
            records = await result.data()

        for rec in records:
            entity = rec["entity_name"]
            rel = rec.get("rel_type")
            target = rec.get("target_name")

            if rel and target:
                fact_str = f"{entity} {rel.lower().replace('_', ' ')} {target}"
                entities = [entity, target]
            else:
                fact_str = f"{entity} ({rec['entity_type']})"
                entities = [entity]

            if fact_str in seen:
                continue
            seen.add(fact_str)

            results.append(
                GraphFact(
                    fact=fact_str,
                    entities=entities,
                    relation_type=rel,
                    confidence=rec.get("confidence"),
                )
            )

        return results[:limit]

    async def close(self) -> None:
        await self._driver.close()


_engine: GraphEngine | None = None


def get_graph_engine() -> GraphEngine:
    """Return (and lazily initialise) the shared graph engine singleton."""
    global _engine  # noqa: PLW0603
    if _engine is not None:
        return _engine

    from config.settings import settings

    name = settings.graph_engine.lower()
    if name == "graphiti":
        _engine = GraphitiEngine()
    elif name == "gliner":
        _engine = GLiNEREngine()
    else:
        msg = f"Unknown graph engine: {name!r}. Use 'graphiti' or 'gliner'."
        raise ValueError(msg)
    return _engine


async def close_graph_engine() -> None:
    """Shut down the shared graph engine singleton."""
    global _engine  # noqa: PLW0603
    if _engine is not None:
        await _engine.close()
        _engine = None
