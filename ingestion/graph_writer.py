"""Knowledge graph writers for Neo4j.

GraphitiWriter — primary writer using Graphiti for temporal
    knowledge graph with automatic entity/relationship extraction.
_LegacyGraphWriter — deprecated raw-Cypher writer kept for
    backward compatibility during transition.
"""

from __future__ import annotations

import logging
from datetime import datetime

from ingestion.graphiti_client import get_graphiti

logger = logging.getLogger(__name__)


class GraphitiWriter:
    """Ingest text chunks as Graphiti episodes.

    Graphiti handles entity extraction, relationship discovery,
    and temporal metadata internally via its LLM pipeline.
    """

    async def ingest_chunk(
        self,
        chunk_text: str,
        source_key: str,
        page_number: int,
        chunk_index: int,
        reference_time: datetime | None = None,
    ) -> None:
        """Add a text chunk as a Graphiti episode."""
        client = await get_graphiti()
        episode_name = f"{source_key}:p{page_number}:c{chunk_index}"
        ref_time = reference_time or datetime.now()

        logger.debug(
            "Ingesting episode %s",
            episode_name,
            extra={"source_key": source_key},
        )
        await client.add_episode(
            name=episode_name,
            episode_body=chunk_text,
            source_description=source_key,
            reference_time=ref_time,
        )

    async def close(self) -> None:
        """Close the underlying Graphiti client."""
        from ingestion.graphiti_client import close_graphiti

        await close_graphiti()


# ------------------------------------------------------------------
# Legacy writer (deprecated — kept for transition / tests)
# ------------------------------------------------------------------

from neo4j import AsyncGraphDatabase  # noqa: E402
from neo4j.exceptions import (  # noqa: E402
    ServiceUnavailable,
    TransientError,
)
from tenacity import (  # noqa: E402
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings  # noqa: E402
from ingestion.entity_extractor import Entity, ExtractionResult  # noqa: E402

_NEO4J_RETRY = retry(
    wait=wait_exponential(multiplier=1, min=1, max=15),
    stop=stop_after_attempt(settings.max_retries),
    retry=retry_if_exception_type(
        (ServiceUnavailable, TransientError, OSError),
    ),
    before_sleep=lambda rs: logger.warning(
        "Neo4j retry attempt %d after %s",
        rs.attempt_number,
        rs.outcome.exception(),
    ),
)


class _LegacyGraphWriter:
    """Deprecated: raw Cypher upserts to Neo4j.

    Use GraphitiWriter instead. This class is kept temporarily
    so existing tests continue to work during the migration.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.driver = AsyncGraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(
                user or settings.neo4j_user,
                password or settings.neo4j_password,
            ),
        )

    async def close(self) -> None:
        await self.driver.close()

    @_NEO4J_RETRY
    async def upsert_document(
        self, s3_key: str, **properties: object
    ) -> None:
        """MERGE Document node on s3_key, SET all props."""
        async with self.driver.session() as session:
            await session.run(
                "MERGE (d:Document {s3_key: $s3_key}) "
                "SET d.filename = $filename, "
                "d.mime_type = $mime_type, "
                "d.ingested_at = datetime(), "
                "d.page_count = $page_count, "
                "d.source_bucket = $source_bucket",
                s3_key=s3_key,
                filename=properties.get("filename", ""),
                mime_type=properties.get("mime_type", ""),
                page_count=properties.get("page_count", 0),
                source_bucket=properties.get("source_bucket", ""),
            )

    @_NEO4J_RETRY
    async def upsert_chunk(
        self,
        chunk_id: str,
        text_preview: str,
        page_number: int,
        s3_key: str,
    ) -> None:
        """MERGE Chunk node, create HAS_CHUNK rel."""
        async with self.driver.session() as session:
            await session.run(
                "MERGE (c:Chunk {id: $chunk_id}) "
                "SET c.text_preview = $text_preview, "
                "c.page_number = $page_number "
                "WITH c "
                "MATCH (d:Document {s3_key: $s3_key}) "
                "MERGE (d)-[:HAS_CHUNK {page_number: $page_number}]->(c)",
                chunk_id=chunk_id,
                text_preview=text_preview,
                page_number=page_number,
                s3_key=s3_key,
            )

    @_NEO4J_RETRY
    async def upsert_entity(self, entity: Entity) -> None:
        """MERGE Entity on (name, type), SET description."""
        async with self.driver.session() as session:
            await session.run(
                "MERGE (e:Entity {name: $name, type: $type}) "
                "SET e.description = $description, "
                "e.last_seen = datetime() "
                "ON CREATE SET e.first_seen = datetime()",
                name=entity.name,
                type=entity.type,
                description=entity.description,
            )

    @_NEO4J_RETRY
    async def upsert_relationship(
        self,
        source: str,
        target: str,
        relation: str,
        properties: dict[str, object],
    ) -> None:
        """APOC merge relationship with dynamic type."""
        async with self.driver.session() as session:
            await session.run(
                "MATCH (s:Entity {name: $source}) "
                "MATCH (t:Entity {name: $target}) "
                "CALL apoc.merge.relationship("
                "s, $relation, $props, {}, t, {}"
                ") YIELD rel RETURN rel",
                source=source,
                target=target,
                relation=relation,
                props=properties,
            )

    @_NEO4J_RETRY
    async def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_name: str,
        confidence: float = 1.0,
    ) -> None:
        """MERGE MENTIONS relationship."""
        async with self.driver.session() as session:
            await session.run(
                "MATCH (c:Chunk {id: $chunk_id}) "
                "MATCH (e:Entity {name: $entity_name}) "
                "MERGE (c)-[r:MENTIONS]->(e) "
                "SET r.confidence = $confidence",
                chunk_id=chunk_id,
                entity_name=entity_name,
                confidence=confidence,
            )

    async def write_extraction_result(
        self,
        s3_key: str,
        chunk_id: str,
        extraction_result: ExtractionResult,
    ) -> None:
        """Orchestrate upserts for one chunk's extraction."""
        for entity in extraction_result.entities:
            await self.upsert_entity(entity)

        for rel in extraction_result.relationships:
            await self.upsert_relationship(
                source=rel.source,
                target=rel.target,
                relation=rel.relation,
                properties=rel.properties,
            )

        for entity in extraction_result.entities:
            await self.link_chunk_to_entity(chunk_id, entity.name)


# Backward-compatible alias so existing imports still work
GraphWriter = _LegacyGraphWriter
