from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase

from config.settings import settings
from ingestion.entity_extractor import Entity, ExtractionResult

logger = logging.getLogger(__name__)


class GraphWriter:
    """Upsert documents, chunks, entities, and relationships to Neo4j."""

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

    async def upsert_document(self, s3_key: str, **properties: object) -> None:
        """MERGE Document node on s3_key, SET all properties."""
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

    async def upsert_chunk(
        self,
        chunk_id: str,
        text_preview: str,
        page_number: int,
        s3_key: str,
    ) -> None:
        """MERGE Chunk node, create HAS_CHUNK rel to Document."""
        async with self.driver.session() as session:
            await session.run(
                "MERGE (c:Chunk {id: $chunk_id}) "
                "SET c.text_preview = $text_preview, "
                "c.page_number = $page_number "
                "WITH c "
                "MATCH (d:Document {s3_key: $s3_key}) "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                chunk_id=chunk_id,
                text_preview=text_preview,
                page_number=page_number,
                s3_key=s3_key,
            )

    async def upsert_entity(self, entity: Entity) -> None:
        """MERGE Entity on (name, type), SET description + timestamps."""
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

    async def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_name: str,
        confidence: float = 1.0,
    ) -> None:
        """MERGE MENTIONS relationship between Chunk and Entity."""
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
        """Orchestrate upserts for one chunk's extraction result."""
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
