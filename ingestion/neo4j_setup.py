from __future__ import annotations

import logging

from neo4j import AsyncDriver, AsyncGraphDatabase

from config.settings import settings

logger = logging.getLogger(__name__)


def get_driver() -> AsyncDriver:
    """Create a Neo4j async driver from settings."""
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


async def create_neo4j_schema(driver: AsyncDriver) -> None:
    """Create Neo4j constraints and verify APOC plugin.

    Idempotent — uses IF NOT EXISTS for all constraints.
    """
    async with driver.session() as session:
        await session.run(
            "CREATE CONSTRAINT doc_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.source_key IS UNIQUE"
        )
        await session.run(
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
        )
        await session.run(
            "CREATE CONSTRAINT chunk_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )

        result = await session.run("RETURN apoc.version() AS version")
        record = await result.single()
        if not record:
            raise RuntimeError("APOC plugin not available")

        logger.info(
            "Neo4j schema created, APOC version: %s",
            record["version"],
        )
