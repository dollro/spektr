from __future__ import annotations

import pytest

from ingestion.neo4j_setup import create_neo4j_schema, get_driver


@pytest.mark.integration
async def test_create_schema_creates_constraints() -> None:
    """Schema creation adds all expected constraints."""
    driver = get_driver()
    try:
        await create_neo4j_schema(driver)

        async with driver.session() as session:
            result = await session.run("SHOW CONSTRAINTS")
            records = [r async for r in result]
            names = {r["name"] for r in records}

        assert "doc_unique" in names
        assert "entity_unique" in names
        assert "chunk_unique" in names
    finally:
        await driver.close()


@pytest.mark.integration
async def test_apoc_available() -> None:
    """APOC plugin is available (verified during schema creation)."""
    driver = get_driver()
    try:
        # create_neo4j_schema raises if APOC is missing
        await create_neo4j_schema(driver)

        async with driver.session() as session:
            result = await session.run("RETURN apoc.version() AS version")
            record = await result.single()
        assert record is not None
        assert record["version"]
    finally:
        await driver.close()


@pytest.mark.integration
async def test_schema_idempotent() -> None:
    """Calling create_neo4j_schema twice does not raise."""
    driver = get_driver()
    try:
        await create_neo4j_schema(driver)
        await create_neo4j_schema(driver)  # should not error
    finally:
        await driver.close()
