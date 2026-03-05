from __future__ import annotations

import pytest

from ingestion.entity_extractor import (
    Entity,
    ExtractionResult,
    Relationship,
)
from ingestion.graph_writer import GraphWriter


@pytest.fixture
async def writer():
    """Create a GraphWriter connected to local Neo4j."""
    gw = GraphWriter()
    yield gw
    # Clean up test data
    async with gw.driver.session() as session:
        await session.run("MATCH (n) WHERE n:Document OR n:Chunk OR n:Entity DETACH DELETE n")
    await gw.close()


@pytest.mark.integration
async def test_upsert_document(writer: GraphWriter) -> None:
    """Upsert creates a Document node with correct properties."""
    await writer.upsert_document(
        "test/doc.pdf",
        filename="doc.pdf",
        mime_type="application/pdf",
        page_count=5,
        source_bucket="my-bucket",
    )
    async with writer.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {source_key: $key}) RETURN d",
            key="test/doc.pdf",
        )
        record = await result.single()

    assert record is not None
    doc = record["d"]
    assert doc["filename"] == "doc.pdf"
    assert doc["mime_type"] == "application/pdf"
    assert doc["page_count"] == 5


@pytest.mark.integration
async def test_upsert_document_idempotent(
    writer: GraphWriter,
) -> None:
    """Upserting same document twice creates only one node."""
    await writer.upsert_document("test/dup.pdf", filename="dup.pdf")
    await writer.upsert_document("test/dup.pdf", filename="dup.pdf")

    async with writer.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {source_key: 'test/dup.pdf'}) RETURN count(d) AS cnt"
        )
        record = await result.single()
    assert record is not None
    assert record["cnt"] == 1


@pytest.mark.integration
async def test_upsert_chunk(writer: GraphWriter) -> None:
    """Upsert chunk creates node and HAS_CHUNK relationship."""
    await writer.upsert_document("test/c.pdf", filename="c.pdf")
    await writer.upsert_chunk(
        chunk_id="c-001",
        text_preview="Hello world",
        page_number=1,
        source_key="test/c.pdf",
    )

    async with writer.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {source_key: 'test/c.pdf'})"
            "-[:HAS_CHUNK]->(c:Chunk {id: 'c-001'}) "
            "RETURN c"
        )
        record = await result.single()
    assert record is not None
    assert record["c"]["text_preview"] == "Hello world"


@pytest.mark.integration
async def test_upsert_entity(writer: GraphWriter) -> None:
    """Upsert entity creates node with timestamps."""
    entity = Entity(
        name="Google",
        type="ORGANIZATION",
        description="Tech company.",
    )
    await writer.upsert_entity(entity)

    async with writer.driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'Google', type: 'ORGANIZATION'}) RETURN e"
        )
        record = await result.single()
    assert record is not None
    node = record["e"]
    assert node["description"] == "Tech company."
    assert node["first_seen"] is not None
    assert node["last_seen"] is not None


@pytest.mark.integration
async def test_upsert_relationship(writer: GraphWriter) -> None:
    """APOC merge creates dynamic relationship type."""
    e1 = Entity(name="Google", type="ORGANIZATION", description="Tech co.")
    e2 = Entity(name="Python", type="TECHNOLOGY", description="Language.")
    await writer.upsert_entity(e1)
    await writer.upsert_entity(e2)
    await writer.upsert_relationship("Google", "Python", "USES_TECHNOLOGY", {})

    async with writer.driver.session() as session:
        result = await session.run(
            "MATCH (s:Entity {name: 'Google'})"
            "-[r:USES_TECHNOLOGY]->"
            "(t:Entity {name: 'Python'}) "
            "RETURN r"
        )
        record = await result.single()
    assert record is not None


@pytest.mark.integration
async def test_link_chunk_to_entity(writer: GraphWriter) -> None:
    """MENTIONS relationship links chunk to entity."""
    await writer.upsert_document("test/m.pdf", filename="m.pdf")
    await writer.upsert_chunk("m-001", "preview", 1, source_key="test/m.pdf")
    entity = Entity(name="Google", type="ORGANIZATION", description="Co.")
    await writer.upsert_entity(entity)
    await writer.link_chunk_to_entity("m-001", "Google", 0.95)

    async with writer.driver.session() as session:
        result = await session.run(
            "MATCH (c:Chunk {id: 'm-001'})"
            "-[r:MENTIONS]->"
            "(e:Entity {name: 'Google'}) "
            "RETURN r.confidence AS conf"
        )
        record = await result.single()
    assert record is not None
    assert record["conf"] == pytest.approx(0.95)


@pytest.mark.integration
async def test_write_extraction_result(
    writer: GraphWriter,
) -> None:
    """End-to-end: creates entities, relationships, MENTIONS."""
    await writer.upsert_document("test/e.pdf", filename="e.pdf")
    await writer.upsert_chunk("e-001", "preview", 1, source_key="test/e.pdf")

    extraction = ExtractionResult(
        entities=[
            Entity(
                name="Apple",
                type="ORGANIZATION",
                description="Tech company.",
            ),
            Entity(
                name="Swift",
                type="TECHNOLOGY",
                description="Programming language.",
            ),
        ],
        relationships=[
            Relationship(
                source="Apple",
                target="Swift",
                relation="PRODUCES",
                properties={},
            ),
        ],
    )
    await writer.write_extraction_result("test/e.pdf", "e-001", extraction)

    async with writer.driver.session() as session:
        # Check entities exist
        result = await session.run(
            "MATCH (e:Entity) WHERE e.name IN ['Apple', 'Swift'] RETURN count(e) AS cnt"
        )
        record = await result.single()
        assert record is not None
        assert record["cnt"] == 2

        # Check PRODUCES relationship
        result = await session.run(
            "MATCH (:Entity {name: 'Apple'})"
            "-[:PRODUCES]->(:Entity {name: 'Swift'}) "
            "RETURN count(*) AS cnt"
        )
        record = await result.single()
        assert record is not None
        assert record["cnt"] == 1

        # Check MENTIONS relationships
        result = await session.run(
            "MATCH (c:Chunk {id: 'e-001'})-[:MENTIONS]->(e:Entity) RETURN count(e) AS cnt"
        )
        record = await result.single()
        assert record is not None
        assert record["cnt"] == 2
