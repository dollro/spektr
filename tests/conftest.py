"""Shared test fixtures for Spektr test suite."""

from __future__ import annotations

import os

# Set required env vars before any config imports trigger Settings validation.
os.environ.setdefault("JINA_API_KEY", "test-jina-key")
os.environ.setdefault("JINA_MODEL", "jina-clip-v4")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.constants import (
    DENSE_COLLECTION,
    DENSE_DIM,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
)
from ingestion.entity_extractor import ExtractionResult

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# --- Sample file fixtures ---


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.txt").read_bytes()


@pytest.fixture
def sample_txt_name() -> str:
    return "sample.txt"


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.pdf").read_bytes()


@pytest.fixture
def sample_pdf_name() -> str:
    return "sample.pdf"


@pytest.fixture
def sample_png_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.png").read_bytes()


@pytest.fixture
def sample_png_name() -> str:
    return "sample.png"


# --- Mock embedder ---


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock JinaV4Embedder returning deterministic vectors."""
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(
        return_value=[[0.1] * DENSE_DIM],
    )
    embedder.embed_text_query = AsyncMock(
        return_value=[0.2] * DENSE_DIM,
    )
    embedder.embed_image = AsyncMock(
        return_value=[0.3] * DENSE_DIM,
    )
    embedder.embed_multi_vector = AsyncMock(
        return_value=[[0.4] * MULTIVEC_DIM] * 10,
    )
    embedder.embed_query_multi_vector = AsyncMock(
        return_value=[[0.5] * MULTIVEC_DIM] * 5,
    )
    embedder.close = AsyncMock()
    return embedder


# --- Mock LLM client ---


def _make_mock_extraction() -> ExtractionResult:
    """Return a deterministic ExtractionResult for testing."""
    return ExtractionResult(
        entities=[
            {
                "name": "Test Entity",
                "type": "CONCEPT",
                "description": "A test entity",
            },
        ],
        relationships=[],
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client returning deterministic extraction JSON."""
    import json

    client = MagicMock()
    result = _make_mock_extraction()
    client.chat = AsyncMock(
        return_value=json.dumps(result.model_dump()),
    )
    return client


# --- Qdrant fixtures ---


@pytest.fixture
def qdrant_client():  # type: ignore[no-untyped-def]
    """Real Qdrant client for integration tests.

    Clears test collections before and after each test.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")

    # Clean before
    for name in (DENSE_COLLECTION, MULTIVEC_COLLECTION):
        if client.collection_exists(name):
            client.delete_collection(name)

    # Provision fresh collections
    from ingestion.qdrant_setup import ensure_collections

    ensure_collections(client)

    yield client

    # Clean after
    for name in (DENSE_COLLECTION, MULTIVEC_COLLECTION):
        if client.collection_exists(name):
            client.delete_collection(name)

    client.close()


# --- Neo4j fixtures ---


@pytest.fixture
async def neo4j_driver():  # type: ignore[no-untyped-def]
    """Real Neo4j driver for integration tests.

    Clears all test data before and after each test.
    """
    from neo4j import AsyncGraphDatabase

    from config.settings import settings
    from ingestion.neo4j_setup import create_neo4j_schema

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    # Clean and provision
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await create_neo4j_schema(driver)

    yield driver

    # Clean after
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await driver.close()


# --- Agent fixtures ---


@pytest.fixture
async def rag_agent():  # type: ignore[no-untyped-def]
    """Create a RAG agent with a dummy API key for testing.

    Returns (agent, server) tuple. Use agent.override() to
    swap model and toolsets in tests.
    """
    import os

    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

    from agent.agent import create_rag_agent

    agent, server = await create_rag_agent()
    return agent, server
