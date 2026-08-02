"""Shared test fixtures for Spektr test suite."""

from __future__ import annotations

import asyncio
import os
import pathlib
from unittest.mock import AsyncMock, MagicMock

import dotenv
import pytest

# Load .env so real credentials are available for integration tests.
# setdefault fills in dummy values only for vars not in .env (unit tests).
dotenv.load_dotenv()
os.environ.setdefault("JINA_API_KEY", "test-jina-key")
os.environ.setdefault("JINA_MODEL", "jina-embeddings-v4")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")

# Redirect every collection reference to throwaway copies BEFORE config is
# imported. The integration fixtures below drop and recreate collections
# wholesale; pointed at the real names they would wipe the developer's
# ingested corpus and any Path B live-session points in documents_dense.
# Assignment, not setdefault — a stray .env value must not defeat isolation.
os.environ["QDRANT_DENSE_COLLECTION"] = "test_documents_dense"
os.environ["QDRANT_MULTIVEC_COLLECTION"] = "test_documents_multivec"

from config.constants import (  # noqa: E402
    DENSE_COLLECTION,
    DENSE_DIM,
    MULTIVEC_COLLECTION,
    MULTIVEC_DIM,
)
from ingestion.entity_extractor import ExtractionResult  # noqa: E402

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
    """Mock embedder returning deterministic vectors."""
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
    embedder.model_name = "mock-model"
    embedder.dim = DENSE_DIM
    return embedder


# --- Mock LLM client ---


def _make_mock_extraction() -> ExtractionResult:
    """Return a deterministic ExtractionResult for testing."""
    return ExtractionResult(
        entities=[
            {
                "name": "Test Entity",
                "type": "concept",
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

    from config.settings import settings

    # Last line of defence: this fixture deletes whatever it is handed, so it
    # must never be handed the real collections. See tests/test_collection_isolation.py.
    for name in (DENSE_COLLECTION, MULTIVEC_COLLECTION):
        assert name.startswith("test_"), (
            f"Refusing to run: integration fixture would drop {name!r}, which is "
            "not a test collection. Check the QDRANT_*_COLLECTION overrides at "
            "the top of tests/conftest.py."
        )

    client = QdrantClient(url=settings.qdrant_url)

    # Clean before
    for name in (DENSE_COLLECTION, MULTIVEC_COLLECTION):
        if client.collection_exists(name):
            client.delete_collection(name)

    # Provision fresh collections (always create both for test coverage)
    from ingestion.qdrant_setup import (
        create_dense_collection,
        create_multivec_collection,
    )

    create_dense_collection(client)
    create_multivec_collection(client)

    yield client

    # Clean after
    for name in (DENSE_COLLECTION, MULTIVEC_COLLECTION):
        if client.collection_exists(name):
            client.delete_collection(name)

    client.close()


# --- Neo4j fixtures ---

# Must match docker-compose.yml so tests exercise the production version.
NEO4J_TEST_IMAGE = "neo4j:5.26-community"

# Set once the session has been redirected at the ephemeral container. Stays
# None for unit-only runs, which never talk to Neo4j at all.
_EPHEMERAL_NEO4J_URI: str | None = None


@pytest.fixture(scope="session")
def neo4j_container():  # type: ignore[no-untyped-def]
    """Ephemeral Neo4j owned by the test run.

    Neo4j Community has a single database, so there is no cheap namespace
    equivalent to the ``test_*`` Qdrant collections and the fixtures below wipe
    whatever they are pointed at. Aimed at the dev instance they destroy the
    developer's knowledge graph; a throwaway container makes those wipes
    correct, because the database genuinely belongs to the tests.
    """
    from testcontainers.community.neo4j import Neo4jContainer

    from config.settings import settings

    container = (
        Neo4jContainer(NEO4J_TEST_IMAGE, password=settings.neo4j_password)
        # APOC is mandatory, not optional: create_neo4j_schema raises without
        # it and both graph engines call apoc.merge.relationship. No network
        # needed — the jar ships in the image under /var/lib/neo4j/labs/ and
        # the entrypoint installs it when NEO4J_PLUGINS is set.
        .with_env("NEO4J_PLUGINS", '["apoc"]')
        .with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*")
    )

    try:
        container.start()
    except Exception as exc:  # noqa: BLE001 - re-raised as a diagnosable failure
        pytest.fail(
            f"Could not start the ephemeral Neo4j test container "
            f"({NEO4J_TEST_IMAGE}): {exc!r}. Integration tests deliberately run "
            "against a throwaway Neo4j so they never touch the dev graph — "
            "Docker must be running and able to provide that image.",
            pytrace=False,
        )

    try:
        yield container
    finally:
        container.stop()


def _reset_neo4j_singletons() -> None:
    """Drop cached clients that captured the previous ``settings.neo4j_uri``.

    Everything else reads the setting at call time, but these two copy it:
    ``GLiNEREngine.__init__`` stores it on the instance, and the Graphiti
    client bakes it into a connected driver.
    """
    import ingestion.graph_engine as graph_engine
    import ingestion.graphiti_client as graphiti_client

    graph_engine._engine = None
    graphiti_client._client = None
    graphiti_client._graphiti_embedder = None


async def _provision_test_schema() -> None:
    """Create constraints and the ``entity_fulltext`` index on the fresh container.

    The per-test ``neo4j_driver`` fixture does this too, but tests that reach
    Neo4j transitively (graph search via ``multi_search``) never request it and
    would otherwise query an unindexed database.
    """
    from ingestion.neo4j_setup import create_neo4j_schema, get_driver

    driver = get_driver()
    try:
        await create_neo4j_schema(driver)
    finally:
        await driver.close()


@pytest.fixture(scope="session", autouse=True)
def _use_ephemeral_neo4j(request):  # type: ignore[no-untyped-def]
    """Point every Neo4j consumer at the ephemeral container.

    Autouse is load-bearing: the graph engine is reachable from tests that
    request no Neo4j fixture at all (``tests/eval/test_retrieval_metrics.py``
    gets there via ``multi_search`` -> ``graph_search``), and an opt-in fixture
    would leave those paths aimed at the developer's database.

    Only starts the container when the session actually collected integration
    tests, so ``task test`` stays fast.
    """
    if not any(item.get_closest_marker("integration") for item in request.session.items):
        yield
        return

    global _EPHEMERAL_NEO4J_URI  # noqa: PLW0603

    container = request.getfixturevalue("neo4j_container")

    from config.settings import settings

    _EPHEMERAL_NEO4J_URI = container.get_connection_url()
    settings.neo4j_uri = _EPHEMERAL_NEO4J_URI
    _reset_neo4j_singletons()
    asyncio.run(_provision_test_schema())

    yield


def assert_ephemeral_neo4j(fixture: str) -> None:
    """Refuse to run a wiping fixture against anything but the test container.

    Last line of defence, mirroring the one in ``qdrant_client``. Every fixture
    that issues a bulk DELETE must call this: a Neo4j test that forgets
    ``@pytest.mark.integration`` never triggers ``_use_ephemeral_neo4j``, so the
    wipe would land on the developer's real graph.

    Shared rather than inlined because ``tests/test_graph_writer.py`` builds its
    own ``GraphWriter`` and needs the identical check — reading the module
    global here means both callers see the live value, not an import-time copy.
    """
    from config.settings import settings

    assert _EPHEMERAL_NEO4J_URI is not None and settings.neo4j_uri == _EPHEMERAL_NEO4J_URI, (
        f"Refusing to run: {fixture} wipes {settings.neo4j_uri!r}, which is "
        "not the ephemeral test container. Neo4j tests must be marked "
        "@pytest.mark.integration so _use_ephemeral_neo4j starts it."
    )


@pytest.fixture
async def neo4j_driver():  # type: ignore[no-untyped-def]
    """Real Neo4j driver for integration tests.

    Clears all test data before and after each test.
    """
    from neo4j import AsyncGraphDatabase

    from config.settings import settings
    from ingestion.neo4j_setup import create_neo4j_schema

    assert_ephemeral_neo4j("the neo4j_driver fixture")

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
