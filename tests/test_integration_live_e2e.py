"""Real end-to-end integration test for the live ingestion path.

No mocks — uses real Qdrant, Neo4j, Graphiti (LLM), and Jina embeddings.
Requires all Docker services running and valid API keys in .env.

Flow:
  1. Start session
  2. Ingest live chunks (vector + Graphiti temporal graph)
  3. Verify vectors in Qdrant with correct payloads
  4. Verify Graphiti created entities/edges in Neo4j
  5. Search via vector_search with session_id
  6. Search via graph_search with session_id (Graphiti temporal facts)
  7. Archive session — verify is_live flipped
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from httpx import ASGITransport, AsyncClient

from config.constants import DENSE_COLLECTION

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration

SESSION_ID = "e2e-live-real"

CHUNKS = [
    {
        "text": (
            "The team proposed migrating the user database from SQLite to "
            "PostgreSQL for better concurrency."
        ),
        "timestamp": "2026-03-07T10:00:00Z",
    },
    {
        "text": (
            "The team agreed and suggested using Alembic for schema migrations "
            "and adding connection pooling."
        ),
        "timestamp": "2026-03-07T10:01:00Z",
    },
    {
        "text": (
            "The team raised concerns about downtime during the migration "
            "and proposed a blue-green deployment strategy."
        ),
        "timestamp": "2026-03-07T10:02:00Z",
    },
]


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons and disable ingest auth for tests."""
    import ingestion.live_ingest as mod
    from config.settings import settings

    mod._active_session = None
    mod._qdrant_client = None
    mod._embedder = None
    original_key = settings.ingest_api_key
    settings.ingest_api_key = ""
    yield
    settings.ingest_api_key = original_key
    mod._active_session = None
    mod._qdrant_client = None
    mod._embedder = None


@pytest.fixture(autouse=True)
async def _reset_graphiti_singleton():
    """Reset Graphiti client singleton between tests."""
    import ingestion.graphiti_client as gc

    yield
    await gc.close_graphiti()


@pytest.fixture
async def _cleanup_session(qdrant_client):
    """Clean up session data from Qdrant and Neo4j after test."""
    yield

    # Remove session points from Qdrant
    from qdrant_client import models

    qdrant_client.delete(
        collection_name=DENSE_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=SESSION_ID),
                    ),
                ],
            ),
        ),
    )

    # Remove Graphiti group data from Neo4j
    try:
        from ingestion.graphiti_client import get_graphiti

        await get_graphiti()  # ensure singleton is initialised before driver use
        # Delete episodes for this group
        from neo4j import AsyncGraphDatabase

        from config.settings import settings

        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        async with driver.session() as session:
            await session.run(
                "MATCH (e:Episodic {group_id: $gid}) DETACH DELETE e",
                gid=SESSION_ID,
            )
        await driver.close()
    except Exception:
        logger.warning("Cleanup of Neo4j Graphiti data failed", exc_info=True)


class TestLiveIngestRealE2E:
    """Full live ingestion e2e with real services — no mocks."""

    async def test_full_live_lifecycle(
        self, qdrant_client, _cleanup_session
    ) -> None:
        """Ingest real live chunks, verify vector + graph, search, archive."""
        from ingestion.live_ingest import app

        # --- 1. Start session ---
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/session/start",
                json={"session_id": SESSION_ID, "metadata": {"title": "DB Migration"}},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "active"

            # --- 2. Ingest chunks ---
            for chunk in CHUNKS:
                resp = await client.post(
                    "/ingest/chunk",
                    json={"session_id": SESSION_ID, **chunk},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["vector_indexed"] is True
                assert data["graph_status"] == "processing"

            # Wait for background Graphiti tasks to complete
            # Graphiti LLM calls take several seconds per chunk
            logger.info("Waiting for Graphiti background tasks...")
            await asyncio.sleep(60)

        # --- 3. Verify vectors in Qdrant ---
        from qdrant_client import models

        points, _ = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=SESSION_ID),
                    ),
                ],
            ),
            limit=10,
            with_payload=True,
        )
        assert len(points) == len(CHUNKS), f"Expected {len(CHUNKS)} points, got {len(points)}"

        payloads = [p.payload for p in points]
        assert all(p["is_live"] is True for p in payloads)
        assert all(p["session_id"] == SESSION_ID for p in payloads)
        assert all(p["content_type"] == "live" for p in payloads)
        logger.info("Qdrant: %d points with correct payloads", len(points))

        # --- 4. Verify Graphiti wrote to Neo4j ---
        from neo4j import AsyncGraphDatabase

        from config.settings import settings

        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        async with driver.session() as session:
            result = await session.run(
                "MATCH (e:Episodic {group_id: $gid}) RETURN count(e) AS cnt",
                gid=SESSION_ID,
            )
            record = await result.single()
            episode_count = record["cnt"]
        await driver.close()

        assert episode_count >= 1, f"Expected Graphiti episodes in Neo4j, got {episode_count}"
        logger.info("Neo4j: %d Graphiti episodes for session", episode_count)

        # --- 5. Vector search with session_id ---
        from ingestion.embedder import create_embedder

        real_embedder = create_embedder()
        try:
            from unittest.mock import patch

            with (
                patch("server.tools.vector_search._qdrant_client", qdrant_client),
                patch("server.tools.vector_search._embedder", real_embedder),
            ):
                from server.tools.vector_search import vector_search

                results = await vector_search(
                    "database migration PostgreSQL", session_id=SESSION_ID
                )

            assert len(results) >= 1
            live_results = [
                r
                for r in results
                if r.get("metadata", {}).get("source_type") == "live"
            ]
            assert len(live_results) >= 1, f"No live results found in {results}"
            logger.info(
                "Vector search: %d results (%d live)",
                len(results),
                len(live_results),
            )
        finally:
            await real_embedder.close()

        # --- 6. Graph search with session_id (Graphiti temporal facts) ---
        from server.tools.graph_search import graph_search

        graph_results = await graph_search(
            "database migration", session_id=SESSION_ID
        )
        # Graphiti should have extracted some facts about the session
        graphiti_facts = [r for r in graph_results if "error" not in r]
        logger.info("Graph search: %d facts returned", len(graphiti_facts))
        # At minimum, Graphiti should return something for these clear entities
        assert len(graphiti_facts) >= 1, f"No graph facts found: {graph_results}"

        # --- 7. Archive session ---
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": SESSION_ID}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/session/end",
                json={"session_id": SESSION_ID, "archive": True},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "archived"

        # Verify is_live flipped
        points, _ = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=SESSION_ID),
                    ),
                ],
            ),
            limit=10,
            with_payload=True,
        )
        assert len(points) == len(CHUNKS)
        assert all(p.payload["is_live"] is False for p in points)
        logger.info("Archive: all %d points flipped to is_live=false", len(points))
