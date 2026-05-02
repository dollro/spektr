"""Integration tests for live ingestion path (Path B).

Full round-trip against real Qdrant (Docker). Graphiti is mocked
because it requires an LLM API key.

Flow: start session -> ingest chunks -> verify Qdrant payloads ->
      search with session_id -> archive -> verify is_live flipped ->
      new session -> ingest -> discard -> verify points deleted.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from config.constants import DENSE_COLLECTION

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_graphiti():
    client = AsyncMock()
    client.add_episode = AsyncMock()
    client.delete_group = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def _reset_live_ingest_state():
    """Reset module-level singletons between tests."""
    import ingestion.live_ingest as mod

    mod._active_session = None
    mod._qdrant_client = None
    mod._embedder = None
    yield
    mod._active_session = None
    mod._qdrant_client = None
    mod._embedder = None


@pytest.fixture
def patched_app(qdrant_client, mock_embedder, mock_graphiti):
    """Live ingest app wired to real Qdrant, mock embedder + Graphiti.

    Disables INGEST_API_KEY auth for the test — auth is covered
    separately in unit tests for the ingest module.
    """
    from config.settings import settings

    with (
        patch.object(settings, "ingest_api_key", ""),
        patch(
            "ingestion.live_ingest._get_qdrant_client",
            return_value=qdrant_client,
        ),
        patch(
            "ingestion.live_ingest._get_embedder",
            return_value=mock_embedder,
        ),
        patch(
            "ingestion.live_ingest.get_graphiti",
            return_value=mock_graphiti,
        ),
    ):
        from ingestion.live_ingest import app

        yield app


class TestLiveIngestE2E:
    """Full lifecycle: ingest -> search -> archive/discard."""

    async def test_ingest_and_search_with_session(
        self, qdrant_client, mock_embedder, mock_graphiti, patched_app
    ) -> None:
        """Ingested live chunks are searchable via vector_search."""
        async with AsyncClient(
            transport=ASGITransport(app=patched_app), base_url="http://test"
        ) as client:
            # Start session
            resp = await client.post(
                "/session/start",
                json={"session_id": "e2e-1", "metadata": {"title": "E2E Test"}},
            )
            assert resp.status_code == 200

            # Ingest two chunks
            for i, text in enumerate(
                [
                    "We should migrate the database to PostgreSQL.",
                    "I agree, the current SQLite setup won't scale.",
                ]
            ):
                resp = await client.post(
                    "/ingest/chunk",
                    json={
                        "session_id": "e2e-1",
                        "text": text,
                        "timestamp": f"2026-03-07T10:0{i}:00Z",
                    },
                )
                assert resp.status_code == 200
                assert resp.json()["vector_indexed"] is True

            # Let background Graphiti tasks finish
            await asyncio.sleep(0.2)

        # Verify points landed in Qdrant with correct payloads
        points, _ = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=_session_filter("e2e-1"),
            limit=10,
            with_payload=True,
        )
        assert len(points) == 2

        payloads = [p.payload for p in points]
        assert all(p["is_live"] is True for p in payloads)
        assert all(p["session_id"] == "e2e-1" for p in payloads)
        assert all(p["content_type"] == "live" for p in payloads)

        # Search with session_id via vector_search
        with (
            patch("server.tools.vector_search._qdrant_client", qdrant_client),
            patch("server.tools.vector_search._embedder", mock_embedder),
        ):
            from server.tools.vector_search import vector_search

            results = await vector_search("database migration", session_id="e2e-1")

        assert len(results) >= 1
        live_results = [
            r for r in results if r.get("metadata", {}).get("source_type") == "live"
        ]
        assert len(live_results) >= 1

        # Graphiti was called for both chunks
        assert mock_graphiti.add_episode.call_count == 2

    async def test_archive_flips_is_live(
        self, qdrant_client, mock_embedder, mock_graphiti, patched_app
    ) -> None:
        """Archiving a session sets is_live=false on all session points."""
        async with AsyncClient(
            transport=ASGITransport(app=patched_app), base_url="http://test"
        ) as client:
            await client.post(
                "/session/start",
                json={"session_id": "e2e-archive"},
            )
            await client.post(
                "/ingest/chunk",
                json={
                    "session_id": "e2e-archive",
                    "text": "Archivable content.",
                    "timestamp": "2026-03-07T11:00:00Z",
                },
            )
            await asyncio.sleep(0.1)

            # Archive
            resp = await client.post(
                "/session/end",
                json={"session_id": "e2e-archive", "archive": True},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "archived"

        # Verify is_live flipped to false
        points, _ = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=_session_filter("e2e-archive"),
            limit=10,
            with_payload=True,
        )
        assert len(points) == 1
        assert points[0].payload["is_live"] is False

    async def test_discard_deletes_points(
        self, qdrant_client, mock_embedder, mock_graphiti, patched_app
    ) -> None:
        """Discarding a session removes all session points from Qdrant."""
        async with AsyncClient(
            transport=ASGITransport(app=patched_app), base_url="http://test"
        ) as client:
            await client.post(
                "/session/start",
                json={"session_id": "e2e-discard"},
            )
            await client.post(
                "/ingest/chunk",
                json={
                    "session_id": "e2e-discard",
                    "text": "Disposable content.",
                    "timestamp": "2026-03-07T12:00:00Z",
                },
            )
            await asyncio.sleep(0.1)

            # Discard
            resp = await client.post(
                "/session/end",
                json={"session_id": "e2e-discard", "archive": False},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "discarded"

        # Verify points are gone
        points, _ = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=_session_filter("e2e-discard"),
            limit=10,
        )
        assert len(points) == 0
        mock_graphiti.delete_group.assert_called_once_with("e2e-discard")


def _session_filter(session_id: str):
    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            ),
        ],
    )
