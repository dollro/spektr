"""Tests for live transcript ingestion FastAPI endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.upsert = MagicMock()
    client.scroll = MagicMock(return_value=([], None))
    client.delete = MagicMock()
    client.set_payload = MagicMock()
    return client


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=[[0.1] * 512])
    embedder.close = AsyncMock()
    return embedder


@pytest.fixture
def mock_graphiti():
    client = AsyncMock()
    client.add_episode = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def _reset_active_session():
    """Reset active session state between tests."""
    import ingestion.live_ingest as mod

    mod._active_session = None
    yield
    mod._active_session = None


class TestSessionStart:
    @pytest.mark.asyncio
    async def test_start_session(self, mock_qdrant, mock_embedder) -> None:
        """POST /session/start creates a new session."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/start",
                    json={
                        "session_id": "meeting-1",
                        "metadata": {"title": "Test Meeting"},
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "meeting-1"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_start_duplicate_session_rejected(self, mock_qdrant, mock_embedder) -> None:
        """Starting a session while one is active returns 409."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "existing"}

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/start",
                    json={"session_id": "meeting-2"},
                )

        assert resp.status_code == 409


class TestIngestTranscript:
    @pytest.mark.asyncio
    async def test_ingest_transcript_chunk(
        self,
        mock_qdrant,
        mock_embedder,
        mock_graphiti,
    ) -> None:
        """POST /ingest/transcript embeds and upserts to Qdrant."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1"}

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.get_graphiti", return_value=mock_graphiti),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ingest/transcript",
                    json={
                        "session_id": "meeting-1",
                        "text": "Alice: Hello everyone.",
                        "timestamp": "2026-03-06T14:30:00Z",
                        "speaker": "Alice",
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["vector_indexed"] is True
        assert data["graph_status"] == "processing"
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_graphiti_background_task_called(
        self,
        mock_qdrant,
        mock_embedder,
        mock_graphiti,
    ) -> None:
        """Background task calls Graphiti add_episode with correct args."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1"}

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.get_graphiti", return_value=mock_graphiti),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(
                    "/ingest/transcript",
                    json={
                        "session_id": "meeting-1",
                        "text": "Alice: Hello everyone.",
                        "timestamp": "2026-03-06T14:30:00Z",
                        "speaker": "Alice",
                    },
                )
                # Let the background task complete
                await asyncio.sleep(0.1)

        mock_graphiti.add_episode.assert_called_once()
        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert call_kwargs["episode_body"] == "Alice: Hello everyone."
        assert call_kwargs["group_id"] == "meeting-1"
        assert "Alice" in call_kwargs["source_description"]

    @pytest.mark.asyncio
    async def test_ingest_session_mismatch_rejected(
        self,
        mock_qdrant,
        mock_embedder,
    ) -> None:
        """Ingesting with a mismatched session_id returns 400."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1"}

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ingest/transcript",
                    json={
                        "session_id": "wrong-session",
                        "text": "Hello",
                        "timestamp": "2026-03-06T14:30:00Z",
                    },
                )

        assert resp.status_code == 400
        assert "mismatch" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_ingest_without_active_session_rejected(
        self,
        mock_qdrant,
        mock_embedder,
    ) -> None:
        """Ingesting without an active session returns 404."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ingest/transcript",
                    json={
                        "session_id": "meeting-1",
                        "text": "Hello",
                        "timestamp": "2026-03-06T14:30:00Z",
                    },
                )

        assert resp.status_code == 404


class TestSessionEnd:
    @pytest.mark.asyncio
    async def test_end_session_archive(self, mock_qdrant, mock_embedder) -> None:
        """POST /session/end with archive=true updates points."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1"}

        mock_qdrant.scroll = MagicMock(return_value=([MagicMock(id="point-1")], None))

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/end",
                    json={"session_id": "meeting-1", "archive": True},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "archived"
        assert mod._active_session is None

    @pytest.mark.asyncio
    async def test_end_session_discard(
        self,
        mock_qdrant,
        mock_embedder,
        mock_graphiti,
    ) -> None:
        """POST /session/end with archive=false deletes points."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1"}

        mock_qdrant.scroll = MagicMock(return_value=([MagicMock(id="point-1")], None))

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.get_graphiti", return_value=mock_graphiti),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/end",
                    json={"session_id": "meeting-1", "archive": False},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "discarded"
