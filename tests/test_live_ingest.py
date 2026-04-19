"""Tests for live transcript ingestion FastAPI endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from config.settings import settings
from fastapi import HTTPException

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

_TEST_API_KEY = "test-ingest-key-42"


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


@pytest.fixture(autouse=True)
def _disable_ingest_auth():
    """Disable ingest auth so non-auth tests aren't affected by .env values."""
    with patch.object(settings, "ingest_api_key", ""):
        yield


def _fake_request(auth_header: str | None = None) -> Request:
    """Build a minimal Starlette Request with an optional Authorization header."""
    headers: list[tuple[bytes, bytes]] = []
    if auth_header is not None:
        headers.append((b"authorization", auth_header.encode()))
    scope = {"type": "http", "method": "GET", "path": "/", "headers": headers}
    return Request(scope)


class TestExtractBearer:
    """Unit tests for _extract_bearer."""

    def test_valid_bearer(self) -> None:
        from ingestion.live_ingest import _extract_bearer

        assert _extract_bearer(_fake_request("Bearer my-token")) == "my-token"

    def test_missing_header_raises_401(self) -> None:
        from ingestion.live_ingest import _extract_bearer

        with pytest.raises(HTTPException) as exc_info:
            _extract_bearer(_fake_request())
        assert exc_info.value.status_code == 401

    def test_malformed_header_raises_401(self) -> None:
        from ingestion.live_ingest import _extract_bearer

        with pytest.raises(HTTPException) as exc_info:
            _extract_bearer(_fake_request("Token abc"))
        assert exc_info.value.status_code == 401


class TestRequireIngestApiKey:
    """Unit tests for _require_ingest_api_key."""

    def test_auth_disabled_allows_any_request(self) -> None:
        from ingestion.live_ingest import _require_ingest_api_key

        with patch.object(settings, "ingest_api_key", ""):
            _require_ingest_api_key(_fake_request())  # no error

    def test_missing_token_raises_401(self) -> None:
        from ingestion.live_ingest import _require_ingest_api_key

        with patch.object(settings, "ingest_api_key", _TEST_API_KEY):
            with pytest.raises(HTTPException) as exc_info:
                _require_ingest_api_key(_fake_request())
            assert exc_info.value.status_code == 401

    def test_wrong_key_raises_403(self) -> None:
        from ingestion.live_ingest import _require_ingest_api_key

        with patch.object(settings, "ingest_api_key", _TEST_API_KEY):
            with pytest.raises(HTTPException) as exc_info:
                _require_ingest_api_key(_fake_request("Bearer wrong"))
            assert exc_info.value.status_code == 403

    def test_correct_key_passes(self) -> None:
        from ingestion.live_ingest import _require_ingest_api_key

        with patch.object(settings, "ingest_api_key", _TEST_API_KEY):
            _require_ingest_api_key(
                _fake_request(f"Bearer {_TEST_API_KEY}")
            )  # no error


class TestRequireSessionToken:
    """Unit tests for _require_session_token."""

    def test_auth_disabled_allows_any_request(self) -> None:
        from ingestion.live_ingest import _require_session_token

        with patch.object(settings, "ingest_api_key", ""):
            _require_session_token(_fake_request())  # no error

    def test_no_active_session_raises_404(self) -> None:
        import ingestion.live_ingest as mod
        from ingestion.live_ingest import _require_session_token

        mod._active_session = None
        with patch.object(settings, "ingest_api_key", _TEST_API_KEY):
            with pytest.raises(HTTPException) as exc_info:
                _require_session_token(_fake_request("Bearer any"))
            assert exc_info.value.status_code == 404

    def test_wrong_token_raises_403(self) -> None:
        import ingestion.live_ingest as mod
        from ingestion.live_ingest import _require_session_token

        mod._active_session = {
            "session_id": "s1",
            "session_token": "real-token",
        }
        with patch.object(settings, "ingest_api_key", _TEST_API_KEY):
            with pytest.raises(HTTPException) as exc_info:
                _require_session_token(_fake_request("Bearer wrong"))
            assert exc_info.value.status_code == 403

    def test_correct_token_passes(self) -> None:
        import ingestion.live_ingest as mod
        from ingestion.live_ingest import _require_session_token

        mod._active_session = {
            "session_id": "s1",
            "session_token": "real-token",
        }
        with patch.object(settings, "ingest_api_key", _TEST_API_KEY):
            _require_session_token(
                _fake_request("Bearer real-token")
            )  # no error


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


class TestSessionAuth:
    @pytest.mark.asyncio
    async def test_start_session_rejected_without_key(self, mock_qdrant, mock_embedder) -> None:
        """POST /session/start without Bearer token returns 401 when auth enabled."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.settings") as mock_settings,
        ):
            mock_settings.ingest_api_key = _TEST_API_KEY
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/start",
                    json={"session_id": "meeting-1"},
                )

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_start_session_rejected_with_wrong_key(
        self, mock_qdrant, mock_embedder
    ) -> None:
        """POST /session/start with wrong API key returns 403."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.settings") as mock_settings,
        ):
            mock_settings.ingest_api_key = _TEST_API_KEY
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/start",
                    json={"session_id": "meeting-1"},
                    headers={"Authorization": "Bearer wrong-key"},
                )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_start_session_returns_session_token(
        self, mock_qdrant, mock_embedder
    ) -> None:
        """POST /session/start with valid API key returns a session_token."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.settings") as mock_settings,
        ):
            mock_settings.ingest_api_key = _TEST_API_KEY
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/session/start",
                    json={"session_id": "meeting-1"},
                    headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "session_token" in data
        assert len(data["session_token"]) > 20

    @pytest.mark.asyncio
    async def test_ingest_rejected_without_session_token(
        self, mock_qdrant, mock_embedder
    ) -> None:
        """POST /ingest/transcript without session token returns 401."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1", "session_token": "real-token"}

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.settings") as mock_settings,
        ):
            mock_settings.ingest_api_key = _TEST_API_KEY
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

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_ingest_rejected_with_wrong_session_token(
        self, mock_qdrant, mock_embedder
    ) -> None:
        """POST /ingest/transcript with wrong session token returns 403."""
        import ingestion.live_ingest as mod

        mod._active_session = {"session_id": "meeting-1", "session_token": "real-token"}

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.settings") as mock_settings,
        ):
            mock_settings.ingest_api_key = _TEST_API_KEY
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
                    headers={"Authorization": "Bearer wrong-token"},
                )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_full_auth_flow(
        self, mock_qdrant, mock_embedder, mock_graphiti
    ) -> None:
        """Full flow: start with API key, ingest with session token, end with session token."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.get_graphiti", return_value=mock_graphiti),
            patch("ingestion.live_ingest.settings") as mock_settings,
        ):
            mock_settings.ingest_api_key = _TEST_API_KEY
            mock_settings.qdrant_url = "http://localhost:6333"
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # 1. Start session with API key
                resp = await client.post(
                    "/session/start",
                    json={"session_id": "meeting-1"},
                    headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
                )
                assert resp.status_code == 200
                session_token = resp.json()["session_token"]

                # 2. Ingest with session token
                resp = await client.post(
                    "/ingest/transcript",
                    json={
                        "session_id": "meeting-1",
                        "text": "Alice: Hello everyone.",
                        "timestamp": "2026-03-06T14:30:00Z",
                        "speaker": "Alice",
                    },
                    headers={"Authorization": f"Bearer {session_token}"},
                )
                assert resp.status_code == 200
                assert resp.json()["status"] == "accepted"

                # 3. End session with session token
                resp = await client.post(
                    "/session/end",
                    json={"session_id": "meeting-1", "archive": True},
                    headers={"Authorization": f"Bearer {session_token}"},
                )
                assert resp.status_code == 200
                assert resp.json()["status"] == "archived"
