from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedders.openrouter import OpenRouterEmbedder

DIMS = 3072


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.openrouter_api_key = "test-or-key"
    s.openrouter_api_url = "https://openrouter.ai/api"
    s.openrouter_model = "google/gemini-embedding-2-preview"
    s.openrouter_dense_dimensions = DIMS
    s.openrouter_rpm = 300
    s.openrouter_max_concurrent = 10
    s.openrouter_http_referer = "https://example.test"
    s.openrouter_x_title = "Spektr-Test"
    s.max_retries = 3
    return s


@pytest.fixture
async def embedder(mock_settings: MagicMock) -> OpenRouterEmbedder:
    with patch("ingestion.embedders.openrouter.settings", mock_settings):
        e = OpenRouterEmbedder(api_key="test-or-key")
        e._ensure_loop_resources()
        return e


def _mock_response(data: list[dict]) -> httpx.Response:  # type: ignore[type-arg]
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestEmbedText:
    async def test_correct_payload(self, embedder: OpenRouterEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}, {"embedding": [0.2] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "google/gemini-embedding-2-preview"
        assert payload["input"] == ["hello", "world"]
        assert payload["encoding_format"] == "float"
        assert payload["dimensions"] == DIMS
        assert len(result) == 2
        assert len(result[0]) == DIMS

    async def test_url_is_embeddings_endpoint(self, embedder: OpenRouterEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"])

        assert mock_post.call_args.args[0] == embedder._url
        assert "openrouter.ai/api/v1/embeddings" in embedder._url

    async def test_dimensions_override(self, embedder: OpenRouterEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 768}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["hi"], dimensions=768)

        assert mock_post.call_args.kwargs["json"]["dimensions"] == 768


class TestRankingHeaders:
    async def test_optional_headers_attached(self, embedder: OpenRouterEmbedder) -> None:
        headers = embedder._client.headers  # type: ignore[union-attr]
        assert headers["Authorization"] == "Bearer test-or-key"
        assert headers["HTTP-Referer"] == "https://example.test"
        assert headers["X-Title"] == "Spektr-Test"


class TestUnsupported:
    async def test_embed_image_raises(self, embedder: OpenRouterEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="text-only"):
            await embedder.embed_image(b"\x89PNG")

    async def test_embed_multi_vector_raises(self, embedder: OpenRouterEmbedder) -> None:
        with pytest.raises(NotImplementedError):
            await embedder.embed_multi_vector(b"\x89PNG")

    async def test_embed_query_multi_vector_raises(self, embedder: OpenRouterEmbedder) -> None:
        with pytest.raises(NotImplementedError):
            await embedder.embed_query_multi_vector("q")


class TestEventLoopIsolation:
    """Regression: one shared embedder is used from several event loops.

    CocoIndex runs file components concurrently on multiple worker threads,
    each with its own asyncio loop, all sharing a single embedder instance.
    The old implementation kept one ``self._client`` slot and recreated it
    whenever the loop changed, so concurrent loops clobbered each other and an
    in-flight request awaited a connection-pool lock owned by a different loop
    (``RuntimeError: the current task is not holding this lock``). Resources
    must instead be isolated per loop.
    """

    def test_client_is_stable_within_a_loop_under_concurrency(
        self, mock_settings: MagicMock
    ) -> None:
        with patch("ingestion.embedders.openrouter.settings", mock_settings):
            emb = OpenRouterEmbedder(api_key="test-or-key")

        mid = threading.Barrier(2)
        results: dict[int, tuple[bool, int]] = {}
        errors: list[BaseException] = []
        keepalive: list[httpx.AsyncClient] = []  # keep clients alive so ids are stable

        def worker(tag: int) -> None:
            async def run() -> None:
                c1, _ = emb._ensure_loop_resources()
                mid.wait()  # hold both loops open & bound at the same time
                c2, _ = emb._ensure_loop_resources()
                keepalive.extend((c1, c2))
                results[tag] = (c1 is c2, id(c1))

            try:
                asyncio.run(run())
            except BaseException as exc:  # noqa: BLE001 — surface the loop-race crash
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"embedder raised across concurrent loops: {errors}"
        # Each loop kept the same client across both calls (no cross-loop clobber).
        assert results[0][0] is True, "client was recreated mid-loop (slot clobbered)"
        assert results[1][0] is True, "client was recreated mid-loop (slot clobbered)"
        # The two loops received genuinely distinct clients (true isolation).
        assert results[0][1] != results[1][1], "two loops shared one client"

    async def test_request_uses_the_current_loops_client(
        self, mock_settings: MagicMock
    ) -> None:
        with patch("ingestion.embedders.openrouter.settings", mock_settings):
            emb = OpenRouterEmbedder(api_key="test-or-key")
            client, _ = emb._ensure_loop_resources()
            mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
            with patch.object(
                client, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                await emb.embed_text(["hi"])
            mock_post.assert_awaited_once()


class TestProtocolCompliance:
    def test_implements_embedder_protocol(self) -> None:
        from ingestion.embedder import Embedder

        assert isinstance(OpenRouterEmbedder.__new__(OpenRouterEmbedder), Embedder)
