from __future__ import annotations

import asyncio
import base64
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedders.voyage import VoyageEmbedder

DIMS = 1024


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.voyage_api_key = "test-voyage-key"
    s.voyage_api_url = "https://api.voyageai.com"
    s.voyage_text_model = "voyage-4-large"
    s.voyage_multimodal_model = "voyage-multimodal-3.5"
    s.dense_dimensions = DIMS
    s.voyage_rpm = 300
    s.voyage_max_concurrent = 10
    s.max_retries = 3
    return s


@pytest.fixture
async def embedder(mock_settings: MagicMock) -> VoyageEmbedder:
    with patch("ingestion.embedders.voyage.settings", mock_settings):
        e = VoyageEmbedder(api_key="test-voyage-key")
        e._ensure_loop_resources()
        return e


def _mock_response(
    data: list[dict],  # type: ignore[type-arg]
) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestEmbedText:
    async def test_correct_payload(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}, {"embedding": [0.2] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "voyage-4-large"
        assert payload["input_type"] == "document"
        assert payload["input"] == ["hello", "world"]
        assert payload["output_dimensions"] == DIMS
        assert len(result) == 2
        assert len(result[0]) == DIMS

    async def test_url_is_text_endpoint(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"])

        assert mock_post.call_args.args[0] == embedder._text_url


class TestEmbedTextQuery:
    async def test_query_task_maps_correctly(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text_query("search query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["input_type"] == "query"
        assert len(result) == DIMS


class TestEmbedImage:
    async def test_uses_multimodal_endpoint_and_model(self, embedder: VoyageEmbedder) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nfakedata"
        expected_b64 = base64.b64encode(image_bytes).decode()
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_image(image_bytes)

        url = mock_post.call_args.args[0]
        assert url == embedder._multimodal_url
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "voyage-multimodal-3.5"
        assert payload["input"] == [
            [
                {
                    "type": "image_base64",
                    "image_base64": expected_b64,
                    "media_type": "image/png",
                }
            ]
        ]
        assert payload["input_type"] == "document"
        assert payload["output_dimensions"] == DIMS
        assert len(result) == DIMS

    async def test_custom_timeout(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_image(b"fake")

        assert mock_post.call_args.kwargs["timeout"] == 120.0


class TestMultiVectorNotSupported:
    async def test_embed_multi_vector_raises(self, embedder: VoyageEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="ColBERT"):
            await embedder.embed_multi_vector(b"fake")

    async def test_embed_query_multi_vector_raises(self, embedder: VoyageEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="ColBERT"):
            await embedder.embed_query_multi_vector("query")


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
        with patch("ingestion.embedders.voyage.settings", mock_settings):
            emb = VoyageEmbedder(api_key="test-voyage-key")

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
        assert results[0][0] is True, "client was recreated mid-loop (slot clobbered)"
        assert results[1][0] is True, "client was recreated mid-loop (slot clobbered)"
        assert results[0][1] != results[1][1], "two loops shared one client"

    async def test_request_uses_the_current_loops_client(
        self, mock_settings: MagicMock
    ) -> None:
        with patch("ingestion.embedders.voyage.settings", mock_settings):
            emb = VoyageEmbedder(api_key="test-voyage-key")
            client, _ = emb._ensure_loop_resources()
            mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
            with patch.object(
                client, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                await emb.embed_text(["hi"])
            mock_post.assert_awaited_once()
