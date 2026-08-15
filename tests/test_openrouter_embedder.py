from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedders.openrouter import OpenRouterEmbedder, _is_retryable

DIMS = 3072


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.openrouter_api_key = "test-or-key"
    s.openrouter_api_url = "https://openrouter.ai/api"
    s.openrouter_model = "google/gemini-embedding-2"
    s.dense_dimensions = DIMS
    s.embedding_model_id = "google/gemini-embedding-2"
    s.openrouter_rpm = 300
    s.openrouter_batch_size = 100
    s.openrouter_max_concurrent = 10
    s.openrouter_http_referer = "https://example.test"
    s.openrouter_x_title = "Spektr-Test"
    s.max_retries = 3
    s.supports_image_embedding = True
    s.embedding_image_model_id = "google/gemini-embedding-2"
    s.embedding_model = "gemini-2"
    s.embedding_route = "openrouter"
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
        assert payload["model"] == "google/gemini-embedding-2"
        assert payload["input"] == ["hello", "world"]
        assert payload["encoding_format"] == "float"
        assert payload["dimensions"] == DIMS
        assert payload["input_type"] == "document"  # default task="passage"
        assert len(result) == 2
        assert len(result[0]) == DIMS

    async def test_query_uses_query_input_type(self, embedder: OpenRouterEmbedder) -> None:
        """Documents and queries must not land in the same embedding mode."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text_query("what is spektr?")

        assert mock_post.call_args.kwargs["json"]["input_type"] == "query"

    async def test_gemini_and_jina_task_spellings_are_not_sent(
        self, embedder: OpenRouterEmbedder
    ) -> None:
        """The gateway silently ignores task_type/task — sending them is a trap."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text(["hi"])

        payload = mock_post.call_args.kwargs["json"]
        assert "task_type" not in payload
        assert "task" not in payload
        assert "output_dimensionality" not in payload

    async def test_unknown_task_raises(self, embedder: OpenRouterEmbedder) -> None:
        """Unmapped tasks must fail loudly — the API would ignore them silently."""
        with pytest.raises(ValueError, match="Unknown embedding task"):
            await embedder.embed_text(["hi"], task="retrieval.passage")

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

    async def test_long_input_is_split_into_batches(
        self, embedder: OpenRouterEmbedder
    ) -> None:
        """Gemini hard-rejects >100 inputs with a non-retryable 400."""
        embedder._batch_size = 100
        texts = [f"chunk {i}" for i in range(250)]

        def _resp(*args: object, **kwargs: object) -> httpx.Response:
            n = len(kwargs["json"]["input"])  # type: ignore[index,call-overload]
            return _mock_response([{"embedding": [0.1] * DIMS, "index": i} for i in range(n)])

        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, side_effect=_resp
        ) as mock_post:
            result = await embedder.embed_text(texts)

        sizes = [len(c.kwargs["json"]["input"]) for c in mock_post.call_args_list]
        assert sizes == [100, 100, 50]
        assert len(result) == 250

    async def test_batches_preserve_order(self, embedder: OpenRouterEmbedder) -> None:
        """Chunk N's vector must stay with chunk N even if the API reorders."""
        embedder._batch_size = 2

        def _resp(*args: object, **kwargs: object) -> httpx.Response:
            n = len(kwargs["json"]["input"])  # type: ignore[index,call-overload]
            # Return them out of order, with explicit indices.
            items = [{"embedding": [float(i)] * DIMS, "index": i} for i in range(n)]
            return _mock_response(list(reversed(items)))

        with patch.object(embedder._client, "post", new_callable=AsyncMock, side_effect=_resp):
            result = await embedder.embed_text(["a", "b", "c", "d"])

        assert [v[0] for v in result] == [0.0, 1.0, 0.0, 1.0]

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


class TestRetryPredicate:
    """ConnectTimeout is a TimeoutException, NOT a ConnectError subclass."""

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            httpx.PoolTimeout("timed out"),
            httpx.ConnectError("refused"),
            httpx.ReadError("reset"),
            httpx.RemoteProtocolError("bad frame"),
        ],
    )
    def test_transient_transport_errors_are_retryable(self, exc: Exception) -> None:
        assert _is_retryable(exc) is True

    def test_client_errors_are_not_retryable(self) -> None:
        """A 400 (e.g. batch too large) must fail fast, not burn retries."""
        resp = httpx.Response(400, request=httpx.Request("POST", "https://x.test"))
        err = httpx.HTTPStatusError("bad", request=resp.request, response=resp)
        assert _is_retryable(err) is False

    def test_server_errors_and_429_are_retryable(self) -> None:
        for code in (429, 500, 503):
            resp = httpx.Response(code, request=httpx.Request("POST", "https://x.test"))
            err = httpx.HTTPStatusError("x", request=resp.request, response=resp)
            assert _is_retryable(err) is True, code


class TestRankingHeaders:
    async def test_optional_headers_attached(self, embedder: OpenRouterEmbedder) -> None:
        headers = embedder._client.headers  # type: ignore[union-attr]
        assert headers["Authorization"] == "Bearer test-or-key"
        assert headers["HTTP-Referer"] == "https://example.test"
        assert headers["X-Title"] == "Spektr-Test"


class TestEmbedImage:
    """Image support on this route (gemini-2 is natively multimodal)."""

    async def test_payload_shape_and_dimensions(self, embedder: OpenRouterEmbedder) -> None:
        """`dimensions` is honoured for images, which is what puts image and
        text points in one collection."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            result = await embedder.embed_image(b"\x89PNG-bytes")

        payload = mock_post.call_args.kwargs["json"]
        item = payload["input"][0]["content"][0]
        assert item["type"] == "image_url"
        assert item["image_url"]["url"].startswith("data:image/png;base64,")
        assert payload["dimensions"] == DIMS
        assert len(result) == DIMS

    async def test_input_type_is_not_sent(self, embedder: OpenRouterEmbedder) -> None:
        """Verified live: the gateway accepts input_type for images and
        returns a byte-identical vector, so sending it would advertise an
        asymmetry the model does not have."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_image(b"\x89PNG")

        assert "input_type" not in mock_post.call_args.kwargs["json"]

    async def test_media_type_is_passed_through(self, embedder: OpenRouterEmbedder) -> None:
        """JPEG is accepted too; the caller's type must reach the data URI."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_image(b"jpegbytes", media_type="image/jpeg")

        url = mock_post.call_args.kwargs["json"]["input"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    async def test_uses_longer_timeout(self, embedder: OpenRouterEmbedder) -> None:
        """A page bitmap trips the 60s text default."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_image(b"\x89PNG")

        assert mock_post.call_args.kwargs["timeout"] == 120.0

    async def test_raises_when_pair_lacks_image_support(
        self, embedder: OpenRouterEmbedder, mock_settings: MagicMock
    ) -> None:
        """Fail loudly rather than send a request the model will reject."""
        mock_settings.supports_image_embedding = False
        with patch("ingestion.embedders.openrouter.settings", mock_settings):
            with pytest.raises(NotImplementedError, match="not available"):
                await embedder.embed_image(b"\x89PNG")


class TestUnsupported:
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
