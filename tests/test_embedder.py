from __future__ import annotations

import asyncio
import base64
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from config.settings import settings
from ingestion.embedder import Embedder, TokenBucket, create_embedder
from ingestion.embedders.jina import JinaV4Embedder

FIXTURES = Path(__file__).parent / "fixtures"
DIMS = settings.dense_dimensions


@pytest.fixture
async def embedder() -> JinaV4Embedder:
    e = JinaV4Embedder(api_key="test-key")
    e._ensure_loop_resources()
    return e


def _mock_response(data: list[dict]) -> httpx.Response:  # type: ignore[type-arg]
    """Create a mock httpx.Response with embedding data."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestEmbedderProtocol:
    def test_jina_implements_protocol(self) -> None:
        assert isinstance(JinaV4Embedder(api_key="k"), Embedder)

    def test_factory_returns_jina(self) -> None:
        """Dispatch is on the route; ambient .env must not decide the test."""
        with patch("ingestion.embedder.settings") as mock_settings:
            mock_settings.embedding_route = "native"
            mock_settings.embedding_model = "jina-v4"
            emb = create_embedder(api_key="k")
        assert isinstance(emb, JinaV4Embedder)

    def test_factory_returns_openrouter_for_openrouter_route(self) -> None:
        from ingestion.embedders.openrouter import OpenRouterEmbedder

        with patch("ingestion.embedder.settings") as mock_settings:
            mock_settings.embedding_route = "openrouter"
            mock_settings.embedding_model = "gemini-2"
            emb = create_embedder(api_key="k")
        assert isinstance(emb, OpenRouterEmbedder)

    def test_factory_unknown_native_model(self) -> None:
        """gemini-2 has no native client; the factory must say so, not guess."""
        with patch("ingestion.embedder.settings") as mock_settings:
            mock_settings.embedding_route = "native"
            mock_settings.embedding_model = "gemini-2"
            with pytest.raises(ValueError, match="No native client"):
                create_embedder()


class TestProtocolCompliance:
    def test_jina_satisfies_protocol(self) -> None:
        """JinaV4Embedder is a valid Embedder."""
        assert isinstance(JinaV4Embedder.__new__(JinaV4Embedder), Embedder)

    def test_voyage_satisfies_protocol(self) -> None:
        """VoyageEmbedder is a valid Embedder."""
        from ingestion.embedders.voyage import VoyageEmbedder

        assert isinstance(VoyageEmbedder.__new__(VoyageEmbedder), Embedder)

    def test_mock_embedder_satisfies_protocol(self) -> None:
        """The test mock_embedder fixture satisfies the protocol."""
        mock = MagicMock()
        mock.embed_text = AsyncMock()
        mock.embed_text_query = AsyncMock()
        mock.embed_image = AsyncMock()
        mock.embed_multi_vector = AsyncMock()
        mock.embed_query_multi_vector = AsyncMock()
        mock.close = AsyncMock()
        mock.tokens_used = 0.0
        mock.reset_token_counter = MagicMock()
        mock.model_name = "mock-model"
        mock.dim = 512
        assert isinstance(mock, Embedder)


class TestEmbedText:
    async def test_correct_payload(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}, {"embedding": [0.2] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "jina-embeddings-v4"
        assert payload["task"] == "retrieval.passage"
        assert payload["dimensions"] == DIMS
        assert payload["normalized"] is True
        assert payload["input"] == [{"text": "hello"}, {"text": "world"}]
        assert len(result) == 2
        assert len(result[0]) == DIMS

    async def test_url_is_correct(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"])

        assert mock_post.call_args.args[0] == embedder._embeddings_url


class TestEmbedTextQuery:
    async def test_uses_retrieval_query_task(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text_query("search query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["task"] == "retrieval.query"
        assert len(result) == DIMS


class TestEmbedImage:
    async def test_correct_payload(self, embedder: JinaV4Embedder) -> None:
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

        payload = mock_post.call_args.kwargs["json"]
        assert payload["task"] == "retrieval.passage"
        expected_uri = f"data:image/png;base64,{expected_b64}"
        assert payload["input"] == [{"image": expected_uri}]
        assert len(result) == DIMS

    async def test_custom_timeout(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_image(b"fake")

        assert mock_post.call_args.kwargs["timeout"] == 120.0


class TestEmbedMultiVector:
    async def test_correct_payload_with_colbert(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [[0.1] * 128, [0.2] * 128]}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_multi_vector(b"fake")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["embedding_type_params"] == {"output_type": "colbert"}
        assert payload["dimensions"] == 128
        assert len(result) == 2
        assert len(result[0]) == 128


class TestEmbedQueryMultiVector:
    async def test_uses_retrieval_query_and_colbert(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [[0.1] * 128, [0.2] * 128]}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_query_multi_vector("query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["task"] == "retrieval.query"
        assert payload["embedding_type_params"] == {"output_type": "colbert"}
        assert payload["input"] == [{"text": "query"}]
        assert len(result) == 2


class TestErrorHandling:
    async def test_http_error_raises(self, embedder: JinaV4Embedder) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_resp.request = MagicMock()
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await embedder.embed_text(["fail"])

    async def test_retry_exhaustion_on_429(
        self,
        embedder: JinaV4Embedder,
    ) -> None:
        """EC-01: Repeated 429s exhaust retries and raise RetryError."""
        from tenacity import RetryError, wait_none

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests"
        mock_resp.headers = {"Retry-After": "0"}
        mock_resp.request = MagicMock()
        # Patch retry wait to avoid 15s+ delay from exponential backoff
        original_wait = embedder._request_with_retry.retry.wait  # type: ignore[attr-defined]
        embedder._request_with_retry.retry.wait = wait_none()  # type: ignore[attr-defined]
        try:
            with patch.object(
                embedder._client,
                "post",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ):
                with pytest.raises(RetryError):
                    await embedder.embed_text(["overloaded"])
        finally:
            embedder._request_with_retry.retry.wait = original_wait  # type: ignore[attr-defined]


DENSE_DIM = DIMS


class TestLateChunking:
    async def test_payload_includes_late_chunking(
        self,
        embedder: JinaV4Embedder,
    ) -> None:
        """When late_chunking=True, payload has 'late_chunking': True."""
        mock_resp = _mock_response(
            [
                {"embedding": [0.1] * DENSE_DIM},
                {"embedding": [0.2] * DENSE_DIM},
            ]
        )
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(
                ["chunk 1", "chunk 2"],
                late_chunking=True,
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["late_chunking"] is True

    async def test_no_late_chunking_by_default(
        self,
        embedder: JinaV4Embedder,
    ) -> None:
        """Default embed_text does NOT include late_chunking."""
        mock_resp = _mock_response([{"embedding": [0.1] * DENSE_DIM}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["hello"])

        payload = mock_post.call_args.kwargs["json"]
        assert "late_chunking" not in payload

    async def test_single_batch_when_late_chunking(
        self,
        embedder: JinaV4Embedder,
    ) -> None:
        """late_chunking=True sends all texts in one API call."""
        n_texts = 50
        mock_resp = _mock_response(
            [{"embedding": [0.1] * DENSE_DIM}] * n_texts,
        )
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            with patch("ingestion.embedders.jina.settings") as mock_s:
                mock_s.jina_batch_size = 10
                mock_s.jina_api_url = "https://api.jina.ai"
                mock_s.jina_model = "jina-embeddings-v4"
                mock_s.dense_dimensions = DENSE_DIM
                await embedder.embed_text(
                    [f"chunk {i}" for i in range(n_texts)],
                    late_chunking=True,
                )

        assert mock_post.call_count == 1
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["input"]) == n_texts


def _fast_limiters(embedder: JinaV4Embedder) -> None:
    """Replace both rate limiters with fast variants for tests."""
    # RPM: burst=2, refill 2 tokens/sec → ~0.5s wait for 3rd request
    embedder._rpm_limiter = TokenBucket(tokens_per_sec=2, burst=2)
    # TPM: effectively unlimited so it doesn't interfere
    embedder._tpm_limiter = TokenBucket(tokens_per_sec=1e9, burst=int(1e9))


class TestRateLimiter:
    async def test_rpm_throttling(self, embedder: JinaV4Embedder) -> None:
        """Requests beyond RPM limit are delayed."""
        _fast_limiters(embedder)
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])

        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            # First 2 should be instant (burst)
            t0 = time.monotonic()
            await embedder.embed_text(["a"])
            await embedder.embed_text(["b"])
            fast_elapsed = time.monotonic() - t0

            # Third should be delayed
            t1 = time.monotonic()
            await embedder.embed_text(["c"])
            slow_elapsed = time.monotonic() - t1

        assert fast_elapsed < 1.0
        assert slow_elapsed >= 0.3  # had to wait for token refill

    async def test_tpm_throttling(self, embedder: JinaV4Embedder) -> None:
        """TPM limiter delays requests that exceed token budget."""
        # RPM: unlimited. TPM: burst=10 tokens, refill 20 tokens/sec.
        embedder._rpm_limiter = TokenBucket(tokens_per_sec=1e9, burst=int(1e9))
        embedder._tpm_limiter = TokenBucket(tokens_per_sec=20, burst=10)
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])

        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            # First call: "a" ≈ 1 token (max(0.25, 1.0) = 1.0), instant
            t0 = time.monotonic()
            await embedder.embed_text(["a"])
            fast_elapsed = time.monotonic() - t0

            # Drain remaining burst with a ~9-token string
            await embedder.embed_text(["x" * 36])  # 36/4 = 9 tokens

            # Next call must wait for refill
            t1 = time.monotonic()
            await embedder.embed_text(["hello"])  # ~1.25 tokens
            slow_elapsed = time.monotonic() - t1

        assert fast_elapsed < 0.5
        assert slow_elapsed >= 0.02  # had to wait for TPM refill

    async def test_concurrent_requests_respect_semaphore(
        self,
        embedder: JinaV4Embedder,
    ) -> None:
        """Concurrency semaphore limits parallel requests."""
        _fast_limiters(embedder)
        call_count = 0
        max_concurrent = 0

        async def slow_post(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count, max_concurrent
            call_count += 1
            current = call_count
            if current > max_concurrent:
                max_concurrent = current
            await asyncio.sleep(0.1)
            call_count -= 1
            return _mock_response([{"embedding": [0.1] * DIMS}])

        embedder._semaphore = asyncio.Semaphore(2)
        with patch.object(
            embedder._client,
            "post",
            side_effect=slow_post,
        ):
            await asyncio.gather(
                embedder.embed_text(["a"]),
                embedder.embed_text(["b"]),
                embedder.embed_text(["c"]),
            )

        assert max_concurrent <= 2


class TestEstimateTokens:
    def test_text_token_estimation(self) -> None:
        """Text tokens estimated as len/4."""
        payload = {"input": [{"text": "hello world test string"}]}
        result = JinaV4Embedder._estimate_tokens(payload)
        assert result == len("hello world test string") / 4.0

    def test_image_token_estimation_uses_tile_calculation(self) -> None:
        """Image tokens estimated via 28x28 tile grid, not base64 length."""
        import io
        import os

        from PIL import Image

        # Use random noise so PNG can't compress it away
        img = Image.frombytes("RGB", (400, 300), os.urandom(400 * 300 * 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        payload = {"input": [{"image": data_uri}]}
        result = JinaV4Embedder._estimate_tokens(payload)
        assert 1000 <= result <= 2500
        old_estimate = len(b64) / 4.0
        assert result < old_estimate * 0.5

    def test_image_token_estimation_small_image(self) -> None:
        """Small images still get a reasonable token count."""
        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        payload = {"input": [{"image": data_uri}]}
        result = JinaV4Embedder._estimate_tokens(payload)
        assert 100 <= result <= 300

    def test_mixed_payload_sums_correctly(self) -> None:
        """Payload with both text and image sums both estimates."""
        payload = {
            "input": [
                {"text": "x" * 400},
                {"image": "data:image/png;base64," + "A" * 1000},
            ]
        }
        result = JinaV4Embedder._estimate_tokens(payload)
        assert result >= 100


class TestTokenCounter:
    async def test_tracks_estimated_tokens(self, embedder: JinaV4Embedder) -> None:
        """Token counter accumulates across calls."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            embedder.reset_token_counter()
            await embedder.embed_text(["hello world"])
            tokens_after_text = embedder.tokens_used

            await embedder.embed_text(["another query"])
            tokens_after_two = embedder.tokens_used

        assert tokens_after_text > 0
        assert tokens_after_two > tokens_after_text

    async def test_reset_clears_counter(self, embedder: JinaV4Embedder) -> None:
        """reset_token_counter zeroes the accumulator."""
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            await embedder.embed_text(["test"])
            embedder.reset_token_counter()

        assert embedder.tokens_used == 0.0


@pytest.mark.integration
class TestEmbedderIntegration:
    @pytest.fixture
    async def live_embedder(self) -> JinaV4Embedder:
        embedder = JinaV4Embedder()
        yield embedder  # type: ignore[misc]
        await embedder.close()

    async def test_embed_text_returns_2048d(self, live_embedder: JinaV4Embedder) -> None:
        result = await live_embedder.embed_text(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == DIMS
        assert all(isinstance(v, float) for v in result[0])

    async def test_embed_text_query_returns_2048d(self, live_embedder: JinaV4Embedder) -> None:
        result = await live_embedder.embed_text_query("test query")
        assert len(result) == DIMS

    async def test_embed_image_returns_2048d(self, live_embedder: JinaV4Embedder) -> None:
        png = (FIXTURES / "sample.png").read_bytes()
        result = await live_embedder.embed_image(png)
        assert len(result) == DIMS

    @pytest.mark.skipif(
        not __import__("config.settings", fromlist=["settings"]).settings.multivec_enabled,
        reason="multivec_enabled=False (requires jina-colbert-v2)",
    )
    async def test_embed_multi_vector_returns_128d(
        self, live_embedder: JinaV4Embedder
    ) -> None:
        png = (FIXTURES / "sample.png").read_bytes()
        result = await live_embedder.embed_multi_vector(png)
        assert len(result) > 0
        assert all(len(v) == 128 for v in result)

    @pytest.mark.skipif(
        not __import__("config.settings", fromlist=["settings"]).settings.multivec_enabled,
        reason="multivec_enabled=False (requires jina-colbert-v2)",
    )
    async def test_embed_query_multi_vector_structure(
        self, live_embedder: JinaV4Embedder
    ) -> None:
        result = await live_embedder.embed_query_multi_vector("test")
        assert len(result) > 0
        assert all(len(v) == 128 for v in result)


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

    def test_client_is_stable_within_a_loop_under_concurrency(self) -> None:
        emb = JinaV4Embedder(api_key="test-key")
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

    async def test_request_uses_the_current_loops_client(self) -> None:
        emb = JinaV4Embedder(api_key="test-key")
        client, _ = emb._ensure_loop_resources()
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await emb.embed_text(["hi"])
        mock_post.assert_awaited_once()
