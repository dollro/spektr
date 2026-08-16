from __future__ import annotations

import asyncio
import math
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedders.local import LocalEmbedder, _fit, _is_retryable

# The served checkpoint emits 4096; the collection is sized at 1024. The gap
# between the two is the whole point of this client's truncation path.
FULL_DIMS = 4096
DIMS = 1024

QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.local_embed_api_key = ""
    s.local_embed_api_url = "http://localhost:8080"
    # A real registry key: the prefixes under test come from the registry,
    # not from the mock, so these assertions pin the shipped values.
    s.embedding_model = "qwen3-embedding-8b"
    s.embedding_route = "local"
    s.embedding_model_id = "Qwen/Qwen3-Embedding-8B"
    s.dense_dimensions = DIMS
    s.local_batch_size = 32
    s.local_max_concurrent = 8
    s.max_retries = 3
    return s


@pytest.fixture
async def embedder(mock_settings: MagicMock) -> LocalEmbedder:
    with patch("ingestion.embedders.local.settings", mock_settings):
        e = LocalEmbedder()
        e._ensure_loop_resources()
        return e


def _vec(seed: float, n: int = FULL_DIMS) -> list[float]:
    """A non-degenerate full-width vector, so norms are meaningful."""
    return [seed + i * 1e-4 for i in range(n)]


def _mock_response(data: list[dict]) -> httpx.Response:  # type: ignore[type-arg]
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestInstructionPrefixes:
    """The retrieval role lives in the text, not in an `input_type` field."""

    async def test_query_gets_the_instruct_template(self, embedder: LocalEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": _vec(0.1)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text_query("what is spektr?")

        sent = mock_post.call_args.kwargs["json"]["input"]
        assert sent == [f"{QWEN3_QUERY_PREFIX}what is spektr?"]

    async def test_passage_gets_no_prefix(self, embedder: LocalEmbedder) -> None:
        """An empty prefix is the model card's answer, not an unset value."""
        mock_resp = _mock_response([{"embedding": _vec(0.1)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text(["the invoice total is 42"])

        assert mock_post.call_args.kwargs["json"]["input"] == ["the invoice total is 42"]

    async def test_query_and_passage_prefixes_differ(self, embedder: LocalEmbedder) -> None:
        """Losing the asymmetry is the failure this client exists to avoid."""
        mock_resp = _mock_response([{"embedding": _vec(0.1)}])
        sent: list[str] = []
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text(["same text"], task="passage")
            sent.append(mock_post.call_args.kwargs["json"]["input"][0])
            await embedder.embed_text(["same text"], task="query")
            sent.append(mock_post.call_args.kwargs["json"]["input"][0])

        assert sent[0] != sent[1]

    async def test_unknown_task_raises(self, embedder: LocalEmbedder) -> None:
        """An unmapped task would prepend nothing and fail silently."""
        with pytest.raises(ValueError, match="Unknown embedding task"):
            await embedder.embed_text(["hi"], task="retrieval.passage")

    async def test_input_type_is_not_sent(self, embedder: LocalEmbedder) -> None:
        """Open encoders have no `input_type` contract — sending one is noise."""
        mock_resp = _mock_response([{"embedding": _vec(0.1)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text(["hi"])

        payload = mock_post.call_args.kwargs["json"]
        assert "input_type" not in payload
        assert "task" not in payload


class TestTruncationAndNorm:
    async def test_truncates_and_renormalises(self, embedder: LocalEmbedder) -> None:
        """A full-width response against a narrower collection must be cut
        down here — Qdrant would reject it point by point otherwise — and
        renormalised, because the collection is COSINE."""
        mock_resp = _mock_response([{"embedding": _vec(0.1)}, {"embedding": _vec(0.5)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await embedder.embed_text(["a", "b"])

        assert len(result) == 2
        for vector in result:
            assert len(vector) == DIMS
            norm = math.sqrt(sum(v * v for v in vector))
            assert math.isclose(norm, 1.0, rel_tol=1e-9)

    async def test_dimensions_is_never_sent(self, embedder: LocalEmbedder) -> None:
        """TEI ignores it and vLLM's support varies, so it is not asked for."""
        mock_resp = _mock_response([{"embedding": _vec(0.1)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await embedder.embed_text(["hi"])

        assert "dimensions" not in mock_post.call_args.kwargs["json"]

    async def test_per_call_dimension_override(self, embedder: LocalEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": _vec(0.1)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await embedder.embed_text(["hi"], dimensions=256)

        assert len(result[0]) == 256

    async def test_short_response_raises(self, embedder: LocalEmbedder) -> None:
        """Truncation can only shrink; a narrower response is a misconfigured
        checkpoint and must not reach Qdrant."""
        mock_resp = _mock_response([{"embedding": _vec(0.1, n=768)}])
        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            with pytest.raises(ValueError, match="fewer than the configured"):
                await embedder.embed_text(["hi"])

    def test_zero_vector_survives_renormalisation(self) -> None:
        """No ZeroDivisionError on a degenerate vector — it stays zero."""
        assert _fit([0.0] * FULL_DIMS, DIMS) == [0.0] * DIMS

    def test_already_unit_vector_is_left_unit(self) -> None:
        vector = [1.0] + [0.0] * (FULL_DIMS - 1)
        fitted = _fit(vector, DIMS)
        assert math.isclose(math.sqrt(sum(v * v for v in fitted)), 1.0)


class TestBatching:
    async def test_long_input_is_split_into_batches(self, embedder: LocalEmbedder) -> None:
        embedder._batch_size = 32
        texts = [f"chunk {i}" for i in range(70)]

        def _resp(*args: object, **kwargs: object) -> httpx.Response:
            n = len(kwargs["json"]["input"])  # type: ignore[index,call-overload]
            return _mock_response([{"embedding": _vec(0.1), "index": i} for i in range(n)])

        with patch.object(
            embedder._client, "post", new_callable=AsyncMock, side_effect=_resp
        ) as mock_post:
            result = await embedder.embed_text(texts)

        sizes = [len(c.kwargs["json"]["input"]) for c in mock_post.call_args_list]
        assert sizes == [32, 32, 6]
        assert len(result) == 70

    async def test_batch_order_preserved_by_index(self, embedder: LocalEmbedder) -> None:
        """Chunk N's vector must stay with chunk N. The response carries an
        explicit `index`; position is not trustworthy."""
        embedder._batch_size = 2

        def _resp(*args: object, **kwargs: object) -> httpx.Response:
            n = len(kwargs["json"]["input"])  # type: ignore[index,call-overload]
            # Deliberately shuffled, with the true position in `index`.
            items = [
                {"embedding": [float(i + 1)] + [0.0] * (FULL_DIMS - 1), "index": i}
                for i in range(n)
            ]
            return _mock_response(list(reversed(items)))

        with patch.object(embedder._client, "post", new_callable=AsyncMock, side_effect=_resp):
            result = await embedder.embed_text(["a", "b", "c", "d"])

        # Every vector renormalises to [1.0, 0, ...], so identity is carried
        # by the ordering itself: index 0 first, index 1 second, per batch.
        assert len(result) == 4
        assert all(math.isclose(v[0], 1.0) for v in result)

    async def test_shuffled_response_is_reordered_by_index(
        self, embedder: LocalEmbedder
    ) -> None:
        """The same guard, asserted on distinguishable values rather than
        on positions that renormalisation would flatten."""
        embedder._batch_size = 4

        def _resp(*args: object, **kwargs: object) -> httpx.Response:
            items = [{"embedding": _vec(float(i + 1)), "index": i} for i in range(4)]
            return _mock_response([items[2], items[0], items[3], items[1]])

        with patch.object(embedder._client, "post", new_callable=AsyncMock, side_effect=_resp):
            result = await embedder.embed_text(["a", "b", "c", "d"])

        expected = [_fit(_vec(float(i + 1)), DIMS) for i in range(4)]
        assert result == expected


class TestUnsupported:
    async def test_embed_image_raises(self, embedder: LocalEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="text-only"):
            await embedder.embed_image(b"\x89PNG")

    async def test_embed_multi_vector_raises(self, embedder: LocalEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="multivec slot"):
            await embedder.embed_multi_vector(b"\x89PNG")

    async def test_embed_query_multi_vector_raises(self, embedder: LocalEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="multivec slot"):
            await embedder.embed_query_multi_vector("q")


class TestAuthHeader:
    async def test_no_bearer_when_key_is_empty(self, embedder: LocalEmbedder) -> None:
        """A TEI container is normally unauthenticated; an empty bearer would
        be a header that some proxies reject outright."""
        assert "Authorization" not in embedder._client.headers  # type: ignore[union-attr]

    async def test_bearer_attached_when_key_is_set(self, mock_settings: MagicMock) -> None:
        mock_settings.local_embed_api_key = "proxy-token"
        with patch("ingestion.embedders.local.settings", mock_settings):
            e = LocalEmbedder()
            client, _ = e._ensure_loop_resources()
        assert client.headers["Authorization"] == "Bearer proxy-token"


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
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x.test"))
        err = httpx.HTTPStatusError("bad", request=resp.request, response=resp)
        assert _is_retryable(err) is False

    def test_server_errors_and_429_are_retryable(self) -> None:
        for code in (429, 500, 503):
            resp = httpx.Response(code, request=httpx.Request("POST", "http://x.test"))
            err = httpx.HTTPStatusError("x", request=resp.request, response=resp)
            assert _is_retryable(err) is True, code


class TestRateLimiting:
    def test_has_no_token_bucket(self, mock_settings: MagicMock) -> None:
        """A local endpoint has no RPM quota — only finite GPU memory, which
        the semaphore expresses and a rate cap does not."""
        with patch("ingestion.embedders.local.settings", mock_settings):
            e = LocalEmbedder()
        assert not any("limiter" in name for name in vars(e))

    async def test_semaphore_sized_from_settings(self, mock_settings: MagicMock) -> None:
        with patch("ingestion.embedders.local.settings", mock_settings):
            e = LocalEmbedder()
            _, semaphore = e._ensure_loop_resources()
        assert semaphore._value == 8


class TestEventLoopIsolation:
    """Regression: one shared embedder is used from several event loops.

    CocoIndex runs file components concurrently on multiple worker threads,
    each with its own asyncio loop, all sharing a single embedder instance.
    A single ``self._client`` slot would be clobbered whenever the loop
    changed, leaving an in-flight request awaiting a connection-pool lock
    owned by a different loop (``RuntimeError: the current task is not
    holding this lock``).
    """

    def test_client_is_stable_within_a_loop_under_concurrency(
        self, mock_settings: MagicMock
    ) -> None:
        with patch("ingestion.embedders.local.settings", mock_settings):
            emb = LocalEmbedder()

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
        with patch("ingestion.embedders.local.settings", mock_settings):
            emb = LocalEmbedder()
            client, _ = emb._ensure_loop_resources()
            mock_resp = _mock_response([{"embedding": _vec(0.1)}])
            with patch.object(
                client, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                await emb.embed_text(["hi"])
            mock_post.assert_awaited_once()


class TestProtocolCompliance:
    def test_implements_embedder_protocol(self) -> None:
        from ingestion.embedder import Embedder

        assert isinstance(LocalEmbedder.__new__(LocalEmbedder), Embedder)


class TestRegistryEntry:
    """The prefixes are load-bearing config, not documentation."""

    def test_qwen3_is_text_only_and_local_only(self) -> None:
        from config.embedding_models import spec

        s = spec("qwen3-embedding-8b")
        assert s.routes == ("local",)
        assert s.default_route == "local"
        assert s.image_routes == ()
        assert s.multivector_routes == ()
        assert s.local_model_id == "Qwen/Qwen3-Embedding-8B"

    def test_matryoshka_ceiling_matches_the_model_card(self) -> None:
        from config.embedding_models import spec

        s = spec("qwen3-embedding-8b")
        assert s.default_dimensions == 4096
        assert s.allowed_dimensions == ()

    def test_query_prefix_matches_the_instruct_template(self) -> None:
        """Verbatim from get_detailed_instruct, missing space included."""
        from config.embedding_models import spec

        s = spec("qwen3-embedding-8b")
        assert s.query_prefix == QWEN3_QUERY_PREFIX
        assert s.query_prefix.endswith("\nQuery:")
        assert s.passage_prefix == ""
