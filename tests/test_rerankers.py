"""Wire-contract tests for the reranker protocol adapters.

These adapters exist because rerank APIs did not converge the way
`/v1/embeddings` did. Every assertion here pins one of the divergences that
would otherwise be a silent wrong answer rather than an error: the documents
key, the response envelope, and the score key all differ between the two
protocols.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from retrieval.rerankers._http import is_retryable
from retrieval.rerankers.jina import JinaReranker
from retrieval.rerankers.tei import TEIReranker


def _response(payload: Any, status: int = 200) -> httpx.Response:
    """Build a real httpx.Response so status handling is not mocked away."""
    return httpx.Response(
        status_code=status,
        content=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        request=httpx.Request("POST", "http://test/rerank"),
    )


def _patch_post(response: httpx.Response) -> Any:
    return patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response))


@pytest.mark.asyncio
class TestJinaProtocol:
    async def test_reads_results_envelope_and_relevance_score(self) -> None:
        payload = {
            "results": [
                {"index": 2, "relevance_score": 4.5, "document": {"text": "c"}},
                {"index": 0, "relevance_score": -1.25, "document": {"text": "a"}},
            ]
        }
        reranker = JinaReranker(base_url="https://api.jina.ai", api_key="k", wire_id="m")
        with _patch_post(_response(payload)):
            out = await reranker.rerank("q", ["a", "b", "c"], top_n=2)
        assert out == [(2, 4.5), (0, -1.25)]

    async def test_negative_scores_survive(self) -> None:
        """v3.5 emits unbounded logits; a negative score is meaningful, not an error."""
        reranker = JinaReranker(base_url="https://api.jina.ai", api_key="k", wire_id="m")
        with _patch_post(_response({"results": [{"index": 0, "relevance_score": -8.0}]})):
            out = await reranker.rerank("q", ["a"], top_n=1)
        assert out == [(0, -8.0)]

    async def test_sends_model_documents_and_top_n(self) -> None:
        reranker = JinaReranker(
            base_url="https://api.jina.ai", api_key="k", wire_id="jina-reranker-v3.5"
        )
        post = AsyncMock(return_value=_response({"results": []}))
        with patch.object(httpx.AsyncClient, "post", post):
            await reranker.rerank("q", ["a", "b"], top_n=5)
        body = post.await_args.kwargs["json"]
        assert body["model"] == "jina-reranker-v3.5"
        assert body["documents"] == ["a", "b"]
        assert body["top_n"] == 5

    async def test_url_has_no_double_slash(self) -> None:
        reranker = JinaReranker(base_url="https://api.jina.ai/", api_key="k", wire_id="m")
        assert reranker._url == "https://api.jina.ai/v1/rerank"


@pytest.mark.asyncio
class TestTEIProtocol:
    async def test_reads_bare_list_and_score_key(self) -> None:
        """TEI returns a bare list with `score`.

        Not `{"results": [...]}`, and not `relevance_score` — both differ from
        the Jina protocol, and both would read as an empty result rather than
        an error if the adapter got them wrong.
        """
        payload = [{"index": 0, "score": 0.2}, {"index": 1, "score": 0.9}]
        reranker = TEIReranker(base_url="http://localhost:8080", wire_id="qwen3-reranker")
        with _patch_post(_response(payload)):
            out = await reranker.rerank("q", ["a", "b"], top_n=2)
        assert out == [(1, 0.9), (0, 0.2)], "must sort by score descending"

    async def test_sends_texts_not_documents(self) -> None:
        reranker = TEIReranker(base_url="http://localhost:8080")
        post = AsyncMock(return_value=_response([]))
        with patch.object(httpx.AsyncClient, "post", post):
            await reranker.rerank("q", ["a", "b"], top_n=2)
        body = post.await_args.kwargs["json"]
        assert body["texts"] == ["a", "b"]
        assert "documents" not in body
        assert "model" not in body, "TEI serves one model, chosen at container start"

    async def test_truncates_to_top_n_client_side(self) -> None:
        """TEI has no top_n parameter, so the adapter must honour it itself."""
        payload = [{"index": i, "score": i / 10} for i in range(5)]
        reranker = TEIReranker(base_url="http://localhost:8080")
        with _patch_post(_response(payload)):
            out = await reranker.rerank("q", ["a"] * 5, top_n=2)
        assert len(out) == 2
        assert out == [(4, 0.4), (3, 0.3)]

    @pytest.mark.parametrize(("api_key", "expected"), [("", False), ("secret", True)])
    async def test_auth_header_only_sent_when_key_configured(
        self, api_key: str, expected: bool
    ) -> None:
        """An empty key must not become the literal header 'Bearer '.

        Captures the constructor kwargs rather than inspecting the mock's
        bound self — patching the method with an AsyncMock detaches it from
        the instance, so the client is not recoverable from the call args.
        """
        captured: dict[str, Any] = {}
        real_init = httpx.AsyncClient.__init__

        def spy_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs.get("headers") or {})
            real_init(self, *args, **kwargs)

        reranker = TEIReranker(base_url="http://localhost:8080", api_key=api_key)
        with (
            patch.object(httpx.AsyncClient, "__init__", spy_init),
            _patch_post(_response([])),
        ):
            await reranker.rerank("q", ["a"], top_n=1)
        assert ("Authorization" in captured) is expected


@pytest.mark.asyncio
class TestErrorHandling:
    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    async def test_client_errors_are_not_retried(self, status: int) -> None:
        """The predicate this replaces retried 401s three times before degrading."""
        reranker = JinaReranker(base_url="https://api.jina.ai", api_key="", wire_id="m")
        post = AsyncMock(return_value=_response({"detail": "no"}, status=status))
        with patch.object(httpx.AsyncClient, "post", post):
            with pytest.raises(httpx.HTTPStatusError):
                await reranker.rerank("q", ["a"], top_n=1)
        assert post.await_count == 1, "4xx must fail on the first attempt"

    @pytest.mark.parametrize("status", [429, 500, 503])
    async def test_server_errors_are_retried(self, status: int) -> None:
        """Exhausted retries must surface the real HTTPStatusError.

        Without `reraise=True` this is a tenacity RetryError, whose str() is
        an opaque Future repr — and `rerank()` degrades via
        `RerankError(str(exc))`, so that repr is what an operator would see.
        """
        reranker = JinaReranker(base_url="https://api.jina.ai", api_key="k", wire_id="m")
        post = AsyncMock(return_value=_response({"detail": "later"}, status=status))
        with patch.object(httpx.AsyncClient, "post", post):
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await reranker.rerank("q", ["a"], top_n=1)
        assert post.await_count > 1, "429/5xx must be retried"
        assert str(status) in str(excinfo.value), "status code must reach the caller"


class TestRetryPredicate:
    def test_timeout_is_retryable(self) -> None:
        """httpx.TimeoutException is a SIBLING of NetworkError, not a ConnectError
        subclass — listing ConnectError alone does not cover a connect timeout."""
        assert is_retryable(httpx.ConnectTimeout("slow"))
        assert is_retryable(httpx.ReadTimeout("slow"))

    def test_transient_network_errors_are_retryable(self) -> None:
        assert is_retryable(httpx.ConnectError("refused"))
        assert is_retryable(httpx.RemoteProtocolError("truncated"))

    def test_value_error_is_not_retryable(self) -> None:
        assert not is_retryable(ValueError("programmer error"))
