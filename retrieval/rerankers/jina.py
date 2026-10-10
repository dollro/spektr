"""Jina rerank protocol: POST {base}/v1/rerank.

Also serves Cohere, whose /v2/rerank uses the same request keys and the same
`{"results": [{"index", "relevance_score"}]}` envelope — only the path and the
base URL differ. Registering a Cohere model against this protocol is therefore
a registry entry plus a path override, not a new adapter.

Verified response shape (retrieval/rerank.py's own module docstring, checked
live 2026-08-01 for both v2 and v3.5): only `index` and `relevance_score` are
read, which is why the v2 -> v3.5 change of the unused `document` field from a
string to an object needed no code change here.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import settings
from retrieval.rerankers._http import is_retryable, raise_for_rerank_status

logger = logging.getLogger(__name__)


class JinaReranker:
    """Listwise reranker over Jina's (and Cohere's) /rerank contract."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        wire_id: str,
        path: str = "/v1/rerank",
    ) -> None:
        self._url = f"{base_url.rstrip('/')}{path}"
        self._api_key = api_key
        self._wire_id = wire_id

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return (original index, score) pairs, best first.

        A fresh client per call is deliberate. The embedder clients need the
        `WeakKeyDictionary` loop-affinity dance (`ingestion/embedders/jina.py:94`)
        because CocoIndex shares one embedder across worker threads, each with
        its own event loop, and a cached `AsyncClient` then deadlocks on a
        connection-pool lock owned by another loop. Reranking is one call per
        *query*, not per chunk, so the pooling that dance protects is worth
        far less here than the failure mode it avoids.
        """
        payload = {
            "model": self._wire_id,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        data = await self._request(payload)
        return [
            (int(item["index"]), float(item["relevance_score"])) for item in data["results"]
        ]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(is_retryable),
        # Surface the real HTTPStatusError rather than tenacity's RetryError.
        # `rerank()` degrades via `RerankError(str(exc))`, and str(RetryError)
        # is "RetryError[<Future at 0x... state=finished raised ...>]" — which
        # is what an operator would see in `degraded` diagnostics instead of
        # the status code and body that would tell them what to fix.
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "Jina rerank retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else None,
        ),
    )
    async def _request(self, payload: dict) -> dict:  # type: ignore[type-arg]
        async with httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        ) as client:
            resp = await client.post(self._url, json=payload)
            raise_for_rerank_status(resp, "Jina")
            return resp.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        """No-op: clients are per-call, so there is nothing to hold open."""
        return
