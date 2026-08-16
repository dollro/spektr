"""TEI rerank protocol: POST {base}/rerank.

The self-hosted path — HuggingFace text-embeddings-inference, Infinity, and
vLLM's rerank endpoint all speak this shape. Three things differ from the Jina
protocol, and every one of them is a silent-wrong-answer if missed:

* the documents key is `texts`, not `documents`;
* the response is a BARE LIST, not `{"results": [...]}`;
* the score key is `score`, not `relevance_score`.

There is also no `model` field: a TEI process serves exactly one model, chosen
at container start. The registry's `wire_id` is therefore recorded for
observability but never sent — mismatching it against what the container
actually loaded produces no error, just embeddings from a different model,
which is why `model_name` surfaces it.

Scores here are [0,1] sigmoid, NOT the unbounded logits jina-reranker-v3.5
emits. That is the whole reason `RerankSpec` carries `score_scale` — see
config/rerank_models.py and retrieval/gate.py.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import settings
from retrieval.rerankers._http import is_retryable, raise_for_rerank_status

logger = logging.getLogger(__name__)


class TEIReranker:
    """Listwise reranker over the TEI / Infinity / vLLM /rerank contract."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        wire_id: str = "",
        path: str = "/rerank",
    ) -> None:
        self._url = f"{base_url.rstrip('/')}{path}"
        self._api_key = api_key
        self._wire_id = wire_id

    @property
    def model_name(self) -> str:
        """The model this adapter BELIEVES it is talking to.

        TEI does not echo the served model, so this is the configured value,
        not an observed one. Treat a mismatch as possible.
        """
        return self._wire_id

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return (original index, score) pairs, best first.

        TEI has no `top_n` parameter — it scores and returns every text. The
        truncation is applied here so the adapter honours the same contract as
        the hosted ones.

        `raw_scores: False` is LOAD-BEARING, not a formatting preference. The
        Qwen3-Reranker checkpoint emits raw logit differences; this flag is
        what makes TEI apply the sigmoid server-side, and that in turn is the
        only reason `config/rerank_models.py` can record
        `score_scale="unit"` for it. Flipping this to True changes the scale
        out from under `RERANK_SCORE_FLOOR` and silently disables the
        relevance gate — the exact failure the registry exists to prevent.
        """
        payload = {"query": query, "texts": documents, "raw_scores": False}
        data = await self._request(payload)
        pairs = [(int(item["index"]), float(item["score"])) for item in data]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:top_n]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(is_retryable),
        # See the same flag in jina.py: propagate the real error so a failure
        # reaches `degraded` as a status code rather than a RetryError repr.
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "TEI rerank retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else None,
        ),
    )
    async def _request(self, payload: dict) -> list[dict]:  # type: ignore[type-arg]
        headers = {"Content-Type": "application/json"}
        # TEI is usually unauthenticated on a private network; send the header
        # only when a key is configured so an empty one is not read as the
        # literal string "Bearer ".
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(30.0)) as client:
            resp = await client.post(self._url, json=payload)
            raise_for_rerank_status(resp, "TEI")
            return resp.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        """No-op: clients are per-call, so there is nothing to hold open."""
        return
