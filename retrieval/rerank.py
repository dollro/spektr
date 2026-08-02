"""Jina Reranker v3.5 — listwise re-scoring of retrieved candidates.

v3.5 ranks the whole candidate list in one forward pass ("last but not late"
interaction) rather than scoring each document independently, which is why it
outperforms the pointwise v2 it replaces. The /v1/rerank request schema is
unchanged, so this is a model-string swap plus typed plumbing.

Response shape verified live for both models on 2026-08-01: both return
``{"results": [{"index": int, "relevance_score": float, ...}]}``. The only
difference is the (unused) ``document`` field — a plain string on v2, an
``{"text": ...}`` object on v3.5. Parsing here only reads ``index`` and
``relevance_score``, so no shape adjustment is needed.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from retrieval.models import FusedResult

logger = logging.getLogger(__name__)

RERANK_URL = f"{settings.jina_api_url}/v1/rerank"


class RerankError(RuntimeError):
    """Raised when reranking fails and the caller must handle degradation."""


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(settings.max_retries),
    retry=retry_if_exception_type((httpx.HTTPStatusError,)),
)
async def _rerank_request(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[dict]:  # type: ignore[type-arg]
    """Call the Jina Reranker API and return its ranked results."""
    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(30.0),
    ) as client:
        resp = await client.post(
            RERANK_URL,
            json={
                "model": settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Jina Reranker error: {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()["results"]  # type: ignore[no-any-return]


async def rerank(
    query: str,
    results: list[FusedResult],
    top_k: int = 5,
) -> list[FusedResult]:
    """Re-score fused results against the query.

    Args:
        query: The original user query, never a sub-query.
        results: Fused candidates to rescore.
        top_k: Number of results to return.

    Returns:
        Results ordered by rerank score.

    Raises:
        RerankError: The API call failed. The caller decides how to degrade —
            it is the only layer that knows whether it can report `degraded`.
    """
    if not results:
        return []

    documents = [r.text for r in results]
    if not any(documents):
        return results[:top_k]

    try:
        ranked = await _rerank_request(query, documents, top_k)
    except Exception as exc:
        logger.exception("Reranking failed")
        raise RerankError(str(exc)) from exc

    out: list[FusedResult] = []
    for item in ranked:
        original = results[item["index"]]
        out.append(original.model_copy(update={"score": item["relevance_score"]}))
    return out


async def rerank_dicts(
    query: str,
    results: list[dict],  # type: ignore[type-arg]
    top_k: int = 5,
) -> list[dict]:  # type: ignore[type-arg]
    """Dict-based reranking for callers that predate the typed pipeline.

    Preserves the original score as 'original_score'. Used by vector_search.
    """
    if not results:
        return results

    documents = [r.get("text", "") for r in results]
    if not any(documents):
        return results[:top_k]

    try:
        ranked = await _rerank_request(query, documents, top_k)
    except Exception:
        logger.exception("Reranking failed, returning originals")
        return results[:top_k]

    reranked: list[dict] = []  # type: ignore[type-arg]
    for item in ranked:
        original = results[item["index"]].copy()
        original["original_score"] = original.get("score")
        original["score"] = item["relevance_score"]
        reranked.append(original)
    return reranked
