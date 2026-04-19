"""Jina Reranker v2 for re-scoring search results.

Calls the Jina Reranker API to re-rank results by relevance
to the query. Enabled via settings.rerank_enabled.
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

logger = logging.getLogger(__name__)

RERANK_URL = f"{settings.jina_api_url}/v1/rerank"
RERANK_MODEL = "jina-reranker-v2-base-multilingual"


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
    """Call Jina Reranker API and return ranked results."""
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
                "model": RERANK_MODEL,
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
    results: list[dict],  # type: ignore[type-arg]
    top_k: int = 5,
) -> list[dict]:  # type: ignore[type-arg]
    """Re-rank search results using Jina Reranker v2.

    Sends result texts to the Jina Reranker API and merges
    reranker scores back into the result dicts. Original scores
    are preserved as 'original_score'.

    Args:
        query: The search query.
        results: Search result dicts (must contain 'text').
        top_k: Number of top results to return.

    Returns:
        Re-ranked results sorted by relevance score.
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
        idx = item["index"]
        original = results[idx].copy()
        original["original_score"] = original.get("score")
        original["score"] = item["relevance_score"]
        reranked.append(original)

    return reranked
