"""Relevance gate deciding whether a retrieval pass needs widening.

The failure this targets is "the right chunk was at rank 60 and we fetched
20". A weak top-1 rerank score is the cheapest available signal for it. The
retry widens the candidate pool and re-ranks; it makes no extra LLM call.
"""

from __future__ import annotations

from retrieval.models import FusedResult


def should_retry(results: list[FusedResult], floor: float) -> bool:
    """Return True when the best result is too weak to trust.

    Args:
        results: Reranked results, best first.
        floor: Minimum acceptable top-1 score. Inclusive.

    Returns:
        True if the caller should widen the pool and retry once.
    """
    if not results:
        return True
    return results[0].score < floor
