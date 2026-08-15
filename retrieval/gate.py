"""Relevance gate deciding whether a retrieval pass needs widening.

The failure this targets is "the right chunk was at rank 60 and we fetched
20". A weak top-1 rerank score is the cheapest available signal for it. The
retry widens the candidate pool and re-ranks; it makes no extra LLM call.

Every gate evaluation is logged — fired or not — so the retry *rate* has a
denominator. ``scripts/retry_stats.py`` aggregates those records.
"""

from __future__ import annotations

import logging

from config.settings import settings
from retrieval.models import FusedResult

logger = logging.getLogger(__name__)


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


def _improved(before: float | None, after: float | None) -> bool:
    """Did the widened pass produce a better top-1 than the first pass?"""
    if after is None:
        return False
    if before is None:
        return True  # nothing -> something
    return after > before


def log_gate_decision(
    *,
    fired: bool,
    top_score: float | None,
    degraded: list[str],
    top_score_after: float | None = None,
    widened_to: int | None = None,
) -> None:
    """Emit one structured record per gate evaluation.

    One record per gated run — fired or not — because a retry count without
    a denominator cannot answer "how often".

    Args:
        fired: Whether the gate triggered a retry.
        top_score: Top-1 score before any retry. None when the first pass
            returned nothing.
        degraded: Channel names that failed during the first pass.
        top_score_after: Top-1 score after the retry. Only set when fired.
        widened_to: Widened candidate pool size. Only set when fired.

    The ``gate_reranked`` field is what makes the rate interpretable. When
    the reranker is disabled or degraded, ``FusedResult.score`` carries the
    RRF fusion score instead, which is always positive and so never falls
    below the default floor of 0.0. Those runs cannot fire by construction
    and must be excluded from the denominator rather than counted as
    "gate did not fire".
    """
    extra: dict[str, object] = {
        "gate_fired": fired,
        "gate_reranked": settings.rerank_enabled and "rerank" not in degraded,
        "top_score": top_score,
    }
    if fired:
        extra["top_score_after"] = top_score_after
        extra["gate_widened_to"] = widened_to
        extra["retry_helped"] = _improved(top_score, top_score_after)
    logger.info("Relevance gate evaluated", extra=extra)
