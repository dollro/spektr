"""Composition of retrieval stages into named pipelines.

fast_pipeline  — channels -> RRF -> rerank. No LLM.
smart_pipeline — decompose -> (channels per sub-query) -> RRF -> rerank
                 -> gate -> optional single retry.

smart_pipeline delegates its core to the same channel/fusion/rerank code
fast_pipeline uses, so "hybrid is multi plus two stages" is true in code and
not only in the docs.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field
from qdrant_client import models

from config.settings import settings
from retrieval.channels import (
    build_filter,
    build_kb_filter,
    build_live_filter,
    dense_channel,
    sparse_channel,
)
from retrieval.decompose import decompose
from retrieval.fusion import rrf
from retrieval.gate import log_gate_decision, should_retry
from retrieval.models import Candidate, FusedResult
from retrieval.rerank import RerankError, rerank

logger = logging.getLogger(__name__)


class PipelineOutput(BaseModel):
    """Result of a retrieval pipeline run."""

    results: list[FusedResult] = Field(default_factory=list)
    live_results: list[FusedResult] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    retried: bool = False


def _merge_degraded(kb: list[str], live: list[str]) -> list[str]:
    """Union of two degraded-channel lists, order-preserving, deduped."""
    merged = list(kb)
    for name in live:
        if name not in merged:
            merged.append(name)
    return merged


async def _gather_channels(
    queries: list[str],
    limit: int,
    query_filter: models.Filter | None,
) -> tuple[list[list[Candidate]], list[str]]:
    """Run dense and sparse for every query concurrently.

    Returns the channel lists plus the names of channels that failed.
    """
    tasks: list[tuple[str, asyncio.Task[list[Candidate]]]] = []
    for query in queries:
        tasks.append(
            (
                "dense",
                asyncio.create_task(
                    dense_channel(query, limit=limit, query_filter=query_filter)
                ),
            )
        )
        if settings.sparse_enabled:
            tasks.append(
                (
                    "sparse",
                    asyncio.create_task(
                        sparse_channel(query, limit=limit, query_filter=query_filter)
                    ),
                )
            )

    channels: list[list[Candidate]] = []
    degraded: list[str] = []
    for name, task in tasks:
        try:
            channels.append(await task)
        except Exception:
            logger.exception("%s channel failed", name)
            if name not in degraded:
                degraded.append(name)
    return channels, degraded


async def _retrieve_and_rank(
    queries: list[str],
    rank_query: str,
    limit: int,
    query_filter: models.Filter | None,
) -> tuple[list[FusedResult], list[str]]:
    """Channels -> RRF -> rerank. The shared core of both pipelines."""
    channels, degraded = await _gather_channels(queries, limit, query_filter)
    if not channels:
        return [], degraded

    fused = rrf(channels, k=settings.rrf_k)
    if not settings.rerank_enabled or not fused:
        return fused[:limit], degraded

    candidates = fused[: settings.rerank_candidates]
    try:
        ranked = await rerank(rank_query, candidates, top_k=limit)
    except RerankError:
        degraded.append("rerank")
        return candidates[:limit], degraded
    return ranked, degraded


async def _run_channel_set(
    queries: list[str],
    rank_query: str,
    limit: int,
    query_filter: models.Filter | None,
    *,
    gate: bool = True,
) -> tuple[list[FusedResult], list[str], bool]:
    """Retrieve+rank with an optional relevance-gated single retry.

    Used by smart_pipeline for the KB and live channel sets independently,
    so each retains its own `limit`-sized pool. gate=False skips the retry
    entirely — used for the live channel set, where an empty or weak result
    just means the session is small, not that the pool needs widening: a
    session's chunk set is fully retrievable at the original limit, so
    retrying gains nothing and only costs an extra round trip.
    """
    results, degraded = await _retrieve_and_rank(queries, rank_query, limit, query_filter)

    if not (gate and settings.retry_enabled):
        return results, degraded, False

    top_score = results[0].score if results else None
    if not should_retry(results, settings.rerank_score_floor):
        log_gate_decision(fired=False, top_score=top_score, degraded=degraded)
        return results, degraded, False

    # The gate judged the FIRST pass, so its health is what makes top_score
    # interpretable. `degraded` is rebound below by the widened pass.
    judged_degraded = degraded
    widened = limit * settings.retry_limit_multiplier
    results, degraded = await _retrieve_and_rank(queries, rank_query, widened, query_filter)
    results = results[:limit]
    log_gate_decision(
        fired=True,
        top_score=top_score,
        degraded=judged_degraded,
        top_score_after=results[0].score if results else None,
        widened_to=widened,
    )
    return results, degraded, True


async def fast_pipeline(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> PipelineOutput:
    """Deterministic retrieval: channels -> RRF -> rerank. No LLM calls.

    When session_id is set, the KB and live-session channel sets are run
    TWICE — once scoped to bulk KB data, once scoped to the session — so
    each gets a full `limit`-sized result set. A single query with
    session_id as a narrowing filter would silently exclude the KB
    entirely; see docs/server/search-tools.md and _dual_query's history
    in server/tools/vector_search.py.
    """
    if session_id is None:
        query_filter = build_filter(content_type, source_file)
        results, degraded = await _retrieve_and_rank([query], query, limit, query_filter)
        return PipelineOutput(results=results, degraded=degraded)

    kb_task = asyncio.create_task(
        _retrieve_and_rank([query], query, limit, build_kb_filter(content_type, source_file))
    )
    live_task = asyncio.create_task(
        _retrieve_and_rank([query], query, limit, build_live_filter(session_id))
    )
    (kb_results, kb_degraded), (live_results, live_degraded) = await asyncio.gather(
        kb_task, live_task
    )
    return PipelineOutput(
        results=kb_results,
        live_results=live_results,
        degraded=_merge_degraded(kb_degraded, live_degraded),
    )


async def smart_pipeline(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> PipelineOutput:
    """LLM-augmented retrieval: decomposition plus a gated single retry.

    Decomposition runs once against the original query; the resulting
    sub-queries drive both the KB and live channel sets when session_id is
    set (see fast_pipeline's docstring for why dual retrieval is needed).
    """
    try:
        sub_queries = await decompose(query)
    except Exception:
        logger.exception("Decomposition failed")
        sub_queries = [query]

    if session_id is None:
        query_filter = build_filter(content_type, source_file)
        results, degraded, retried = await _run_channel_set(
            sub_queries, query, limit, query_filter
        )
        return PipelineOutput(
            results=results, degraded=degraded, sub_queries=sub_queries, retried=retried
        )

    kb_task = asyncio.create_task(
        _run_channel_set(sub_queries, query, limit, build_kb_filter(content_type, source_file))
    )
    live_task = asyncio.create_task(
        _run_channel_set(sub_queries, query, limit, build_live_filter(session_id), gate=False)
    )
    (
        (kb_results, kb_degraded, kb_retried),
        (live_results, live_degraded, live_retried),
    ) = await asyncio.gather(kb_task, live_task)
    return PipelineOutput(
        results=kb_results,
        live_results=live_results,
        degraded=_merge_degraded(kb_degraded, live_degraded),
        sub_queries=sub_queries,
        retried=kb_retried or live_retried,
    )
