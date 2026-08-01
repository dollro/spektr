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
from retrieval.channels import build_filter, dense_channel, sparse_channel
from retrieval.decompose import decompose
from retrieval.fusion import rrf
from retrieval.gate import should_retry
from retrieval.models import Candidate, FusedResult
from retrieval.rerank import RerankError, rerank

logger = logging.getLogger(__name__)


class PipelineOutput(BaseModel):
    """Result of a retrieval pipeline run."""

    results: list[FusedResult] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    retried: bool = False


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


async def fast_pipeline(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> PipelineOutput:
    """Deterministic retrieval: channels -> RRF -> rerank. No LLM calls."""
    query_filter = build_filter(content_type, source_file, session_id)
    results, degraded = await _retrieve_and_rank([query], query, limit, query_filter)
    return PipelineOutput(results=results, degraded=degraded)


async def smart_pipeline(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> PipelineOutput:
    """LLM-augmented retrieval: decomposition plus a gated single retry."""
    query_filter = build_filter(content_type, source_file, session_id)

    try:
        sub_queries = await decompose(query)
    except Exception:
        logger.exception("Decomposition failed")
        sub_queries = [query]

    results, degraded = await _retrieve_and_rank(sub_queries, query, limit, query_filter)

    retried = False
    if settings.retry_enabled and should_retry(results, settings.rerank_score_floor):
        widened = limit * settings.retry_limit_multiplier
        logger.info("Relevance gate fired, widening pool to %d", widened)
        results, degraded = await _retrieve_and_rank(sub_queries, query, widened, query_filter)
        results = results[:limit]
        retried = True

    return PipelineOutput(
        results=results,
        degraded=degraded,
        sub_queries=sub_queries,
        retried=retried,
    )
