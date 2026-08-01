"""Tests for retrieval pipeline composition and degradation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.models import Candidate
from retrieval.pipeline import fast_pipeline, smart_pipeline


def _cand(doc_id: str, channel: str, score: float = 0.9) -> Candidate:
    return Candidate(
        id=doc_id, text=f"t-{doc_id}", source_file="d.pdf", score=score, channel=channel
    )


def _passthrough_rerank(query, results, top_k):  # type: ignore[no-untyped-def]
    return results[:top_k]


@pytest.mark.asyncio
async def test_fast_pipeline_fuses_both_channels() -> None:
    """Dense and sparse both contribute to the fused output."""
    with (
        patch(
            "retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])
        ),
        patch(
            "retrieval.pipeline.sparse_channel", AsyncMock(return_value=[_cand("b", "sparse")])
        ),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
    ):
        out = await fast_pipeline("q", limit=10)

    assert {r.id for r in out.results} == {"a", "b"}
    assert out.degraded == []


@pytest.mark.asyncio
async def test_sparse_failure_degrades_to_dense() -> None:
    """A sparse outage still returns dense results, flagged."""
    with (
        patch(
            "retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])
        ),
        patch("retrieval.pipeline.sparse_channel", AsyncMock(side_effect=RuntimeError)),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
    ):
        out = await fast_pipeline("q", limit=10)

    assert [r.id for r in out.results] == ["a"]
    assert out.degraded == ["sparse"]


@pytest.mark.asyncio
async def test_dense_failure_degrades_to_sparse() -> None:
    """A dense outage still returns sparse results, flagged."""
    with (
        patch("retrieval.pipeline.dense_channel", AsyncMock(side_effect=RuntimeError)),
        patch(
            "retrieval.pipeline.sparse_channel", AsyncMock(return_value=[_cand("b", "sparse")])
        ),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
    ):
        out = await fast_pipeline("q", limit=10)

    assert [r.id for r in out.results] == ["b"]
    assert out.degraded == ["dense"]


@pytest.mark.asyncio
async def test_both_channels_failing_returns_empty_and_flags_both() -> None:
    """Total retrieval failure is reported, not raised."""
    with (
        patch("retrieval.pipeline.dense_channel", AsyncMock(side_effect=RuntimeError)),
        patch("retrieval.pipeline.sparse_channel", AsyncMock(side_effect=RuntimeError)),
    ):
        out = await fast_pipeline("q", limit=10)

    assert out.results == []
    assert sorted(out.degraded) == ["dense", "sparse"]


@pytest.mark.asyncio
async def test_rerank_failure_degrades_to_fusion_order() -> None:
    """A reranker outage keeps the fused ordering and flags it."""
    from retrieval.rerank import RerankError

    with (
        patch("retrieval.pipeline.settings.rerank_enabled", True),
        patch(
            "retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])
        ),
        patch("retrieval.pipeline.sparse_channel", AsyncMock(return_value=[])),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=RerankError("down"))),
    ):
        out = await fast_pipeline("q", limit=10)

    assert [r.id for r in out.results] == ["a"]
    assert out.degraded == ["rerank"]


@pytest.mark.asyncio
async def test_sparse_disabled_skips_channel_without_degrading() -> None:
    """Turning sparse off is a configuration choice, not a degradation."""
    with (
        patch("retrieval.pipeline.settings.sparse_enabled", False),
        patch(
            "retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])
        ),
        patch("retrieval.pipeline.sparse_channel", AsyncMock()) as sparse,
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
    ):
        out = await fast_pipeline("q", limit=10)

    sparse.assert_not_called()
    assert out.degraded == []


@pytest.mark.asyncio
async def test_smart_pipeline_runs_a_channel_pair_per_subquery() -> None:
    """Two sub-queries produce two dense calls and two sparse calls."""
    dense = AsyncMock(return_value=[_cand("a", "dense")])
    sparse = AsyncMock(return_value=[_cand("b", "sparse")])
    with (
        patch("retrieval.pipeline.decompose", AsyncMock(return_value=["q1", "q2"])),
        patch("retrieval.pipeline.dense_channel", dense),
        patch("retrieval.pipeline.sparse_channel", sparse),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
        patch("retrieval.pipeline.should_retry", return_value=False),
    ):
        out = await smart_pipeline("q", limit=10)

    assert dense.await_count == 2
    assert sparse.await_count == 2
    assert out.sub_queries == ["q1", "q2"]


@pytest.mark.asyncio
async def test_smart_pipeline_reranks_against_original_query() -> None:
    """Sub-queries drive retrieval; the original query drives ranking."""
    captured: dict = {}  # type: ignore[type-arg]

    async def _capture(query, results, top_k):  # type: ignore[no-untyped-def]
        captured["query"] = query
        return results[:top_k]

    with (
        patch("retrieval.pipeline.settings.rerank_enabled", True),
        patch("retrieval.pipeline.decompose", AsyncMock(return_value=["sub1", "sub2"])),
        patch(
            "retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])
        ),
        patch("retrieval.pipeline.sparse_channel", AsyncMock(return_value=[])),
        patch("retrieval.pipeline.rerank", _capture),
        patch("retrieval.pipeline.should_retry", return_value=False),
    ):
        await smart_pipeline("ORIGINAL", limit=10)

    assert captured["query"] == "ORIGINAL"


@pytest.mark.asyncio
async def test_gate_triggers_exactly_one_retry() -> None:
    """A weak result set widens the pool once, never twice."""
    dense = AsyncMock(return_value=[_cand("a", "dense")])
    with (
        patch("retrieval.pipeline.decompose", AsyncMock(return_value=["q"])),
        patch("retrieval.pipeline.dense_channel", dense),
        patch("retrieval.pipeline.sparse_channel", AsyncMock(return_value=[])),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
        patch("retrieval.pipeline.should_retry", return_value=True),
    ):
        out = await smart_pipeline("q", limit=10)

    assert out.retried is True
    assert dense.await_count == 2  # initial pass + one retry
    assert dense.await_args_list[-1].kwargs["limit"] > dense.await_args_list[0].kwargs["limit"]
