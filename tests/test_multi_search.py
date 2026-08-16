"""Tests for the multi_search and hybrid_search MCP adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.models import FusedResult
from retrieval.pipeline import PipelineOutput
from server.tools.hybrid_search import hybrid_search
from server.tools.multi_search import multi_search


def _fused(doc_id: str) -> FusedResult:
    return FusedResult(
        id=doc_id,
        text=f"t-{doc_id}",
        source_file="d.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
        fusion_score=0.03,
        channels=["dense"],
        metadata={},
    )


@pytest.mark.asyncio
async def test_multi_search_returns_fused_schema() -> None:
    """Results carry score, fusion_score, and channel provenance."""
    out_pipeline = PipelineOutput(results=[_fused("a")])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)

    assert out["results"][0]["fusion_score"] == 0.03
    assert out["results"][0]["channels"] == ["dense"]
    assert out["graph_facts"] == []
    assert "degraded" not in out
    assert "sub_queries" not in out


@pytest.mark.asyncio
async def test_multi_search_omits_degraded_when_healthy() -> None:
    """A clean run has no degraded key at all."""
    with (
        patch(
            "server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())
        ),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)
    assert "degraded" not in out


@pytest.mark.asyncio
async def test_multi_search_reports_degraded_channels() -> None:
    """A failed channel surfaces in degraded."""
    out_pipeline = PipelineOutput(results=[_fused("a")], degraded=["sparse"])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)
    assert out["degraded"] == ["sparse"]


@pytest.mark.asyncio
async def test_total_channel_failure_sets_error_key() -> None:
    """Both channels down is distinguishable from a genuine zero-hit query."""
    out_pipeline = PipelineOutput(results=[], degraded=["dense", "sparse"])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)

    assert out["results"] == []
    assert "error" in out


@pytest.mark.asyncio
async def test_zero_hits_has_no_error_key() -> None:
    """A healthy query that simply matches nothing is not an error."""
    with (
        patch(
            "server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())
        ),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)

    assert out["results"] == []
    assert "error" not in out


@pytest.mark.asyncio
async def test_graph_failure_degrades_not_raises() -> None:
    """A graph outage yields empty facts and a degraded flag."""
    with (
        patch(
            "server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())
        ),
        patch("server.tools.multi_search.search_graph", AsyncMock(side_effect=RuntimeError)),
    ):
        out = await multi_search("q", limit=5)
    assert out["graph_facts"] == []
    assert "graph" in out["degraded"]


@pytest.mark.asyncio
async def test_real_graph_channel_propagates_engine_failure() -> None:
    """The REAL search_graph must raise, not return an error dict.

    The test above mocks `search_graph` with a raising stub, so it passed
    even while the real implementation caught everything and RETURNED
    `[{"error": ...}]`. That made `run_graph` see a successful call, so
    "graph" never reached `degraded` and the error dict was handed back
    inside `graph_facts` dressed as a fact. This exercises the real function
    with only the engine faulted, which is what the mocked test cannot do.
    """
    with (
        patch(
            "server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())
        ),
        patch(
            "server.tools.graph_search.get_graph_engine",
            side_effect=RuntimeError("neo4j down"),
        ),
    ):
        out = await multi_search("q", limit=5)

    assert "graph" in out["degraded"], "an engine outage must degrade the channel"
    assert out["graph_facts"] == [], "a failed channel contributes no facts"


@pytest.mark.asyncio
async def test_mcp_graph_search_still_returns_error_dict() -> None:
    """The MCP-facing tool keeps its soft-failure shape.

    A direct tool call has no `degraded` channel to report into, so the
    caught, dict-shaped response is right there — only the fused callers
    need the raising variant.
    """
    from server.tools.graph_search import graph_search

    with patch(
        "server.tools.graph_search.get_graph_engine",
        side_effect=RuntimeError("neo4j down"),
    ):
        out = await graph_search("q", limit=5)

    assert len(out) == 1
    assert "error" in out[0]


@pytest.mark.asyncio
async def test_mcp_graph_search_still_raises_on_bad_search_type() -> None:
    """An invalid argument is the caller's bug, not a degraded backend."""
    from server.tools.graph_search import graph_search

    with pytest.raises(ValueError, match="not yet implemented"):
        await graph_search("q", search_type="semantic")


@pytest.mark.asyncio
async def test_live_results_split_out_when_session_active() -> None:
    """Live-session chunks are separated from KB results.

    The pipeline itself owns the KB/live split (dual retrieval — see
    retrieval/pipeline.py); shape_response must trust PipelineOutput's
    results/live_results fields rather than re-deriving the split from
    result metadata. A prior version sniffed
    `item.metadata.get("source_type") == "live"`, which production code
    never sets on stored payloads (ingestion/live_ingest.py writes
    metadata={}) — that fixture asserted data that can't occur for real,
    which is how the session_id dual-retrieval bug shipped undetected.
    """
    out_pipeline = PipelineOutput(results=[_fused("b")], live_results=[_fused("a")])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5, session_id="s1")

    assert [r["id"] for r in out["live_results"]] == ["a"]
    assert [r["id"] for r in out["results"]] == ["b"]


@pytest.mark.asyncio
async def test_session_id_does_not_exclude_kb_results() -> None:
    """A session_id must not confine the whole query to that session.

    Regression test for defect (b): the old build_filter treated session_id
    as a narrowing `must` condition on the single query, so setting
    session_id silently returned zero KB results. Simulates the pipeline
    returning both a KB and a live hit (as dual retrieval now does) and
    checks both survive into the response.
    """
    out_pipeline = PipelineOutput(results=[_fused("kb-1")], live_results=[_fused("live-1")])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5, session_id="nonexistent-session")

    assert [r["id"] for r in out["results"]] == ["kb-1"]
    assert [r["id"] for r in out["live_results"]] == ["live-1"]


@pytest.mark.asyncio
async def test_empty_query_returns_empty_without_calling_pipeline() -> None:
    """A blank query short-circuits."""
    with patch("server.tools.multi_search.fast_pipeline") as pipeline:
        out = await multi_search("   ", limit=5)
    pipeline.assert_not_called()
    assert out["results"] == []


@pytest.mark.asyncio
async def test_hybrid_search_exposes_subqueries_and_retried() -> None:
    """hybrid_search adds the two LLM-stage fields."""
    out_pipeline = PipelineOutput(
        results=[_fused("a")], sub_queries=["q1", "q2"], retried=True
    )
    with (
        patch(
            "server.tools.hybrid_search.smart_pipeline", AsyncMock(return_value=out_pipeline)
        ),
        patch("server.tools.hybrid_search.search_graph", AsyncMock(return_value=[])),
    ):
        out = await hybrid_search("q", limit=5)

    assert out["sub_queries"] == ["q1", "q2"]
    assert out["retried"] is True
    assert out["results"][0]["channels"] == ["dense"]


@pytest.mark.asyncio
async def test_limit_is_clamped() -> None:
    """Limits are bounded to protect the caller's token budget."""
    captured: dict = {}  # type: ignore[type-arg]

    async def _capture(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return PipelineOutput()

    with (
        patch("server.tools.multi_search.fast_pipeline", _capture),
        patch("server.tools.multi_search.search_graph", AsyncMock(return_value=[])),
    ):
        await multi_search("q", limit=5000)
    assert captured["limit"] == 100
