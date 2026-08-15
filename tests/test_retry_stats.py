"""Tests for the relevance-gate log aggregator."""

from __future__ import annotations

import json

from scripts.retry_stats import _percentile, _render, collect


def _line(**fields: object) -> str:
    base: dict[str, object] = {"message": "Relevance gate evaluated", "gate_reranked": True}
    base.update(fields)
    return json.dumps(base)


def test_ignores_unrelated_and_malformed_lines() -> None:
    """Log streams carry plenty that is not gate telemetry."""
    stats = collect(
        [
            "not json at all",
            json.dumps({"message": "something else"}),
            "{broken",
            "",
            _line(gate_fired=False, top_score=0.4),
        ]
    )
    assert stats.total == 1


def test_container_log_prefix_is_tolerated() -> None:
    """`docker compose logs` prefixes each line with the service name."""
    stats = collect([f"mcp-1  | {_line(gate_fired=False, top_score=0.4)}"])
    assert stats.total == 1


def test_counts_fired_against_gated_denominator() -> None:
    """Inert runs are excluded — they could not have fired."""
    stats = collect(
        [
            _line(gate_fired=False, top_score=0.4),
            _line(gate_fired=True, top_score=-0.1, retry_helped=True),
            _line(gate_fired=False, top_score=0.016, gate_reranked=False),
        ]
    )
    assert stats.total == 3
    assert stats.inert == 1
    assert stats.gated == 2
    assert stats.fired == 1
    assert stats.helped == 1


def test_inert_runs_do_not_contribute_scores() -> None:
    """RRF scores are a different scale and would corrupt the percentiles."""
    stats = collect(
        [
            _line(gate_fired=False, top_score=0.016, gate_reranked=False),
            _line(gate_fired=False, top_score=0.4),
        ]
    )
    assert stats.scores == [0.4]


def test_empty_first_pass_is_tracked_separately() -> None:
    """A null top_score means the pass returned nothing at all."""
    stats = collect([_line(gate_fired=True, top_score=None, retry_helped=True)])
    assert stats.fired == 1
    assert stats.empty_first_pass == 1
    assert stats.scores == []


def test_render_explains_empty_input() -> None:
    """Zero records is ambiguous, so the report must say what to check."""
    assert "No relevance-gate records found" in _render(collect([]))


def test_render_flags_all_inert() -> None:
    """All-inert means the reranker is broken, not that retrieval is healthy."""
    report = _render(collect([_line(gate_fired=False, gate_reranked=False, top_score=0.01)]))
    assert "RERANK_ENABLED" in report


def test_percentile_is_nearest_rank() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert _percentile(values, 0.0) == 0.1
    assert _percentile(values, 0.5) == 0.3
    assert _percentile(values, 1.0) == 0.5
