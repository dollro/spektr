"""Tests for the relevance gate."""

from __future__ import annotations

from retrieval.gate import should_retry
from retrieval.models import FusedResult


def _result(score: float) -> FusedResult:
    return FusedResult(
        id="a", text="t", source_file="doc.pdf", score=score, fusion_score=0.01
    )


def test_low_top_score_triggers_retry() -> None:
    """A weak best hit means the pool was probably too narrow."""
    assert should_retry([_result(0.1)], floor=0.3) is True


def test_high_top_score_does_not_retry() -> None:
    """A strong best hit needs no widening."""
    assert should_retry([_result(0.9)], floor=0.3) is False


def test_score_exactly_at_floor_does_not_retry() -> None:
    """The floor is inclusive — at the floor is good enough."""
    assert should_retry([_result(0.3)], floor=0.3) is False


def test_empty_results_trigger_retry() -> None:
    """Nothing found is the strongest signal to widen."""
    assert should_retry([], floor=0.3) is True


def test_only_top_result_matters() -> None:
    """A weak tail below a strong head is normal, not a failure."""
    assert should_retry([_result(0.9), _result(0.01)], floor=0.3) is False
