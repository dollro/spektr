"""Tests for the relevance gate."""

from __future__ import annotations

import logging

import pytest

from retrieval.gate import log_gate_decision, should_retry
from retrieval.models import FusedResult


def _result(score: float) -> FusedResult:
    return FusedResult(id="a", text="t", source_file="doc.pdf", score=score, fusion_score=0.01)


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


def _record(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    assert len(caplog.records) == 1
    return caplog.records[0]


def test_non_fired_decision_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The denominator only exists if quiet runs are recorded too."""
    with caplog.at_level(logging.INFO, logger="retrieval.gate"):
        log_gate_decision(fired=False, top_score=0.4, degraded=[])
    record = _record(caplog)
    assert record.gate_fired is False
    assert record.top_score == 0.4
    assert not hasattr(record, "retry_helped")


def test_fired_decision_records_before_and_after(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Improvement is the signal that first-stage recall was the problem."""
    with caplog.at_level(logging.INFO, logger="retrieval.gate"):
        log_gate_decision(
            fired=True, top_score=-0.2, degraded=[], top_score_after=0.35, widened_to=30
        )
    record = _record(caplog)
    assert record.gate_fired is True
    assert record.retry_helped is True
    assert record.gate_widened_to == 30


def test_retry_that_did_not_help_is_marked(caplog: pytest.LogCaptureFixture) -> None:
    """A retry that found nothing better means the chunk is absent, not misranked."""
    with caplog.at_level(logging.INFO, logger="retrieval.gate"):
        log_gate_decision(
            fired=True, top_score=-0.2, degraded=[], top_score_after=-0.3, widened_to=30
        )
    assert _record(caplog).retry_helped is False


def test_empty_then_results_counts_as_helped(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing -> something is the clearest possible improvement."""
    with caplog.at_level(logging.INFO, logger="retrieval.gate"):
        log_gate_decision(
            fired=True, top_score=None, degraded=[], top_score_after=0.1, widened_to=30
        )
    assert _record(caplog).retry_helped is True


def test_degraded_rerank_marks_run_inert(caplog: pytest.LogCaptureFixture) -> None:
    """Without a rerank score the gate compares RRF scores, which never fire."""
    with caplog.at_level(logging.INFO, logger="retrieval.gate"):
        log_gate_decision(fired=False, top_score=0.016, degraded=["rerank"])
    assert _record(caplog).gate_reranked is False
