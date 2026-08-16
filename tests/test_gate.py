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


class TestScoreScaleCoupling:
    """The floor is meaningless without the score scale that produced it.

    A logit-scaled reranker (jina-reranker-v3.5) centres near zero, so 0.0 is
    a meaningful "worse than neutral" boundary. A unit-scaled one (a TEI
    cross-encoder with `raw_scores: false`) emits [0, 1], where nothing can
    ever fall below 0.0 — so carrying the logit floor across a model switch
    does not merely mistune the gate, it disables it, with no exception and
    no log line. These tests are the regression guard for that.
    """

    def test_logit_floor_on_unit_scores_never_fires(self) -> None:
        """The failure mode itself: a [0,1] score can never sit below 0.0."""
        for score in (0.0, 0.01, 0.5, 0.99):
            assert should_retry([_result(score)], floor=0.0) is False

    def test_unit_floor_fires_on_weak_unit_scores(self) -> None:
        """With the model's own floor, a weak unit-scaled hit still widens."""
        assert should_retry([_result(0.2)], floor=0.5) is True
        assert should_retry([_result(0.8)], floor=0.5) is False

    def test_logit_floor_fires_on_negative_logits(self) -> None:
        """A negative logit is 'judged less relevant than neutral'."""
        assert should_retry([_result(-2.5)], floor=0.0) is True

    def test_each_registry_model_floor_can_fire(self) -> None:
        """Every registered reranker must have a floor its scores can cross.

        Catches a new entry whose default_score_floor was copied from a model
        on the other scale — the mistake this whole field exists to prevent.
        """
        from config.rerank_models import REGISTRY

        for name, model_spec in REGISTRY.items():
            floor = model_spec.default_score_floor
            if model_spec.score_scale == "unit":
                assert 0.0 < floor < 1.0, (
                    f"{name}: a unit-scaled floor of {floor} is outside the "
                    "reachable [0,1] range, so the gate could never fire"
                )
                assert should_retry([_result(floor / 2)], floor=floor) is True
            else:
                assert should_retry([_result(floor - 1.0)], floor=floor) is True


def test_resolved_floor_follows_the_model() -> None:
    """An unset RERANK_SCORE_FLOOR adopts the active model's default."""
    from config.rerank_models import REGISTRY
    from config.settings import Settings

    for name, model_spec in REGISTRY.items():
        s = Settings(  # type: ignore[call-arg]
            _env_file=None,
            neo4j_password="x",
            rerank_model=name,
            rerank_enabled=False,
            rerank_score_floor=None,
        )
        assert s.resolved_score_floor == model_spec.default_score_floor

    explicit = Settings(  # type: ignore[call-arg]
        _env_file=None,
        neo4j_password="x",
        rerank_enabled=False,
        rerank_score_floor=-3.0,
    )
    assert explicit.resolved_score_floor == -3.0, "an explicit override wins"


def test_explicit_zero_floor_is_not_treated_as_unset() -> None:
    """0.0 is a real logit floor, which is why the sentinel had to be None.

    A falsy check here would make `RERANK_SCORE_FLOOR=0` indistinguishable
    from leaving it blank — harmless while jina's default is also 0.0, and a
    silent surprise the moment a model with a non-zero default is selected.
    """
    from config.settings import Settings

    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        neo4j_password="x",
        rerank_model="qwen3-reranker",
        rerank_enabled=False,
        rerank_score_floor=0.0,
    )
    assert s.resolved_score_floor == 0.0, "explicit 0.0 must not fall back to 0.5"


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
