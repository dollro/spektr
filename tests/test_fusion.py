"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from retrieval.fusion import rrf
from retrieval.models import Candidate


def _cand(doc_id: str, score: float, channel: str) -> Candidate:
    return Candidate(
        id=doc_id,
        text=f"text-{doc_id}",
        source_file="doc.pdf",
        score=score,
        channel=channel,
    )


def test_single_channel_preserves_order() -> None:
    """One channel in, same order out."""
    channel = [_cand("a", 0.9, "dense"), _cand("b", 0.5, "dense")]
    fused = rrf([channel], k=60)
    assert [f.id for f in fused] == ["a", "b"]
    assert fused[0].channels == ["dense"]


def test_document_in_both_channels_outranks_singletons() -> None:
    """Agreement across channels beats a strong hit in one channel."""
    dense = [_cand("a", 0.99, "dense"), _cand("shared", 0.5, "dense")]
    sparse = [_cand("b", 0.99, "sparse"), _cand("shared", 0.5, "sparse")]
    fused = rrf([dense, sparse], k=60)
    # 'shared' scores 1/62 + 1/62; 'a' and 'b' score 1/61 each.
    assert fused[0].id == "shared"
    assert sorted(fused[0].channels) == ["dense", "sparse"]


def test_empty_channel_is_ignored() -> None:
    """An empty channel does not affect the ranking."""
    dense = [_cand("a", 0.9, "dense")]
    fused = rrf([dense, []], k=60)
    assert [f.id for f in fused] == ["a"]


def test_no_channels_returns_empty() -> None:
    """No input channels yields no results."""
    assert rrf([], k=60) == []
    assert rrf([[], []], k=60) == []


def test_zero_overlap_interleaves_by_rank() -> None:
    """Disjoint channels interleave — both rank-1 hits tie, then both rank-2."""
    dense = [_cand("a", 0.9, "dense"), _cand("c", 0.3, "dense")]
    sparse = [_cand("b", 0.9, "sparse"), _cand("d", 0.3, "sparse")]
    fused = rrf([dense, sparse], k=60)
    assert set(f.id for f in fused[:2]) == {"a", "b"}
    assert set(f.id for f in fused[2:]) == {"c", "d"}


def test_fusion_score_is_recorded_and_score_mirrors_it() -> None:
    """Pre-rerank, score equals fusion_score."""
    fused = rrf([[_cand("a", 0.9, "dense")]], k=60)
    assert fused[0].fusion_score == 1 / 61
    assert fused[0].score == fused[0].fusion_score


def test_k_changes_rank_sensitivity() -> None:
    """A smaller k weights top ranks more heavily."""
    dense = [_cand("a", 0.9, "dense")]
    assert rrf([dense], k=1)[0].fusion_score > rrf([dense], k=60)[0].fusion_score
