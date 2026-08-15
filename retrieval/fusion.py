"""Reciprocal Rank Fusion over independent retrieval channels.

RRF scores a document as sum(1 / (k + rank)) across every channel that
returned it, using 1-based rank and discarding raw scores. This is what lets
cosine similarity and miniCOIL scores merge without normalisation — their
scales are incompatible, their ranks are not.
"""

from __future__ import annotations

from retrieval.models import Candidate, FusedResult


def rrf(channels: list[list[Candidate]], k: int = 60) -> list[FusedResult]:
    """Fuse ranked channels into one list ordered by RRF score.

    Args:
        channels: One ranked candidate list per channel. Empty lists are ignored.
        k: Rank-damping constant. Lower values weight top ranks more heavily.

    Returns:
        Fused results sorted by descending fusion score.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Candidate] = {}
    origins: dict[str, list[str]] = {}

    for channel in channels:
        for rank, cand in enumerate(channel, start=1):
            scores[cand.id] = scores.get(cand.id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(cand.id, cand)
            origins.setdefault(cand.id, [])
            if cand.channel not in origins[cand.id]:
                origins[cand.id].append(cand.channel)

    fused = [
        FusedResult(
            id=doc_id,
            text=seen[doc_id].text,
            source_file=seen[doc_id].source_file,
            page_number=seen[doc_id].page_number,
            chunk_index=seen[doc_id].chunk_index,
            score=score,
            fusion_score=score,
            channels=origins[doc_id],
            metadata=seen[doc_id].metadata,
        )
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda f: f.fusion_score, reverse=True)
    return fused
