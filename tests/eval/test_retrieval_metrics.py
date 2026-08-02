"""Retrieval-only quality metrics. No LLM in the loop."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

SET_PATH = Path(__file__).parent / "retrieval_set.yaml"


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents appearing in the top k."""
    if not relevant:
        return 0.0
    hits = len([d for d in retrieved[:k] if d in relevant])
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG over the top k."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def test_recall_perfect() -> None:
    assert recall_at_k(["a", "b"], {"a", "b"}, k=10) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(["a", "x"], {"a", "b"}, k=10) == 0.5


def test_recall_respects_k() -> None:
    assert recall_at_k(["x", "x", "a"], {"a"}, k=2) == 0.0


def test_mrr_first_position() -> None:
    assert mrr(["a", "x"], {"a"}) == 1.0


def test_mrr_second_position() -> None:
    assert mrr(["x", "a"], {"a"}) == 0.5


def test_mrr_absent() -> None:
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ordering_is_one() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=10) == pytest.approx(1.0)


def test_ndcg_penalises_late_hits() -> None:
    early = ndcg_at_k(["a", "x"], {"a"}, k=10)
    late = ndcg_at_k(["x", "a"], {"a"}, k=10)
    assert early > late


def test_retrieval_set_is_wellformed() -> None:
    """Every entry has a query, a category, and at least one label.

    The size floor is deliberately low: the corpus is currently two
    documents, so this set is a smoke check on the harness, not a
    statistically meaningful benchmark. Raise the floor when the corpus
    grows — see docs/eval/golden-set.md.
    """
    data = yaml.safe_load(SET_PATH.read_text())
    assert len(data["queries"]) >= 6, "need at least 6 labelled queries"
    valid_categories = {"exact_id", "multi_hop", "semantic"}
    for entry in data["queries"]:
        assert entry["query"].strip()
        assert entry["category"] in valid_categories, entry["query"]
        assert entry["relevant"], f"no labels for: {entry['query']}"
        for rel in entry["relevant"]:
            assert "source_file" in rel
            assert "chunk_index" in rel
            # page_number is required too: chunk_index resets per page in
            # this corpus (see _doc_key below), so source_file + chunk_index
            # alone is not a unique key.
            assert "page_number" in rel


def test_all_three_categories_are_represented() -> None:
    """Each channel's reason for existing has at least one probe.

    exact_id exercises sparse, semantic exercises dense, multi_hop
    exercises decomposition. A set missing one cannot detect a regression
    in that stage.
    """
    data = yaml.safe_load(SET_PATH.read_text())
    present = {entry["category"] for entry in data["queries"]}
    assert present == {"exact_id", "multi_hop", "semantic"}, f"missing: {present}"


def _doc_key(result: dict) -> str:  # type: ignore[type-arg]
    # chunk_index resets per page in this corpus's chunking scheme (see
    # ingestion/pipeline.py's _make_chunk_id), so source_file + chunk_index
    # alone collides across pages of the same document. page_number is
    # required for a unique key — this is a deliberate deviation from the
    # brief's verbatim _doc_key, verified against the real corpus (arxiv.pdf
    # repeats chunk_index 0-5 on every one of its 5 pages).
    return f"{result['source_file']}#{result['page_number']}#{result['chunk_index']}"


def _relevant_keys(entry: dict) -> set[str]:  # type: ignore[type-arg]
    return {
        f"{r['source_file']}#{r['page_number']}#{r['chunk_index']}" for r in entry["relevant"]
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_quality_meets_thresholds(frozen_corpus: int) -> None:
    """Score the full labelled set against multi_search and gate on it.

    `frozen_corpus` loads the committed snapshot into the throwaway collection
    first — without it these metrics score an empty collection and report a
    vacuous 0.0 for everything.
    """
    from server.tools.multi_search import multi_search

    data = yaml.safe_load(SET_PATH.read_text())
    thresholds = yaml.safe_load(
        (Path(__file__).parent / "thresholds.yaml").read_text()
    )

    recalls: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []

    for entry in data["queries"]:
        out = await multi_search(entry["query"], limit=10)
        retrieved = [_doc_key(r) for r in out["results"]]
        relevant = _relevant_keys(entry)
        recalls.append(recall_at_k(retrieved, relevant, k=10))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k=10))
        rrs.append(mrr(retrieved, relevant))

    scores = {
        "recall_at_10": sum(recalls) / len(recalls),
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "mrr": sum(rrs) / len(rrs),
    }
    print(f"\nRetrieval scores: {scores}")

    for metric, value in scores.items():
        floor = thresholds.get(metric)
        if floor is not None:
            assert value >= floor, f"{metric} {value:.3f} below floor {floor}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ablation_matrix(frozen_corpus: int) -> None:
    """Score each stage combination so wins can be attributed.

    Not a gate — it prints a table. Run when deciding whether a stage earns
    its cost.
    """
    from unittest.mock import patch

    from server.tools.multi_search import multi_search

    data = yaml.safe_load(SET_PATH.read_text())
    configs = {
        "dense-only": {"sparse_enabled": False, "rerank_enabled": False},
        "dense+sparse": {"sparse_enabled": True, "rerank_enabled": False},
        "dense+rerank": {"sparse_enabled": False, "rerank_enabled": True},
        "all": {"sparse_enabled": True, "rerank_enabled": True},
    }

    print("\n| config | recall@10 | ndcg@10 | mrr |")
    print("|-|-|-|-|")
    for name, overrides in configs.items():
        patches = [
            patch(f"retrieval.pipeline.settings.{key}", value)
            for key, value in overrides.items()
        ]
        for p in patches:
            p.start()
        try:
            recalls, ndcgs, rrs = [], [], []
            for entry in data["queries"]:
                out = await multi_search(entry["query"], limit=10)
                retrieved = [_doc_key(r) for r in out["results"]]
                relevant = _relevant_keys(entry)
                recalls.append(recall_at_k(retrieved, relevant, k=10))
                ndcgs.append(ndcg_at_k(retrieved, relevant, k=10))
                rrs.append(mrr(retrieved, relevant))
            n = len(recalls)
            print(
                f"| {name} | {sum(recalls)/n:.3f} | "
                f"{sum(ndcgs)/n:.3f} | {sum(rrs)/n:.3f} |"
            )
        finally:
            for p in patches:
                p.stop()
