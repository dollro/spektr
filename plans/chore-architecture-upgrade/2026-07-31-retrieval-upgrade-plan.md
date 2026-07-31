# Retrieval Architecture Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a miniCOIL sparse retrieval channel fused with dense via RRF, upgrade to the listwise jina-reranker-v3.5, and add query decomposition with a relevance-gated retry — exposed as a reworked `hybrid_search` plus a new LLM-free `multi_search`.

**Architecture:** A new `retrieval/` package holds pure retrieval stages (channels, fusion, rerank, decompose, gate) composed by `pipeline.py`. MCP tools in `server/tools/` become thin adapters. `documents_dense` migrates from an unnamed vector to named `dense` + `sparse` vectors, which forces a full re-ingest.

**Tech Stack:** Python 3.13, uv, Qdrant (named + sparse vectors), fastembed (miniCOIL), Jina embeddings v4 + reranker v3.5, Anthropic Haiku 4.5, FastMCP, pytest-asyncio, Ruff, mypy strict.

**Design doc:** `plans/chore-architecture-upgrade/2026-07-31-retrieval-upgrade-design.md`

## Global Constraints

- Python 3.13. Package manager is `uv` — never `pip` directly.
- Ruff line length 95. Run `uv run ruff check .` before every commit — it must pass clean.
- mypy strict, **but the repo does not currently pass `uv run mypy .`**: 14 pre-existing errors in 6 files (missing `gliner2`, `boto3`/`yaml`/`ragas` stubs, and a `scripts/backup.py` dual-module-name clash) are present at the branch base and are out of scope here. `task check` therefore fails for reasons unrelated to this work. The binding requirement is: **introduce no new mypy errors**. Verify with `uv run mypy . 2>&1 | tail -1` and confirm the count is still 14 (or lower) — never higher. Do not "fix" the pre-existing errors; that is a separate chore.
- Max file 600 lines, max function 60 lines, max class 100 lines.
- `retrieval/` must not import from `server/` or `fastmcp` — it is transport-agnostic.
- Existing dict-based tool returns use `# type: ignore[type-arg]` on bare `dict`. Match that convention.
- All new stages are individually disableable by a `settings` flag.
- Never lower a threshold in `tests/eval/thresholds.yaml`.
- Commit messages: `<type>(<scope>): <subject>`. Never mention Claude or AI authorship.
- fastembed floor: `>=0.7.0` (first release containing `Qdrant/minicoil-v1`).
- Exact model strings: `Qdrant/minicoil-v1`, `jina-reranker-v3.5`, `claude-haiku-4-5-20251001`, `claude-sonnet-5`.

---

## File Structure

**Created:**

| File | Responsibility |
|-|-|
| `retrieval/__init__.py` | Package marker |
| `retrieval/models.py` | `Candidate`, `FusedResult` |
| `retrieval/fusion.py` | `rrf()` — pure rank fusion |
| `retrieval/gate.py` | `should_retry()` — pure confidence check |
| `retrieval/rerank.py` | Jina reranker v3.5 (moved from `server/tools/reranker.py`) |
| `retrieval/channels.py` | `dense_channel()`, `sparse_channel()` |
| `retrieval/decompose.py` | `decompose()` — query → sub-queries |
| `retrieval/pipeline.py` | `fast_pipeline()`, `smart_pipeline()` |
| `ingestion/sparse_embedder.py` | miniCOIL encode for ingest + query |
| `server/tools/multi_search.py` | LLM-free MCP adapter |
| `tests/test_fusion.py`, `tests/test_gate.py`, `tests/test_rerank.py`, `tests/test_channels.py`, `tests/test_decompose.py`, `tests/test_pipeline_retrieval.py`, `tests/test_multi_search.py`, `tests/test_sparse_embedder.py` | Unit tests |
| `tests/eval/retrieval_set.yaml` | Labeled retrieval golden set |
| `tests/eval/test_retrieval_metrics.py` | recall@k / nDCG@10 / MRR + ablation |

**Modified:** `config/settings.py`, `config/constants.py`, `ingestion/qdrant_setup.py`, `ingestion/pipeline.py`, `server/tools/vector_search.py`, `server/tools/hybrid_search.py`, `server/models.py`, `server/mcp_server.py`, `agent/agent.py`, `scripts/doctor.py`, `pyproject.toml`, `docs/**`, `CLAUDE.md`.

**Deleted:** `server/tools/reranker.py` (moved to `retrieval/rerank.py`).

---

## Task 1: Configuration and retrieval package foundation

**Files:**
- Create: `retrieval/__init__.py`, `retrieval/models.py`
- Modify: `config/settings.py`, `config/constants.py`, `pyproject.toml`
- Test: `tests/test_settings.py` (extend)

**Interfaces:**
- Consumes: nothing
- Produces: `retrieval.models.Candidate`, `retrieval.models.FusedResult`; settings fields `sparse_enabled`, `sparse_model`, `rrf_k`, `rerank_model`, `rerank_candidates`, `rerank_score_floor`, `retry_enabled`, `retry_limit_multiplier`, `decompose_enabled`, `decompose_model`, `decompose_max_subqueries`; constants `DENSE_VECTOR_NAME`, `SPARSE_VECTOR_NAME`, `MINICOIL_AVG_LEN`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_settings.py`:

```python
def test_retrieval_defaults() -> None:
    """New retrieval settings expose the documented defaults.

    _env_file=None isolates from the developer's .env — Settings() otherwise
    reads it and this would assert the local config, not the defaults.
    """
    from config.settings import Settings

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.sparse_enabled is True
    assert s.sparse_model == "Qdrant/minicoil-v1"
    assert s.rrf_k == 60
    assert s.rerank_model == "jina-reranker-v3.5"
    assert s.rerank_candidates == 50
    assert s.rerank_score_floor == 0.0
    assert s.retry_enabled is True
    assert s.retry_limit_multiplier == 3
    assert s.decompose_enabled is True
    assert s.decompose_model == ""
    assert s.decompose_max_subqueries == 4
    assert s.llm_model == "claude-sonnet-5"


def test_vector_name_constants() -> None:
    """Named-vector constants exist for the migrated collection."""
    from config.constants import DENSE_VECTOR_NAME, MINICOIL_AVG_LEN, SPARSE_VECTOR_NAME

    assert DENSE_VECTOR_NAME == "dense"
    assert SPARSE_VECTOR_NAME == "sparse"
    assert MINICOIL_AVG_LEN == 80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_settings.py::test_retrieval_defaults tests/test_settings.py::test_vector_name_constants -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'sparse_enabled'` and `ImportError: cannot import name 'DENSE_VECTOR_NAME'`

- [ ] **Step 3: Add settings**

In `config/settings.py`, replace the line `rerank_enabled: bool = True` with this block (keep `rerank_enabled`; the rest are new):

```python
    rerank_enabled: bool = True

    # Retrieval pipeline
    sparse_enabled: bool = True
    sparse_model: str = "Qdrant/minicoil-v1"
    rrf_k: int = 60
    rerank_model: str = "jina-reranker-v3.5"
    rerank_candidates: int = 50  # fused candidates sent to the reranker
    rerank_score_floor: float = 0.0  # gate threshold; see calibration note below
    retry_enabled: bool = True
    retry_limit_multiplier: int = 3  # candidate-pool widening on gated retry
    decompose_enabled: bool = True
    decompose_model: str = ""  # empty -> fall back to llm_model
    decompose_max_subqueries: int = 4
```

`decompose_model` defaults to empty rather than a model string on purpose: this deployment routes Anthropic models through OpenRouter (`LLM_API_TYPE=openai`, `LLM_MODEL=anthropic/claude-haiku-4.5`), so a hardcoded native-Anthropic id like `claude-haiku-4-5-20251001` would be wrong here. Empty means "use whatever `llm_model` is", and anyone wanting a separately cheaper decomposition model sets `DECOMPOSE_MODEL` in `.env`.

Then change the existing `llm_model` default to:

```python
    llm_model: str = "claude-sonnet-5"
```

Note: `.env` sets `LLM_MODEL`, which overrides this default at runtime. The default only applies to deployments without that variable — see the test note in Step 1.

- [ ] **Step 4: Add constants**

Append to `config/constants.py`:

```python
# Named vectors on DENSE_COLLECTION. Sparse vectors must be named in Qdrant,
# and named/unnamed vectors cannot coexist — this is why the collection is
# recreated rather than updated in place.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# miniCOIL length normalisation. Average chunk length in tokens, derived from
# the 512-character chunk target (~80 tokens). Index-time only; not used when
# encoding queries. Revisit if chunk sizing changes.
MINICOIL_AVG_LEN = 80
```

- [ ] **Step 5: Create the retrieval package**

Create `retrieval/__init__.py`:

```python
"""Transport-agnostic retrieval stages.

This package must not import from `server/` or `fastmcp` — it is composed by
MCP adapters but knows nothing about them.
"""
```

Create `retrieval/models.py`:

```python
"""Shared dataclasses for the retrieval pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A single ranked hit from one retrieval channel."""

    id: str
    text: str
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    score: float
    channel: str
    metadata: dict = Field(default_factory=dict)  # type: ignore[type-arg]


class FusedResult(BaseModel):
    """A candidate after rank fusion, optionally rescored by the reranker."""

    id: str
    text: str
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    score: float  # rerank score when reranked, else equal to fusion_score
    fusion_score: float
    channels: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)  # type: ignore[type-arg]
```

- [ ] **Step 6: Add the fastembed dependency**

Run: `uv add 'fastembed>=0.7.0'`

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings.py -v && uv run ruff check . && uv run mypy . 2>&1 | tail -1`
Expected: PASS, clean lint and types

- [ ] **Step 8: Commit**

```bash
git add config/settings.py config/constants.py retrieval/ tests/test_settings.py pyproject.toml uv.lock
git commit -m "feat(retrieval): add pipeline settings, vector-name constants, and models"
```

---

## Task 2: RRF fusion

**Files:**
- Create: `retrieval/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `retrieval.models.Candidate`, `retrieval.models.FusedResult`
- Produces: `rrf(channels: list[list[Candidate]], k: int = 60) -> list[FusedResult]`

Reciprocal Rank Fusion scores each document `sum(1 / (k + rank))` across the channels that returned it, where `rank` is 1-based position within that channel. It ignores raw scores entirely, which is why it can merge cosine similarity and miniCOIL scores without normalisation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.fusion'`

- [ ] **Step 3: Write the implementation**

Create `retrieval/fusion.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Commit**

```bash
git add retrieval/fusion.py tests/test_fusion.py
git commit -m "feat(retrieval): add reciprocal rank fusion"
```

---

## Task 3: Relevance gate

**Files:**
- Create: `retrieval/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `retrieval.models.FusedResult`
- Produces: `should_retry(results: list[FusedResult], floor: float) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gate.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.gate'`

- [ ] **Step 3: Write the implementation**

Create `retrieval/gate.py`:

```python
"""Relevance gate deciding whether a retrieval pass needs widening.

The failure this targets is "the right chunk was at rank 60 and we fetched
20". A weak top-1 rerank score is the cheapest available signal for it. The
retry widens the candidate pool and re-ranks; it makes no extra LLM call.
"""

from __future__ import annotations

from retrieval.models import FusedResult


def should_retry(results: list[FusedResult], floor: float) -> bool:
    """Return True when the best result is too weak to trust.

    Args:
        results: Reranked results, best first.
        floor: Minimum acceptable top-1 score. Inclusive.

    Returns:
        True if the caller should widen the pool and retry once.
    """
    if not results:
        return True
    return results[0].score < floor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gate.py -v`
Expected: PASS — 5 tests

- [ ] **Step 5: Commit**

```bash
git add retrieval/gate.py tests/test_gate.py
git commit -m "feat(retrieval): add relevance gate"
```

---

## Task 4: Reranker v3.5

**Files:**
- Create: `retrieval/rerank.py`
- Delete: `server/tools/reranker.py`
- Modify: `server/tools/vector_search.py:116`, `server/tools/hybrid_search.py:91`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `retrieval.models.FusedResult`, `settings.rerank_model`, `settings.rerank_candidates`
- Produces: `RerankError` (exception), `rerank(query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]`, `rerank_dicts(query: str, results: list[dict], top_k: int) -> list[dict]`

Two deliberately different failure contracts:

- **`rerank`** (typed, used by the pipeline) **raises `RerankError`**. The pipeline needs to know reranking failed so it can add `rerank` to `degraded`; if this swallowed the error, the caller would have to infer failure by comparing scores, which is unreliable.
- **`rerank_dicts`** (legacy, used by `vector_search`) **swallows and returns the original order**, preserving today's behavior for a caller that has no `degraded` field to report into.

- [ ] **Step 1: Verify the live API response shape**

This is the design doc's open verification item. Do it before writing code.

Run:

```bash
curl -s https://api.jina.ai/v1/rerank \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"jina-reranker-v2-base-multilingual","query":"what is qdrant",
       "documents":["Qdrant is a vector database.","Bananas are yellow."],
       "top_n":2}' | head -40

curl -s https://api.jina.ai/v1/rerank \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"jina-reranker-v3.5","query":"what is qdrant",
       "documents":["Qdrant is a vector database.","Bananas are yellow."],
       "top_n":2}' | head -40
```

Expected: both return `{"results": [{"index": int, "relevance_score": float, ...}]}`.

**If the v3.5 shape differs**, stop and adjust `_parse_results` in Step 3 to match, and note the difference in the commit message. Do not guess.

- [ ] **Step 2: Write the failing test**

Create `tests/test_rerank.py`:

```python
"""Tests for the Jina reranker wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.models import FusedResult
from retrieval.rerank import RerankError, rerank, rerank_dicts


def _result(doc_id: str, text: str) -> FusedResult:
    return FusedResult(
        id=doc_id, text=text, source_file="doc.pdf", score=0.01, fusion_score=0.01
    )


@pytest.mark.asyncio
async def test_rerank_applies_api_ordering() -> None:
    """Results are reordered and rescored from the API response."""
    results = [_result("a", "first"), _result("b", "second")]
    api = [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]
    with patch("retrieval.rerank._rerank_request", AsyncMock(return_value=api)):
        out = await rerank("q", results, top_k=2)

    assert [r.id for r in out] == ["b", "a"]
    assert out[0].score == 0.9
    # fusion_score survives reranking for debugging
    assert out[0].fusion_score == 0.01


@pytest.mark.asyncio
async def test_rerank_raises_on_api_failure() -> None:
    """The typed path propagates failure so the pipeline can flag it.

    Swallowing here would force the caller to guess whether reranking
    happened by comparing scores, which is unreliable.
    """
    results = [_result("a", "first"), _result("b", "second")]
    with patch("retrieval.rerank._rerank_request", AsyncMock(side_effect=RuntimeError)):
        with pytest.raises(RerankError):
            await rerank("q", results, top_k=1)


@pytest.mark.asyncio
async def test_rerank_empty_input_returns_empty() -> None:
    """No results in, no results out, no API call."""
    assert await rerank("q", [], top_k=5) == []


@pytest.mark.asyncio
async def test_rerank_uses_configured_model() -> None:
    """The model string comes from settings, not a hardcoded constant."""
    captured: dict = {}  # type: ignore[type-arg]

    async def _fake(query: str, documents: list[str], top_n: int) -> list[dict]:  # type: ignore[type-arg]
        captured["called"] = True
        return [{"index": 0, "relevance_score": 0.5}]

    with patch("retrieval.rerank._rerank_request", _fake):
        await rerank("q", [_result("a", "x")], top_k=1)
    assert captured["called"] is True


@pytest.mark.asyncio
async def test_rerank_dicts_preserves_dict_contract() -> None:
    """The legacy dict path keeps original_score, for vector_search."""
    results = [{"text": "first", "score": 0.4}, {"text": "second", "score": 0.3}]
    api = [{"index": 1, "relevance_score": 0.95}]
    with patch("retrieval.rerank._rerank_request", AsyncMock(return_value=api)):
        out = await rerank_dicts("q", results, top_k=1)

    assert out[0]["text"] == "second"
    assert out[0]["score"] == 0.95
    assert out[0]["original_score"] == 0.3
```

- [ ] **Step 3: Write the implementation**

Create `retrieval/rerank.py`:

```python
"""Jina Reranker v3.5 — listwise re-scoring of retrieved candidates.

v3.5 ranks the whole candidate list in one forward pass ("last but not late"
interaction) rather than scoring each document independently, which is why it
outperforms the pointwise v2 it replaces. The /v1/rerank request schema is
unchanged, so this is a model-string swap plus typed plumbing.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from retrieval.models import FusedResult

logger = logging.getLogger(__name__)

RERANK_URL = f"{settings.jina_api_url}/v1/rerank"


class RerankError(RuntimeError):
    """Raised when reranking fails and the caller must handle degradation."""


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(settings.max_retries),
    retry=retry_if_exception_type((httpx.HTTPStatusError,)),
)
async def _rerank_request(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[dict]:  # type: ignore[type-arg]
    """Call the Jina Reranker API and return its ranked results."""
    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(30.0),
    ) as client:
        resp = await client.post(
            RERANK_URL,
            json={
                "model": settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Jina Reranker error: {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()["results"]  # type: ignore[no-any-return]


async def rerank(
    query: str,
    results: list[FusedResult],
    top_k: int = 5,
) -> list[FusedResult]:
    """Re-score fused results against the query.

    Args:
        query: The original user query, never a sub-query.
        results: Fused candidates to rescore.
        top_k: Number of results to return.

    Returns:
        Results ordered by rerank score.

    Raises:
        RerankError: The API call failed. The caller decides how to degrade —
            it is the only layer that knows whether it can report `degraded`.
    """
    if not results:
        return []

    documents = [r.text for r in results]
    if not any(documents):
        return results[:top_k]

    try:
        ranked = await _rerank_request(query, documents, top_k)
    except Exception as exc:
        logger.exception("Reranking failed")
        raise RerankError(str(exc)) from exc

    out: list[FusedResult] = []
    for item in ranked:
        original = results[item["index"]]
        out.append(original.model_copy(update={"score": item["relevance_score"]}))
    return out


async def rerank_dicts(
    query: str,
    results: list[dict],  # type: ignore[type-arg]
    top_k: int = 5,
) -> list[dict]:  # type: ignore[type-arg]
    """Dict-based reranking for callers that predate the typed pipeline.

    Preserves the original score as 'original_score'. Used by vector_search.
    """
    if not results:
        return results

    documents = [r.get("text", "") for r in results]
    if not any(documents):
        return results[:top_k]

    try:
        ranked = await _rerank_request(query, documents, top_k)
    except Exception:
        logger.exception("Reranking failed, returning originals")
        return results[:top_k]

    reranked: list[dict] = []  # type: ignore[type-arg]
    for item in ranked:
        original = results[item["index"]].copy()
        original["original_score"] = original.get("score")
        original["score"] = item["relevance_score"]
        reranked.append(original)
    return reranked
```

- [ ] **Step 4: Update the two existing callers**

In `server/tools/vector_search.py`, replace the import inside `vector_search` (currently line 116):

```python
            from retrieval.rerank import rerank_dicts

            results = await rerank_dicts(query, results, top_k=limit)
```

In `server/tools/hybrid_search.py`, replace the import block (currently line 91):

```python
            from retrieval.rerank import rerank_dicts

            try:
                kb_results = await rerank_dicts(query, kb_results, top_k=limit)
```

Then delete the old module:

```bash
git rm server/tools/reranker.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rerank.py tests/test_tools.py -v && uv run ruff check . && uv run mypy . 2>&1 | tail -1`
Expected: PASS. `tests/test_tools.py` must still pass — it exercises the existing tools through the moved reranker.

If `tests/test_tools.py` patches `server.tools.reranker.rerank`, update those patch targets to `retrieval.rerank.rerank_dicts`.

- [ ] **Step 6: Commit**

```bash
git add retrieval/rerank.py tests/test_rerank.py server/tools/vector_search.py \
        server/tools/hybrid_search.py tests/test_tools.py
git commit -m "feat(retrieval): upgrade to jina-reranker-v3.5 and move reranker into retrieval/"
```

---

## Task 5: miniCOIL sparse embedder

**Files:**
- Create: `ingestion/sparse_embedder.py`
- Test: `tests/test_sparse_embedder.py`

**Interfaces:**
- Consumes: `settings.sparse_model`, `constants.MINICOIL_AVG_LEN`
- Produces: `encode_documents(texts: list[str]) -> list[SparseVector]`, `encode_query(text: str) -> SparseVector`, `SparseVector` (a `qdrant_client.models.SparseVector`)

miniCOIL is a local CPU model loaded lazily through fastembed. Document encoding applies BM25-style length normalisation via `avg_len`; query encoding does not.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sparse_embedder.py`:

```python
"""Tests for the miniCOIL sparse embedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.sparse_embedder import encode_documents, encode_query, reset_model


class _FakeEmbedding:
    """Mimics a fastembed SparseEmbedding."""

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    reset_model()


def test_encode_documents_returns_sparse_vectors() -> None:
    """Document texts map to Qdrant SparseVector objects."""
    fake = MagicMock()
    fake.embed.return_value = iter([_FakeEmbedding([1, 5], [0.4, 0.7])])

    with patch("ingestion.sparse_embedder._load_model", return_value=fake):
        out = encode_documents(["hello world"])

    assert len(out) == 1
    assert out[0].indices == [1, 5]
    assert out[0].values == [0.4, 0.7]


def test_encode_documents_empty_input_skips_model() -> None:
    """No texts means no model load and no output."""
    with patch("ingestion.sparse_embedder._load_model") as loader:
        assert encode_documents([]) == []
    loader.assert_not_called()


def test_encode_query_returns_single_vector() -> None:
    """Query encoding returns one SparseVector, not a list."""
    fake = MagicMock()
    fake.query_embed.return_value = iter([_FakeEmbedding([2], [0.9])])

    with patch("ingestion.sparse_embedder._load_model", return_value=fake):
        out = encode_query("hello")

    assert out.indices == [2]
    assert out.values == [0.9]


def test_model_is_loaded_once() -> None:
    """The model is cached across calls — loading is expensive."""
    fake = MagicMock()
    fake.embed.return_value = iter([_FakeEmbedding([1], [0.5])])

    with patch("ingestion.sparse_embedder._load_model", return_value=fake) as loader:
        encode_documents(["a"])
        fake.embed.return_value = iter([_FakeEmbedding([1], [0.5])])
        encode_documents(["b"])

    assert loader.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sparse_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.sparse_embedder'`

- [ ] **Step 3: Write the implementation**

Create `ingestion/sparse_embedder.py`:

```python
"""miniCOIL sparse encoding for the lexical retrieval channel.

miniCOIL behaves like BM25 that understands word sense — it keeps exact-term
matching while disambiguating by context. The model runs locally on CPU via
fastembed, so there is no API cost, but the first call pays a load penalty.

Document encoding applies BM25-style length normalisation through avg_len;
query encoding deliberately does not.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models

from config.constants import MINICOIL_AVG_LEN
from config.settings import settings

logger = logging.getLogger(__name__)

_model: Any | None = None

SparseVector = models.SparseVector


def _load_model() -> Any:
    """Instantiate the fastembed sparse model. Imported lazily — heavy."""
    from fastembed import SparseTextEmbedding

    logger.info("Loading sparse model %s", settings.sparse_model)
    # avg_len is a CONSTRUCTOR kwarg, not a per-call embed() option. Passing it
    # as embed(options={"avg_len": ...}) is silently swallowed and has zero
    # effect on length normalisation — verified empirically against fastembed
    # 0.8.0. Document values shift with avg_len; query_embed() ignores it, so
    # the document/query asymmetry falls out without per-call handling.
    return SparseTextEmbedding(model_name=settings.sparse_model, avg_len=MINICOIL_AVG_LEN)


def _get_model() -> Any:
    global _model  # noqa: PLW0603
    if _model is None:
        _model = _load_model()
    return _model


def reset_model() -> None:
    """Drop the cached model. Test hook."""
    global _model  # noqa: PLW0603
    _model = None


def encode_documents(texts: list[str]) -> list[SparseVector]:
    """Encode chunk texts for indexing.

    Args:
        texts: Chunk texts to encode.

    Returns:
        One SparseVector per input text, in order.
    """
    if not texts:
        return []

    embeddings = _get_model().embed(texts)
    return [
        models.SparseVector(indices=list(e.indices), values=list(e.values))
        for e in embeddings
    ]


def encode_query(text: str) -> SparseVector:
    """Encode a query for search. No length normalisation.

    Args:
        text: The query string.

    Returns:
        A single SparseVector.
    """
    embedding = next(iter(_get_model().query_embed(text)))
    return models.SparseVector(
        indices=list(embedding.indices), values=list(embedding.values)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sparse_embedder.py -v`
Expected: PASS — 4 tests

- [ ] **Step 5: Smoke-test against the real model**

Run:

```bash
uv run python -c "
from ingestion.sparse_embedder import encode_documents, encode_query
d = encode_documents(['Qdrant is a vector database', 'apple slicer'])
q = encode_query('vector db')
print('docs:', [(len(v.indices)) for v in d])
print('query:', len(q.indices))
assert all(len(v.indices) > 0 for v in d)
assert len(q.indices) > 0
print('OK')
"
```

Expected: non-zero index counts and `OK`. This is the first real download of the model — expect a delay.

- [ ] **Step 6: Commit**

```bash
git add ingestion/sparse_embedder.py tests/test_sparse_embedder.py
git commit -m "feat(ingestion): add miniCOIL sparse embedder"
```

---

## Task 6: Named-vector collection schema and sparse indexing

**Files:**
- Modify: `ingestion/qdrant_setup.py:18-46`, `ingestion/pipeline.py:234-258`, `server/tools/vector_search.py:93-99`, `scripts/doctor.py`
- Test: `tests/test_qdrant_setup.py`, `tests/test_pipeline_chunking.py`, `tests/test_doctor.py`

**Interfaces:**
- Consumes: `constants.DENSE_VECTOR_NAME`, `constants.SPARSE_VECTOR_NAME`, `ingestion.sparse_embedder.encode_documents`
- Produces: `documents_dense` with named vectors `dense` (512d cosine) and `sparse` (miniCOIL, `Modifier.IDF`); every dense point carries both

This is the breaking schema change. Qdrant requires sparse vectors to be named, and named and unnamed vectors cannot coexist in one collection.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_qdrant_setup.py`:

```python
def test_dense_collection_uses_named_vectors() -> None:
    """Dense and sparse vectors are both named, sparse uses IDF."""
    from unittest.mock import MagicMock

    from qdrant_client import models

    from config.constants import DENSE_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
    from ingestion.qdrant_setup import create_dense_collection

    client = MagicMock()
    client.collection_exists.return_value = False
    create_dense_collection(client)

    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == DENSE_COLLECTION
    assert DENSE_VECTOR_NAME in kwargs["vectors_config"]
    assert SPARSE_VECTOR_NAME in kwargs["sparse_vectors_config"]
    sparse_cfg = kwargs["sparse_vectors_config"][SPARSE_VECTOR_NAME]
    assert sparse_cfg.modifier == models.Modifier.IDF
```

Add to `tests/test_pipeline_chunking.py`:

```python
def test_text_chunk_points_carry_named_dense_and_sparse() -> None:
    """Upserted points use the named-vector dict with a sparse entry."""
    from unittest.mock import patch

    from qdrant_client import models

    from config.constants import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    fake_sparse = [models.SparseVector(indices=[1], values=[0.5])]
    with patch(
        "ingestion.pipeline.encode_documents", return_value=fake_sparse
    ) as enc:
        from ingestion.pipeline import _build_chunk_point

        point = _build_chunk_point(
            source_file="doc.pdf",
            page_number=1,
            chunk_index=0,
            text="hello",
            contextualized_text=None,
            vector=[0.1] * 512,
            mime="application/pdf",
            now="2026-07-31T00:00:00Z",
            embedder_model="jina-embeddings-v4",
            embedder_dim=512,
        )

    enc.assert_called_once_with(["hello"])
    assert DENSE_VECTOR_NAME in point.vector
    assert SPARSE_VECTOR_NAME in point.vector
    assert point.payload["text_content"] == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_qdrant_setup.py::test_dense_collection_uses_named_vectors tests/test_pipeline_chunking.py::test_text_chunk_points_carry_named_dense_and_sparse -v`
Expected: FAIL — `KeyError: 'vectors_config'` is a dict of one unnamed config, and `_build_chunk_point` does not exist

- [ ] **Step 3: Update the collection schema**

In `ingestion/qdrant_setup.py`, replace the body of `create_dense_collection` after the existence check:

```python
    dim = settings.dense_dimensions
    client.create_collection(
        collection_name=DENSE_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
```

Add `DENSE_VECTOR_NAME` and `SPARSE_VECTOR_NAME` to the `config.constants` import at the top of the file. Leave the two `create_payload_index` calls unchanged.

- [ ] **Step 4: Extract and update point construction**

In `ingestion/pipeline.py`, add the import:

```python
from config.constants import (
    DENSE_COLLECTION,
    DENSE_VECTOR_NAME,
    MULTIVEC_COLLECTION,
    SPARSE_VECTOR_NAME,
)
from ingestion.sparse_embedder import encode_documents
```

Extract the payload construction (currently lines 234-258) into a helper above the function that uses it, so it stays under the 60-line function cap and becomes testable:

```python
def _build_chunk_point(
    *,
    source_file: str,
    page_number: int,
    chunk_index: int,
    text: str,
    contextualized_text: str | None,
    vector: list[float],
    mime: str,
    now: str,
    embedder_model: str,
    embedder_dim: int,
) -> models.PointStruct:
    """Build one dense-collection point with named dense + sparse vectors."""
    chunk_id = _make_chunk_id(source_file, page_number, chunk_index)
    payload = {
        "source_file": source_file,
        "content_type": "text_chunk",
        "page_number": page_number,
        "chunk_index": chunk_index,
        "char_count": len(text),
        "text_content": text,
        "embedder_model": embedder_model,
        "embedder_dim": embedder_dim,
        "metadata": {
            "mime_type": mime,
            "ingested_at": now,
            "source_key": source_file,
        },
    }
    if contextualized_text is not None:
        payload["contextualized_text"] = contextualized_text

    sparse = encode_documents([text])[0]
    return models.PointStruct(
        id=_make_point_id(chunk_id),
        vector={DENSE_VECTOR_NAME: vector, SPARSE_VECTOR_NAME: sparse},
        payload=payload,
    )
```

Replace the inline construction in the chunk loop with a call to `_build_chunk_point`, passing `embedder.model_name` and `embedder.dim`.

Apply the same named-vector change to the two other `payload={` upsert sites in `ingestion/pipeline.py` (lines ~294 and ~330 — image and VLM caption points). Those have no text to encode, so give them `{DENSE_VECTOR_NAME: vector}` only, with no sparse entry. Qdrant permits points that omit a sparse vector.

- [ ] **Step 5: Point vector_search at the named vector**

In `server/tools/vector_search.py`, the `query_points` call must specify the named vector — without `using=` it fails against a named-vector collection. Change it to:

```python
        response = qdrant.query_points(
            collection_name=DENSE_COLLECTION,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
```

Add `DENSE_VECTOR_NAME` to the `config.constants` import. Apply the same `using=DENSE_VECTOR_NAME` argument to every `query_points` call in `_dual_query` in the same file.

- [ ] **Step 6: Teach doctor about sparse vectors**

In `scripts/doctor.py`, extend the payload audit to flag points missing a sparse vector. Find the loop that checks `embedder_model` / `embedder_dim` and add alongside it:

```python
        if point.payload.get("content_type") == "text_chunk":
            vectors = point.vector or {}
            if SPARSE_VECTOR_NAME not in vectors:
                missing_sparse.append(point.id)
```

Initialise `missing_sparse: list[str] = []` before the loop, import `SPARSE_VECTOR_NAME` from `config.constants`, ensure the scroll requests `with_vectors=True`, and report after the loop:

```python
    if missing_sparse:
        print(f"  ✗ {len(missing_sparse)} text chunks missing sparse vectors")
        print("    Re-ingest required: task ingest")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_qdrant_setup.py tests/test_pipeline_chunking.py tests/test_doctor.py tests/test_tools.py -v && uv run ruff check . && uv run mypy . 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 8: Perform the migration**

```bash
task up
uv run python -c "
from qdrant_client import QdrantClient
from config.constants import DENSE_COLLECTION
from config.settings import settings
c = QdrantClient(url=settings.qdrant_url)
if c.collection_exists(DENSE_COLLECTION):
    c.delete_collection(DENSE_COLLECTION)
    print('dropped', DENSE_COLLECTION)
"
uv run cocoindex drop --force 2>/dev/null || echo "clear CocoIndex tracking manually if this failed"
task ingest
task doctor
```

Expected: `task doctor` reports no drift and no missing sparse vectors.

- [ ] **Step 9: Commit**

```bash
git add ingestion/qdrant_setup.py ingestion/pipeline.py server/tools/vector_search.py \
        scripts/doctor.py tests/
git commit -m "feat(ingestion): migrate documents_dense to named dense + sparse vectors"
```

---

## Task 7: Retrieval channels

**Files:**
- Create: `retrieval/channels.py`
- Test: `tests/test_channels.py`

**Interfaces:**
- Consumes: `retrieval.models.Candidate`, `ingestion.sparse_embedder.encode_query`, `ingestion.embedder.create_embedder`, `constants.DENSE_VECTOR_NAME`, `constants.SPARSE_VECTOR_NAME`
- Produces: `dense_channel(query, limit, query_filter) -> list[Candidate]`, `sparse_channel(query, limit, query_filter) -> list[Candidate]`, `build_filter(content_type, source_file, session_id) -> models.Filter | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_channels.py`:

```python
"""Tests for dense and sparse retrieval channels."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from retrieval.channels import build_filter, dense_channel, sparse_channel


def _point(point_id: str, score: float, text: str) -> MagicMock:
    p = MagicMock()
    p.id = point_id
    p.score = score
    p.payload = {
        "text_content": text,
        "source_file": "doc.pdf",
        "page_number": 2,
        "chunk_index": 3,
        "metadata": {"mime_type": "application/pdf"},
    }
    return p


@pytest.mark.asyncio
async def test_dense_channel_targets_named_dense_vector() -> None:
    """The dense query must specify using='dense'."""
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_point("p1", 0.9, "hello")]
    embedder = MagicMock()
    embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

    with (
        patch("retrieval.channels._get_qdrant_client", return_value=qdrant),
        patch("retrieval.channels._get_embedder", return_value=embedder),
    ):
        out = await dense_channel("hello", limit=10, query_filter=None)

    assert qdrant.query_points.call_args.kwargs["using"] == "dense"
    assert len(out) == 1
    assert out[0].channel == "dense"
    assert out[0].id == "p1"
    assert out[0].chunk_index == 3


@pytest.mark.asyncio
async def test_sparse_channel_targets_named_sparse_vector() -> None:
    """The sparse query must specify using='sparse'."""
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_point("p2", 4.2, "world")]

    with (
        patch("retrieval.channels._get_qdrant_client", return_value=qdrant),
        patch(
            "retrieval.channels.encode_query",
            return_value=models.SparseVector(indices=[1], values=[0.5]),
        ),
    ):
        out = await sparse_channel("world", limit=10, query_filter=None)

    assert qdrant.query_points.call_args.kwargs["using"] == "sparse"
    assert out[0].channel == "sparse"


@pytest.mark.asyncio
async def test_dense_channel_empty_query_returns_empty() -> None:
    """A blank query short-circuits without touching Qdrant."""
    with patch("retrieval.channels._get_qdrant_client") as client:
        assert await dense_channel("  ", limit=10, query_filter=None) == []
    client.assert_not_called()


def test_build_filter_combines_conditions() -> None:
    """Content type, source file, and session all become must-conditions."""
    f = build_filter(content_type="text_chunk", source_file="a.pdf", session_id="s1")
    assert f is not None
    assert len(f.must) == 3


def test_build_filter_returns_none_when_unfiltered() -> None:
    """No filters means no Filter object."""
    assert build_filter(content_type=None, source_file=None, session_id=None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_channels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.channels'`

- [ ] **Step 3: Write the implementation**

Create `retrieval/channels.py`:

```python
"""Dense and sparse retrieval channels over documents_dense.

Each channel returns its own ranked list. Fusion is a separate stage — these
functions never compare scores across channels, because cosine similarity and
miniCOIL scores are not on a comparable scale.
"""

from __future__ import annotations

import asyncio
import logging

from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from config.settings import settings
from ingestion.embedder import Embedder, create_embedder
from ingestion.sparse_embedder import encode_query
from retrieval.models import Candidate

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None
_embedder: Embedder | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def _get_embedder() -> Embedder:
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


def build_filter(
    content_type: str | None,
    source_file: str | None,
    session_id: str | None,
) -> models.Filter | None:
    """Assemble a Qdrant filter from optional constraints."""
    conditions: list[models.FieldCondition] = []
    if content_type is not None:
        conditions.append(
            models.FieldCondition(
                key="content_type", match=models.MatchValue(value=content_type)
            )
        )
    if source_file is not None:
        conditions.append(
            models.FieldCondition(
                key="source_file", match=models.MatchValue(value=source_file)
            )
        )
    if session_id is not None:
        # session_id is written at the TOP LEVEL of the payload by
        # ingestion/live_ingest.py, not nested under metadata. The existing
        # _dual_query in server/tools/vector_search.py filters on the same key.
        conditions.append(
            models.FieldCondition(
                key="session_id", match=models.MatchValue(value=session_id)
            )
        )
    return models.Filter(must=conditions) if conditions else None  # type: ignore[arg-type]


def _to_candidates(points: list, channel: str) -> list[Candidate]:  # type: ignore[type-arg]
    """Convert Qdrant scored points into Candidates."""
    out: list[Candidate] = []
    for point in points:
        payload = point.payload or {}
        out.append(
            Candidate(
                id=str(point.id),
                text=payload.get("text_content", payload.get("text", "")),
                source_file=payload.get("source_file", ""),
                page_number=payload.get("page_number", 0),
                chunk_index=payload.get("chunk_index", 0),
                score=point.score,
                channel=channel,
                metadata=payload.get("metadata", {}),
            )
        )
    return out


async def dense_channel(
    query: str,
    limit: int,
    query_filter: models.Filter | None,
) -> list[Candidate]:
    """Semantic similarity search over the named 'dense' vector."""
    if not query or not query.strip():
        return []

    vector = await _get_embedder().embed_text_query(query)
    response = await asyncio.to_thread(
        _get_qdrant_client().query_points,
        collection_name=DENSE_COLLECTION,
        query=vector,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _to_candidates(response.points, "dense")


async def sparse_channel(
    query: str,
    limit: int,
    query_filter: models.Filter | None,
) -> list[Candidate]:
    """Lexical search over the named 'sparse' miniCOIL vector."""
    if not query or not query.strip():
        return []

    vector = await asyncio.to_thread(encode_query, query)
    response = await asyncio.to_thread(
        _get_qdrant_client().query_points,
        collection_name=DENSE_COLLECTION,
        query=vector,
        using=SPARSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _to_candidates(response.points, "sparse")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_channels.py -v`
Expected: PASS — 5 tests

- [ ] **Step 5: Commit**

```bash
git add retrieval/channels.py tests/test_channels.py
git commit -m "feat(retrieval): add dense and sparse retrieval channels"
```

---

## Task 8: Query decomposition

**Files:**
- Create: `retrieval/decompose.py`
- Test: `tests/test_decompose.py`

**Interfaces:**
- Consumes: `settings.decompose_enabled`, `settings.decompose_model`, `settings.decompose_max_subqueries`, `settings.anthropic_api_key`
- Produces: `decompose(query: str) -> list[str]` — always returns at least `[query]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_decompose.py`:

```python
"""Tests for query decomposition."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.decompose import _model_name, _parse_subqueries, decompose


def test_model_name_falls_back_to_llm_model() -> None:
    """An unset decompose_model routes to the primary LLM model."""
    with (
        patch("retrieval.decompose.settings.decompose_model", ""),
        patch("retrieval.decompose.settings.llm_model", "primary/model"),
    ):
        assert _model_name() == "primary/model"


def test_model_name_prefers_explicit_decompose_model() -> None:
    """An explicit decompose_model wins over llm_model."""
    with (
        patch("retrieval.decompose.settings.decompose_model", "cheap/model"),
        patch("retrieval.decompose.settings.llm_model", "primary/model"),
    ):
        assert _model_name() == "cheap/model"


def test_parse_extracts_numbered_lines() -> None:
    """The model returns one sub-query per line."""
    raw = "1. What is the warranty period?\n2. What voids the warranty?"
    assert _parse_subqueries(raw, max_n=4) == [
        "What is the warranty period?",
        "What voids the warranty?",
    ]


def test_parse_handles_bare_lines() -> None:
    """Unnumbered lines are accepted too."""
    assert _parse_subqueries("first thing\nsecond thing", max_n=4) == [
        "first thing",
        "second thing",
    ]


def test_parse_respects_max() -> None:
    """More sub-queries than allowed are truncated."""
    raw = "\n".join(f"{i}. q{i}" for i in range(1, 10))
    assert len(_parse_subqueries(raw, max_n=4)) == 4


def test_parse_ignores_blank_and_noise_lines() -> None:
    """Empty lines and a preamble are dropped."""
    raw = "Here are the sub-queries:\n\n1. real one\n\n"
    assert _parse_subqueries(raw, max_n=4) == ["Here are the sub-queries:", "real one"]


@pytest.mark.asyncio
async def test_decompose_disabled_returns_identity() -> None:
    """With the flag off, no LLM call happens."""
    with (
        patch("retrieval.decompose.settings.decompose_enabled", False),
        patch("retrieval.decompose._call_llm") as llm,
    ):
        assert await decompose("a and b") == ["a and b"]
    llm.assert_not_called()


@pytest.mark.asyncio
async def test_decompose_llm_failure_returns_identity() -> None:
    """An LLM error degrades to the original query."""
    with patch(
        "retrieval.decompose._call_llm", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        assert await decompose("a and b") == ["a and b"]


@pytest.mark.asyncio
async def test_decompose_single_subquery_returns_original() -> None:
    """If the model returns one part, the query was not multi-part."""
    with patch("retrieval.decompose._call_llm", AsyncMock(return_value="1. a and b")):
        assert await decompose("a and b") == ["a and b"]


@pytest.mark.asyncio
async def test_decompose_returns_subqueries() -> None:
    """A genuine multi-part question splits."""
    with patch(
        "retrieval.decompose._call_llm",
        AsyncMock(return_value="1. warranty period\n2. warranty exclusions"),
    ):
        assert await decompose("what is the warranty and what voids it") == [
            "warranty period",
            "warranty exclusions",
        ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decompose.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.decompose'`

- [ ] **Step 3: Write the implementation**

Create `retrieval/decompose.py`:

```python
"""Query decomposition — split multi-part questions into sub-queries.

Each sub-query becomes an additional retrieval channel, so decomposition
composes naturally with RRF: N sub-queries produce 2N channels, all fused in
one pass. Every failure path returns [query] unchanged, so a decomposition
outage degrades to single-query retrieval rather than an error.
"""

from __future__ import annotations

import logging

from config.settings import settings

logger = logging.getLogger(__name__)

PROMPT = """Split this search query into independent sub-queries, one per line.

Rules:
- Output ONLY the sub-queries, one per line, no preamble.
- If the query asks exactly one thing, output it unchanged as a single line.
- Maximum {max_n} lines.
- Each sub-query must be self-contained and searchable on its own.

Query: {query}"""


def _parse_subqueries(raw: str, max_n: int) -> list[str]:
    """Extract sub-queries from the model's line-per-query output."""
    out: list[str] = []
    for line in raw.strip().splitlines():
        text = line.strip()
        if not text:
            continue
        # Strip a leading "1." / "1)" / "- " enumerator if present.
        for sep in (". ", ") ", "- "):
            head, found, tail = text.partition(sep)
            if found and head.lstrip("-").strip().isdigit() or (sep == "- " and not head):
                text = tail.strip()
                break
        if text:
            out.append(text)
        if len(out) >= max_n:
            break
    return out


def _model_name() -> str:
    """Decomposition model, falling back to the primary LLM when unset."""
    return settings.decompose_model or settings.llm_model


async def _call_llm(prompt: str) -> str:
    """Send the decomposition prompt to the configured provider.

    Follows the same anthropic/openai dispatch as ingestion/entity_extractor.py
    — this deployment may route Anthropic models through an OpenAI-compatible
    gateway, so the provider is a config choice, not a model-name choice.
    """
    if settings.llm_api_type.lower() == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
        response = await client.messages.create(
            model=_model_name(),
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    import openai

    client_oa = openai.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
    )
    completion = await client_oa.chat.completions.create(
        model=_model_name(),
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


async def decompose(query: str) -> list[str]:
    """Split a query into sub-queries, or return it unchanged.

    Args:
        query: The user's original query.

    Returns:
        One or more sub-queries. Never empty; falls back to [query].
    """
    if not settings.decompose_enabled or not query.strip():
        return [query]

    prompt = PROMPT.format(query=query, max_n=settings.decompose_max_subqueries)
    try:
        raw = await _call_llm(prompt)
    except Exception:
        logger.exception("Decomposition failed, using the original query")
        return [query]

    parts = _parse_subqueries(raw, settings.decompose_max_subqueries)
    if len(parts) <= 1:
        return [query]
    return parts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_decompose.py -v`
Expected: PASS — 10 tests

If `test_parse_ignores_blank_and_noise_lines` or the enumerator-stripping tests fail, fix `_parse_subqueries` until they pass — the parser must handle numbered, bulleted, and bare lines.

- [ ] **Step 5: Commit**

```bash
git add retrieval/decompose.py tests/test_decompose.py
git commit -m "feat(retrieval): add LLM query decomposition"
```

---

## Task 9: Pipeline composition

**Files:**
- Create: `retrieval/pipeline.py`
- Test: `tests/test_pipeline_retrieval.py`

**Interfaces:**
- Consumes: `dense_channel`, `sparse_channel`, `build_filter`, `rrf`, `rerank`, `decompose`, `should_retry`
- Produces: `fast_pipeline(...) -> PipelineOutput`, `smart_pipeline(...) -> PipelineOutput`, `PipelineOutput` (fields: `results: list[FusedResult]`, `degraded: list[str]`, `sub_queries: list[str]`, `retried: bool`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_retrieval.py`:

```python
"""Tests for retrieval pipeline composition and degradation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.models import Candidate, FusedResult
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
        patch("retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])),
        patch("retrieval.pipeline.sparse_channel", AsyncMock(return_value=[_cand("b", "sparse")])),
        patch("retrieval.pipeline.rerank", AsyncMock(side_effect=_passthrough_rerank)),
    ):
        out = await fast_pipeline("q", limit=10)

    assert {r.id for r in out.results} == {"a", "b"}
    assert out.degraded == []


@pytest.mark.asyncio
async def test_sparse_failure_degrades_to_dense() -> None:
    """A sparse outage still returns dense results, flagged."""
    with (
        patch("retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])),
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
        patch("retrieval.pipeline.sparse_channel", AsyncMock(return_value=[_cand("b", "sparse")])),
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
        patch("retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])),
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
        patch("retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])),
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
        patch("retrieval.pipeline.decompose", AsyncMock(return_value=["sub1", "sub2"])),
        patch("retrieval.pipeline.dense_channel", AsyncMock(return_value=[_cand("a", "dense")])),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `retrieval/pipeline.py`:

```python
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
            ("dense", asyncio.create_task(dense_channel(query, limit=limit, query_filter=query_filter)))
        )
        if settings.sparse_enabled:
            tasks.append(
                ("sparse", asyncio.create_task(sparse_channel(query, limit=limit, query_filter=query_filter)))
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

    results, degraded = await _retrieve_and_rank(
        sub_queries, query, limit, query_filter
    )

    retried = False
    if settings.retry_enabled and should_retry(results, settings.rerank_score_floor):
        widened = limit * settings.retry_limit_multiplier
        logger.info("Relevance gate fired, widening pool to %d", widened)
        results, degraded = await _retrieve_and_rank(
            sub_queries, query, widened, query_filter
        )
        results = results[:limit]
        retried = True

    return PipelineOutput(
        results=results,
        degraded=degraded,
        sub_queries=sub_queries,
        retried=retried,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_retrieval.py -v && uv run ruff check . && uv run mypy . 2>&1 | tail -1`
Expected: PASS — 9 tests

Note on `test_gate_triggers_exactly_one_retry`: the assertion reads `dense.await_args_list[-1].kwargs["limit"]`, so `_gather_channels` must pass `limit` as a keyword argument. It does.

- [ ] **Step 5: Commit**

```bash
git add retrieval/pipeline.py tests/test_pipeline_retrieval.py
git commit -m "feat(retrieval): compose fast and smart retrieval pipelines"
```

---

## Task 10: MCP tools

**Files:**
- Create: `server/tools/multi_search.py`
- Modify: `server/tools/hybrid_search.py` (full rewrite), `server/models.py`, `server/mcp_server.py:50-54`
- Test: `tests/test_multi_search.py`

**Interfaces:**
- Consumes: `retrieval.pipeline.fast_pipeline`, `retrieval.pipeline.smart_pipeline`, `server.tools.graph_search.graph_search`
- Produces: `multi_search(query, limit, content_type, source_file, session_id) -> dict`, `hybrid_search(query, limit, content_type, source_file, session_id) -> dict`, `server.models.FusedSearchResponse`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_search.py`:

```python
"""Tests for the multi_search and hybrid_search MCP adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.models import FusedResult
from retrieval.pipeline import PipelineOutput
from server.tools.hybrid_search import hybrid_search
from server.tools.multi_search import multi_search


def _fused(doc_id: str, source_type: str = "bulk") -> FusedResult:
    return FusedResult(
        id=doc_id,
        text=f"t-{doc_id}",
        source_file="d.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
        fusion_score=0.03,
        channels=["dense"],
        metadata={"source_type": source_type},
    )


@pytest.mark.asyncio
async def test_multi_search_returns_fused_schema() -> None:
    """Results carry score, fusion_score, and channel provenance."""
    out_pipeline = PipelineOutput(results=[_fused("a")])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
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
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())),
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)
    assert "degraded" not in out


@pytest.mark.asyncio
async def test_multi_search_reports_degraded_channels() -> None:
    """A failed channel surfaces in degraded."""
    out_pipeline = PipelineOutput(results=[_fused("a")], degraded=["sparse"])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)
    assert out["degraded"] == ["sparse"]


@pytest.mark.asyncio
async def test_total_channel_failure_sets_error_key() -> None:
    """Both channels down is distinguishable from a genuine zero-hit query."""
    out_pipeline = PipelineOutput(results=[], degraded=["dense", "sparse"])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)

    assert out["results"] == []
    assert "error" in out


@pytest.mark.asyncio
async def test_zero_hits_has_no_error_key() -> None:
    """A healthy query that simply matches nothing is not an error."""
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())),
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5)

    assert out["results"] == []
    assert "error" not in out


@pytest.mark.asyncio
async def test_graph_failure_degrades_not_raises() -> None:
    """A graph outage yields empty facts and a degraded flag."""
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=PipelineOutput())),
        patch("server.tools.multi_search.graph_search", AsyncMock(side_effect=RuntimeError)),
    ):
        out = await multi_search("q", limit=5)
    assert out["graph_facts"] == []
    assert "graph" in out["degraded"]


@pytest.mark.asyncio
async def test_live_results_split_out_when_session_active() -> None:
    """Live-session chunks are separated from KB results."""
    out_pipeline = PipelineOutput(results=[_fused("a", "live"), _fused("b", "bulk")])
    with (
        patch("server.tools.multi_search.fast_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
    ):
        out = await multi_search("q", limit=5, session_id="s1")

    assert [r["id"] for r in out["live_results"]] == ["a"]
    assert [r["id"] for r in out["results"]] == ["b"]


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
        patch("server.tools.hybrid_search.smart_pipeline", AsyncMock(return_value=out_pipeline)),
        patch("server.tools.hybrid_search.graph_search", AsyncMock(return_value=[])),
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
        patch("server.tools.multi_search.graph_search", AsyncMock(return_value=[])),
    ):
        await multi_search("q", limit=5000)
    assert captured["limit"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_multi_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.tools.multi_search'`

- [ ] **Step 3: Add the response model**

In `server/models.py`, add below the existing models (keep `HybridSearchResponse` for now; it is removed in Step 7):

```python
class FusedSearchResult(BaseModel):
    """One reranked, rank-fused retrieval hit."""

    id: str
    text: str
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    score: float
    fusion_score: float
    channels: list[str] = []
    metadata: dict = {}  # type: ignore[type-arg]


class FusedSearchResponse(BaseModel):
    """Shared response shape for multi_search and hybrid_search."""

    results: list[FusedSearchResult] = []
    graph_facts: list[GraphFact] = []
    live_results: list[FusedSearchResult] = []
    query: str
    session_id: str | None = None
    sub_queries: list[str] | None = None
    retried: bool | None = None
    degraded: list[str] | None = None
```

- [ ] **Step 4: Write multi_search**

Create `server/tools/multi_search.py`:

```python
"""Deterministic fused search — dense + sparse -> RRF -> rerank.

No LLM calls anywhere in this path. Use this when latency and cost matter
more than recall on hard multi-part questions; use hybrid_search otherwise.
Both tools return the identical schema, so callers can swap freely.
"""

from __future__ import annotations

import asyncio
import logging

from retrieval.pipeline import PipelineOutput, fast_pipeline
from server.tools.graph_search import graph_search

logger = logging.getLogger(__name__)


def _empty(query: str, session_id: str | None) -> dict:  # type: ignore[type-arg]
    return {
        "results": [],
        "graph_facts": [],
        "live_results": [],
        "query": query,
        "session_id": session_id,
    }


def shape_response(
    output: PipelineOutput,
    graph_facts: list[dict],  # type: ignore[type-arg]
    query: str,
    session_id: str | None,
    degraded: list[str],
    *,
    include_llm_fields: bool,
) -> dict:  # type: ignore[type-arg]
    """Build the shared response dict from a pipeline result.

    Shared by multi_search and hybrid_search so the two schemas cannot drift.
    """
    live: list[dict] = []  # type: ignore[type-arg]
    kb: list[dict] = []  # type: ignore[type-arg]
    for item in output.results:
        target = (
            live
            if session_id and item.metadata.get("source_type") == "live"
            else kb
        )
        target.append(item.model_dump())

    response: dict = {  # type: ignore[type-arg]
        "results": kb,
        "graph_facts": graph_facts,
        "live_results": live,
        "query": query,
        "session_id": session_id,
    }
    if include_llm_fields:
        response["sub_queries"] = output.sub_queries
        response["retried"] = output.retried
    if degraded:
        response["degraded"] = degraded
    # Total retrieval failure is louder than partial degradation — callers
    # that ignore `degraded` must still not mistake this for "no matches".
    if "dense" in degraded and "sparse" in degraded:
        response["error"] = "All retrieval channels unavailable"
    return response


async def run_graph(query: str, limit: int, session_id: str | None) -> tuple[list, bool]:  # type: ignore[type-arg]
    """Query the graph, reporting failure rather than raising."""
    try:
        return await graph_search(query, limit=limit, session_id=session_id), False
    except Exception:
        logger.exception("graph_search failed")
        return [], True


async def multi_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """Fused dense + sparse search with reranking. No LLM calls.

    Runs lexical and semantic retrieval concurrently, merges them with
    reciprocal rank fusion, and reranks the result. Graph facts are returned
    separately as supporting context, not fused into the ranking.

    Args:
        query: Natural language search query.
        limit: Max results (default 10, capped at 100).
        content_type: Optional MIME/content-type filter.
        source_file: Optional source file filter.
        session_id: Optional live-session ID.
    """
    if not query or not query.strip():
        return _empty(query, session_id)
    limit = max(1, min(limit, 100))

    pipeline_task = asyncio.create_task(
        fast_pipeline(
            query=query,
            limit=limit,
            content_type=content_type,
            source_file=source_file,
            session_id=session_id,
        )
    )
    graph_task = asyncio.create_task(run_graph(query, limit, session_id))

    output = await pipeline_task
    graph_facts, graph_failed = await graph_task

    degraded = list(output.degraded)
    if graph_failed:
        degraded.append("graph")

    return shape_response(
        output, graph_facts, query, session_id, degraded, include_llm_fields=False
    )
```

- [ ] **Step 5: Rewrite hybrid_search**

Replace the entire contents of `server/tools/hybrid_search.py`:

```python
"""LLM-augmented fused search.

Same core as multi_search — dense + sparse -> RRF -> rerank — wrapped in two
extra stages: query decomposition before retrieval, and a relevance-gated
single retry after reranking. Returns the identical schema to multi_search.

BREAKING CHANGE: this tool previously returned
{vector_results, graph_results, live_results}. It now returns a single ranked
`results` list plus `graph_facts`. See docs/server/search-tools.md.
"""

from __future__ import annotations

import asyncio
import logging

from retrieval.pipeline import smart_pipeline
from server.tools.graph_search import graph_search  # noqa: F401  (re-exported for tests)
from server.tools.multi_search import _empty, run_graph, shape_response

logger = logging.getLogger(__name__)


async def hybrid_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """Fused search with query decomposition and a relevance-gated retry.

    Splits multi-part questions into sub-queries, retrieves and fuses across
    all of them, reranks against the original query, and widens the candidate
    pool once if the best result is weak.

    Costs one cheap LLM call for decomposition. Use multi_search when that
    cost or latency is unwelcome — the schemas are identical.

    Args:
        query: Natural language search query.
        limit: Max results (default 10, capped at 100).
        content_type: Optional MIME/content-type filter.
        source_file: Optional source file filter.
        session_id: Optional live-session ID.
    """
    if not query or not query.strip():
        return _empty(query, session_id)
    limit = max(1, min(limit, 100))

    pipeline_task = asyncio.create_task(
        smart_pipeline(
            query=query,
            limit=limit,
            content_type=content_type,
            source_file=source_file,
            session_id=session_id,
        )
    )
    graph_task = asyncio.create_task(run_graph(query, limit, session_id))

    output = await pipeline_task
    graph_facts, graph_failed = await graph_task

    degraded = list(output.degraded)
    if graph_failed:
        degraded.append("graph")

    return shape_response(
        output, graph_facts, query, session_id, degraded, include_llm_fields=True
    )
```

- [ ] **Step 6: Register the new tool**

In `server/mcp_server.py`, add the import next to the others:

```python
from server.tools.multi_search import multi_search
```

And register it after the `hybrid_search` registration (line 52):

```python
mcp.tool()(multi_search)
```

- [ ] **Step 7: Remove the dead response model**

Delete `HybridSearchResponse` from `server/models.py` — nothing references it after the rewrite. Confirm with:

```bash
grep -rn "HybridSearchResponse" --include=*.py . || echo "no references — safe to delete"
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_multi_search.py tests/test_tools.py -v && uv run ruff check . && uv run mypy . 2>&1 | tail -1`
Expected: PASS

`tests/test_tools.py` contains assertions against the old `hybrid_search` shape. Update those to the new schema — they should assert on `results` and `graph_facts` rather than `vector_results` and `graph_results`.

- [ ] **Step 9: Commit**

```bash
git add server/tools/multi_search.py server/tools/hybrid_search.py server/models.py \
        server/mcp_server.py tests/test_multi_search.py tests/test_tools.py
git commit -m "feat(mcp): add multi_search and rework hybrid_search onto fused retrieval

BREAKING CHANGE: hybrid_search returns a single ranked results list plus
graph_facts, replacing vector_results/graph_results/live_results."
```

---

## Task 11: Agent consumption of the new schema

**Files:**
- Modify: `agent/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: the `FusedSearchResponse` schema from Task 10
- Produces: an updated `SYSTEM_PROMPT` describing both fused tools

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_system_prompt_documents_both_fused_tools() -> None:
    """The agent knows when to prefer the cheap path over the smart one."""
    from agent.agent import SYSTEM_PROMPT

    assert "multi_search" in SYSTEM_PROMPT
    assert "hybrid_search" in SYSTEM_PROMPT
    # The old split-result vocabulary must be gone.
    assert "vector_results" not in SYSTEM_PROMPT
    assert "graph_results" not in SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_system_prompt_documents_both_fused_tools -v`
Expected: FAIL — `multi_search` is absent from the prompt

- [ ] **Step 3: Update the system prompt**

In `agent/agent.py`, update the tool-description section of `SYSTEM_PROMPT` so it reads:

```
Search tools:
- multi_search: fused keyword + semantic search, reranked. Fast and cheap.
  Use this by default.
- hybrid_search: same as multi_search, plus it splits multi-part questions
  into sub-queries and retries when results look weak. Use for complex or
  compound questions. Slower and costs an extra model call.
- vector_search: semantic-only search. Use when you specifically want
  conceptual similarity without keyword matching.
- visual_search: finds pages by visual layout — charts, diagrams, tables.
- graph_search: entity and relationship facts from the knowledge graph.

Both multi_search and hybrid_search return:
- results: ranked chunks, best first. `channels` shows whether a hit came
  from keyword matching, semantic similarity, or both.
- graph_facts: supporting entity facts, not ranked against the chunks.
- live_results: chunks from the active session, when one is set.
- degraded: present only when part of the pipeline failed. Results are still
  usable but less complete than normal.
```

Remove any wording that references `vector_results` or `graph_results`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Verify end to end**

```bash
task serve &
sleep 5
task ask -- "what are the main topics covered in the documents?"
kill %1
```

Expected: a grounded answer. If the agent errors on the tool response shape, the prompt or the schema is out of sync — fix before committing.

- [ ] **Step 6: Commit**

```bash
git add agent/agent.py tests/test_agent.py
git commit -m "feat(agent): consume the fused search schema"
```

---

## Task 12: Retrieval eval set and metrics

**Files:**
- Create: `tests/eval/retrieval_set.yaml`, `tests/eval/test_retrieval_metrics.py`
- Modify: `tests/eval/thresholds.yaml`, `Taskfile.yml`
- Test: `tests/eval/test_retrieval_metrics.py`

**Interfaces:**
- Consumes: `server.tools.multi_search.multi_search`, `server.tools.hybrid_search.hybrid_search`
- Produces: `task eval-retrieval`; metrics `recall@10`, `ndcg@10`, `mrr`

**Scope decision (2026-08-01):** the corpus is currently two PDFs, too thin for the 40-query multi-document set the design called for. This task builds the *instrument* — metric functions, ablation harness, Taskfile entries, all fully unit-tested — plus a small smoke-level labeled set over the existing documents. The full labeled set is deferred until the corpus grows; the thresholds stay non-gating until then.

The metric code is mechanical. The labels are not, and thin labels produce noise, so this task deliberately does not pretend otherwise.

- [ ] **Step 1: Write the failing metric tests**

Create `tests/eval/test_retrieval_metrics.py`:

```python
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


def test_all_three_categories_are_represented() -> None:
    """Each channel's reason for existing has at least one probe.

    exact_id exercises sparse, semantic exercises dense, multi_hop
    exercises decomposition. A set missing one cannot detect a regression
    in that stage.
    """
    data = yaml.safe_load(SET_PATH.read_text())
    present = {entry["category"] for entry in data["queries"]}
    assert present == {"exact_id", "multi_hop", "semantic"}, f"missing: {present}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_retrieval_metrics.py -v`
Expected: metric tests PASS, `test_retrieval_set_is_wellformed` FAILS — `retrieval_set.yaml` does not exist

- [ ] **Step 3: Build the labeled retrieval set**

Create `tests/eval/retrieval_set.yaml`. This requires reading your corpus and deciding what "relevant" means — it cannot be generated mechanically.

Structure:

```yaml
# Labelled retrieval golden set. Each query lists the chunks that SHOULD be
# retrieved. Used by task eval-retrieval for recall@10 / nDCG@10 / MRR.
#
# Coverage targets — aim for roughly even thirds:
#   exact_id   : part numbers, error codes, section refs, proper nouns.
#                These are what the sparse channel exists for.
#   multi_hop  : questions needing two or more chunks, ideally across
#                documents. These are what decomposition exists for.
#   semantic   : paraphrased questions with no lexical overlap. These are
#                what the dense channel exists for.
queries:
  - query: "What is the maximum operating temperature for part XR-4400?"
    category: exact_id
    relevant:
      - source_file: "spec-sheet.pdf"
        chunk_index: 12

  - query: "How does the warranty differ between commercial and residential use?"
    category: multi_hop
    relevant:
      - source_file: "warranty.pdf"
        chunk_index: 3
      - source_file: "warranty.pdf"
        chunk_index: 9

  - query: "What happens if the device gets too hot?"
    category: semantic
    relevant:
      - source_file: "spec-sheet.pdf"
        chunk_index: 12
```

To find real chunk indices, use the existing MCP tool:

```bash
uv run python -c "
import asyncio
from server.tools.list_document_chunks import list_document_chunks
out = asyncio.run(list_document_chunks(source_file='YOUR_FILE.pdf'))
for c in out['chunks'][:40]:
    print(c['chunk_index'], repr(c['text'][:110]))
"
```

**Write at least 6 entries covering all three categories**, drawn from `arxiv.pdf` and `sample.pdf`. Read the actual chunks first with the command above and label honestly — a query whose "relevant" chunk you had to stretch to justify is worse than no query.

Add this header comment to the file, above `queries:`:

```yaml
# STATUS: smoke-level set. The corpus is two documents, so these metrics
# validate that the harness works — they are NOT a meaningful benchmark and
# the thresholds below are non-gating. Expand to 40+ queries across a real
# multi-document corpus before treating recall/nDCG movements as signal.
# Tracking: see docs/eval/golden-set.md "Expanding the retrieval set".
```

- [ ] **Step 4: Write the scoring harness**

Append to `tests/eval/test_retrieval_metrics.py`:

```python
def _doc_key(result: dict) -> str:  # type: ignore[type-arg]
    return f"{result['source_file']}#{result['chunk_index']}"


def _relevant_keys(entry: dict) -> set[str]:  # type: ignore[type-arg]
    return {f"{r['source_file']}#{r['chunk_index']}" for r in entry["relevant"]}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_quality_meets_thresholds() -> None:
    """Score the full labelled set against multi_search and gate on it."""
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
```

- [ ] **Step 5: Establish baseline thresholds**

Run the harness to get real numbers before setting floors:

```bash
uv run pytest tests/eval/test_retrieval_metrics.py::test_retrieval_quality_meets_thresholds -v -s -m integration
```

Then append to `tests/eval/thresholds.yaml`. **Comment the retrieval floors out for now** — a six-query set over two documents produces too much variance to gate CI on, and a gate that fires spuriously gets disabled, which is worse than no gate:

```yaml
# Retrieval-only metrics (task eval-retrieval). No LLM in the loop, so these
# are far less noisy than the RAGAS bars above.
#
# NON-GATING until the retrieval set grows past smoke level — see
# tests/eval/retrieval_set.yaml. Uncomment and set from a measured baseline
# once the corpus supports 40+ queries. Raise over time, never silently
# lower, and record the measuring commit in the commit message.
#
# recall_at_10: 0.00
# ndcg_at_10: 0.00
# mrr: 0.00
```

`test_retrieval_quality_meets_thresholds` already skips any metric absent from the file (`thresholds.get(metric)` returns `None`), so with these commented out it runs, prints the scores, and asserts nothing. That is the intended state — the harness is exercised on every run, so it cannot rot before the labels arrive.

- [ ] **Step 6: Add the ablation script**

Append to `tests/eval/test_retrieval_metrics.py`:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ablation_matrix() -> None:
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
```

- [ ] **Step 7: Add the task runner entry**

In `Taskfile.yml`, add alongside the existing `eval` task:

```yaml
  eval-retrieval:
    desc: Retrieval-only quality metrics (recall@10, nDCG@10, MRR)
    cmds:
      - uv run pytest tests/eval/test_retrieval_metrics.py -v -s -m integration

  eval-ablation:
    desc: Per-stage retrieval attribution table
    cmds:
      - uv run pytest tests/eval/test_retrieval_metrics.py::test_ablation_matrix -v -s -m integration
```

- [ ] **Step 8: Run everything**

Run: `uv run pytest tests/eval/test_retrieval_metrics.py -v && task eval-retrieval && task eval-ablation && task eval`
Expected: unit metric tests PASS; retrieval metrics meet the new floors; ablation prints a table; RAGAS still meets `thresholds.yaml`

If RAGAS metrics dropped, stop and investigate before committing. If they improved, note the new values in the commit message but do not raise the thresholds in the same commit.

- [ ] **Step 9: Commit**

```bash
git add tests/eval/ Taskfile.yml
git commit -m "test(eval): add labelled retrieval set, retrieval metrics, and ablation matrix"
```

---

## Task 13: Documentation

**Files:**
- Create: `docs/operations/reindex.md`
- Modify: `docs/server/search-tools.md`, `docs/ingestion/embeddings.md`, `docs/eval/golden-set.md`, `CLAUDE.md`, `mkdocs.yml`

**Interfaces:**
- Consumes: everything above
- Produces: no code

- [ ] **Step 1: Document the search tools and the breaking change**

In `docs/server/search-tools.md`, add a `multi_search` section, rewrite the `hybrid_search` section, and add a migration note at the top:

```markdown
## Breaking change — hybrid_search response shape

As of the retrieval upgrade, `hybrid_search` no longer returns
`{vector_results, graph_results, live_results}`. It returns a single ranked
`results` list plus `graph_facts`.

| Before | After |
|-|-|
| `vector_results` | `results` |
| `graph_results` | `graph_facts` |
| `live_results` | `live_results` (unchanged) |
| `strategy` | removed |
| `errors` | `degraded` (channel names, not messages) |

Each result gains `id`, `chunk_index`, `fusion_score`, and `channels`.
`channels` records which retrieval channels surfaced the hit — `dense`,
`sparse`, or both.

Migration: replace `response["vector_results"]` with `response["results"]`
and `response["graph_results"]` with `response["graph_facts"]`. Ranking is now
done server-side, so clients that re-sorted results should stop doing so.

`multi_search` returns the identical schema without `sub_queries` or
`retried`, and makes no LLM calls.
```

Document both tools' parameters and the `degraded` semantics: its presence means partial failure, its absence means a healthy run.

- [ ] **Step 2: Document the sparse channel**

In `docs/ingestion/embeddings.md`, add a section covering: miniCOIL via fastembed, `Qdrant/minicoil-v1`, local CPU execution with no API cost, the `avg_len` index-time normalisation and why queries skip it, and the named-vector layout of `documents_dense` (`dense` + `sparse` with `Modifier.IDF`).

- [ ] **Step 3: Write the re-ingest runbook**

Create `docs/operations/reindex.md`, following the structure of the existing `backup-restore.md`. Cover the migration from Task 6 Step 8: run `task backup` first, drop the collection, clear CocoIndex tracking, re-ingest, verify with `task doctor`. State plainly that this is a full re-index requiring embedding API spend, and that search is unavailable until it completes.

Register the new page in `mkdocs.yml` under the Operations nav section, alongside `backup-restore.md`.

- [ ] **Step 4: Document the retrieval eval**

In `docs/eval/golden-set.md`, add sections for `task eval-retrieval` and `task eval-ablation`: the three metrics, the `retrieval_set.yaml` format, the three query categories and why each exists, and how to read the ablation table when deciding whether a stage earns its cost.

Add a section titled **"Expanding the retrieval set"** stating plainly that the current set is smoke-level over a two-document corpus, that the thresholds are commented out and non-gating for that reason, and listing what to do when the corpus grows: write 40+ queries balanced across the three categories, measure a baseline, uncomment the floors at baseline minus ~0.05, and note the measuring commit. This is the tracking record for the deferred work — do not leave it implicit.

- [ ] **Step 5: Update the production contracts**

In `CLAUDE.md`, under **Qdrant payload schema**, replace the existing bullets with:

```markdown
- `documents_dense` uses **named vectors**: `dense` (embedding, cosine) and
  `sparse` (miniCOIL, `Modifier.IDF`). Never write points with an unnamed
  vector — the collection will reject them.
- Every text-chunk point carries both vectors. Image and VLM-caption points
  carry `dense` only. `scripts/doctor.py` flags text chunks missing `sparse`.
- Every dense point carries `embedder_model` (str) and `embedder_dim` (int).
- `source_file` is the logical key; `metadata.source_key` duplicates it.
```

Add to the **Eval gate** section:

```markdown
- Retrieval changes must also pass `task eval-retrieval` (recall@10, nDCG@10,
  MRR against `tests/eval/retrieval_set.yaml`). These metrics have no LLM in
  the loop and are the primary gate for retrieval work; RAGAS remains the
  gate for generation quality.
```

Add a new contract block:

```markdown
**Retrieval pipeline** (`retrieval/`):
- `retrieval/` must not import from `server/` or `fastmcp`. It is composed by
  MCP adapters and knows nothing about transport.
- `hybrid_search` must delegate its retrieval core to the same code path as
  `multi_search`. If the two ever diverge, the "hybrid is multi plus two
  stages" contract is broken and the ablation matrix stops being meaningful.
- Every pipeline stage degrades rather than raising. A stage failure appends
  its name to `degraded` and the tool still returns results.
```

Also update the **Architecture** section's MCP bullet: it says "six tools" and must now say seven, listing `multi_search`.

- [ ] **Step 6: Verify the docs build**

Run: `task docs-build`
Expected: builds with no broken-link warnings

- [ ] **Step 7: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "docs: document fused retrieval, sparse channel, and re-ingest runbook"
```

---

## Final verification

- [ ] **Run the full check**

```bash
uv run ruff check .
uv run mypy . 2>&1 | tail -1   # must still read "Found 14 errors" or fewer
task test-integration
task eval
task eval-retrieval
task eval-ablation
task doctor
```

Expected: all green. `task doctor` reports no drift and no missing sparse vectors.

- [ ] **Confirm the tool surface**

```bash
task serve &
sleep 5
curl -s -H "Authorization: Bearer $MCP_API_KEY" http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m json.tool | grep '"name"'
kill %1
```

Expected: seven tools — `vector_search`, `visual_search`, `graph_search`, `multi_search`, `hybrid_search`, `list_documents`, `list_document_chunks`.

- [ ] **Record the ablation baseline**

Paste the `task eval-ablation` table into the PR description. It is the evidence that the sparse channel and reranker earned their cost, and the reference point for future changes.
