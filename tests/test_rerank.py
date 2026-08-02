"""Tests for the Jina reranker wrapper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from retrieval.models import FusedResult
from retrieval.rerank import RERANK_URL, RerankError, rerank, rerank_dicts
from retrieval.rerank import settings as rerank_settings


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
@respx.mock
async def test_rerank_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model string in the request body comes from settings, not a hardcoded constant.

    Patches settings.rerank_model to a sentinel unrelated to the real default, then
    inspects the actual HTTP request body the Jina client sent — proving the
    settings -> request wiring, not just that some function got called.
    """
    monkeypatch.setattr(rerank_settings, "rerank_model", "sentinel-model-xyz")
    route = respx.post(RERANK_URL).mock(
        return_value=httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
        )
    )

    await rerank("q", [_result("a", "x")], top_k=1)

    assert route.called
    sent_payload = json.loads(route.calls.last.request.content)
    assert sent_payload["model"] == "sentinel-model-xyz"


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
