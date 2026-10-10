"""Tests for the pluggable reranker facade.

The wire-level behaviour of each protocol lives in `tests/test_rerankers.py`;
this file covers the facade — factory dispatch, the FusedResult/dict contracts,
and the degradation split between `rerank()` and `rerank_dicts()`.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from config import rerank_models
from retrieval.models import FusedResult
from retrieval.rerank import (
    RerankError,
    create_reranker,
    rerank,
    rerank_dicts,
    reset_reranker,
)
from retrieval.rerank import settings as rerank_settings
from retrieval.rerankers.jina import JinaReranker
from retrieval.rerankers.tei import TEIReranker


@pytest.fixture(autouse=True)
def _clear_reranker_cache() -> None:
    """The reranker is a module-level singleton; a test that repoints
    settings must not inherit an instance built from the previous config."""
    reset_reranker()


def _result(doc_id: str, text: str) -> FusedResult:
    return FusedResult(
        id=doc_id, text=text, source_file="doc.pdf", score=0.01, fusion_score=0.01
    )


@pytest.mark.asyncio
async def test_rerank_applies_api_ordering() -> None:
    """Results are reordered and rescored from the reranker's response."""
    results = [_result("a", "first"), _result("b", "second")]
    with patch("retrieval.rerank._score", AsyncMock(return_value=[(1, 0.9), (0, 0.2)])):
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
    with patch(
        "retrieval.rerank._get_reranker",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RerankError):
            await rerank("q", results, top_k=1)


@pytest.mark.asyncio
async def test_rerank_empty_input_returns_empty() -> None:
    """No results in, no results out, no API call."""
    assert await rerank("q", [], top_k=5) == []


@pytest.mark.asyncio
@respx.mock
async def test_registry_wire_id_reaches_the_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model on the wire comes from the registry's `wire_id`.

    Registers a sentinel spec whose key and wire_id differ, so the assertion
    cannot pass by accident if the code sent `RERANK_MODEL` itself. That
    distinction is load-bearing: for TEI they are deliberately different
    things (a checkpoint to launch, not a request field).
    """
    monkeypatch.setitem(
        rerank_models.REGISTRY,
        "sentinel-model-xyz",
        rerank_models.RerankSpec(
            protocol="jina",
            routes=("native",),
            default_route="native",
            wire_id="wire-id-abc",
            score_scale="logit",
            default_score_floor=0.0,
        ),
    )
    monkeypatch.setattr(rerank_settings, "rerank_model", "sentinel-model-xyz")
    monkeypatch.setattr(rerank_settings, "rerank_route", "native")

    base_url, _ = rerank_settings.rerank_endpoint
    route = respx.post(f"{base_url}/v1/rerank").mock(
        return_value=httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
        )
    )

    await rerank("q", [_result("a", "x")], top_k=1)

    assert route.called
    sent_payload = json.loads(route.calls.last.request.content)
    assert sent_payload["model"] == "wire-id-abc"


@pytest.mark.asyncio
async def test_rerank_dicts_preserves_dict_contract() -> None:
    """The legacy dict path keeps original_score, for vector_search."""
    results = [{"text": "first", "score": 0.4}, {"text": "second", "score": 0.3}]
    with patch("retrieval.rerank._score", AsyncMock(return_value=[(1, 0.95)])):
        out = await rerank_dicts("q", results, top_k=1)

    assert out[0]["text"] == "second"
    assert out[0]["score"] == 0.95
    assert out[0]["original_score"] == 0.3


@pytest.mark.asyncio
async def test_rerank_dicts_swallows_failure() -> None:
    """vector_search has no `degraded` channel, so this path degrades in place."""
    results = [{"text": "first", "score": 0.4}, {"text": "second", "score": 0.3}]
    with patch("retrieval.rerank._get_reranker", side_effect=RuntimeError("boom")):
        out = await rerank_dicts("q", results, top_k=2)
    assert [r["text"] for r in out] == ["first", "second"]
    assert "original_score" not in out[0], "input order returned untouched"


class TestFactoryDispatch:
    def test_jina_protocol_builds_jina_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rerank_settings, "rerank_model", "jina-reranker-v3.5")
        monkeypatch.setattr(rerank_settings, "rerank_route", "native")
        assert isinstance(create_reranker(), JinaReranker)

    def test_tei_protocol_builds_tei_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rerank_settings, "rerank_model", "qwen3-reranker")
        monkeypatch.setattr(rerank_settings, "rerank_route", "local")
        monkeypatch.setattr(rerank_settings, "rerank_api_url", "http://localhost:8080")
        assert isinstance(create_reranker(), TEIReranker)

    def test_unknown_model_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rerank_settings, "rerank_model", "not-a-reranker")
        with pytest.raises(ValueError, match="Unknown rerank model"):
            create_reranker()

    def test_local_route_does_not_fall_back_to_jina(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A self-hosted slot pointed at api.jina.ai would be worse than failing."""
        monkeypatch.setattr(rerank_settings, "rerank_route", "local")
        monkeypatch.setattr(rerank_settings, "rerank_api_url", "")
        monkeypatch.setattr(rerank_settings, "rerank_api_key", "")
        assert rerank_settings.rerank_endpoint == ("", "")
