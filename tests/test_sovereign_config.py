"""The configuration the capability-slot work exists to make expressible.

Before the slots, `EMBEDDING_MODEL` selected one model that had to supply
text embedding, image embedding and ColBERT together, and the reranker was
hard-wired to Jina. No self-hostable model supplies all three, so a fully
self-hosted deployment was unreachable *by construction* — a shape problem,
not a missing-provider problem.

These tests pin the resulting configuration. They are cheap and they are the
only thing standing between a future refactor and quietly re-coupling the
slots.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings

SOVEREIGN = dict(
    _env_file=None,
    neo4j_password="x",
    # dense: Apache-2.0 weights behind your own TEI/vLLM host
    embedding_model="qwen3-embedding-8b",
    embedding_route="local",
    local_embed_api_url="http://localhost:8080",
    embedding_dimensions=1024,
    # a text-only dense model cannot carry images; the validator enforces it
    image_embed_strategy="none",
    # late interaction: ColQwen via colpali-engine
    multivec_model="colqwen2.5",
    # rerank: Qwen3-Reranker behind TEI
    rerank_model="qwen3-reranker",
    rerank_route="local",
    rerank_api_url="http://localhost:8081",
    # Unset so the model's own floor applies. Pinned explicitly because
    # conftest's load_dotenv() leaks the developer's RERANK_SCORE_FLOOR into
    # os.environ, where it beats _env_file=None.
    rerank_score_floor=None,
    # graph: local CPU extraction, no LLM call
    graph_engine="gliner",
)


def _sovereign(**overrides: object) -> Settings:
    return Settings(**{**SOVEREIGN, **overrides})  # type: ignore[arg-type]


class TestFullySelfHostedStack:
    def test_it_validates(self) -> None:
        """The whole point: this combination must be legal."""
        s = _sovereign()
        assert s.embedding_route == "local"
        assert s.multivec_route == "local"
        assert s.rerank_route == "local"

    def test_no_vendor_host_in_the_retrieval_path(self) -> None:
        """Every endpoint resolves to the operator's own host.

        The rerank slot deliberately does NOT fall back to api.jina.ai on the
        local route — silently pointing a self-hosted deployment at a vendor
        would be worse than failing.
        """
        s = _sovereign()
        base_url, _ = s.rerank_endpoint
        assert base_url == "http://localhost:8081"
        assert "jina.ai" not in base_url
        assert s.local_embed_api_url.startswith("http://localhost")

    def test_local_route_resolves_a_checkpoint_not_a_vendor_id(self) -> None:
        s = _sovereign()
        assert s.embedding_model_id == "Qwen/Qwen3-Embedding-8B"

    def test_score_floor_follows_the_self_hosted_reranker(self) -> None:
        """Carrying jina's logit floor here would disable the gate outright."""
        s = _sovereign()
        assert s.resolved_score_floor == 0.5, "unit scale, not the 0.0 logit default"

    @pytest.mark.parametrize("stale_floor", [0.0, -1.0, 1.0, 4.5])
    def test_a_floor_on_the_wrong_scale_is_rejected(self, stale_floor: float) -> None:
        """The likelier version of the score-scale mistake.

        The registry default protects the *unset* case. An operator who
        pinned 0.0 while running jina-reranker-v3.5 and then switched to a
        unit-scaled model keeps that value, and a [0,1] score can never fall
        below it — the gate goes dark just as silently.
        """
        with pytest.raises(ValidationError, match="unreachable for"):
            _sovereign(rerank_score_floor=stale_floor)

    def test_a_valid_unit_floor_is_accepted(self) -> None:
        assert _sovereign(rerank_score_floor=0.3).resolved_score_floor == 0.3

    def test_logit_scale_accepts_any_floor(self) -> None:
        """Unbounded scores make no value implausible — no guard applies."""
        s = _sovereign(
            rerank_model="jina-reranker-v3.5",
            rerank_route="native",
            rerank_api_key="k",
            rerank_api_url="",
            rerank_score_floor=-2.0,
        )
        assert s.resolved_score_floor == -2.0

    def test_dense_slot_is_text_only_and_says_so(self) -> None:
        s = _sovereign()
        assert s.supports_image_embedding is False

    def test_image_strategy_must_be_none_for_a_text_only_dense_model(self) -> None:
        """The cost of sovereignty, enforced at startup rather than discovered.

        Cross-modal retrieval from one shared vector space is a gemini-2
        privilege. With a text-only dense model, images reach search through
        visual_search/multivec instead — and asking for `smart` here fails
        loudly rather than silently dropping every page.
        """
        with pytest.raises(ValidationError, match="IMAGE_EMBED_STRATEGY"):
            _sovereign(image_embed_strategy="smart")


class TestSlotsAreIndependent:
    """Each slot can be self-hosted or vendor-hosted without the others."""

    def test_vendor_dense_with_self_hosted_multivec(self) -> None:
        """Sovereign visual retrieval on top of a managed text embedder."""
        s = Settings(  # type: ignore[call-arg]
            _env_file=None,
            neo4j_password="x",
            openrouter_api_key="k",
            embedding_model="gemini-2",
            embedding_route="openrouter",
            multivec_model="colqwen2.5",
            rerank_enabled=False,
            graph_engine="gliner",
        )
        assert s.multivec_route == "local"
        assert s.embedding_route == "openrouter"

    def test_self_hosted_dense_with_vendor_rerank(self) -> None:
        """The reverse split — no slot constrains another."""
        s = _sovereign(
            rerank_model="jina-reranker-v3.5",
            rerank_route="native",
            rerank_api_key="k",
            rerank_api_url="",
        )
        assert s.embedding_route == "local"
        assert s.rerank_route == "native"
        assert s.resolved_score_floor == 0.0, "back to the logit floor"

    def test_multivec_does_not_constrain_the_dense_model(self) -> None:
        """The exact combination the retired MULTIVEC_ENABLED flag rejected."""
        s = Settings(  # type: ignore[call-arg]
            _env_file=None,
            neo4j_password="x",
            openrouter_api_key="k",
            embedding_model="gemini-2",
            embedding_route="openrouter",
            multivec_model="jina-v4-colbert",
            rerank_enabled=False,
            graph_engine="gliner",
        )
        assert s.multivec_slot_filled is True
