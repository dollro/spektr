from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from config.rerank_models import REGISTRY, RerankSpec, spec
from config.settings import Settings


def _shipped_defaults() -> Settings:
    """Settings as a fresh deployment sees them, with no `.env` in play."""
    return Settings(
        jina_api_key="test",
        neo4j_password="test",
        _env_file=None,  # type: ignore[call-arg]
    )


class TestRegistryLookup:
    def test_spec_returns_the_registered_entry(self) -> None:
        """A known name resolves to its spec object."""
        assert spec("jina-reranker-v3.5") is REGISTRY["jina-reranker-v3.5"]

    def test_unknown_model_raises_value_error(self) -> None:
        """An unknown name fails loudly and names the valid options."""
        with pytest.raises(ValueError) as exc:
            spec("jina-reranker-v2")

        message = str(exc.value)
        assert "Unknown rerank model 'jina-reranker-v2'" in message
        for known in REGISTRY:
            assert known in message

    def test_unknown_model_does_not_chain_the_key_error(self) -> None:
        """`from None` keeps the traceback about the config, not the dict."""
        with pytest.raises(ValueError) as exc:
            spec("nope")

        assert exc.value.__cause__ is None
        assert exc.value.__suppress_context__ is True


class TestRegistryInvariants:
    @pytest.mark.parametrize("model", sorted(REGISTRY))
    def test_default_route_is_a_member_of_routes(self, model: str) -> None:
        """RERANK_ROUTE unset must resolve to a route the model actually has."""
        entry = REGISTRY[model]
        assert entry.default_route in entry.routes

    @pytest.mark.parametrize("model", sorted(REGISTRY))
    def test_routes_are_non_empty(self, model: str) -> None:
        """A model with no route cannot be reached and must not be advertised."""
        assert REGISTRY[model].routes

    @pytest.mark.parametrize("model", sorted(REGISTRY))
    def test_unit_scaled_floor_is_within_the_unit_interval(self, model: str) -> None:
        """A [0,1] reranker with a floor outside [0,1] is a dead or always-on gate."""
        entry = REGISTRY[model]
        if entry.score_scale == "unit":
            assert 0.0 <= entry.default_score_floor <= 1.0

    def test_specs_are_frozen(self) -> None:
        """Registry entries are shared config; mutating one must not be possible."""
        with pytest.raises(FrozenInstanceError):
            REGISTRY["jina-reranker-v3.5"].default_score_floor = 1.0  # type: ignore[misc]


class TestJinaEntryMatchesCurrentSettings:
    """The jina entry exists to reproduce today's behaviour exactly.

    These assertions are the regression guard for Phase 1: if the registry
    and the shipped `Settings` defaults ever disagree, wiring the registry
    into `Settings` changes retrieval behaviour silently.
    """

    def test_default_rerank_model_is_in_the_registry(self) -> None:
        assert _shipped_defaults().rerank_model in REGISTRY

    def test_default_score_floor_matches_the_spec(self) -> None:
        defaults = _shipped_defaults()
        entry = spec(defaults.rerank_model)
        assert entry.default_score_floor == defaults.rerank_score_floor

    def test_jina_entry_values(self) -> None:
        """Pin the whole spec, so a later edit to any field is deliberate."""
        assert REGISTRY["jina-reranker-v3.5"] == RerankSpec(
            protocol="jina",
            routes=("native",),
            default_route="native",
            wire_id="jina-reranker-v3.5",
            score_scale="logit",
            default_score_floor=0.0,
        )

    def test_wire_id_matches_the_model_string_sent_today(self) -> None:
        """`retrieval/rerank.py` sends `settings.rerank_model` as the model id."""
        defaults = _shipped_defaults()
        assert spec(defaults.rerank_model).wire_id == defaults.rerank_model


class TestSelfHostedEntry:
    def test_qwen3_is_tei_on_the_local_route(self) -> None:
        entry = spec("qwen3-reranker")
        assert entry.protocol == "tei"
        assert entry.routes == ("local",)
        assert entry.default_route == "local"

    def test_qwen3_scale_differs_from_jina(self) -> None:
        """The whole point of `score_scale`: the two entries disagree.

        A floor tuned on the logit-scaled model can never fire on the
        unit-scaled one, which is the silent-gate-disable failure this
        registry exists to make impossible to hit by accident.
        """
        assert spec("qwen3-reranker").score_scale == "unit"
        assert spec("jina-reranker-v3.5").score_scale == "logit"
        jina_floor = spec("jina-reranker-v3.5").default_score_floor
        assert spec("qwen3-reranker").default_score_floor != jina_floor
