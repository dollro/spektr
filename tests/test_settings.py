from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestMultivecSlot:
    """The late-interaction slot is independent of the dense model.

    These replace two tests that asserted the opposite — that
    `MULTIVEC_ENABLED=true` was legal with `jina-v4` and illegal with
    `voyage-4`. That coupling was the defect: the two share no vector space
    (`documents_multivec` has its own collection, dimensionality and MaxSim
    comparator), so gating one on the other meant changing the text embedder
    silently removed the `visual_search` tool.
    """

    def test_retired_flag_raises_with_migration_path(self) -> None:
        """A stale MULTIVEC_ENABLED must fail loudly, not be ignored.

        `extra="ignore"` would drop it and boot with the slot unfilled —
        visual_search dark on a deployment that believed it was on.
        """
        from config.settings import Settings

        with pytest.raises(ValidationError, match="MULTIVEC_ENABLED has been replaced"):
            Settings(
                jina_api_key="test",
                neo4j_password="test",
                multivec_enabled=True,
                rerank_enabled=False,
                _env_file=None,
            )

    def test_slot_is_independent_of_the_dense_model(self) -> None:
        """The combination the old validator rejected outright.

        A gemini-2 dense slot with a jina-v4-colbert multivector slot is
        exactly what the coupling made inexpressible, and there is nothing
        wrong with it: different models, different collections.
        """
        from config.settings import Settings

        s = Settings(
            jina_api_key="test",
            neo4j_password="test",
            openrouter_api_key="test",
            embedding_model="gemini-2",
            embedding_route="openrouter",
            multivec_model="jina-v4-colbert",
            rerank_enabled=False,
            _env_file=None,
        )
        assert s.multivec_slot_filled is True
        assert s.multivec_route == "native", "unset route adopts the model's default"
        assert s.multivec_dimensions == 128

    def test_empty_model_means_slot_unfilled(self) -> None:
        from config.settings import Settings

        s = Settings(
            jina_api_key="test",
            neo4j_password="test",
            rerank_enabled=False,
            _env_file=None,
        )
        assert s.multivec_slot_filled is False

    def test_illegal_route_for_model_is_rejected(self) -> None:
        from config.settings import Settings

        with pytest.raises(ValidationError, match="is not served via"):
            Settings(
                jina_api_key="test",
                neo4j_password="test",
                multivec_model="colqwen2.5",
                multivec_route="native",
                rerank_enabled=False,
                _env_file=None,
            )

    def test_unknown_model_fails_loudly(self) -> None:
        from config.settings import Settings

        with pytest.raises(ValidationError, match="Unknown multivector model"):
            Settings(
                jina_api_key="test",
                neo4j_password="test",
                multivec_model="not-a-model",
                rerank_enabled=False,
                _env_file=None,
            )


class TestSettingsValidation:
    def test_live_ingest_port_default(self) -> None:
        """LIVE_INGEST_PORT defaults to 8001."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="k",
            neo4j_password="p",
            _env_file=None,
        )
        assert s.live_ingest_port == 8001

    def test_schema_induction_enabled_default(self) -> None:
        """SCHEMA_INDUCTION_ENABLED defaults to True."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="k",
            neo4j_password="p",
            _env_file=None,
        )
        assert s.schema_induction_enabled is True

    def test_schema_induction_model_default(self) -> None:
        """SCHEMA_INDUCTION_MODEL defaults to empty, falling back to LLM_MODEL."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="k",
            neo4j_password="p",
            _env_file=None,
        )
        assert s.schema_induction_model == ""

    def test_schema_cache_ttl_default(self) -> None:
        """SCHEMA_CACHE_TTL defaults to 3600."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="k",
            neo4j_password="p",
            _env_file=None,
        )
        assert s.schema_cache_ttl == 3600

    def test_voyage_without_multivec_is_valid(self) -> None:
        """Voyage without multivec is fine."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="test",
            neo4j_password="test",
            embedding_model="voyage-4",
            embedding_route="native",
            voyage_api_key="test",
            rerank_enabled=False,
            # conftest's load_dotenv() leaks the developer's
            # EMBEDDING_DIMENSIONS into os.environ, and 768 is not one of
            # voyage-4's four sizes. Pin it so this asserts the code, not .env.
            embedding_dimensions=0,
            image_embed_strategy="none",
            _env_file=None,
        )
        assert s.embedding_model == "voyage-4"
        assert s.dense_dimensions == 1024  # model default


def test_retrieval_defaults(monkeypatch):  # type: ignore[no-untyped-def]
    """New retrieval settings expose the documented defaults.

    _env_file=None isolates from the developer's .env — Settings() otherwise
    reads it and this would assert the local config, not the defaults.
    monkeypatch removes LLM_MODEL from os.environ set by conftest.load_dotenv()
    to verify the code-level default, not constructor override.
    """
    from config.settings import Settings

    monkeypatch.delenv("LLM_MODEL", raising=False)
    # Same leak, same fix: the developer's .env pins RERANK_SCORE_FLOOR, and
    # os.environ beats _env_file=None.
    monkeypatch.delenv("RERANK_SCORE_FLOOR", raising=False)
    s = Settings(
        neo4j_password="test",
        _env_file=None,
    )  # type: ignore[call-arg]
    assert s.sparse_enabled is True
    assert s.sparse_model == "Qdrant/minicoil-v1"
    assert s.rrf_k == 60
    assert s.rerank_model == "jina-reranker-v3.5"
    assert s.rerank_candidates == 50
    # Unset, because the floor is only meaningful paired with the active
    # model's score_scale; `resolved_score_floor` is what callers read.
    assert s.rerank_score_floor is None
    assert s.resolved_score_floor == 0.0  # jina-reranker-v3.5's logit default
    assert s.retry_enabled is True
    assert s.retry_limit_multiplier == 3
    assert s.decompose_enabled is True
    assert s.decompose_model == ""
    assert s.decompose_max_subqueries == 4
    # Haiku by default: this model drives Graphiti entity extraction, which is
    # one LLM call per episode and dominates the cost of a full ingest.
    assert s.llm_model == "claude-haiku-4.5"


def test_vector_name_constants() -> None:
    """Named-vector constants exist for the migrated collection."""
    from config.constants import DENSE_VECTOR_NAME, MINICOIL_AVG_LEN, SPARSE_VECTOR_NAME

    assert DENSE_VECTOR_NAME == "dense"
    assert SPARSE_VECTOR_NAME == "sparse"
    assert MINICOIL_AVG_LEN == 80


class TestEmbeddingSelection:
    """The model/route split: capabilities follow the model, not the route."""

    def _settings(self, **kw: object):  # type: ignore[no-untyped-def]
        from config.settings import Settings

        base = dict(
            neo4j_password="test",
            openrouter_api_key="test",
            image_embed_strategy="none",
            embedding_dimensions=0,
            _env_file=None,
        )
        base.update(kw)
        return Settings(**base)  # type: ignore[arg-type]

    def test_dimensions_default_to_the_model(self) -> None:
        """Unset means the model's RECOMMENDED size, not its full size."""
        s = self._settings(embedding_model="gemini-2", embedding_route="openrouter")
        assert s.dense_dimensions == 768  # not 3072

    def test_switching_model_alone_resolves_route_and_dimensions(self) -> None:
        """The one-line switch: change the model, touch nothing else."""
        expected = {
            "gemini-2": ("openrouter", 768),
            "voyage-4": ("native", 1024),
            "jina-v4": ("native", 512),
        }
        for model, (route, dims) in expected.items():
            s = self._settings(
                embedding_model=model,
                embedding_route="",
                embedding_dimensions=0,
                voyage_api_key="k",
                jina_api_key="k",
            )
            assert (s.embedding_route, s.dense_dimensions) == (route, dims), model

    def test_explicit_route_overrides_the_default(self) -> None:
        """Pinning still wins, for the case where the registry choice is wrong."""
        s = self._settings(embedding_model="voyage-4", embedding_route="openrouter")
        assert s.embedding_route == "openrouter"
        assert s.embedding_model_id == "voyageai/voyage-4-large"

    def test_explicit_dimensions_win(self) -> None:
        s = self._settings(
            embedding_model="gemini-2", embedding_route="openrouter", embedding_dimensions=768
        )
        assert s.dense_dimensions == 768

    def test_dimensions_validated_against_the_model(self) -> None:
        """A value tuned for one model must not silently carry to another.

        768 is fine for gemini-2 (Matryoshka, 1..3072) but voyage-4 emits
        only 256/512/1024/2048 — the exact trap a single global knob would
        otherwise spring on an EMBEDDING_MODEL switch.
        """
        assert (
            self._settings(
                embedding_model="gemini-2",
                embedding_route="openrouter",
                embedding_dimensions=768,
            ).dense_dimensions
            == 768
        )

        with pytest.raises(ValidationError, match="not supported by voyage-4"):
            self._settings(
                embedding_model="voyage-4",
                embedding_route="native",
                voyage_api_key="k",
                embedding_dimensions=768,
            )

    def test_dimensions_above_model_maximum_rejected(self) -> None:
        """Matryoshka truncates down, never up."""
        with pytest.raises(ValidationError, match="out of range"):
            self._settings(
                embedding_model="gemini-2",
                embedding_route="openrouter",
                embedding_dimensions=4096,
            )

    def test_illegal_pair_is_rejected(self) -> None:
        """jina-v4 is not served by OpenRouter — the licence forbids it."""
        with pytest.raises(ValidationError, match="not served via"):
            self._settings(embedding_model="jina-v4", embedding_route="openrouter")

    def test_gemini_has_no_native_route(self) -> None:
        with pytest.raises(ValidationError, match="not served via"):
            self._settings(embedding_model="gemini-2", embedding_route="native")

    def test_model_id_resolves_per_route(self) -> None:
        openrouter = self._settings(embedding_model="voyage-4", embedding_route="openrouter")
        native = self._settings(
            embedding_model="voyage-4", embedding_route="native", voyage_api_key="k"
        )
        assert openrouter.embedding_model_id == "voyageai/voyage-4-large"
        assert native.embedding_model_id == "voyage-4-large"

    def test_image_strategy_rejected_when_unimplemented(self) -> None:
        """The old silent-drop failure mode is now a startup error.

        voyage-4 can embed images, but only on its native route — the
        OpenRouter client does not send them. Capability is per model AND
        route, which is exactly what the registry encodes.
        """
        with pytest.raises(ValidationError, match="IMAGE_EMBED_STRATEGY"):
            self._settings(
                embedding_model="voyage-4",
                embedding_route="openrouter",
                image_embed_strategy="smart",
            )

    def test_image_strategy_allowed_for_gemini_on_openrouter(self) -> None:
        """gemini-2 is natively multimodal; images work on this route."""
        s = self._settings(
            embedding_model="gemini-2",
            embedding_route="openrouter",
            image_embed_strategy="smart",
        )
        assert s.supports_image_embedding is True
        # Natively multimodal: one id serves both modalities.
        assert s.embedding_image_model_id == s.embedding_model_id

    def test_image_strategy_allowed_on_native_route(self) -> None:
        s = self._settings(
            embedding_model="voyage-4",
            embedding_route="native",
            voyage_api_key="k",
            image_embed_strategy="smart",
        )
        assert s.supports_image_embedding is True

    def test_retired_provider_setting_fails_loudly(self) -> None:
        """extra=ignore would otherwise boot prod on the wrong model."""
        with pytest.raises(ValidationError, match="EMBEDDING_PROVIDER has been replaced"):
            self._settings(embedding_provider="openrouter")


class TestGraphEngineLlmSelection:
    """Graphiti only speaks OpenAI; the config must not promise otherwise."""

    def _settings(self, **kw: object):  # type: ignore[no-untyped-def]
        from config.settings import Settings

        # Every LLM/graph field is pinned: conftest's load_dotenv() leaks the
        # developer's .env into os.environ, which _env_file=None does not
        # shadow, and these four fields are exactly what is under test.
        base = dict(
            neo4j_password="test",
            openrouter_api_key="test",
            image_embed_strategy="none",
            embedding_dimensions=0,
            graph_enabled=True,
            graph_engine="graphiti",
            llm_api_type="anthropic",
            llm_model="claude-haiku-4.5",
            llm_base_url="",
            _env_file=None,
        )
        base.update(kw)
        return Settings(**base)  # type: ignore[arg-type]

    def test_graphiti_requires_openai_compatible_llm(self) -> None:
        """The shipped default combination is a silent cross-provider call.

        graphiti_client builds OpenAIGenericClient unconditionally and an
        empty base_url resolves to api.openai.com inside the OpenAI SDK, so
        this would post a Claude model id there with an Anthropic key.
        """
        with pytest.raises(ValidationError) as exc:
            self._settings()
        message = str(exc.value)
        assert "GRAPH_ENGINE=graphiti" in message
        assert "LLM_API_TYPE=anthropic" in message
        # Both escape hatches must be named, or the error is not actionable.
        assert "LLM_BASE_URL" in message
        assert "GRAPH_ENGINE=gliner" in message

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"graph_engine": "gliner"}, "local CPU extraction needs no LLM"),
            ({"graph_enabled": False}, "no graph, no Graphiti LLM calls"),
            ({"llm_api_type": "openai"}, "the OpenAI client is the right client"),
            (
                {"llm_base_url": "https://openrouter.ai/api/v1"},
                "an OpenAI-compatible gateway can serve a Claude model id",
            ),
        ],
    )
    def test_valid_combinations_still_pass(
        self, overrides: dict[str, object], reason: str
    ) -> None:
        s = self._settings(**overrides)
        assert s.neo4j_password == "test", reason

    def test_api_type_match_is_case_insensitive(self) -> None:
        """get_llm_client() lowercases llm_api_type; this must agree with it."""
        s = self._settings(llm_api_type="OpenAI")
        assert s.llm_api_type == "OpenAI"
