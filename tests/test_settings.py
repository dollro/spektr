from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestSettingsValidation:
    def test_voyage_with_multivec_raises(self) -> None:
        """Voyage + multivec_enabled=True is an invalid combination."""
        from config.settings import Settings

        with pytest.raises(ValidationError, match="[Vv]oyage.*ColBERT|multivec"):
            Settings(
                jina_api_key="test",
                neo4j_password="test",
                embedding_model="voyage-4",
                embedding_route="native",
                voyage_api_key="test",
                multivec_enabled=True,
                _env_file=None,
            )

    def test_jina_with_multivec_is_valid(self) -> None:
        """Jina + multivec_enabled=True is fine."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="test",
            neo4j_password="test",
            embedding_model="jina-v4",
            embedding_route="native",
            multivec_enabled=True,
            image_embed_strategy="none",
            _env_file=None,
        )
        assert s.multivec_enabled is True

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
            multivec_enabled=False,
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
    s = Settings(
        neo4j_password="test",
        _env_file=None,
    )  # type: ignore[call-arg]
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
