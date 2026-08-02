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
                embedding_provider="voyage",
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
            embedding_provider="jina",
            multivec_enabled=True,
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
        """SCHEMA_INDUCTION_MODEL defaults to claude-haiku."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="k",
            neo4j_password="p",
            _env_file=None,
        )
        assert "haiku" in s.schema_induction_model

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
            embedding_provider="voyage",
            voyage_api_key="test",
            multivec_enabled=False,
            _env_file=None,
        )
        assert s.embedding_provider == "voyage"


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
    assert s.llm_model == "claude-sonnet-5"


def test_vector_name_constants() -> None:
    """Named-vector constants exist for the migrated collection."""
    from config.constants import DENSE_VECTOR_NAME, MINICOIL_AVG_LEN, SPARSE_VECTOR_NAME

    assert DENSE_VECTOR_NAME == "dense"
    assert SPARSE_VECTOR_NAME == "sparse"
    assert MINICOIL_AVG_LEN == 80
