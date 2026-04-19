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
