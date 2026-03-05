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
