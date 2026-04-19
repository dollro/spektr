"""Tests for the LLM-based schema inducer."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestInducedSchema:
    def test_induced_schema_creation(self) -> None:
        """InducedSchema holds entity and relationship types."""
        from ingestion.schema_inducer import InducedSchema

        schema = InducedSchema(
            entity_types={"clause": "A legal clause"},
            relationship_types={"governs": "X governs Y"},
        )
        assert "clause" in schema.entity_types
        assert "governs" in schema.relationship_types


class TestMergedSchema:
    def test_merged_schema_creation(self) -> None:
        """MergedSchema holds merged entity and relationship types."""
        from ingestion.schema_inducer import MergedSchema

        schema = MergedSchema(
            entity_types={"person": "A person", "clause": "A clause"},
            relationship_types={"uses": "X uses Y", "governs": "X governs Y"},
        )
        assert len(schema.entity_types) == 2
        assert len(schema.relationship_types) == 2


class TestSchemaInducer:
    @pytest.mark.asyncio
    async def test_induce_returns_schema(self) -> None:
        """SchemaInducer.induce calls LLM and returns InducedSchema."""
        from ingestion.schema_inducer import SchemaInducer

        mock_response = (
            '{"entity_types": {"clause": "A legal clause"}, '
            '"relationship_types": {"governs": "X governs Y"}}'
        )

        with patch(
            "ingestion.schema_inducer._call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            inducer = SchemaInducer()
            result = await inducer.induce("This agreement governs the terms..." + "x" * 200)

        assert "clause" in result.entity_types
        assert "governs" in result.relationship_types

    @pytest.mark.asyncio
    async def test_induce_short_text_returns_empty(self) -> None:
        """Text shorter than 200 chars returns empty schema (no LLM call)."""
        from ingestion.schema_inducer import SchemaInducer

        inducer = SchemaInducer()
        result = await inducer.induce("Short text.")

        assert result.entity_types == {}
        assert result.relationship_types == {}

    @pytest.mark.asyncio
    async def test_induce_malformed_json_returns_empty(self) -> None:
        """Malformed LLM response returns empty schema."""
        from ingestion.schema_inducer import SchemaInducer

        with patch(
            "ingestion.schema_inducer._call_llm",
            new_callable=AsyncMock,
            return_value="not json",
        ):
            inducer = SchemaInducer()
            result = await inducer.induce("x" * 300)

        assert result.entity_types == {}

    def test_merge_with_base_combines_schemas(self) -> None:
        """merge_with_base adds induced types on top of base schema."""
        from ingestion.schema_inducer import InducedSchema, SchemaInducer

        inducer = SchemaInducer()
        induced = InducedSchema(
            entity_types={"clause": "A legal clause"},
            relationship_types={"governs": "X governs Y"},
        )
        merged = inducer.merge_with_base(induced)

        # Should have all base types plus induced types
        assert "person" in merged.entity_types  # from base
        assert "clause" in merged.entity_types  # from induced
        assert "uses" in merged.relationship_types  # from base
        assert "governs" in merged.relationship_types  # from induced

    def test_merge_induced_cannot_override_base(self) -> None:
        """Induced types with same name as base types don't override."""
        from ingestion.schema_inducer import InducedSchema, SchemaInducer

        inducer = SchemaInducer()
        induced = InducedSchema(
            entity_types={"person": "OVERRIDDEN"},
            relationship_types={},
        )
        merged = inducer.merge_with_base(induced)

        # Base description should be preserved
        assert merged.entity_types["person"] != "OVERRIDDEN"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self) -> None:
        """Second call with similar text hits cache, skips LLM."""
        from ingestion.schema_inducer import SchemaInducer

        mock_llm = AsyncMock(
            return_value='{"entity_types": {"clause": "A clause"}, "relationship_types": {}}'
        )

        with patch("ingestion.schema_inducer._call_llm", mock_llm):
            inducer = SchemaInducer()
            text = "x" * 300
            await inducer.induce(text)
            await inducer.induce(text)  # same text

        # LLM called only once
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expires(self) -> None:
        """Expired cache entries trigger a new LLM call."""
        from ingestion.schema_inducer import SchemaInducer

        mock_llm = AsyncMock(
            return_value='{"entity_types": {"clause": "A clause"}, "relationship_types": {}}'
        )

        with patch("ingestion.schema_inducer._call_llm", mock_llm):
            inducer = SchemaInducer(cache_ttl=0)  # instant expiry
            text = "x" * 300
            await inducer.induce(text)
            await inducer.induce(text)

        assert mock_llm.call_count == 2
