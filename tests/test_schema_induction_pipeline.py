"""Tests for schema induction integration in the pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk


class TestSchemaInductionInPipeline:
    @pytest.mark.asyncio
    async def test_ingest_to_graph_calls_inducer_when_enabled(self) -> None:
        """When schema_induction_enabled, inducer is called before graph ingest."""
        from ingestion.schema_inducer import InducedSchema, MergedSchema

        chunks = [
            TextChunk(text="x" * 300, chunk_index=0, page_number=1),
        ]

        mock_engine = AsyncMock()
        mock_inducer = MagicMock()
        induced = InducedSchema(
            entity_types={"clause": "A clause"},
            relationship_types={},
        )
        merged = MergedSchema(
            entity_types={"person": "A person", "clause": "A clause"},
            relationship_types={"uses": "X uses Y"},
        )
        mock_inducer.induce = AsyncMock(return_value=induced)
        mock_inducer.merge_with_base.return_value = merged

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch("ingestion.pipeline.SchemaInducer", return_value=mock_inducer),
        ):
            mock_settings.schema_induction_enabled = True
            mock_settings.graph_engine = "gliner"

            from ingestion.pipeline import _ingest_to_graph_with_schema

            await _ingest_to_graph_with_schema("doc.pdf", chunks, mock_engine)

        mock_inducer.induce.assert_called_once()
        mock_engine.ingest.assert_called_once()
        # Schema should be passed to engine
        call_kwargs = mock_engine.ingest.call_args
        assert call_kwargs.kwargs.get("schema") == merged

    @pytest.mark.asyncio
    async def test_ingest_to_graph_skips_inducer_for_graphiti(self) -> None:
        """Graphiti engine ignores schema induction (it has its own LLM)."""
        chunks = [
            TextChunk(text="x" * 300, chunk_index=0, page_number=1),
        ]
        mock_engine = AsyncMock()

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.schema_induction_enabled = True
            mock_settings.graph_engine = "graphiti"

            from ingestion.pipeline import _ingest_to_graph_with_schema

            await _ingest_to_graph_with_schema("doc.pdf", chunks, mock_engine)

        # Engine should be called without schema
        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args
        assert call_kwargs.kwargs.get("schema") is None

    @pytest.mark.asyncio
    async def test_ingest_to_graph_skips_inducer_when_disabled(self) -> None:
        """When schema_induction_enabled=False, no inducer call."""
        chunks = [TextChunk(text="x" * 300, chunk_index=0, page_number=1)]
        mock_engine = AsyncMock()

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.schema_induction_enabled = False
            mock_settings.graph_engine = "gliner"

            from ingestion.pipeline import _ingest_to_graph_with_schema

            await _ingest_to_graph_with_schema("doc.pdf", chunks, mock_engine)

        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args
        assert call_kwargs.kwargs.get("schema") is None
