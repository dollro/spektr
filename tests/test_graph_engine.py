"""Tests for modular graph engine protocol and factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk
from server.models import GraphFact


class TestGraphFact:
    def test_graphfact_minimal(self) -> None:
        """GraphFact works with just fact field (backward compat)."""
        gf = GraphFact(fact="Apple is a tech company")
        assert gf.fact == "Apple is a tech company"
        assert gf.entities is None
        assert gf.relation_type is None
        assert gf.confidence is None

    def test_graphfact_with_structured_fields(self) -> None:
        """GraphFact accepts optional structured fields from GLiNER."""
        gf = GraphFact(
            fact="Tim Cook works for Apple",
            entities=["Tim Cook", "Apple"],
            relation_type="works_for",
            confidence=0.95,
        )
        assert gf.entities == ["Tim Cook", "Apple"]
        assert gf.relation_type == "works_for"
        assert gf.confidence == 0.95

    def test_graphfact_with_temporal_fields(self) -> None:
        """GraphFact accepts temporal fields from Graphiti."""
        gf = GraphFact(
            fact="Apple acquired NeXT",
            source="report.pdf",
            created_at="2026-01-01T00:00:00",
            expired_at="2026-06-01T00:00:00",
        )
        assert gf.expired_at == "2026-06-01T00:00:00"


class TestGetGraphEngine:
    def test_factory_returns_graphiti_by_default(self) -> None:
        """Factory returns GraphitiEngine when graph_engine='graphiti'."""
        from ingestion.graph_engine import GraphitiEngine, get_graph_engine

        with patch("config.settings.settings") as mock_settings:
            mock_settings.graph_engine = "graphiti"
            engine = get_graph_engine()
        assert isinstance(engine, GraphitiEngine)

    def test_factory_returns_gliner_when_configured(self) -> None:
        """Factory returns GLiNEREngine when graph_engine='gliner'."""
        from ingestion.graph_engine import GLiNEREngine, get_graph_engine

        with (
            patch("config.settings.settings") as mock_settings,
            patch.object(GLiNEREngine, "__init__", lambda self: None),
        ):
            mock_settings.graph_engine = "gliner"
            engine = get_graph_engine()
        assert isinstance(engine, GLiNEREngine)

    def test_factory_raises_on_unknown_engine(self) -> None:
        """Factory raises ValueError for unknown engine."""
        from ingestion.graph_engine import get_graph_engine

        with (
            patch("config.settings.settings") as mock_settings,
            pytest.raises(ValueError, match="Unknown graph engine"),
        ):
            mock_settings.graph_engine = "invalid"
            get_graph_engine()


class TestGraphitiEngine:
    @pytest.mark.asyncio
    async def test_ingest_delegates_to_graphiti_writer(self) -> None:
        """GraphitiEngine.ingest calls GraphitiWriter.ingest_bulk."""
        from ingestion.graph_engine import GraphitiEngine

        chunks = [TextChunk(text="hello world", chunk_index=0, page_number=1)]
        engine = GraphitiEngine()

        with patch.object(engine._writer, "ingest_bulk", new_callable=AsyncMock) as mock_bulk:
            await engine.ingest(chunks, "test.pdf")
            mock_bulk.assert_called_once()
            assert mock_bulk.call_args.kwargs["source_key"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_search_delegates_to_graphiti_client(self) -> None:
        """GraphitiEngine.search calls Graphiti client.search."""
        from ingestion.graph_engine import GraphitiEngine

        mock_edge = MagicMock()
        mock_edge.fact = "Apple is a company"
        mock_edge.source_description = "report.pdf"
        mock_edge.created_at = "2026-01-01"
        mock_edge.expired_at = None

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[mock_edge])

        engine = GraphitiEngine()
        with patch(
            "ingestion.graphiti_client.get_graphiti",
            return_value=mock_client,
        ):
            results = await engine.search("Apple", limit=5)

        assert len(results) == 1
        assert results[0].fact == "Apple is a company"
        assert results[0].source == "report.pdf"

    @pytest.mark.asyncio
    async def test_close_delegates(self) -> None:
        """GraphitiEngine.close calls GraphitiWriter.close."""
        from ingestion.graph_engine import GraphitiEngine

        engine = GraphitiEngine()
        with patch.object(engine._writer, "close", new_callable=AsyncMock) as mock_close:
            await engine.close()
            mock_close.assert_called_once()


class TestGLiNEREngineIngest:
    @pytest.mark.asyncio
    async def test_ingest_extracts_entities_and_writes_neo4j(self) -> None:
        """GLiNEREngine.ingest extracts entities + relations and writes to Neo4j."""
        from ingestion.graph_engine import GLiNEREngine

        chunks = [
            TextChunk(
                text="Tim Cook works for Apple Inc.",
                chunk_index=0,
                page_number=1,
            ),
        ]

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": {
                "person": ["Tim Cook"],
                "organization": ["Apple Inc."],
            },
            "relation_extraction": {
                "works_for": [("Tim Cook", "Apple Inc.")],
            },
        }

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._extractor = mock_extractor
        engine._driver = mock_driver
        engine._schema = MagicMock()
        engine._entity_map = {
            "person": "PERSON",
            "organization": "ORGANIZATION",
        }
        engine._relation_map = {"works_for": "WORKS_AT"}

        await engine.ingest(chunks, "test.pdf")

        # Should have called session.run for MERGE entities (2) + relationships (1)
        assert mock_session.run.call_count >= 3

    @pytest.mark.asyncio
    async def test_ingest_empty_chunks_is_noop(self) -> None:
        """GLiNEREngine.ingest with empty chunks does nothing."""
        from ingestion.graph_engine import GLiNEREngine

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._extractor = MagicMock()
        engine._driver = MagicMock()
        engine._schema = MagicMock()
        engine._entity_map = {}
        engine._relation_map = {}

        await engine.ingest([], "test.pdf")
        engine._extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_normalizes_entity_names(self) -> None:
        """Entity names are normalized (stripped, title-cased) before MERGE."""
        from ingestion.graph_engine import GLiNEREngine

        chunks = [TextChunk(text="test", chunk_index=0, page_number=1)]

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": {"person": ["  john doe  "]},
            "relation_extraction": {},
        }

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._extractor = mock_extractor
        engine._driver = mock_driver
        engine._schema = MagicMock()
        engine._entity_map = {"person": "PERSON"}
        engine._relation_map = {}

        await engine.ingest(chunks, "test.pdf")

        # Check that the MERGE call used normalized name "John Doe"
        calls = mock_session.run.call_args_list
        assert any("John Doe" in str(c) for c in calls)
