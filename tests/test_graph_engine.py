"""Tests for modular graph engine protocol and factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk
from server.models import GraphFact


class TestExpandedSchema:
    def test_entity_types_count(self) -> None:
        """Base schema has 14 entity types covering diverse domains."""
        from config.constants import ENTITY_TYPES

        assert len(ENTITY_TYPES) == 14

    def test_relationship_types_count(self) -> None:
        """Base schema has 12 relationship types."""
        from config.constants import RELATIONSHIP_TYPES

        assert len(RELATIONSHIP_TYPES) == 12

    def test_entity_types_have_descriptions(self) -> None:
        """Every entity type has a non-empty description."""
        from config.constants import ENTITY_TYPES

        for name, desc in ENTITY_TYPES.items():
            assert desc.strip(), f"Empty description for entity type '{name}'"

    def test_relationship_types_have_descriptions(self) -> None:
        """Every relationship type has a non-empty description."""
        from config.constants import RELATIONSHIP_TYPES

        for name, desc in RELATIONSHIP_TYPES.items():
            assert desc.strip(), f"Empty description for rel type '{name}'"


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
    def setup_method(self) -> None:
        """Reset singleton between tests."""
        import ingestion.graph_engine as mod

        mod._engine = None

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

    def test_factory_returns_singleton(self) -> None:
        """Factory returns same instance on repeated calls."""
        from ingestion.graph_engine import get_graph_engine

        with patch("config.settings.settings") as mock_settings:
            mock_settings.graph_engine = "graphiti"
            engine1 = get_graph_engine()
            engine2 = get_graph_engine()
        assert engine1 is engine2

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
        mock_edge.configure_mock(name="report.pdf")
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

        long_text = (
            "Tim Cook works for Apple Inc. as the CEO. "
            "He has been leading the company since 2011 when Steve Jobs stepped down. "
            "Under his leadership Apple has grown to become one of the most valuable "
            "companies in the world with products like iPhone and MacBook."
        )
        chunks = [
            TextChunk(
                text=long_text,
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

        long_text = (
            "John Doe is a senior engineer at the company. "
            "He specializes in distributed systems and has contributed to many "
            "open source projects over the past decade of his career in technology "
            "and software engineering across multiple organizations worldwide."
        )
        chunks = [TextChunk(text=long_text, chunk_index=0, page_number=1)]

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


class TestGLiNEREngineSearch:
    @pytest.mark.asyncio
    async def test_search_returns_graph_facts(self) -> None:
        """GLiNEREngine.search queries Neo4j and returns GraphFacts."""
        from ingestion.graph_engine import GLiNEREngine

        mock_result = AsyncMock()
        mock_result.data = AsyncMock(
            return_value=[
                {
                    "entity_name": "Apple Inc.",
                    "entity_type": "ORGANIZATION",
                    "rel_type": "USES_TECHNOLOGY",
                    "target_name": "Python",
                    "confidence": 0.95,
                    "score": 2.5,
                },
            ]
        )

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._driver = mock_driver
        engine._extractor = MagicMock()
        engine._schema = MagicMock()
        engine._entity_map = {}
        engine._relation_map = {}

        results = await engine.search("Apple", limit=5)

        assert len(results) == 1
        assert results[0].fact == "Apple Inc. uses technology Python"
        assert results[0].entities == ["Apple Inc.", "Python"]
        assert results[0].relation_type == "USES_TECHNOLOGY"
        assert results[0].confidence == 0.95

        # Verify Cypher uses fulltext index
        query_str = mock_session.run.call_args.args[0]
        assert "entity_fulltext" in query_str

    @pytest.mark.asyncio
    async def test_search_entity_without_relations(self) -> None:
        """Search returns entity-only facts when no relationships exist."""
        from ingestion.graph_engine import GLiNEREngine

        mock_result = AsyncMock()
        mock_result.data = AsyncMock(
            return_value=[
                {
                    "entity_name": "Google",
                    "entity_types": ["ORGANIZATION"],
                    "rel_type": None,
                    "target_name": None,
                    "confidence": None,
                    "score": 1.8,
                },
            ]
        )

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._driver = mock_driver
        engine._extractor = MagicMock()
        engine._schema = MagicMock()
        engine._entity_map = {}
        engine._relation_map = {}

        results = await engine.search("Google", limit=5)

        assert len(results) == 1
        assert results[0].fact == "Google (ORGANIZATION)"
        assert results[0].entities == ["Google"]
        assert results[0].relation_type is None

    @pytest.mark.asyncio
    async def test_search_deduplicates_facts(self) -> None:
        """Search deduplicates identical facts from multiple records."""
        from ingestion.graph_engine import GLiNEREngine

        mock_result = AsyncMock()
        mock_result.data = AsyncMock(
            return_value=[
                {
                    "entity_name": "Apple",
                    "entity_type": "ORGANIZATION",
                    "rel_type": "PRODUCES",
                    "target_name": "iPhone",
                    "confidence": 1.0,
                    "score": 3.0,
                },
                {
                    "entity_name": "Apple",
                    "entity_type": "ORGANIZATION",
                    "rel_type": "PRODUCES",
                    "target_name": "iPhone",
                    "confidence": 1.0,
                    "score": 2.5,
                },
            ]
        )

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._driver = mock_driver
        engine._extractor = MagicMock()
        engine._schema = MagicMock()
        engine._entity_map = {}
        engine._relation_map = {}

        results = await engine.search("Apple", limit=5)

        # Should deduplicate to 1 result
        assert len(results) == 1


class TestRelationConstraints:
    def test_constraints_cover_all_relationship_types(self) -> None:
        """Every relationship type has a domain/range constraint entry."""
        from config.constants import RELATION_CONSTRAINTS, RELATIONSHIP_TYPES

        for rel in RELATIONSHIP_TYPES:
            assert rel in RELATION_CONSTRAINTS, f"Missing constraint for '{rel}'"

    def test_constraint_types_are_valid_entity_types(self) -> None:
        """All types referenced in constraints exist in ENTITY_TYPES."""
        from config.constants import ENTITY_TYPES, RELATION_CONSTRAINTS

        valid = set(ENTITY_TYPES)
        for rel, (sources, targets) in RELATION_CONSTRAINTS.items():
            invalid_src = sources - valid
            invalid_tgt = targets - valid
            assert not invalid_src, f"{rel} has invalid source types: {invalid_src}"
            assert not invalid_tgt, f"{rel} has invalid target types: {invalid_tgt}"

    def test_valued_at_requires_monetary_target(self) -> None:
        """valued_at only allows monetary_value as target."""
        from config.constants import RELATION_CONSTRAINTS

        _, targets = RELATION_CONSTRAINTS["valued_at"]
        assert targets == frozenset({"monetary_value"})

    def test_scheduled_for_requires_datetime_target(self) -> None:
        """scheduled_for only allows date_time as target."""
        from config.constants import RELATION_CONSTRAINTS

        _, targets = RELATION_CONSTRAINTS["scheduled_for"]
        assert targets == frozenset({"date_time"})

    def test_created_by_target_is_agent(self) -> None:
        """created_by target must be person or organization."""
        from config.constants import RELATION_CONSTRAINTS

        _, targets = RELATION_CONSTRAINTS["created_by"]
        assert targets == frozenset({"person", "organization"})


class TestGLiNERConstraintValidation:
    def test_violates_constraint_drops_invalid_triple(self) -> None:
        """Invalid domain/range pair is rejected."""
        from ingestion.graph_engine import GLiNEREngine

        # concept -[valued_at]-> date_time should be rejected
        assert GLiNEREngine._violates_constraint(
            "valued_at",
            head_types={"concept"},
            tail_types={"date_time"},
        )

    def test_violates_constraint_allows_valid_triple(self) -> None:
        """Valid domain/range pair is accepted."""
        from ingestion.graph_engine import GLiNEREngine

        # product -[valued_at]-> monetary_value should pass
        assert not GLiNEREngine._violates_constraint(
            "valued_at",
            head_types={"product"},
            tail_types={"monetary_value"},
        )

    def test_violates_constraint_allows_unknown_rel(self) -> None:
        """Schema-induced relationship types (not in base) are allowed."""
        from ingestion.graph_engine import GLiNEREngine

        assert not GLiNEREngine._violates_constraint(
            "governs",
            head_types={"document"},
            tail_types={"organization"},
        )

    def test_violates_constraint_multi_typed_entity(self) -> None:
        """Entity with multiple types passes if ANY type matches."""
        from ingestion.graph_engine import GLiNEREngine

        # entity typed as both concept and product — product is valid source for valued_at
        assert not GLiNEREngine._violates_constraint(
            "valued_at",
            head_types={"concept", "product"},
            tail_types={"monetary_value"},
        )

    @pytest.mark.asyncio
    async def test_ingest_drops_constraint_violating_relations(self) -> None:
        """Relations violating domain/range constraints are not written to Neo4j."""
        from ingestion.graph_engine import GLiNEREngine

        long_text = (
            "The quarterly revenue metric of $5M was reported on January 15th 2026. "
            "The financial report covers all divisions across North America and Europe "
            "including manufacturing facilities and corporate offices worldwide."
        )
        chunks = [TextChunk(text=long_text, chunk_index=0, page_number=1)]

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": {
                "metric": ["Revenue"],
                "date_time": ["January 15Th 2026"],
                "organization": ["Acme Corp"],
            },
            "relation_extraction": {
                # valid: organization -[valued_at]-> monetary_value? NO — metric is not
                # monetary_value; invalid: metric -[valued_at]-> date_time
                "valued_at": [("Revenue", "January 15th 2026")],
                # valid: organization -[scheduled_for]-> date_time? NO — org not in sources
                "scheduled_for": [("Acme Corp", "January 15th 2026")],
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

        await engine.ingest(chunks, "report.pdf")

        # Only entity MERGEs should be written (3 entities), zero relationships
        calls = [str(c) for c in mock_session.run.call_args_list]
        rel_calls = [c for c in calls if "apoc.merge.relationship" in c]
        assert len(rel_calls) == 0


class TestGLiNEREngineDynamicSchema:
    @pytest.mark.asyncio
    async def test_ingest_with_custom_schema(self) -> None:
        """GLiNEREngine.ingest uses provided schema instead of base."""
        from ingestion.graph_engine import GLiNEREngine
        from ingestion.schema_inducer import MergedSchema

        long_text = (
            "The Master Service Agreement between Acme Corp and Widget Inc "
            "establishes payment terms of net-30 for all deliverables under "
            "the contract scope including software development and consulting "
            "services. This agreement is effective from January 2026 onwards."
        )
        chunks = [TextChunk(text=long_text, chunk_index=0, page_number=1)]

        custom_schema = MergedSchema(
            entity_types={"clause": "A legal clause", "person": "A person"},
            relationship_types={"governs": "X governs Y"},
        )

        mock_extractor = MagicMock()
        mock_extractor.create_schema.return_value = mock_extractor
        mock_extractor.entities.return_value = mock_extractor
        mock_extractor.relations.return_value = mock_extractor
        mock_extractor.extract.return_value = {
            "entities": {"clause": ["Net-30"]},
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
        engine._schema = MagicMock()  # base schema (should NOT be used)
        engine._entity_map = {}
        engine._relation_map = {}

        await engine.ingest(chunks, "contract.pdf", schema=custom_schema)

        # Verify extractor.create_schema was called (for custom schema)
        mock_extractor.create_schema.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_without_schema_uses_base(self) -> None:
        """GLiNEREngine.ingest without schema parameter uses base schema."""
        from ingestion.graph_engine import GLiNEREngine

        long_text = "x" * 250
        chunks = [TextChunk(text=long_text, chunk_index=0, page_number=1)]

        mock_extractor = MagicMock()
        base_schema = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": {},
            "relation_extraction": {},
        }

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._extractor = mock_extractor
        engine._driver = mock_driver
        engine._schema = base_schema

        await engine.ingest(chunks, "test.pdf")

        # Should use base schema (self._schema), not create_schema
        mock_extractor.extract.assert_called_once_with(long_text, base_schema)
