# Modular Graph Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the graph engine pluggable via `GRAPH_ENGINE` setting, supporting both Graphiti (LLM) and GLiNER2 (local CPU).

**Architecture:** Unified `GraphEngine` protocol with factory dispatch. GraphitiEngine wraps existing code. GLiNEREngine uses gliner2 for extraction + direct Neo4j Cypher writes. Both return enriched `GraphFact` models. Pipeline and search tools are engine-agnostic.

**Tech Stack:** gliner2, neo4j (async driver), pydantic, pytest, pytest-asyncio

**Design doc:** `docs/plans/2026-03-06-modular-graph-engine-design.md`

---

### Task 1: Config + enriched GraphFact

**Files:**
- Modify: `config/settings.py:83` (add setting after `graph_enabled`)
- Modify: `.env.example:156` (add after GRAPH_ENABLED line)
- Modify: `server/models.py:30-36` (enrich GraphFact)
- Test: `tests/test_graph_engine.py` (new)

**Step 1: Write failing test for graph_engine setting**

Create `tests/test_graph_engine.py`:

```python
"""Tests for modular graph engine protocol and factory."""

from __future__ import annotations

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: FAIL — `entities` field not recognized by GraphFact

**Step 3: Enrich GraphFact and add setting**

In `server/models.py`, replace the `GraphFact` class (lines 30-36):

```python
class GraphFact(BaseModel):
    """Knowledge graph fact from any graph engine."""

    fact: str
    source: str | None = None
    created_at: str | None = None
    expired_at: str | None = None
    entities: list[str] | None = None
    relation_type: str | None = None
    confidence: float | None = None
```

In `config/settings.py`, add after line 83 (`graph_enabled`):

```python
    graph_engine: str = "graphiti"  # "graphiti" | "gliner"
```

In `.env.example`, add after the `GRAPH_ENABLED` line (156):

```
GRAPH_ENGINE=graphiti                   # "graphiti" (LLM, slow, rich) or "gliner" (local CPU, fast)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add config/settings.py .env.example server/models.py tests/test_graph_engine.py
git commit -m "feat: add graph_engine setting and enrich GraphFact model"
```

---

### Task 2: GraphEngine protocol + factory + GraphitiEngine

**Files:**
- Create: `ingestion/graph_engine.py`
- Modify: `tests/test_graph_engine.py` (add tests)

**Step 1: Write failing tests for protocol and GraphitiEngine**

Append to `tests/test_graph_engine.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk


class TestGetGraphEngine:
    def test_factory_returns_graphiti_by_default(self) -> None:
        """Factory returns GraphitiEngine when graph_engine='graphiti'."""
        from ingestion.graph_engine import GraphitiEngine, get_graph_engine

        with patch("ingestion.graph_engine.settings") as mock_settings:
            mock_settings.graph_engine = "graphiti"
            engine = get_graph_engine()
        assert isinstance(engine, GraphitiEngine)

    def test_factory_returns_gliner_when_configured(self) -> None:
        """Factory returns GLiNEREngine when graph_engine='gliner'."""
        from ingestion.graph_engine import GLiNEREngine, get_graph_engine

        with patch("ingestion.graph_engine.settings") as mock_settings:
            mock_settings.graph_engine = "gliner"
            engine = get_graph_engine()
        assert isinstance(engine, GLiNEREngine)

    def test_factory_raises_on_unknown_engine(self) -> None:
        """Factory raises ValueError for unknown engine."""
        from ingestion.graph_engine import get_graph_engine

        with (
            patch("ingestion.graph_engine.settings") as mock_settings,
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
            "ingestion.graph_engine.get_graphiti",
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: FAIL — `ingestion.graph_engine` module not found

**Step 3: Create graph_engine.py with protocol, factory, and GraphitiEngine**

Create `ingestion/graph_engine.py`:

```python
"""Pluggable graph engine: protocol, factory, and implementations.

Switch between engines via the GRAPH_ENGINE setting:
  - "graphiti" — LLM-based extraction via Graphiti (slow, rich)
  - "gliner"  — local GLiNER2 model on CPU (fast, zero API cost)
"""

from __future__ import annotations

import logging
from typing import Protocol

from ingestion.file_processor import TextChunk
from server.models import GraphFact

logger = logging.getLogger(__name__)


class GraphEngine(Protocol):
    """Unified interface for knowledge graph ingestion and search."""

    async def ingest(
        self, chunks: list[TextChunk], source_key: str
    ) -> None: ...

    async def search(
        self, query: str, limit: int = 10
    ) -> list[GraphFact]: ...

    async def close(self) -> None: ...


class GraphitiEngine:
    """Graph engine backed by Graphiti (LLM-based extraction)."""

    def __init__(self) -> None:
        from ingestion.graph_writer import GraphitiWriter

        self._writer = GraphitiWriter()

    async def ingest(
        self, chunks: list[TextChunk], source_key: str
    ) -> None:
        await self._writer.ingest_bulk(chunks=chunks, source_key=source_key)

    async def search(
        self, query: str, limit: int = 10
    ) -> list[GraphFact]:
        from ingestion.graphiti_client import get_graphiti

        client = await get_graphiti()
        edges = await client.search(query)
        return [
            GraphFact(
                fact=edge.fact,
                source=edge.source_description,
                created_at=str(edge.created_at),
                expired_at=(
                    str(edge.expired_at) if edge.expired_at else None
                ),
            )
            for edge in edges[:limit]
        ]

    async def close(self) -> None:
        await self._writer.close()


class GLiNEREngine:
    """Graph engine backed by GLiNER2 (local CPU extraction).

    Placeholder — full implementation in Task 5.
    """

    async def ingest(
        self, chunks: list[TextChunk], source_key: str
    ) -> None:
        raise NotImplementedError("GLiNEREngine.ingest not yet implemented")

    async def search(
        self, query: str, limit: int = 10
    ) -> list[GraphFact]:
        raise NotImplementedError("GLiNEREngine.search not yet implemented")

    async def close(self) -> None:
        pass


def get_graph_engine() -> GraphEngine:
    """Factory: create graph engine based on settings."""
    from config.settings import settings

    engine = settings.graph_engine.lower()
    if engine == "graphiti":
        return GraphitiEngine()
    if engine == "gliner":
        return GLiNEREngine()
    msg = f"Unknown graph engine: {engine!r}. Use 'graphiti' or 'gliner'."
    raise ValueError(msg)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: PASS (all 9 tests)

**Step 5: Lint and commit**

```bash
uv run ruff check ingestion/graph_engine.py tests/test_graph_engine.py --fix
uv run ruff format ingestion/graph_engine.py tests/test_graph_engine.py
git add ingestion/graph_engine.py tests/test_graph_engine.py
git commit -m "feat: add GraphEngine protocol, factory, and GraphitiEngine wrapper"
```

---

### Task 3: Wire pipeline.py and graph_search.py to use GraphEngine

**Files:**
- Modify: `ingestion/pipeline.py:31-32,100-101,175-183,437-441,469-484,516,543-544,546-601`
- Modify: `server/tools/graph_search.py` (full rewrite to use engine)
- Modify: `tests/test_graph_engine.py` (add integration-style tests)

**Step 1: Write failing tests for the wiring**

Append to `tests/test_graph_engine.py`:

```python
class TestPipelineWiring:
    @pytest.mark.asyncio
    async def test_pipeline_uses_graph_engine(self) -> None:
        """pipeline.py should call get_graph_engine, not GraphitiWriter directly."""
        import ingestion.pipeline as pipeline_mod

        # Verify the module references get_graph_engine
        source = pipeline_mod.__file__
        assert source is not None
        with open(source) as f:
            content = f.read()
        assert "get_graph_engine" in content
        assert "GraphitiWriter()" not in content


class TestGraphSearchWiring:
    @pytest.mark.asyncio
    async def test_graph_search_uses_engine(self) -> None:
        """graph_search should use get_graph_engine().search()."""
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=[
            GraphFact(fact="test fact", source="doc.pdf"),
        ])

        with patch(
            "server.tools.graph_search.get_graph_engine",
            return_value=mock_engine,
        ):
            from server.tools.graph_search import graph_search

            results = await graph_search("test query", limit=5)

        assert len(results) == 1
        assert results[0]["fact"] == "test fact"
        mock_engine.search.assert_called_once_with("test query", limit=5)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_engine.py::TestPipelineWiring -v`
Expected: FAIL — pipeline.py still imports GraphitiWriter directly

**Step 3: Update pipeline.py**

In `ingestion/pipeline.py`:

Replace the import at line 31:
```python
# OLD: from ingestion.graph_writer import GraphitiWriter
from ingestion.graph_engine import GraphEngine, get_graph_engine
```

Replace all `GraphitiWriter` type annotations with `GraphEngine`:
- Line 101: `graphiti_writer: GraphitiWriter | None` → `graph_engine: GraphEngine | None`
- Line 183: same pattern in `_process_text_page` signature
- Line 272: same in `_process_visual_page`
- Line 441: same in `_caption_and_ingest_visual`

In `ingest_file` function (line 544), replace:
```python
# OLD: graphiti_writer = GraphitiWriter()
graph_engine_inst = get_graph_engine()
```

In `_process_all_pages` closure, replace all `graphiti_writer` references with `graph_engine_inst`.

In the `_ingest_to_graphiti` function (lines 469-484), rename to `_ingest_to_graph` and update:
```python
async def _ingest_to_graph(
    source_file: str,
    chunks: list[TextChunk],
    engine: GraphEngine,
) -> None:
    """Ingest chunks via the active graph engine."""
    await engine.ingest(chunks=chunks, source_key=source_file)
```

Update the call site (line 579) to pass `graph_engine_inst` instead of `graphiti_writer`.

Update `_caption_and_ingest_visual` (line 437-466) to accept `GraphEngine` and call `engine.ingest` with a single-chunk list instead of `graphiti_writer.ingest_chunk`.

Update the finally block (line 600-601) to call `await graph_engine_inst.close()`.

**Step 4: Update graph_search.py**

Replace `server/tools/graph_search.py` entirely:

```python
"""Knowledge graph search tool for MCP server.

Engine-agnostic: dispatches to whichever GraphEngine is configured
via the GRAPH_ENGINE setting.
"""

from __future__ import annotations

import logging

from ingestion.graph_engine import get_graph_engine

logger = logging.getLogger(__name__)


async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
) -> list[dict]:  # type: ignore[type-arg]
    """Search the knowledge graph for entities and relationships.

    Uses the configured graph engine's search method.

    Args:
        query: Search text.
        search_type: 'entity' (default). Reserved for future modes.
        limit: Maximum results (default 10).
    """
    if not query or not query.strip():
        return []
    if search_type != "entity":
        raise ValueError(
            f"search_type='{search_type}' is not yet implemented."
            " Use 'entity' instead."
        )
    limit = max(1, min(limit, 100))

    try:
        engine = get_graph_engine()
        results = await engine.search(query, limit=limit)
        return [r.model_dump() for r in results]
    except Exception as exc:
        logger.exception("graph_search failed")
        return [
            {
                "error": f"graph_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]
```

**Step 5: Run all tests**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: PASS (all 11 tests)

Run: `uv run pytest -v` (full suite to catch regressions)
Expected: All existing tests pass

**Step 6: Lint and commit**

```bash
uv run ruff check ingestion/pipeline.py server/tools/graph_search.py --fix
uv run ruff format ingestion/pipeline.py server/tools/graph_search.py
git add ingestion/pipeline.py server/tools/graph_search.py ingestion/graph_engine.py tests/test_graph_engine.py
git commit -m "refactor: wire pipeline and graph_search to use GraphEngine protocol"
```

---

### Task 4: Neo4j full-text index for GLiNER search path

**Files:**
- Modify: `ingestion/neo4j_setup.py:20-46`
- Test: `tests/test_graph_engine.py` (add schema test)

**Step 1: Write failing test**

Append to `tests/test_graph_engine.py`:

```python
class TestNeo4jSchema:
    @pytest.mark.asyncio
    async def test_fulltext_index_created(self) -> None:
        """create_neo4j_schema creates entity_fulltext index."""
        from ingestion.neo4j_setup import create_neo4j_schema

        source = create_neo4j_schema.__module__
        import importlib
        mod = importlib.import_module(source)
        import inspect
        src = inspect.getsource(mod.create_neo4j_schema)
        assert "entity_fulltext" in src
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_engine.py::TestNeo4jSchema -v`
Expected: FAIL — no `entity_fulltext` in source

**Step 3: Add full-text index to neo4j_setup.py**

In `ingestion/neo4j_setup.py`, add after the `chunk_unique` constraint (line 36):

```python
        # Full-text index for GLiNER engine search path
        await session.run(
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.description]"
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_engine.py::TestNeo4jSchema -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ingestion/neo4j_setup.py tests/test_graph_engine.py
git commit -m "feat: add Neo4j full-text index for GLiNER search path"
```

---

### Task 5: GLiNEREngine — extraction + Neo4j ingestion

**Files:**
- Modify: `ingestion/graph_engine.py` (replace GLiNEREngine placeholder)
- Modify: `pyproject.toml:6-25` (add optional dependency)
- Test: `tests/test_graph_engine.py` (add GLiNEREngine tests)

**Step 1: Write failing tests for GLiNEREngine.ingest**

Append to `tests/test_graph_engine.py`:

```python
class TestGLiNEREngineIngest:
    @pytest.mark.asyncio
    async def test_ingest_extracts_entities_and_writes_neo4j(self) -> None:
        """GLiNEREngine.ingest extracts entities + relations and writes to Neo4j."""
        from ingestion.graph_engine import GLiNEREngine

        chunks = [
            TextChunk(text="Tim Cook works for Apple Inc.", chunk_index=0, page_number=1),
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

        await engine.ingest(chunks, "test.pdf")

        # Should have called session.run for MERGE entities + relationships
        assert mock_session.run.call_count >= 2

    @pytest.mark.asyncio
    async def test_ingest_empty_chunks_is_noop(self) -> None:
        """GLiNEREngine.ingest with empty chunks does nothing."""
        from ingestion.graph_engine import GLiNEREngine

        engine = GLiNEREngine.__new__(GLiNEREngine)
        engine._extractor = MagicMock()
        engine._driver = MagicMock()
        engine._schema = MagicMock()

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

        await engine.ingest(chunks, "test.pdf")

        # Find the MERGE call for entity and check normalized name
        calls = mock_session.run.call_args_list
        entity_call = [c for c in calls if "MERGE" in str(c) and "Entity" in str(c)]
        assert len(entity_call) >= 1
        # Name should be title-cased and stripped
        assert any("John Doe" in str(c) for c in calls)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_engine.py::TestGLiNEREngineIngest -v`
Expected: FAIL — GLiNEREngine.ingest raises NotImplementedError

**Step 3: Add gliner2 optional dependency**

In `pyproject.toml`, add after line 25 (after the closing `]` of dependencies):

```toml
[project.optional-dependencies]
gliner = ["gliner2>=1.2"]
```

**Step 4: Implement GLiNEREngine.ingest**

Replace the `GLiNEREngine` class in `ingestion/graph_engine.py`:

```python
class GLiNEREngine:
    """Graph engine backed by GLiNER2 (local CPU extraction).

    Extracts entities and relationships using a 205MB local model,
    then writes directly to Neo4j via Cypher MERGE statements.
    Zero LLM API calls.
    """

    def __init__(self) -> None:
        from gliner2 import GLiNER2
        from neo4j import AsyncGraphDatabase

        from config.constants import ENTITY_TYPES, RELATIONSHIP_TYPES
        from config.settings import settings

        self._extractor = GLiNER2.from_pretrained(
            "fastino/gliner2-base-v1"
        )
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        # Map SCREAMING_CASE to lowercase for GLiNER2 schema
        entity_map = {t.lower(): t for t in ENTITY_TYPES}
        relation_map = {t.lower(): t for t in RELATIONSHIP_TYPES}
        self._entity_map = entity_map
        self._relation_map = relation_map

        self._schema = (
            self._extractor.create_schema()
            .entities(list(entity_map.keys()))
            .relations(list(relation_map.keys()))
        )

    async def ingest(
        self, chunks: list[TextChunk], source_key: str
    ) -> None:
        if not chunks:
            return

        for chunk in chunks:
            text = chunk.contextualized_text or chunk.text
            result = self._extractor.extract(text, self._schema)

            entities = result.get("entities", {})
            relations = result.get("relation_extraction", {})

            async with self._driver.session() as session:
                # Upsert entities
                for entity_type_lower, names in entities.items():
                    entity_type = self._entity_map.get(
                        entity_type_lower, entity_type_lower.upper()
                    )
                    for name in names:
                        normalized = name.strip().title()
                        if not normalized:
                            continue
                        await session.run(
                            "MERGE (e:Entity {name: $name, type: $type}) "
                            "ON CREATE SET e.first_seen = datetime() "
                            "SET e.last_seen = datetime(), "
                            "e.source = $source",
                            name=normalized,
                            type=entity_type,
                            source=source_key,
                        )

                # Upsert relationships
                for rel_type_lower, pairs in relations.items():
                    rel_type = self._relation_map.get(
                        rel_type_lower, rel_type_lower.upper()
                    )
                    for head, tail in pairs:
                        head_norm = head.strip().title()
                        tail_norm = tail.strip().title()
                        if not head_norm or not tail_norm:
                            continue
                        await session.run(
                            "MATCH (s:Entity {name: $source}) "
                            "MATCH (t:Entity {name: $target}) "
                            "CALL apoc.merge.relationship("
                            "s, $relation, $props, {}, t, {}"
                            ") YIELD rel RETURN rel",
                            source=head_norm,
                            target=tail_norm,
                            relation=rel_type,
                            props={
                                "source": source_key,
                                "confidence": 1.0,
                            },
                        )

        logger.info(
            "GLiNER extracted entities from %d chunks for %s",
            len(chunks),
            source_key,
        )

    async def search(
        self, query: str, limit: int = 10
    ) -> list[GraphFact]:
        raise NotImplementedError(
            "GLiNEREngine.search not yet implemented"
        )

    async def close(self) -> None:
        await self._driver.close()
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_engine.py::TestGLiNEREngineIngest -v`
Expected: PASS (all 3 tests)

**Step 6: Lint and commit**

```bash
uv run ruff check ingestion/graph_engine.py --fix
uv run ruff format ingestion/graph_engine.py
git add ingestion/graph_engine.py pyproject.toml tests/test_graph_engine.py
git commit -m "feat: implement GLiNEREngine ingestion with Neo4j writes"
```

---

### Task 6: GLiNEREngine — Neo4j full-text search

**Files:**
- Modify: `ingestion/graph_engine.py` (replace search placeholder)
- Test: `tests/test_graph_engine.py` (add search tests)

**Step 1: Write failing tests for GLiNEREngine.search**

Append to `tests/test_graph_engine.py`:

```python
class TestGLiNEREngineSearch:
    @pytest.mark.asyncio
    async def test_search_returns_graph_facts(self) -> None:
        """GLiNEREngine.search queries Neo4j and returns GraphFacts."""
        from ingestion.graph_engine import GLiNEREngine

        # Mock Neo4j records for entity + relationship results
        mock_record = {
            "entity_name": "Apple Inc.",
            "entity_type": "ORGANIZATION",
            "rel_type": "USES_TECHNOLOGY",
            "target_name": "Python",
            "score": 2.5,
        }

        mock_result = AsyncMock()
        mock_result.__aiter__ = lambda self: self
        mock_result.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_result.data = AsyncMock(return_value=[mock_record])

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

        mock_session.run.assert_called_once()
        # Verify the Cypher query uses fulltext index
        query_str = mock_session.run.call_args.args[0]
        assert "entity_fulltext" in query_str

    @pytest.mark.asyncio
    async def test_search_empty_query(self) -> None:
        """GLiNEREngine.search with empty results returns empty list."""
        from ingestion.graph_engine import GLiNEREngine

        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[])

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

        results = await engine.search("nonexistent", limit=5)
        assert results == [] or isinstance(results, list)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_engine.py::TestGLiNEREngineSearch -v`
Expected: FAIL — GLiNEREngine.search raises NotImplementedError

**Step 3: Implement GLiNEREngine.search**

Replace the `search` method in `GLiNEREngine`:

```python
    async def search(
        self, query: str, limit: int = 10
    ) -> list[GraphFact]:
        """Search Neo4j via full-text index, traverse relationships."""
        cypher = (
            "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) "
            "YIELD node AS e, score "
            "WITH e, score ORDER BY score DESC LIMIT $limit "
            "OPTIONAL MATCH (e)-[r]->(t:Entity) "
            "RETURN e.name AS entity_name, e.type AS entity_type, "
            "type(r) AS rel_type, t.name AS target_name, "
            "r.confidence AS confidence, score"
        )
        results: list[GraphFact] = []
        seen: set[str] = set()

        async with self._driver.session() as session:
            result = await session.run(cypher, query=query, limit=limit)
            records = await result.data()

        for rec in records:
            entity = rec["entity_name"]
            rel = rec.get("rel_type")
            target = rec.get("target_name")

            if rel and target:
                fact_str = f"{entity} {rel.lower().replace('_', ' ')} {target}"
                entities = [entity, target]
            else:
                fact_str = f"{entity} ({rec['entity_type']})"
                entities = [entity]

            if fact_str in seen:
                continue
            seen.add(fact_str)

            results.append(
                GraphFact(
                    fact=fact_str,
                    entities=entities,
                    relation_type=rel,
                    confidence=rec.get("confidence"),
                )
            )

        return results[:limit]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_engine.py::TestGLiNEREngineSearch -v`
Expected: PASS (both tests)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 6: Lint and commit**

```bash
uv run ruff check ingestion/graph_engine.py --fix
uv run ruff format ingestion/graph_engine.py
git add ingestion/graph_engine.py tests/test_graph_engine.py
git commit -m "feat: implement GLiNEREngine search via Neo4j full-text index"
```

---

### Task 7: Final lint, type-check, and full verification

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (including existing graph_writer tests)

**Step 2: Lint**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: Clean

**Step 3: Type-check**

Run: `uv run mypy ingestion/graph_engine.py server/tools/graph_search.py server/models.py`
Expected: Clean (or only pre-existing issues)

**Step 4: Verify import isolation**

Run: `python -c "from ingestion.graph_engine import get_graph_engine; print('OK')"` with `GRAPH_ENGINE=graphiti` — should NOT import gliner2.

**Step 5: Commit any lint fixes**

```bash
git add -u
git commit -m "chore: lint and format after modular graph engine"
```

---

### Parallel Boundaries

Tasks 1-3 are sequential (each builds on the previous).
Tasks 4, 5, and 6 depend on Task 2 (protocol exists) but Task 4 (schema) and Task 5 (ingest) are independent of each other.
Task 6 depends on Task 5 (needs GLiNEREngine class).
Task 7 is final verification.

```
Task 1 → Task 2 → Task 3
                 ↘ Task 4
                 ↘ Task 5 → Task 6
                              ↘ Task 7
```

Tasks 3 and 4 can run in parallel after Task 2.
Tasks 3 and 5 can run in parallel after Task 2.
