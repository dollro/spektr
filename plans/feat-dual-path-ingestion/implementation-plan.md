# Dual-Path Ingestion & Dynamic Schema Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add live transcript ingestion (Path B) via FastAPI + Graphiti, improve GLiNER2 extraction quality via dynamic schema induction (Path A), and make all MCP search tools session-aware.

**Architecture:** Two ingestion paths sharing Qdrant + Neo4j. Path A (bulk KB) enhanced with per-document LLM schema induction before GLiNER2 extraction. Path B (live transcripts) is a new FastAPI app with immediate Qdrant upsert + background Graphiti episode ingestion. MCP tools gain optional `session_id` for dual-query behavior.

**Tech Stack:** Python 3.13, FastAPI, Qdrant, Neo4j, Graphiti, GLiNER2, Jina embeddings, Pydantic

---

## Task 1: Expand Base Schema in constants.py

**Files:**
- Modify: `config/constants.py`
- Test: `tests/test_graph_engine.py` (existing tests still pass)

**Step 1: Write a test that verifies the expanded schema**

Add to `tests/test_graph_engine.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_engine.py::TestExpandedSchema -v`
Expected: FAIL — currently 7 entity types and 8 relationship types

**Step 3: Update constants.py with expanded schema**

Replace `ENTITY_TYPES` and `RELATIONSHIP_TYPES` in `config/constants.py` with:

```python
ENTITY_TYPES: dict[str, str] = {
    "person": "A named individual, author, speaker, or public figure",
    "organization": "A company, institution, government body, or team",
    "location": "A physical place, address, region, country, or facility",
    "date_time": "A specific date, time period, deadline, or schedule reference",
    "monetary_value": "An amount of money, price, fee, budget, or financial figure",
    "document": "A named contract, agreement, report, policy, regulation, or standard",
    "product": "A named product, service, platform, or deliverable",
    "technology": "A programming language, framework, tool, protocol, or system",
    "metric": "A quantitative measure, KPI, percentage, statistic, or benchmark",
    "event": "A named meeting, conference, milestone, incident, or release",
    "legal_term": "A clause, obligation, right, liability, warranty, or legal concept",
    "role": "A job title, department, committee, or functional responsibility",
    "concept": "An abstract idea, methodology, strategy, design pattern, or practice",
    "requirement": "A specification, condition, constraint, criterion, or deliverable requirement",
}

RELATIONSHIP_TYPES: dict[str, str] = {
    "created_by": "X was created, authored, or produced by Y",
    "owned_by": "X is owned, managed, or governed by Y",
    "uses": "X uses, depends on, or integrates Y",
    "part_of": "X is a component, section, or subset of Y",
    "related_to": "X is associated with or relevant to Y",
    "measured_by": "X is measured, evaluated, or quantified by Y",
    "requires": "X requires, mandates, or depends on Y",
    "applies_to": "X applies to, governs, or regulates Y",
    "succeeds": "X replaces, supersedes, or follows Y",
    "conflicts_with": "X contradicts, opposes, or is incompatible with Y",
    "valued_at": "X has a monetary value, cost, or price of Y",
    "scheduled_for": "X is planned, due, or scheduled for Y",
}
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: ALL PASS (including existing tests — GLiNEREngine uses constants at init time, tests mock `__init__`)

**Step 5: Commit**

```bash
git add config/constants.py tests/test_graph_engine.py
git commit -m "feat: expand base schema to 14 entity types and 12 relationship types"
```

---

## Task 2: Add New Settings

**Files:**
- Modify: `config/settings.py`
- Test: `tests/test_settings.py`

**Step 1: Write tests for new settings**

Add to `tests/test_settings.py`:

```python
class TestDualPathSettings:
    def test_live_ingest_port_default(self) -> None:
        """LIVE_INGEST_PORT defaults to 8001."""
        from config.settings import Settings
        s = Settings(jina_api_key="k", neo4j_password="p")
        assert s.live_ingest_port == 8001

    def test_schema_induction_enabled_default(self) -> None:
        """SCHEMA_INDUCTION_ENABLED defaults to True."""
        from config.settings import Settings
        s = Settings(jina_api_key="k", neo4j_password="p")
        assert s.schema_induction_enabled is True

    def test_schema_induction_model_default(self) -> None:
        """SCHEMA_INDUCTION_MODEL defaults to claude-haiku."""
        from config.settings import Settings
        s = Settings(jina_api_key="k", neo4j_password="p")
        assert "haiku" in s.schema_induction_model

    def test_schema_cache_ttl_default(self) -> None:
        """SCHEMA_CACHE_TTL defaults to 3600."""
        from config.settings import Settings
        s = Settings(jina_api_key="k", neo4j_password="p")
        assert s.schema_cache_ttl == 3600
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_settings.py::TestDualPathSettings -v`
Expected: FAIL — settings don't exist yet

**Step 3: Add settings to Settings class**

Add these fields to the `Settings` class in `config/settings.py`, after the `graph_engine` line:

```python
    # Live ingestion
    live_ingest_port: int = 8001

    # Schema induction
    schema_induction_enabled: bool = True
    schema_induction_model: str = "claude-haiku-4-5-20251001"
    schema_cache_ttl: int = 3600  # seconds
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_settings.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: add settings for live ingestion and schema induction"
```

---

## Task 3: Add New Pydantic Models

**Files:**
- Modify: `server/models.py`
- Test: `tests/test_tools.py` (add model tests)

**Step 1: Write tests for new models**

Add to `tests/test_tools.py`:

```python
class TestLiveIngestModels:
    def test_transcript_chunk_model(self) -> None:
        """TranscriptChunk validates required fields."""
        from server.models import TranscriptChunk
        chunk = TranscriptChunk(
            session_id="meeting-1",
            text="Alice: Hello",
            timestamp="2026-03-06T14:30:00Z",
            speaker="Alice",
        )
        assert chunk.session_id == "meeting-1"
        assert chunk.speaker == "Alice"

    def test_transcript_chunk_optional_speaker(self) -> None:
        """TranscriptChunk speaker is optional."""
        from server.models import TranscriptChunk
        chunk = TranscriptChunk(
            session_id="meeting-1",
            text="Hello",
            timestamp="2026-03-06T14:30:00Z",
        )
        assert chunk.speaker is None

    def test_session_start_request(self) -> None:
        """SessionStartRequest validates fields."""
        from server.models import SessionStartRequest
        req = SessionStartRequest(
            session_id="meeting-1",
            metadata={"title": "Q1 Review"},
        )
        assert req.session_id == "meeting-1"

    def test_session_end_request(self) -> None:
        """SessionEndRequest defaults archive to False."""
        from server.models import SessionEndRequest
        req = SessionEndRequest(session_id="meeting-1")
        assert req.archive is False

    def test_ingest_response(self) -> None:
        """IngestResponse model."""
        from server.models import IngestResponse
        resp = IngestResponse(
            status="accepted",
            vector_indexed=True,
            graph_status="processing",
        )
        assert resp.status == "accepted"

    def test_hybrid_search_response_with_session(self) -> None:
        """HybridSearchResponse supports session_id and transcript_results."""
        from server.models import HybridSearchResponse
        resp = HybridSearchResponse(
            query="test",
            session_id="meeting-1",
            transcript_results=[],
        )
        assert resp.session_id == "meeting-1"
        assert resp.transcript_results == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py::TestLiveIngestModels -v`
Expected: FAIL — models don't exist

**Step 3: Add models to server/models.py**

Append to `server/models.py`:

```python
from datetime import datetime


class TranscriptChunk(BaseModel):
    """A single transcript chunk from a live meeting."""

    session_id: str
    text: str
    timestamp: datetime
    speaker: str | None = None


class SessionStartRequest(BaseModel):
    """Request to start a new meeting session."""

    session_id: str
    metadata: dict = {}  # type: ignore[type-arg]


class SessionEndRequest(BaseModel):
    """Request to end a meeting session."""

    session_id: str
    archive: bool = False


class IngestResponse(BaseModel):
    """Response from transcript ingestion."""

    status: str
    vector_indexed: bool
    graph_status: str
```

Also update `HybridSearchResponse` to add optional fields:

```python
class HybridSearchResponse(BaseModel):
    """Combined vector + graph search response."""

    vector_results: list[SearchResult] = []
    graph_results: list[GraphFact] = []
    transcript_results: list[SearchResult] = []
    query: str
    session_id: str | None = None
    strategy: str = "parallel"
    errors: list[str] | None = None
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tools.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server/models.py tests/test_tools.py
git commit -m "feat: add Pydantic models for live transcript ingestion"
```

---

## Task 4: Schema Inducer

**Files:**
- Create: `ingestion/schema_inducer.py`
- Test: `tests/test_schema_inducer.py`

**Step 1: Write tests**

Create `tests/test_schema_inducer.py`:

```python
"""Tests for the LLM-based schema inducer."""

from __future__ import annotations

import hashlib
import time
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

        mock_response = '{"entity_types": {"clause": "A legal clause"}, "relationship_types": {"governs": "X governs Y"}}'

        with patch(
            "ingestion.schema_inducer._call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            inducer = SchemaInducer()
            result = await inducer.induce("This agreement governs the terms...")

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema_inducer.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement schema_inducer.py**

Create `ingestion/schema_inducer.py`:

```python
"""LLM-based per-document schema induction for GLiNER2.

Analyzes a text sample from each document and proposes domain-specific
entity and relationship types. Results are cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

from config.constants import ENTITY_TYPES, RELATIONSHIP_TYPES
from config.settings import settings

logger = logging.getLogger(__name__)

_MIN_TEXT_LEN = 200


@dataclass
class InducedSchema:
    """Schema proposed by the LLM for a specific document."""

    entity_types: dict[str, str] = field(default_factory=dict)
    relationship_types: dict[str, str] = field(default_factory=dict)


@dataclass
class MergedSchema:
    """Base schema + induced types merged together."""

    entity_types: dict[str, str] = field(default_factory=dict)
    relationship_types: dict[str, str] = field(default_factory=dict)


_PROMPT_TEMPLATE = """\
You are a knowledge graph schema designer. Given the following document excerpt,
propose entity types and relationship types that would capture the key information.

Return JSON with two keys:
- "entity_types": {{"type_name": "description of what this type represents"}}
- "relationship_types": {{"rel_name": "description: X [rel] Y means..."}}

Rules:
- Propose 3-8 entity types specific to this document's domain
- Propose 3-6 relationship types
- Descriptions must be clear enough for a non-expert NER model to use
- Do not duplicate these base types (they are already included): {base_types}

Document excerpt:
---
{sample_text}
---"""


async def _call_llm(prompt: str) -> str:
    """Call the schema induction LLM (cheap/fast model)."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=settings.llm_api_key,
    )
    resp = await client.messages.create(
        model=settings.schema_induction_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


class SchemaInducer:
    """Proposes domain-specific entity/relationship types for a document."""

    def __init__(self, cache_ttl: int | None = None) -> None:
        self._cache: dict[str, tuple[InducedSchema, float]] = {}
        self._ttl = cache_ttl if cache_ttl is not None else settings.schema_cache_ttl

    @staticmethod
    def _cache_key(sample_text: str) -> str:
        """Hash first 500 chars as domain signature."""
        return hashlib.sha256(sample_text[:500].encode()).hexdigest()

    async def induce(self, sample_text: str) -> InducedSchema:
        """Analyze sample text, return entity + relationship types."""
        if len(sample_text) < _MIN_TEXT_LEN:
            return InducedSchema()

        key = self._cache_key(sample_text)
        now = time.monotonic()

        # Check cache
        if key in self._cache:
            schema, cached_at = self._cache[key]
            if now - cached_at < self._ttl:
                return schema

        # Call LLM
        base_type_names = ", ".join(ENTITY_TYPES.keys())
        prompt = _PROMPT_TEMPLATE.format(
            base_types=base_type_names,
            sample_text=sample_text[:2000],
        )

        try:
            raw = await _call_llm(prompt)
            # Extract JSON from response (may be wrapped in markdown)
            json_str = raw
            if "```" in raw:
                json_str = raw.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            data = json.loads(json_str)
            schema = InducedSchema(
                entity_types=data.get("entity_types", {}),
                relationship_types=data.get("relationship_types", {}),
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            logger.warning("Schema induction returned malformed response")
            schema = InducedSchema()
        except Exception:
            logger.exception("Schema induction LLM call failed")
            schema = InducedSchema()

        self._cache[key] = (schema, now)
        return schema

    def merge_with_base(self, induced: InducedSchema) -> MergedSchema:
        """Merge induced types with base schema. Base types take priority."""
        merged_entities = dict(ENTITY_TYPES)
        for name, desc in induced.entity_types.items():
            if name not in merged_entities:
                merged_entities[name] = desc

        merged_rels = dict(RELATIONSHIP_TYPES)
        for name, desc in induced.relationship_types.items():
            if name not in merged_rels:
                merged_rels[name] = desc

        return MergedSchema(
            entity_types=merged_entities,
            relationship_types=merged_rels,
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_schema_inducer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ingestion/schema_inducer.py tests/test_schema_inducer.py
git commit -m "feat: add LLM-based per-document schema inducer with caching"
```

---

## Task 5: GLiNEREngine Accepts Dynamic Schema

**Files:**
- Modify: `ingestion/graph_engine.py`
- Modify: `tests/test_graph_engine.py`

**Step 1: Write tests for dynamic schema support**

Add to `tests/test_graph_engine.py`:

```python
class TestGLiNEREngineDynamicSchema:
    @pytest.mark.asyncio
    async def test_ingest_with_custom_schema(self) -> None:
        """GLiNEREngine.ingest uses provided schema instead of base."""
        from ingestion.graph_engine import GLiNEREngine
        from ingestion.schema_inducer import MergedSchema

        long_text = (
            "The Master Service Agreement between Acme Corp and Widget Inc "
            "establishes payment terms of net-30 for all deliverables under "
            "the contract scope including software development and consulting."
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
        mock_extractor.extract.return_value = {"entities": {}, "relation_extraction": {}}

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_engine.py::TestGLiNEREngineDynamicSchema -v`
Expected: FAIL — `ingest()` doesn't accept `schema` parameter

**Step 3: Modify GraphEngine protocol and GLiNEREngine**

In `ingestion/graph_engine.py`:

1. Update `GraphEngine` protocol `ingest` signature:
```python
class GraphEngine(Protocol):
    async def ingest(
        self,
        chunks: list[TextChunk],
        source_key: str,
        schema: MergedSchema | None = None,
    ) -> None: ...
```

2. Add import at top:
```python
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
if TYPE_CHECKING:
    from ingestion.schema_inducer import MergedSchema
```

3. Update `GraphitiEngine.ingest` to accept and ignore `schema`:
```python
async def ingest(
    self,
    chunks: list[TextChunk],
    source_key: str,
    schema: MergedSchema | None = None,
) -> None:
    await self._writer.ingest_bulk(chunks=chunks, source_key=source_key)
```

4. Update `GLiNEREngine.ingest` to use dynamic schema when provided:
```python
async def ingest(
    self,
    chunks: list[TextChunk],
    source_key: str,
    schema: MergedSchema | None = None,
) -> None:
    if not chunks:
        return

    merged_texts = self._merge_chunks(chunks)
    if not merged_texts:
        return

    # Build schema for extraction
    if schema is not None:
        active_schema = (
            self._extractor.create_schema()
            .entities(schema.entity_types)
            .relations(schema.relationship_types)
        )
    else:
        active_schema = self._schema

    async with self._driver.session() as session:
        for text in merged_texts:
            result = self._extractor.extract(text, active_schema)
            # ... rest of extraction logic unchanged
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_graph_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ingestion/graph_engine.py tests/test_graph_engine.py
git commit -m "feat: GLiNEREngine accepts optional dynamic schema for extraction"
```

---

## Task 6: Integrate Schema Inducer into Pipeline

**Files:**
- Modify: `ingestion/pipeline.py`
- Test: `tests/test_pipeline_chunking.py` (or new test file)

**Step 1: Write test**

Create `tests/test_schema_induction_pipeline.py`:

```python
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
            patch("config.settings.settings") as mock_settings,
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
        assert call_kwargs.kwargs.get("schema") == merged or call_kwargs[1].get("schema") == merged

    @pytest.mark.asyncio
    async def test_ingest_to_graph_skips_inducer_for_graphiti(self) -> None:
        """Graphiti engine ignores schema induction (it has its own LLM)."""
        chunks = [
            TextChunk(text="x" * 300, chunk_index=0, page_number=1),
        ]
        mock_engine = AsyncMock()

        with patch("config.settings.settings") as mock_settings:
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

        with patch("config.settings.settings") as mock_settings:
            mock_settings.schema_induction_enabled = False
            mock_settings.graph_engine = "gliner"

            from ingestion.pipeline import _ingest_to_graph_with_schema

            await _ingest_to_graph_with_schema("doc.pdf", chunks, mock_engine)

        mock_engine.ingest.assert_called_once()
        call_kwargs = mock_engine.ingest.call_args
        assert call_kwargs.kwargs.get("schema") is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema_induction_pipeline.py -v`
Expected: FAIL — `_ingest_to_graph_with_schema` doesn't exist

**Step 3: Add schema induction to pipeline**

In `ingestion/pipeline.py`:

1. Add import at top (after other imports):
```python
from ingestion.schema_inducer import SchemaInducer
```

2. Add a module-level inducer singleton:
```python
_schema_inducer: SchemaInducer | None = None

def _get_schema_inducer() -> SchemaInducer:
    global _schema_inducer  # noqa: PLW0603
    if _schema_inducer is None:
        _schema_inducer = SchemaInducer()
    return _schema_inducer
```

3. Replace `_ingest_to_graph` with `_ingest_to_graph_with_schema`:
```python
async def _ingest_to_graph_with_schema(
    source_file: str,
    chunks: list[TextChunk],
    engine: GraphEngine,
) -> None:
    """Ingest chunks via graph engine, optionally with induced schema."""
    schema = None

    if (
        settings.schema_induction_enabled
        and settings.graph_engine == "gliner"
        and chunks
    ):
        # Use first chunk's text as sample for schema induction
        sample = " ".join(
            (c.contextualized_text or c.text) for c in chunks[:3]
        )
        inducer = _get_schema_inducer()
        induced = await inducer.induce(sample)
        if induced.entity_types or induced.relationship_types:
            schema = inducer.merge_with_base(induced)

    await engine.ingest(chunks=chunks, source_key=source_file, schema=schema)
```

4. Update the call site in `_process_all_pages()` (around line 564-568):
Change `_ingest_to_graph` to `_ingest_to_graph_with_schema`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_schema_induction_pipeline.py tests/test_graph_engine.py -v`
Expected: ALL PASS

**Step 5: Run full test suite to check for regressions**

Run: `uv run pytest --ignore=tests/test_integration_ingestion.py --ignore=tests/test_integration_delete.py --ignore=tests/test_e2e.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add ingestion/pipeline.py tests/test_schema_induction_pipeline.py
git commit -m "feat: integrate schema induction into bulk ingestion pipeline"
```

---

## Task 7: Live Transcript Ingestion Endpoint

**Files:**
- Create: `ingestion/live_ingest.py`
- Test: `tests/test_live_ingest.py`

**Step 1: Write tests**

Create `tests/test_live_ingest.py`:

```python
"""Tests for live transcript ingestion FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.upsert = MagicMock()
    client.scroll = MagicMock(return_value=([], None))
    client.delete = MagicMock()
    return client


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=[[0.1] * 512])
    embedder.close = AsyncMock()
    return embedder


@pytest.fixture
def mock_graphiti():
    client = AsyncMock()
    client.add_episode = AsyncMock()
    return client


class TestSessionStart:
    @pytest.mark.asyncio
    async def test_start_session(self, mock_qdrant, mock_embedder) -> None:
        """POST /session/start creates a new session."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/session/start", json={
                    "session_id": "meeting-1",
                    "metadata": {"title": "Test Meeting"},
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "meeting-1"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_start_duplicate_session_rejected(self, mock_qdrant, mock_embedder) -> None:
        """Starting a session while one is active returns 409."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app
            import ingestion.live_ingest as mod
            mod._active_session = {"session_id": "existing"}

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/session/start", json={
                    "session_id": "meeting-2",
                })

            mod._active_session = None  # cleanup

        assert resp.status_code == 409


class TestIngestTranscript:
    @pytest.mark.asyncio
    async def test_ingest_transcript_chunk(
        self, mock_qdrant, mock_embedder, mock_graphiti,
    ) -> None:
        """POST /ingest/transcript embeds and upserts to Qdrant."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.get_graphiti", return_value=mock_graphiti),
        ):
            from ingestion.live_ingest import app
            import ingestion.live_ingest as mod
            mod._active_session = {"session_id": "meeting-1"}

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/ingest/transcript", json={
                    "session_id": "meeting-1",
                    "text": "Alice: Hello everyone.",
                    "timestamp": "2026-03-06T14:30:00Z",
                    "speaker": "Alice",
                })

            mod._active_session = None

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["vector_indexed"] is True
        assert data["graph_status"] == "processing"
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_without_active_session_rejected(
        self, mock_qdrant, mock_embedder,
    ) -> None:
        """Ingesting without an active session returns 404."""
        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app
            import ingestion.live_ingest as mod
            mod._active_session = None

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/ingest/transcript", json={
                    "session_id": "meeting-1",
                    "text": "Hello",
                    "timestamp": "2026-03-06T14:30:00Z",
                })

        assert resp.status_code == 404


class TestSessionEnd:
    @pytest.mark.asyncio
    async def test_end_session_archive(self, mock_qdrant, mock_embedder) -> None:
        """POST /session/end with archive=true updates points."""
        mock_qdrant.scroll = MagicMock(return_value=([
            MagicMock(id="point-1"),
        ], None))

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
        ):
            from ingestion.live_ingest import app
            import ingestion.live_ingest as mod
            mod._active_session = {"session_id": "meeting-1"}

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/session/end", json={
                    "session_id": "meeting-1",
                    "archive": True,
                })

            assert mod._active_session is None

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "archived"

    @pytest.mark.asyncio
    async def test_end_session_discard(
        self, mock_qdrant, mock_embedder, mock_graphiti,
    ) -> None:
        """POST /session/end with archive=false deletes points."""
        mock_qdrant.scroll = MagicMock(return_value=([
            MagicMock(id="point-1"),
        ], None))

        with (
            patch("ingestion.live_ingest._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.live_ingest._get_embedder", return_value=mock_embedder),
            patch("ingestion.live_ingest.get_graphiti", return_value=mock_graphiti),
        ):
            from ingestion.live_ingest import app
            import ingestion.live_ingest as mod
            mod._active_session = {"session_id": "meeting-1"}

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/session/end", json={
                    "session_id": "meeting-1",
                    "archive": False,
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "discarded"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_live_ingest.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement live_ingest.py**

Create `ingestion/live_ingest.py`:

```python
"""Live transcript ingestion via FastAPI.

Provides HTTP endpoints for real-time meeting transcript ingestion:
- POST /session/start — create a meeting session
- POST /ingest/transcript — ingest a transcript chunk
- POST /session/end — end session (archive or discard)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION
from config.settings import settings
from ingestion.embedder import Embedder, create_embedder
from ingestion.graphiti_client import get_graphiti
from server.models import (
    IngestResponse,
    SessionEndRequest,
    SessionStartRequest,
    TranscriptChunk,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Spektr Live Ingest")

_qdrant_client: QdrantClient | None = None
_embedder: Embedder | None = None
_active_session: dict | None = None  # type: ignore[type-arg]


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def _get_embedder() -> Embedder:
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


def _make_point_id(key: str) -> str:
    """Deterministic UUID from a string key."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@app.post("/session/start")
async def start_session(req: SessionStartRequest) -> dict:  # type: ignore[type-arg]
    """Create a new meeting session."""
    global _active_session  # noqa: PLW0603

    if _active_session is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Active session already exists: {_active_session['session_id']}",
        )

    _active_session = {
        "session_id": req.session_id,
        "metadata": req.metadata,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }

    logger.info("Session started: %s", req.session_id)
    return {
        "session_id": req.session_id,
        "status": "active",
        "created_at": _active_session["created_at"],
    }


@app.post("/ingest/transcript")
async def ingest_transcript(chunk: TranscriptChunk) -> IngestResponse:
    """Ingest a single transcript chunk."""
    if _active_session is None:
        raise HTTPException(status_code=404, detail="No active session")
    if _active_session["session_id"] != chunk.session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Session mismatch: active={_active_session['session_id']}",
        )

    # 1. Embed and upsert to Qdrant (immediate)
    embedder = _get_embedder()
    vectors = await embedder.embed_text([chunk.text])
    point_key = f"{chunk.session_id}::{chunk.timestamp.isoformat()}"

    _get_qdrant_client().upsert(
        collection_name=DENSE_COLLECTION,
        points=[
            models.PointStruct(
                id=_make_point_id(point_key),
                vector=vectors[0],
                payload={
                    "source_file": f"session:{chunk.session_id}",
                    "content_type": "transcript",
                    "is_live": True,
                    "session_id": chunk.session_id,
                    "speaker": chunk.speaker,
                    "timestamp": chunk.timestamp.isoformat(),
                    "text_content": chunk.text,
                    "page_number": 0,
                    "metadata": {},
                },
            ),
        ],
    )

    # 2. Graphiti ingest (background)
    asyncio.create_task(_graphiti_ingest(chunk))

    return IngestResponse(
        status="accepted",
        vector_indexed=True,
        graph_status="processing",
    )


async def _graphiti_ingest(chunk: TranscriptChunk) -> None:
    """Background task: ingest chunk as Graphiti episode."""
    try:
        client = await get_graphiti()
        episode_name = f"{chunk.session_id}:t{chunk.timestamp.isoformat()}"
        await client.add_episode(
            name=episode_name,
            episode_body=chunk.text,
            source_description=f"Meeting transcript, speaker: {chunk.speaker or 'unknown'}",
            reference_time=chunk.timestamp,
            group_id=chunk.session_id,
        )
        logger.info("Graphiti episode ingested: %s", episode_name)
    except Exception:
        logger.exception("Graphiti background ingest failed for %s", chunk.session_id)


@app.post("/session/end")
async def end_session(req: SessionEndRequest) -> dict:  # type: ignore[type-arg]
    """End a session: archive (keep data) or discard (delete data)."""
    global _active_session  # noqa: PLW0603

    if _active_session is None or _active_session["session_id"] != req.session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    qdrant = _get_qdrant_client()

    if req.archive:
        # Set is_live=false on all session points
        points, _ = qdrant.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=req.session_id),
                    ),
                ],
            ),
            limit=10000,
        )
        if points:
            qdrant.set_payload(
                collection_name=DENSE_COLLECTION,
                payload={"is_live": False},
                points=[p.id for p in points],
            )
        status = "archived"
    else:
        # Delete all points for this session
        qdrant.delete(
            collection_name=DENSE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=req.session_id),
                        ),
                    ],
                ),
            ),
        )
        # Delete Graphiti data
        try:
            client = await get_graphiti()
            # Graphiti group deletion
            await client.delete_group(req.session_id)
        except Exception:
            logger.exception("Failed to delete Graphiti group %s", req.session_id)
        status = "discarded"

    _active_session = None
    logger.info("Session %s: %s", req.session_id, status)
    return {"session_id": req.session_id, "status": status}
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_live_ingest.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ingestion/live_ingest.py tests/test_live_ingest.py
git commit -m "feat: add live transcript ingestion FastAPI endpoints"
```

---

## Task 8: Session-Aware Vector Search

**Files:**
- Modify: `server/tools/vector_search.py`
- Test: `tests/test_tools.py`

**Step 1: Write tests**

Add to `tests/test_tools.py`:

```python
class TestSessionAwareVectorSearch:
    async def test_vector_search_with_session_id(self) -> None:
        """When session_id is set, runs dual Qdrant queries."""
        from server.models import SearchResult

        mock_embedder = MagicMock()
        mock_embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

        # Mock Qdrant to return different results per filter
        mock_qdrant = MagicMock()
        transcript_point = MagicMock()
        transcript_point.score = 0.9
        transcript_point.payload = {
            "text": "Alice: Contract terms",
            "source_file": "session:meeting-1",
            "page_number": 0,
            "content_type": "transcript",
            "metadata": {},
            "timestamp": "2026-03-06T14:32:00Z",
            "speaker": "Alice",
        }
        kb_point = MagicMock()
        kb_point.score = 0.85
        kb_point.payload = {
            "text": "Payment policy doc",
            "source_file": "policies/payment.pdf",
            "page_number": 1,
            "content_type": "text_chunk",
            "metadata": {},
        }

        mock_response_transcript = MagicMock()
        mock_response_transcript.points = [transcript_point]
        mock_response_kb = MagicMock()
        mock_response_kb.points = [kb_point]

        mock_qdrant.query_points = MagicMock(
            side_effect=[mock_response_transcript, mock_response_kb]
        )

        with (
            patch("server.tools.vector_search._qdrant_client", mock_qdrant),
            patch("server.tools.vector_search._embedder", mock_embedder),
        ):
            from server.tools.vector_search import vector_search
            results = await vector_search("contract", session_id="meeting-1")

        assert len(results) == 2
        assert mock_qdrant.query_points.call_count == 2

    async def test_vector_search_without_session_unchanged(self) -> None:
        """Without session_id, behavior is unchanged (single query)."""
        mock_embedder = MagicMock()
        mock_embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

        mock_response = MagicMock()
        mock_response.points = []

        mock_qdrant = MagicMock()
        mock_qdrant.query_points = MagicMock(return_value=mock_response)

        with (
            patch("server.tools.vector_search._qdrant_client", mock_qdrant),
            patch("server.tools.vector_search._embedder", mock_embedder),
        ):
            from server.tools.vector_search import vector_search
            results = await vector_search("test")

        assert mock_qdrant.query_points.call_count == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py::TestSessionAwareVectorSearch -v`
Expected: FAIL — `vector_search` doesn't accept `session_id`

**Step 3: Modify vector_search**

In `server/tools/vector_search.py`, update the function signature and add dual-query logic:

```python
async def vector_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,
    *,
    _skip_rerank: bool = False,
) -> list[dict]:  # type: ignore[type-arg]
    """Search documents by semantic similarity.

    When session_id is provided, runs two parallel queries:
    1. Transcript chunks from the active session
    2. Bulk KB results (excluding live data)

    Args:
        query: Natural language search query.
        limit: Maximum number of results (default 10).
        content_type: Optional MIME type filter.
        source_file: Optional source file name filter.
        session_id: Optional session ID for live meeting context.
        _skip_rerank: Internal flag — skip reranking.
    """
    if not query or not query.strip():
        return []
    limit = max(1, min(limit, 100))

    try:
        query_vector = await _get_embedder().embed_text_query(query)
        qdrant = _get_qdrant_client()

        if session_id is not None:
            # Dual query: transcript + KB
            transcript_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    ),
                ],
            )
            kb_conditions: list[models.Condition] = [
                models.FieldCondition(
                    key="is_live",
                    match=models.MatchValue(value=False),
                ),
            ]
            # Add user filters to KB query only
            if content_type is not None:
                kb_conditions.append(
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value=content_type),
                    )
                )
            if source_file is not None:
                kb_conditions.append(
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_file),
                    )
                )

            kb_filter = models.Filter(
                should=[
                    models.Filter(must=kb_conditions),
                    # Also match points without is_live field (bulk KB)
                    models.Filter(
                        must_not=[
                            models.HasIdCondition(has_id=[]),  # dummy
                        ],
                        must=[
                            models.IsNullCondition(
                                is_null=models.PayloadField(key="is_live"),
                            ),
                        ],
                    ),
                ],
            )

            transcript_resp = qdrant.query_points(
                collection_name=DENSE_COLLECTION,
                query=query_vector,
                query_filter=transcript_filter,
                limit=limit,
                with_payload=True,
            )
            kb_resp = qdrant.query_points(
                collection_name=DENSE_COLLECTION,
                query=query_vector,
                query_filter=kb_filter,
                limit=limit,
                with_payload=True,
            )

            results: list[dict] = []  # type: ignore[type-arg]
            # Transcript results sorted by timestamp
            transcript_points = sorted(
                transcript_resp.points,
                key=lambda p: (p.payload or {}).get("timestamp", ""),
            )
            for point in transcript_points:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        score=point.score,
                        text=payload.get("text_content", ""),
                        source_file=payload.get("source_file", ""),
                        page_number=payload.get("page_number", 0),
                        content_type=payload.get("content_type", ""),
                        metadata={
                            **payload.get("metadata", {}),
                            "source_type": "transcript",
                            "speaker": payload.get("speaker"),
                            "timestamp": payload.get("timestamp"),
                        },
                    ).model_dump()
                )
            # KB results sorted by score
            for point in kb_resp.points:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        score=point.score,
                        text=payload.get("text_content", payload.get("text", "")),
                        source_file=payload.get("source_file", ""),
                        page_number=payload.get("page_number", 0),
                        content_type=payload.get("content_type", ""),
                        metadata=payload.get("metadata", {}),
                    ).model_dump()
                )
            return results

        # Standard single-query path (no session)
        conditions: list[models.FieldCondition] = []
        if content_type is not None:
            conditions.append(
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value=content_type),
                )
            )
        if source_file is not None:
            conditions.append(
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        response = qdrant.query_points(
            collection_name=DENSE_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    score=point.score,
                    text=payload.get("text", ""),
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 0),
                    content_type=payload.get("content_type", ""),
                    metadata=payload.get("metadata", {}),
                ).model_dump()
            )

        if settings.rerank_enabled and results and not _skip_rerank:
            from server.tools.reranker import rerank
            results = await rerank(query, results, top_k=limit)

        return results
    except Exception as exc:
        logger.exception("vector_search failed")
        return [
            {
                "error": f"vector_search failed: {exc}",
                "query": query,
                "partial_results": [],
            }
        ]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tools.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server/tools/vector_search.py tests/test_tools.py
git commit -m "feat: session-aware vector search with dual Qdrant queries"
```

---

## Task 9: Session-Aware Graph Search

**Files:**
- Modify: `server/tools/graph_search.py`
- Test: `tests/test_tools.py`

**Step 1: Write tests**

Add to `tests/test_tools.py`:

```python
class TestSessionAwareGraphSearch:
    async def test_graph_search_with_session_id(self) -> None:
        """When session_id is set, queries both Graphiti and GLiNER."""
        from server.models import GraphFact

        graphiti_facts = [
            GraphFact(
                fact="Contract valued at 1.2M",
                source="graphiti",
                created_at="2026-03-06T14:32:00Z",
            ),
        ]
        gliner_facts = [
            GraphFact(
                fact="Acme Corp (organization)",
                entities=["Acme Corp"],
                confidence=0.9,
            ),
        ]

        mock_graphiti = AsyncMock()
        mock_edge = MagicMock()
        mock_edge.fact = "Contract valued at 1.2M"
        mock_edge.source_description = "graphiti"
        mock_edge.created_at = "2026-03-06T14:32:00Z"
        mock_edge.expired_at = None
        mock_graphiti.search = AsyncMock(return_value=[mock_edge])

        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=gliner_facts)

        with (
            patch("server.tools.graph_search.get_graphiti", return_value=mock_graphiti),
            patch("server.tools.graph_search.get_graph_engine", return_value=mock_engine),
        ):
            from server.tools.graph_search import graph_search
            results = await graph_search("contract", session_id="meeting-1")

        # Should have results from both engines
        assert len(results) >= 2

    async def test_graph_search_without_session_unchanged(self) -> None:
        """Without session_id, behavior delegates to engine only."""
        from server.models import GraphFact

        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=[
            GraphFact(fact="Test fact"),
        ])

        with patch(
            "server.tools.graph_search.get_graph_engine",
            return_value=mock_engine,
        ):
            from server.tools.graph_search import graph_search
            results = await graph_search("test")

        assert len(results) == 1
        mock_engine.search.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py::TestSessionAwareGraphSearch -v`
Expected: FAIL — `graph_search` doesn't accept `session_id`

**Step 3: Modify graph_search**

In `server/tools/graph_search.py`:

```python
"""Knowledge graph search tool for MCP server.

Engine-agnostic: dispatches to whichever GraphEngine is configured
via the GRAPH_ENGINE setting. When session_id is provided, also
queries Graphiti for temporal meeting context.
"""

from __future__ import annotations

import logging

from ingestion.graph_engine import get_graph_engine
from server.models import GraphFact

logger = logging.getLogger(__name__)


async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
    session_id: str | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """Search the knowledge graph for entities and relationships.

    When session_id is provided, queries both:
    1. Graphiti (temporal facts from live meeting, filtered by group_id)
    2. The configured graph engine (GLiNER2 entities from bulk KB)

    Args:
        query: Search text.
        search_type: 'entity' (default). Reserved for future modes.
        limit: Maximum results (default 10).
        session_id: Optional session ID for live meeting context.
    """
    if not query or not query.strip():
        return []
    if search_type != "entity":
        raise ValueError(
            f"search_type='{search_type}' is not yet implemented. Use 'entity' instead."
        )
    limit = max(1, min(limit, 100))

    try:
        results: list[dict] = []  # type: ignore[type-arg]

        if session_id is not None:
            # Query Graphiti for temporal meeting facts
            try:
                from ingestion.graphiti_client import get_graphiti

                client = await get_graphiti()
                edges = await client.search(query, group_ids=[session_id])
                for edge in edges[:limit]:
                    results.append(
                        GraphFact(
                            fact=edge.fact,
                            source=edge.source_description,
                            created_at=str(edge.created_at),
                            expired_at=(
                                str(edge.expired_at) if edge.expired_at else None
                            ),
                        ).model_dump()
                    )
            except Exception:
                logger.exception("Graphiti search failed for session %s", session_id)

        # Query the configured graph engine (GLiNER2 or Graphiti)
        engine = get_graph_engine()
        engine_results = await engine.search(query, limit=limit)
        results.extend(r.model_dump() for r in engine_results)

        return results
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

**Step 4: Run tests**

Run: `uv run pytest tests/test_tools.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server/tools/graph_search.py tests/test_tools.py
git commit -m "feat: session-aware graph search queries both Graphiti and GLiNER"
```

---

## Task 10: Session-Aware Hybrid Search

**Files:**
- Modify: `server/tools/hybrid_search.py`
- Test: `tests/test_tools.py`

**Step 1: Write tests**

Add to `tests/test_tools.py`:

```python
class TestSessionAwareHybridSearch:
    async def test_hybrid_search_with_session_id(self) -> None:
        """Hybrid search passes session_id to both sub-searches."""
        mock_vector = AsyncMock(return_value=[
            {"score": 0.9, "text": "transcript", "metadata": {"source_type": "transcript"}},
            {"score": 0.8, "text": "kb doc", "metadata": {}},
        ])
        mock_graph = AsyncMock(return_value=[
            {"fact": "X related Y"},
        ])

        with (
            patch("server.tools.hybrid_search.vector_search", mock_vector),
            patch("server.tools.hybrid_search.graph_search", mock_graph),
        ):
            from server.tools.hybrid_search import hybrid_search
            result = await hybrid_search("test", session_id="meeting-1")

        # vector_search should receive session_id
        mock_vector.assert_called_once()
        assert mock_vector.call_args.kwargs.get("session_id") == "meeting-1"
        # graph_search should receive session_id
        mock_graph.assert_called_once()
        assert mock_graph.call_args.kwargs.get("session_id") == "meeting-1"
        assert result["session_id"] == "meeting-1"

    async def test_hybrid_search_separates_transcript_results(self) -> None:
        """Hybrid search separates transcript from KB results."""
        mock_vector = AsyncMock(return_value=[
            {"score": 0.9, "text": "transcript chunk", "metadata": {"source_type": "transcript"}},
            {"score": 0.8, "text": "kb doc", "metadata": {}},
        ])
        mock_graph = AsyncMock(return_value=[])

        with (
            patch("server.tools.hybrid_search.vector_search", mock_vector),
            patch("server.tools.hybrid_search.graph_search", mock_graph),
        ):
            from server.tools.hybrid_search import hybrid_search
            result = await hybrid_search("test", session_id="meeting-1")

        assert len(result["transcript_results"]) == 1
        assert len(result["vector_results"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py::TestSessionAwareHybridSearch -v`
Expected: FAIL

**Step 3: Modify hybrid_search**

In `server/tools/hybrid_search.py`:

```python
"""Combined vector + knowledge graph search tool.

Runs dense vector search and graph search in parallel,
returning results from both for comprehensive retrieval.
When session_id is provided, separates transcript results
from KB results.
"""

from __future__ import annotations

import asyncio
import logging

from config.settings import settings
from server.tools.graph_search import graph_search
from server.tools.vector_search import vector_search

logger = logging.getLogger(__name__)


async def hybrid_search(
    query: str,
    limit: int = 10,
    session_id: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """Combined vector and knowledge graph search.

    Runs semantic vector search and graph entity search in
    parallel, returning results from both sources.

    When session_id is provided, transcript results are separated
    from KB results for clearer presentation to the LLM.

    Args:
        query: Natural language search query.
        limit: Max results per backend (default 10).
        session_id: Optional session ID for live meeting context.
    """
    if not query or not query.strip():
        return {
            "vector_results": [],
            "graph_results": [],
            "transcript_results": [],
            "query": query,
            "session_id": session_id,
            "strategy": "parallel",
        }
    limit = max(1, min(limit, 100))

    vector_task = asyncio.create_task(
        vector_search(query, limit=limit, session_id=session_id, _skip_rerank=True)
    )
    graph_task = asyncio.create_task(
        graph_search(query, limit=limit, session_id=session_id)
    )

    vector_results: list[dict] = []  # type: ignore[type-arg]
    graph_results: list[dict] = []  # type: ignore[type-arg]
    errors: list[str] = []

    try:
        vector_results = await vector_task
    except Exception as exc:
        logger.exception("Vector search failed in hybrid_search")
        vector_results = [{"error": "Vector search unavailable"}]
        errors.append(f"vector_search: {exc}")

    try:
        graph_results = await graph_task
    except Exception as exc:
        logger.exception("Graph search failed in hybrid_search")
        graph_results = [{"error": "Graph search unavailable"}]
        errors.append(f"graph_search: {exc}")

    # Separate transcript from KB results when session is active
    transcript_results: list[dict] = []  # type: ignore[type-arg]
    kb_results: list[dict] = []  # type: ignore[type-arg]

    has_vector_error = any("error" in r for r in vector_results)
    if session_id and not has_vector_error:
        for r in vector_results:
            meta = r.get("metadata", {})
            if meta.get("source_type") == "transcript":
                transcript_results.append(r)
            else:
                kb_results.append(r)
    else:
        kb_results = vector_results

    if settings.rerank_enabled and kb_results:
        has_error = any("error" in r for r in kb_results)
        if not has_error:
            from server.tools.reranker import rerank

            try:
                kb_results = await rerank(query, kb_results, top_k=limit)
            except Exception as exc:
                logger.warning("Rerank failed in hybrid: %s", exc)

    # Deduplicate graph facts
    has_graph_error = any("error" in r for r in graph_results)
    if not has_vector_error and not has_graph_error:
        vector_sources = {r.get("source_file") for r in kb_results if r.get("source_file")}
        graph_results = [g for g in graph_results if g.get("source") not in vector_sources]

    result: dict = {  # type: ignore[type-arg]
        "vector_results": kb_results,
        "transcript_results": transcript_results,
        "graph_results": graph_results,
        "query": query,
        "session_id": session_id,
        "strategy": "parallel",
    }
    if errors:
        result["errors"] = errors
    return result
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_tools.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server/tools/hybrid_search.py tests/test_tools.py
git commit -m "feat: session-aware hybrid search with transcript separation"
```

---

## Task 11: Mount Live Ingest in MCP Server

**Files:**
- Modify: `server/mcp_server.py`

**Step 1: Write test**

Add to `tests/test_tools.py`:

```python
class TestLiveIngestMount:
    def test_live_ingest_app_importable(self) -> None:
        """Live ingest FastAPI app can be imported."""
        from ingestion.live_ingest import app
        assert app is not None
        assert app.title == "Spektr Live Ingest"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_tools.py::TestLiveIngestMount -v`
Expected: PASS (app already exists from Task 7)

**Step 3: Document the mount approach**

The MCP server (FastMCP on SSE/stdio) and the live ingest server (FastAPI on HTTP) run as **separate processes**. The MCP server doesn't mount the FastAPI app — they are independent services.

Update `server/mcp_server.py` to log the live ingest port:

```python
if __name__ == "__main__":
    logger.info(
        "MCP server starting on %s:%d with tools: %s",
        settings.mcp_transport,
        settings.mcp_port,
        ", ".join(_REGISTERED_TOOLS),
    )
    logger.info(
        "Live ingest available on port %d (run separately)",
        settings.live_ingest_port,
    )
    mcp.run(
        transport=settings.mcp_transport,
        port=settings.mcp_port,
    )
```

**Step 4: Commit**

```bash
git add server/mcp_server.py tests/test_tools.py
git commit -m "chore: document live ingest as separate process"
```

---

## Task 12: Lint, Type Check, Full Test Suite

**Step 1: Run linter**

Run: `uv run ruff check .`
Fix any issues.

**Step 2: Run formatter**

Run: `uv run ruff format .`

**Step 3: Run type checker**

Run: `uv run mypy .`
Fix any type errors in new/modified files.

**Step 4: Run full test suite**

Run: `uv run pytest --ignore=tests/test_integration_ingestion.py --ignore=tests/test_integration_delete.py --ignore=tests/test_e2e.py -v`
Expected: ALL PASS

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint and format after dual-path ingestion"
```

---

## Task 13: Update Documentation

**Files:**
- Modify: `CLAUDE.md` (if layout changed)
- Modify: `docs/` pages as needed

**Step 1: Update CLAUDE.md project layout**

Add `ingestion/live_ingest.py` and `ingestion/schema_inducer.py` to the layout section.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update project layout for dual-path ingestion"
```

---

## Dependency Summary

```
Task 1 (constants) ──────────────┐
Task 2 (settings) ───────────────┤
Task 3 (models) ─────────────────┤
                                  ├─→ Task 4 (schema inducer) ──→ Task 5 (GLiNER schema) ──→ Task 6 (pipeline integration)
                                  ├─→ Task 7 (live ingest) ──────────────────────────────────→ Task 11 (mount)
                                  ├─→ Task 8 (vector search)
                                  ├─→ Task 9 (graph search)
                                  └─→ Task 10 (hybrid search)
                                                                                               ↓
                                                                                         Task 12 (lint/test)
                                                                                               ↓
                                                                                         Task 13 (docs)
```

**Parallelizable:** Tasks 1-3 can run in parallel. Tasks 4, 7, 8, 9, 10 can run in parallel (after 1-3). Tasks 5-6 depend on 4. Task 11 depends on 7. Tasks 12-13 are final.
