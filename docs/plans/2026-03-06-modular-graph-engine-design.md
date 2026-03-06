# Modular Graph Engine Design

**Date:** 2026-03-06
**Status:** Approved
**Branch:** TBD (feat/modular-graph-engine)

## Problem

The current knowledge graph pipeline is tightly coupled to Graphiti. Ingestion takes ~29 minutes for a 25-page PDF due to ~160 LLM API calls. There is no way to swap the extraction engine without rewriting pipeline and search code.

## Goal

Make the graph engine pluggable via a single config setting (`GRAPH_ENGINE`). Support both Graphiti (LLM-based, rich temporal extraction) and GLiNER2 (local CPU model, zero API cost, seconds not minutes). The rest of the system (MCP tools, hybrid search, agent) should be engine-agnostic.

## Speed & Effectiveness Analysis

| Metric | Graphiti | GLiNER2 |
|-|-|-|
| 25-page PDF (74 chunks) | ~29 min, ~160 LLM calls | Est. 5-15 seconds, 0 API calls |
| NER quality (zero-shot F1) | GPT-4o level (~0.60) | 0.59 F1 (CrossNER benchmark) |
| Inference per chunk | 358ms + API latency | 130ms CPU (single forward pass) |
| Cost per document | ~$0.50-2.00 LLM API | $0.00 (local) |
| Model size | N/A (API) | 205MB, <500MB RAM |
| Context window | Full LLM context | 2,048 tokens (fine for chunks) |
| Entity deduplication | Automatic (Graphiti resolves) | MERGE on normalized (name, type) |
| Temporal tracking | Built-in | ingested_at only |
| Relation extraction maturity | Battle-tested via LLM | Newer, schema-driven, typed tuples |

GLiNER2 matches GPT-4o NER quality at 2.6x faster on CPU. Relation extraction is less battle-tested but sufficient for most RAG use cases. Keeping both engines toggleable lets users choose cost/quality tradeoff.

Sources:
- GLiNER2 paper: https://arxiv.org/html/2507.18546v1
- GLiNER2 repo: https://github.com/fastino-ai/GLiNER2
- GLiNER vs LLM evaluation: https://sease.io/2025/10/gliner-as-an-alternative-to-llms-for-query-parsing-evaluation.html

## Architecture

### Unified GraphEngine Protocol

```python
class GraphEngine(Protocol):
    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None: ...
    async def search(self, query: str, limit: int = 10) -> list[GraphFact]: ...
    async def close(self) -> None: ...
```

Factory function dispatches based on `settings.graph_engine`:

```python
def get_graph_engine() -> GraphEngine:
    if settings.graph_engine == "gliner":
        return GLiNEREngine()
    return GraphitiEngine()
```

### Enriched GraphFact

```python
class GraphFact(BaseModel):
    fact: str                              # always populated
    source: str | None = None
    created_at: str | None = None
    expired_at: str | None = None          # Graphiti temporal
    entities: list[str] | None = None      # structured entity names
    relation_type: str | None = None       # typed relationship
    confidence: float | None = None        # extraction confidence
```

- Graphiti fills `fact`, `source`, temporal fields (same as today)
- GLiNER fills `fact` (formatted), plus `entities`, `relation_type`, `confidence`
- Consumers use `GraphFact` uniformly

### GraphitiEngine

Thin wrapper around existing `GraphitiWriter` + `graph_search._search_entities`. No behavior change from current implementation.

### GLiNEREngine

- **Model loading:** singleton via module-level lazy init (like `get_graphiti()`)
- **Model:** `GLiNER2.from_pretrained("fastino/gliner2-base-v1")` — 205MB
- **Schema:** maps `constants.ENTITY_TYPES` and `constants.RELATIONSHIP_TYPES` to GLiNER2 schema format
- **Ingest flow:** chunks -> `extractor.extract(text, schema)` -> Neo4j MERGE via Cypher (reusing `_LegacyGraphWriter` patterns)
- **Batch processing:** `batch_extract_entities` + `batch_extract_relations` for multi-chunk throughput
- **Search flow:** Neo4j full-text index on `Entity.name`/`Entity.description` -> traverse relationships -> format as `GraphFact`
- **Lazy import:** `gliner2` only imported when `graph_engine="gliner"`, so Graphiti users don't need the dependency

### Neo4j Schema (GLiNER path)

Reuses existing schema from `neo4j_setup.py`:
- `Entity(name, type, description)` with uniqueness constraint on `(name, type)`
- `Document(source_key)`, `Chunk(id)` nodes
- Typed relationships via APOC `merge.relationship`
- **Addition:** full-text index on `Entity.name` + `Entity.description` for search
- **Addition:** `confidence` property on relationship edges

### Pipeline Integration

`pipeline.py` replaces `GraphitiWriter()` with `get_graph_engine()`. The engine handles grouping, extraction, and Neo4j writes internally.

`graph_search.py` replaces direct Graphiti import with `get_graph_engine().search()`.

## Files Changed

| File | Change |
|-|-|
| `config/settings.py` | Add `graph_engine: str = "graphiti"` |
| `.env.example` | Add `GRAPH_ENGINE=graphiti` |
| `ingestion/graph_engine.py` | **New** — Protocol, factory, GraphitiEngine, GLiNEREngine |
| `ingestion/graph_writer.py` | Keep as-is (wrapped by GraphitiEngine) |
| `ingestion/pipeline.py` | Use `get_graph_engine()` instead of `GraphitiWriter()` |
| `server/tools/graph_search.py` | Use `get_graph_engine().search()` |
| `server/models.py` | Enrich `GraphFact` with optional fields |
| `ingestion/neo4j_setup.py` | Add full-text index for GLiNER search path |
| `config/constants.py` | Add GLiNER schema mappings if needed |
| `pyproject.toml` | Add `gliner2 >= 1.2` as optional dependency |
| `tests/` | New tests for GLiNEREngine + protocol conformance |

## Dependencies

```toml
[project.optional-dependencies]
gliner = ["gliner2>=1.2"]
```

Lazy import in `GLiNEREngine.__init__` so the base install doesn't require it.

## Risks & Mitigations

| Risk | Mitigation |
|-|-|
| GLiNER2 RE quality lower than LLM | Keep Graphiti as default; GLiNER is opt-in |
| 2,048 token context limit | Chunks are 500-1500 chars; group_chunks respects this |
| Entity dedup less sophisticated | MERGE on normalized (name, type) + description update |
| GLiNER2 library stability | Pin version, lazy import, Graphiti fallback always available |
| Neo4j full-text search quality | Test with real queries; can add embedding-based entity search later |
