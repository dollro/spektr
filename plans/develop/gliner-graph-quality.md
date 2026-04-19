# GLiNER Graph Quality Improvement Plan

## Problem Statement

The GLiNER2 engine produces low-quality knowledge graphs:
- Entity types are corporate-focused (WORKS_AT, ACQUIRED) — irrelevant for general content
- Same entity extracted as multiple types (Freedium as PERSON, TECHNOLOGY, PRODUCT) creating duplicate nodes
- Cartesian product of relationships between duplicated entities (18 WORKS_AT, 17 USES_TECHNOLOGY for a dev tooling article)
- Self-referential relationships (Freedium WORKS_AT Freedium)
- No post-processing or deduplication
- Schema lacks descriptions (GLiNER2 supports them for better accuracy)

Test PDF: article about "10 Claude Code Commands" — graph should have entities like Claude Code, /analyze-issue, GitHub, Reza Rezvani; not "Junior Dev WORKS_AT Atlassian".

---

## Phase 1: Short-Term — Fix GLiNER2 Schema + Post-Processing

### 1.1 Broaden Entity Types with Descriptions

**File: `config/constants.py`**

Replace the current narrow types with general-purpose ones. GLiNER2 accepts
`{type: description}` dicts — descriptions significantly improve extraction accuracy.

```python
# Old
ENTITY_TYPES = ["PERSON", "ORGANIZATION", "PRODUCT", "TECHNOLOGY", "LOCATION", "CONCEPT", "EVENT"]

# New — broader, with descriptions for GLiNER2 schema
ENTITY_TYPES = {
    "person": "A named individual, author, developer, researcher, or public figure",
    "organization": "A company, institution, open-source project, or team",
    "technology": "A programming language, framework, library, tool, CLI command, or protocol",
    "concept": "An idea, methodology, design pattern, workflow, or practice",
    "metric": "A quantitative measure, statistic, benchmark, or KPI",
    "location": "A physical place, region, or URL/domain",
    "event": "A named event, release, conference, or incident",
}
```

Key changes:
- `PRODUCT` merged into `TECHNOLOGY` (tools, CLIs, libraries are tech)
- Added `METRIC` for quantitative data ("+62% fewer bugs", "$47k per quarter")
- Descriptions guide GLiNER2 to classify correctly
- Lowercase keys match GLiNER2's expected format (no need for entity_map conversion)

### 1.2 Broaden Relationship Types with Descriptions

```python
# Old
RELATIONSHIP_TYPES = ["WORKS_AT", "PARTNERS_WITH", "PRODUCES", "USES_TECHNOLOGY",
                      "LOCATED_IN", "ACQUIRED", "COMPETES_WITH", "REFERENCES"]

# New — open-domain relationships with descriptions
RELATIONSHIP_TYPES = {
    "created_by": "X was created, authored, or developed by Y",
    "uses": "X uses, depends on, or integrates Y",
    "part_of": "X is a component, feature, or subset of Y",
    "related_to": "X is conceptually related to or associated with Y",
    "improves": "X improves, enhances, or optimizes Y",
    "measured_by": "X is measured or quantified by Y",
    "located_in": "X is geographically or organizationally located in Y",
    "describes": "X describes, documents, or explains Y",
}
```

Key changes:
- Removed corporate-only relations (ACQUIRED, COMPETES_WITH, PARTNERS_WITH, WORKS_AT)
- Added open-domain relations that work for any content type
- Descriptions help GLiNER2 pick the right relation

### 1.3 Use Descriptions in GLiNER2 Schema

**File: `ingestion/graph_engine.py` — `GLiNEREngine.__init__`**

Currently entities/relations pass as flat lists. GLiNER2's `create_schema()` supports
descriptions via dict format:

```python
# Old
self._schema = (
    self._extractor.create_schema()
    .entities(list(entity_map.keys()))
    .relations(list(relation_map.keys()))
)

# New — pass dicts with descriptions
self._schema = (
    self._extractor.create_schema()
    .entities(ENTITY_TYPES)       # dict: {type: description}
    .relations(RELATIONSHIP_TYPES) # dict: {type: description}
)
```

Need to verify GLiNER2's API accepts dicts. Check `create_schema().entities()` signature.
If it only accepts lists, pass descriptions via the alternate API.

### 1.4 Entity Deduplication — MERGE on Name Only

**File: `ingestion/graph_engine.py` — `GLiNEREngine.ingest`**

Current MERGE key is `{name, type}`, so "Freedium" as PERSON and "Freedium" as TECHNOLOGY
are separate nodes. Fix: MERGE on `name` only, store `type` as a property (or list of types).

```python
# Old
"MERGE (e:Entity {name: $name, type: $type}) "

# New — merge on name only, accumulate types
"MERGE (e:Entity {name: $name}) "
"ON CREATE SET e.types = [$type], e.first_seen = datetime(), "
"e.description = $description "
"SET e.last_seen = datetime(), e.source = $source, "
"e.types = CASE WHEN NOT $type IN coalesce(e.types, []) "
"THEN coalesce(e.types, []) + $type ELSE e.types END"
```

Also update the Neo4j constraint in `neo4j_setup.py`:
```
# Old: CREATE CONSTRAINT entity_unique ... REQUIRE (e.name, e.type) IS UNIQUE
# New: CREATE CONSTRAINT entity_unique ... REQUIRE (e.name) IS UNIQUE
```

### 1.5 Post-Processing Filters

**File: `ingestion/graph_engine.py` — `GLiNEREngine.ingest`**

Add filtering before writing to Neo4j:

```python
# After extraction, before Neo4j writes:

# 1. Skip self-referential relationships
if head_norm == tail_norm:
    continue

# 2. Skip very short entity names (noise)
if len(normalized) < 2:
    continue

# 3. Skip entities that are just common words (stopword filter)
STOPWORDS = {"the", "a", "an", "this", "that", "it", "is", "are", "was"}
if normalized.lower() in STOPWORDS:
    continue
```

### 1.6 Update Neo4j Schema

**File: `ingestion/neo4j_setup.py`**

- Change entity uniqueness constraint from `(name, type)` to `(name)` only
- Update fulltext index to include `types` (array) instead of single `type`

### 1.7 Files Changed Summary

| File | Change |
|-|-|
| `config/constants.py` | Entity/relationship types as dicts with descriptions |
| `ingestion/graph_engine.py` | Schema with descriptions, MERGE on name, post-processing filters |
| `ingestion/neo4j_setup.py` | Update entity uniqueness constraint |
| `server/tools/graph_search.py` | Update search to use `e.types` (array) instead of `e.type` |

### 1.8 Testing

1. Delete Neo4j volumes: `docker compose down -v && docker compose up -d`
2. Run pipeline: `uv run python -m ingestion.pipeline`
3. Verify in Neo4j Browser: `MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 30`
4. Expected: entities like "Claude Code", "Reza Rezvani", "/analyze-issue", "GitHub"
5. Expected: no self-referential, no duplicate entities, no WORKS_AT nonsense

---

## Phase 2: Medium-Term — KGGen as Alternative Graph Engine

### 2.1 What is KGGen?

KGGen (NeurIPS 2025) is an LLM-based text-to-knowledge-graph generator. Key advantages:
- **Open-domain**: no predefined entity/relationship types needed — extracts whatever is relevant
- **Built-in entity resolution**: clusters related entities (e.g. "NYC" + "New York City"), reducing graph sparsity. This is its killer feature vs GLiNER2.
- **96% entity recall** on benchmarks, with richer descriptions than human annotations
- **pip install kg-gen**, clean API

### 2.2 Integration with Existing Stack

KGGen uses **LiteLLM** for model routing. Our stack already uses OpenRouter:

```
LLM_API_TYPE=openai
LLM_MODEL=anthropic/claude-haiku-4.5
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
```

KGGen model format: `openrouter/anthropic/claude-haiku-4.5`

LiteLLM maps `openrouter/*` to OpenRouter's API automatically. So KGGen would use
the same model and API key we already pay for — **no new provider needed**.

For cost efficiency, Claude Haiku 4.5 is ideal: fast, cheap ($0.80/MTok input),
and good enough for entity/relation extraction.

### 2.3 Implementation as GraphEngine

Add `KGGenEngine` implementing the existing `GraphEngine` protocol:

```python
class KGGenEngine:
    def __init__(self) -> None:
        from kg_gen import KGGen
        from config.settings import settings

        # Map our settings to LiteLLM model format
        model = f"openrouter/{settings.llm_model}"
        self._kg = KGGen(
            model=model,
            api_key=settings.llm_api_key,
            temperature=0.0,
        )
        self._driver = AsyncGraphDatabase.driver(...)

    async def ingest(self, chunks: list[TextChunk], source_key: str) -> None:
        # Concatenate chunks into page-level texts
        # Call self._kg.generate(text, context="Document extraction")
        # Write entities + edges to Neo4j
        # KGGen returns: graph.entities (list[str]), graph.edges (list[tuple])
        ...

    async def search(self, query: str, limit: int = 10) -> list[GraphFact]:
        # Same Neo4j fulltext search as GLiNEREngine
        ...

    async def close(self) -> None:
        await self._driver.close()
```

### 2.4 Settings Addition

```python
# config/settings.py
graph_engine: str = "gliner"  # "gliner" | "graphiti" | "kggen"
```

### 2.5 Cost Estimate

For a 25-page PDF (~74 chunks, ~15k tokens after merging):
- Claude Haiku 4.5: ~$0.012 input + ~$0.010 output = **~$0.02 per document**
- Comparable to Graphiti but with better entity resolution
- GLiNER2 remains the zero-cost option

### 2.6 Tradeoffs vs GLiNER2

| Aspect | GLiNER2 (Phase 1) | KGGen (Phase 2) |
|-|-|-|
| Cost per doc | $0 | ~$0.02 (Haiku 4.5 via OpenRouter) |
| Entity resolution | Manual (name normalization) | Automatic (LLM clustering) |
| Relation types | Predefined schema | Open-domain (LLM decides) |
| Offline capable | Yes | No (needs API) |
| Quality | Good with post-processing | Best (NeurIPS benchmark) |
| Latency | ~1s per chunk | ~5-10s per chunk (API round-trips) |
| Context understanding | Limited (512 token window) | Full (LLM context window) |

### 2.7 Recommendation

Keep GLiNER2 as the default (free, offline, fast). Add KGGen as an opt-in engine
for users who want higher quality graphs and are willing to pay per-document LLM costs.
The `GRAPH_ENGINE=kggen` setting makes it a one-line switch.

### 2.8 Dependencies

```
kg-gen >= 0.3
```

Add to `[dependency-groups]` or main dependencies in `pyproject.toml`.

---

## Execution Order

1. Phase 1.4 + 1.6 first (schema/constraint changes — requires Neo4j volume reset)
2. Phase 1.1 + 1.2 (new entity/relationship types)
3. Phase 1.3 (descriptions in GLiNER2 schema)
4. Phase 1.5 (post-processing filters)
5. Phase 1.7 (update graph_search tool)
6. Test end-to-end with test.pdf
7. Phase 2 in a separate branch/PR
