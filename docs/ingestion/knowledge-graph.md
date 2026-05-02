# Knowledge Graph

Spektr builds a knowledge graph in **Neo4j** with a pluggable extraction engine. The `GRAPH_ENGINE` setting selects between two backends:

| Engine | Setting | Extraction | Speed | Cost |
|-|-|-|-|-|
| **Graphiti** | `graphiti` (default) | LLM-based (GPT-4o class) | ~29 min / 74 chunks | ~$0.50-2.00 per doc |
| **GLiNER2** | `gliner` | Local 205MB model on CPU | ~5-15 sec / 74 chunks | $0.00 |

**Source:** `ingestion/graph_engine.py`, `ingestion/graph_writer.py`, `ingestion/graphiti_client.py`

## Architecture

```mermaid
flowchart LR
    Chunks["Text chunks"] --> Factory["get_graph_engine()"]
    Factory -->|graphiti| GE["GraphitiEngine"]
    Factory -->|gliner| GL["GLiNEREngine"]
    GE --> Graphiti["Graphiti Core"]
    Graphiti --> LLM["LLM\n(entity extraction)"]
    Graphiti --> Neo4j["Neo4j"]
    GL --> GLiNER["GLiNER2\n(local model)"]
    GLiNER --> Neo4j

    style Neo4j fill:#d4edda
```

Both engines implement the `GraphEngine` protocol and return `GraphFact` results. Pipeline and search tools are engine-agnostic.

## `GraphEngine` Protocol

```python
class GraphEngine(Protocol):
    async def ingest(
        self,
        chunks: list[TextChunk],
        source_key: str,
        schema: MergedSchema | None = None,
    ) -> None: ...
    async def search(self, query: str, limit: int = 10) -> list[GraphFact]: ...
    async def close(self) -> None: ...
```

The optional `schema` parameter allows per-document dynamic schemas from the schema inducer. GLiNER2 uses it when provided; Graphiti ignores it (discovers entity types autonomously via its own LLM).

The factory `get_graph_engine()` lazily creates a singleton based on `settings.graph_engine`. Call `close_graph_engine()` at shutdown.

## GraphitiEngine (`GRAPH_ENGINE=graphiti`)

Wraps the existing `GraphitiWriter` and Graphiti client. No behavior change from the pre-modular implementation.

### Ingestion

Each chunk becomes a Graphiti **episode** with name format: `{source_key}:p{page_number}:c{chunk_index}`. Graphiti's `add_episode()` internally:

1. Calls an LLM to extract entities and relationships
2. Merges discovered entities into the graph (deduplication by name)
3. Creates typed relationship edges with temporal metadata
4. Sets `created_at` on new edges and `expired_at` when relationships are superseded

Entity and relationship types are **LLM-discovered dynamically** -- no hardcoded taxonomy.

### Search

Delegates to `graphiti_client.search()` which returns edges with facts, sources, and timestamps. Results are mapped to `GraphFact` with `fact`, `source`, `created_at`, and `expired_at` fields.

### Temporal Awareness

Graphiti tracks time on all edges:

| Field | Description |
|-|-|
| `created_at` | When the relationship was first observed |
| `expired_at` | When the relationship was superseded by newer information |
| `reference_time` | The temporal anchor provided at ingestion time |

### Graphiti Client Singleton

`graphiti_client.py` manages the shared Graphiti client lifecycle.

|Function|Description|
|-|-|
|`get_graphiti()`|Returns (and lazily initializes) the singleton. On first call, connects to Neo4j, builds indices/constraints, and wires up LLM + embedder + cross-encoder.|
|`close_graphiti()`|Closes the embedder and client, resets both singletons to `None`.|

The client is constructed with three pluggable components:

- `llm_client=OpenAIGenericClient(config=llm_config)` — entity/relationship extraction (Chat Completions API, OpenRouter-compatible).
- `embedder=_JinaGraphitiEmbedder()` — adapts the project embedder via `create_embedder()`.
- `cross_encoder=OpenAIRerankerClient(config=llm_config)` — Graphiti's reranker for search relevance, sharing the same `LLMConfig` as the extraction client.

> **Side effect at import time:** `graphiti_client.py` calls `os.environ.setdefault("EMBEDDING_DIM", "512")` because `graphiti_core` reads `EMBEDDING_DIM` at import to size its Neo4j vector indexes. Importing this module mutates the process environment — keep this in mind when running tests in isolation.

### LLM Configuration

Graphiti's entity extraction uses `OpenAIGenericClient` (Chat Completions API), compatible with OpenRouter and any OpenAI-compatible endpoint:

| Setting | Env var | Description |
|-|-|-|
| `llm_api_key` | `LLM_API_KEY` | API key for the LLM provider |
| `llm_model` | `LLM_MODEL` | Model name (e.g. `openai/gpt-4.1`) |
| `llm_base_url` | `LLM_BASE_URL` | Base URL (e.g. `https://openrouter.ai/api/v1`) or omit for direct OpenAI |

> **Note:** `OpenAIGenericClient` is used instead of `OpenAIClient` because the Responses API is not supported by OpenRouter.
>
> **Warning:** Reasoning models (e.g. `o4-mini`) may not work -- they can return empty `content` fields, breaking Graphiti's extraction pipeline.

### Embedder Adapter

The client uses `_JinaGraphitiEmbedder` -- an adapter that wraps the project's own `JinaV4Embedder`, reusing the Jina v4 API for graph vector search.

## GLiNEREngine (`GRAPH_ENGINE=gliner`)

Uses the [GLiNER2](https://github.com/fastino-ai/GLiNER2) model for entity and relation extraction in a single forward pass. Zero LLM API calls.

### Setup

Install the optional dependency:

```bash
uv sync --extra gliner
```

The `gliner2` package is lazy-imported only when `GRAPH_ENGINE=gliner`, so Graphiti users don't need it installed.

### Model

- **Model:** `fastino/gliner2-base-v1` (205MB, downloaded on first use)
- **Context window:** 2,048 tokens (sufficient for typical chunks of 500-1500 chars)
- **NER quality:** 0.59 F1 on CrossNER (matches GPT-4o zero-shot)
- **Inference:** ~130ms per chunk on CPU

### Schema

#### Base schema

Entity and relationship types are defined in `config/constants.py` as dicts with natural language descriptions that guide GLiNER2's extraction accuracy. The base schema covers 14 entity types and 12 relationship types across business, legal, financial, and technical domains:

**Entity types (14):** `person`, `organization`, `location`, `date_time`, `monetary_value`, `document`, `product`, `technology`, `metric`, `event`, `legal_term`, `role`, `concept`, `requirement`

**Relationship types (12):** `created_by`, `owned_by`, `uses`, `part_of`, `mentions`, `measured_by`, `requires`, `applies_to`, `succeeds`, `conflicts_with`, `valued_at`, `scheduled_for`

These descriptions are passed directly to `GLiNER2.create_schema().entities()` and `.relations()`, which accept `dict[str, str]` for description-enhanced extraction.

#### Domain/range constraints

Each relationship type has domain/range constraints (`RELATION_CONSTRAINTS` in `constants.py`) that define which entity types are valid as source and target. During GLiNER ingestion, extracted triples that violate these constraints are silently dropped. This prevents nonsensical edges (e.g. `metric -[valued_at]-> date_time`) from polluting the graph.

Key constraint examples:

| Relationship | Valid sources | Valid targets |
|-|-|-|
| `created_by` | person, org, product, tech, document, event | person, org |
| `valued_at` | product, org, document, event | monetary_value |
| `scheduled_for` | event, product, document, requirement | date_time |
| `mentions` | person, org, document, event, product, tech | any |

Schema-induced relationship types (from dynamic induction) bypass constraint checks, since their domain/range is not known at compile time.

#### Dynamic schema induction

When `SCHEMA_INDUCTION_ENABLED=true` (default) and `GRAPH_ENGINE=gliner`, the pipeline runs a single LLM call per document to propose additional domain-specific types.

**Module:** `ingestion/schema_inducer.py`

| Class | Description |
|-|-|
| `SchemaInducer` | Orchestrator: calls LLM with a sample of document text, parses proposed types |
| `InducedSchema` | Dataclass holding LLM-proposed entity and relationship types with descriptions |
| `MergedSchema` | Base schema + induced types merged together (induced types are additive only) |

**How it works:**

1. The pipeline takes a sample from the first 3 chunks of a document
2. `SchemaInducer.induce(sample_text)` calls the LLM (`SCHEMA_INDUCTION_MODEL`, default Haiku) with a prompt asking for 3-8 entity types and 3-6 relationship types specific to the document's domain
3. `merge_with_base(induced)` merges proposed types on top of the base schema (base types are never removed)
4. The merged schema is passed to `GLiNEREngine.ingest(chunks, source_key, schema=merged)`

**Caching:** Results are cached by SHA256 hash of the first 500 chars with a configurable TTL (`SCHEMA_CACHE_TTL`, default 3600s). Documents with similar openings (e.g. contracts from the same template) hit the cache and skip the LLM call.

**Fallbacks:** If the document has < 200 chars of text (badly degraded scan), or if the LLM call fails, the base schema is used without induction.

#### Schema bootstrapping from sample documents

Dynamic schema induction runs per-document at ingestion time, which adds an LLM call per doc. For stable corpora (e.g. a company's standard contract templates, technical documentation), you can eliminate this runtime cost by bootstrapping the base schema once:

1. Run schema induction on ~500 representative documents from your target corpus
2. Aggregate all proposed entity and relationship types across documents
3. Rank by frequency — types proposed across many documents are domain-stable
4. Add the top 5-10 most frequent domain-specific types to the base schema in `config/constants.py`
5. Disable `SCHEMA_INDUCTION_ENABLED` (or leave it on for long-tail documents)

This gives Claude-level schema quality at GLiNER's $0.00 extraction cost. The base schema becomes tuned to your domain without any runtime LLM dependency.

### Ingestion

For each merged page text in a single Neo4j session:

1. `extractor.extract(text, schema)` returns entities and relations
2. Entities are upserted via `MERGE` on `name` only (not type), with normalized names (`.strip().title()`)
3. Entity `types` is stored as an array — multiple types accumulate if the same entity is extracted with different types across pages
4. Entity `description` is set to the first 500 chars of source text (used by full-text search)
5. Relationships are upserted via `apoc.merge.relationship` with `confidence` and `source` properties

### Post-Processing

Before writing to Neo4j, extracted data is filtered:

- **Domain/range constraint violations** — triples where entity types don't match the relationship's allowed sources/targets are dropped
- **Self-referential relationships** (entity → itself) are skipped
- **Short entity names** (< 2 chars) are discarded
- **Stopword entities** (common English words like "the", "is", "copy") are filtered out

### Search

Uses a Neo4j full-text index (`entity_fulltext`) on `Entity.name` and `Entity.description`:

1. Query the full-text index for matching entities
2. Traverse outgoing relationships to connected entities
3. Format as `GraphFact` with `entities`, `relation_type`, and `confidence` fields
4. Deduplicate by fact string

The full-text index is created automatically by `neo4j_setup.py`.

### Neo4j Schema

Reuses the existing schema from `neo4j_setup.py`:

- `Entity(name, types[], description)` with uniqueness constraint on `(name)`
- `types` is an array property — entities accumulate types across extractions
- Typed relationship edges via APOC with `source` and `confidence` properties
- Full-text index on `Entity.name` + `Entity.description` for search

### Comparison

|Aspect|Graphiti|GLiNER2|
|-|-|-|
|Entity extraction|LLM-discovered types|Schema-driven with descriptions from `constants.py`|
|Relation extraction|LLM-discovered, temporal|Schema-driven with descriptions, post-processed|
|Entity dedup|Automatic (Graphiti resolves)|MERGE on normalized `name`, types accumulated as array|
|Temporal tracking|Built-in `created_at`/`expired_at` on every edge|**Entity-level only**: `Entity.first_seen` (set on create) and `Entity.last_seen` (updated on every ingest). Relationships carry `source` (file key) and `confidence` (always `1.0`) — no per-edge timestamps and no expiry.|
|Dependencies|LLM API key, network|None (local model)|

## `_LegacyGraphWriter` (Deprecated)

The `_LegacyGraphWriter` class uses raw Cypher queries to manually upsert documents, chunks, entities, and relationships. It is **deprecated** and kept only for backward compatibility. GLiNEREngine reuses its Cypher patterns but with GLiNER2 extraction instead of external entity extraction.

## Concurrency & Tuning

### Graphiti engine

| Setting | Env var | Default | Scope |
|-|-|-|-|
| `graphiti_concurrency` | `GRAPHITI_CONCURRENCY` | `3` | Max concurrent episode ingestions per page |
| `graph_semaphore_limit` | `GRAPH_SEMAPHORE_LIMIT` | `10` | Max concurrent LLM calls within Graphiti |

### GLiNER engine

GLiNER2 extraction is synchronous and CPU-bound (~130ms/chunk). No concurrency settings needed -- chunks are processed sequentially within a single Neo4j session for optimal throughput.

## Integration

The pipeline uses `get_graph_engine()` to obtain the singleton engine instance:

1. If any page has `content_type == "text"`, the graph engine is obtained via `get_graph_engine()`
2. For each text page, `semantic_chunk()` produces chunks
3. Chunks are passed to `engine.ingest(chunks, source_key)`
4. At pipeline end, `close_graph_engine()` shuts down the singleton

See also: [Pipeline Overview](overview.md) | [File Processing](file-processing.md) | [Architecture Data Flow](../architecture/data-flow.md)
