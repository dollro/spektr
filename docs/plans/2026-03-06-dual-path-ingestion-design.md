# Dual-Path Ingestion & Dynamic Schema Design

**Date:** 2026-03-06
**Status:** Proposed
**Branch:** TBD

---

## 1. Context & Motivation

Spektr is a RAG-as-MCP-Server pipeline that ingests documents into Qdrant (vector) and Neo4j (knowledge graph), then exposes search tools to LLM agents via MCP. It currently operates as a single batch pipeline — CocoIndex processes files from S3 or local storage, embeds them with Jina v4, and extracts entities via a pluggable graph engine (Graphiti or GLiNER2).

This design addresses two gaps:

1. **The pipeline has no real-time path.** There is no way to ingest streaming data and make it searchable within seconds.
2. **GLiNER2 entity extraction quality is poor** because the schema in `constants.py` is hardcoded with 7 tech-biased entity types and 8 relationship types. A legal contract, financial report, or medical document gets the same narrow schema as a software architecture doc.

### 1.1 Use Cases

**Use Case A — Bulk Knowledge Base**
PDFs, text files, scanned documents (OCR via Docling), and images with tables. Volume: 100–1,000 documents, up to ~10k pages. This is the persistent knowledge base that an LLM draws on during meetings. Content types are diverse: contracts, policies, financial reports, technical docs. The pipeline must handle multimodal content — Docling OCR extracts text from scans, and visual pages (charts, diagrams, tables) are image-embedded via dense + ColBERT multi-vectors.

**Use Case B — Live Meeting Transcripts**
An external transcription service pushes text chunks via HTTP POST every ~30 seconds. Target latency: < 5 seconds from chunk arrival to searchability. Meetings are typically 60 minutes but can run up to 3 hours (~120–360 transcript chunks). One active meeting at a time. After the meeting, the user decides whether to archive the transcript into the permanent KB or discard it.

**Combined requirement:** During a meeting, the LLM must provide low-latency recommendations by combining:
- Real-time transcript context (what was said, by whom, when)
- Relevant knowledge base documents (policies, contracts, reference material)
- Temporal awareness of how facts evolved during the conversation

---

## 2. The Core Retrieval Problem

The most demanding retrieval scenario clarifies why certain architectural choices are necessary.

**Scenario:** In a 60-minute meeting, a speaker explains a contract from a legal perspective at t=2min, a financial perspective at t=15min, and a risk perspective at t=45min. Later at t=50min, someone asks a follow-up about that contract. The LLM needs to reconstruct the full chronological picture of everything said about the contract across the meeting, while also pulling relevant contract documents from the bulk KB.

### 2.1 Why Vector Search Alone Is Insufficient

Vector search retrieves chunks ranked by semantic similarity to the current query. If the query at t=50min is about pricing, it finds the financial chunk — but may miss the legal terms and risk discussion because they're semantically distant from "pricing." Vector search returns a relevance-ranked list, not a timeline. The LLM would have to piece together temporal context from scattered chunk metadata.

### 2.2 Why GLiNER2's Graph Is Insufficient for Live Data

GLiNER2 creates flat entity nodes (`Entity{name: "Contract"}`) connected by typed relationships. But it has no concept of episodic memory — it doesn't know that the entity was discussed in different contexts at different times during the meeting. The full-text search on `Entity.name` returns the node and its outgoing edges, but there's no temporal ordering, no fact evolution tracking, and no way to reconstruct the conversation timeline from the graph alone.

### 2.3 Why Graphiti Solves This

Graphiti was designed specifically for this pattern — streaming episodic memory:

- Each transcript chunk becomes an **episode** with a `created_at` timestamp and `reference_time`
- When the contract is mentioned at t=2min, Graphiti creates edges: `Contract --has_terms--> Payment Net-60`, `Contract --involves--> Acme Corp`
- When the value is updated at t=15min from €1.2M to €1.5M, Graphiti doesn't overwrite — it sets `expired_at` on the old edge and creates a new one. **Fact evolution is tracked.**
- One `graph_search("contract")` returns the full relational context with temporal ordering
- Graphiti's bi-temporal model distinguishes between when an event occurred (valid time) and when it was ingested (ingestion time)

The critical insight: **ingestion latency ≠ query latency.** Graphiti's search is fast (graph traversal + vector index, no LLM calls, ~50ms). It's only the ingestion that involves an LLM call (~2–5s per chunk for a short transcript). With 30-second chunk intervals, this fits comfortably.

---

## 3. Approaches Considered

### 3.1 Approach A: "Dual-Path Pipeline" (Selected)

Two separate ingestion paths sharing the same storage backends:
- **Path A (Bulk):** Existing CocoIndex pipeline with GLiNER2, enhanced with dynamic schema induction
- **Path B (Live):** New lightweight HTTP endpoint with Graphiti for temporal episodic memory

GLiNER2 entity quality improved via richer schema descriptions + per-document LLM schema induction (one cheap LLM call per document, not per chunk).

**Trade-offs:** Two entry points to maintain. But the use cases are fundamentally different (batch vs streaming, multimodal vs text-only, flat entities vs temporal episodes). Clean separation is simpler than forced unification.

### 3.2 Approach B: "Unified Pipeline with Priority Queue" (Rejected)

Single pipeline with a priority queue routing data by urgency: high-priority transcripts skip CocoIndex, low-priority bulk data goes through the normal pipeline.

**Rejected because:** CocoIndex is batch-oriented — it wasn't designed for real-time priority routing. Implementing this would require either bypassing CocoIndex for live data (making it identical to Approach A but with more abstraction overhead) or fundamentally modifying the pipeline framework. The unification adds complexity for no benefit since the two paths share almost no processing logic (multimodal OCR + visual embedding vs. text-only + temporal episodes).

### 3.3 Approach C: "Graphiti for Everything" (Rejected)

Use Graphiti for both bulk KB and live transcripts. Best graph quality everywhere, temporal awareness on all data.

**Rejected because:**
- Graphiti ingestion takes ~29 minutes for a 74-chunk PDF (LLM call per chunk). For a bulk KB of 100–1,000 documents, this means hours/days of ingestion time and $50–2,000 in LLM API costs.
- GLiNER2 processes the same 74 chunks in ~15 seconds at $0.00. For bulk static content that doesn't need temporal tracking, this is a 100x speed improvement.
- The research confirms this as the recommended "Tiered Extraction Strategy": GLiNER2 for high-volume/low-complexity (Tier 1), Graphiti for high-complexity/low-volume with temporal needs (Tier 2).

### 3.4 Approach D: "GLiNER2 for Everything" (Rejected)

Use GLiNER2 for both paths. Fastest and cheapest.

**Rejected because** of the core retrieval problem described in Section 2. GLiNER2 cannot track fact evolution, temporal ordering, or episodic context across a conversation. For live meeting recommendations where the LLM needs to reconstruct what was said about a topic over time, this is a fundamental limitation. Vector search partially compensates but cannot provide the structured temporal graph that Graphiti's episodic model delivers.

---

## 4. Research Context: Graphiti vs GLiNER2

This design draws on comparative analysis of both architectures. Key findings that informed the design:

### 4.1 Graphiti: Temporal Memory for Agents

Graphiti is a real-time, incremental memory layer built on temporal knowledge graphs. Its core innovation is a **bi-temporal model** — every edge tracks both when the fact occurred (`valid_at`) and when it was ingested. When new information contradicts existing knowledge, Graphiti doesn't overwrite; it updates validity intervals, preserving historical records.

Retrieval uses a hybrid approach: semantic embeddings + keyword search (BM25) + graph traversal. Neo4j's native vector and full-text indexes provide near-constant time access. No LLM calls during retrieval — this is critical for low-latency meeting recommendations.

In the Deep Memory Retrieval (DMR) benchmark, Zep (powered by Graphiti) outperformed MemGPT by up to 18.5% accuracy while reducing response latency by 90%.

### 4.2 GLiNER2: Schema-Driven CPU Extraction

GLiNER2 uses a bidirectional transformer encoder (not an autoregressive decoder like GPT-4). This gives it three production advantages:
- **Deterministic:** Non-generative, so no hallucinated entities — every output maps directly to input text
- **Fast:** 100–250ms on CPU, no GPU or API calls needed
- **Dynamic schema:** Entity types are not hardcoded — the model accepts natural-language descriptions per type, generalizing zero-shot to new domains

On CrossNER, GLiNER2 achieves 0.59 F1 (matching GPT-4o zero-shot). The key to extraction quality is **description richness** — the more specific and semantically grounded the type descriptions, the better the extraction.

### 4.3 Why Both Engines in One System

The research recommends a "Tiered Extraction Strategy" for production systems:
- **Tier 1 (High Volume):** GLiNER2 on CPU for rapid extraction of common entities
- **Tier 2 (High Complexity):** Graphiti with a frontier LLM for temporal resolution and complex relations

This maps directly to our architecture: GLiNER2 for bulk KB (Tier 1), Graphiti for live transcripts (Tier 2).

Both can write to the same Neo4j instance using strategic namespacing — Graphiti uses `group_id` for partitioning, GLiNER2 entities carry a `source_engine` property. No cross-engine entity resolution is needed in v1; search tools query both independently and merge results.

### 4.4 Dynamic Entity Types in GLiNER2

A common misconception is that small encoder models are restricted to fixed entity types. GLiNER2's `create_schema().entities()` accepts `dict[str, str]` — type names mapped to natural-language descriptions. The `extract_json` method goes further, supporting `field_name::type::description` syntax for structured extraction. Descriptions are the primary lever for extraction accuracy in specialized domains.

The current Spektr schema (`constants.py`) has only 7 entity types with tech-biased descriptions. Expanding this base schema and adding per-document LLM schema induction leverages GLiNER2's dynamic schema capability to close the quality gap with LLM-based extraction, at zero additional inference latency.

---

## 5. Design Decisions

### 5.1 Why Two Paths Instead of One

| Concern | Bulk KB | Live Transcript |
|-|-|-|
| Latency tolerance | Minutes | < 5 seconds |
| Content types | PDF, images, scans, text | Text only |
| Graph engine | GLiNER2 (fast, flat entities) | Graphiti (temporal episodes) |
| Graph needs | Entity/relation extraction | Temporal episodic memory, fact evolution |
| Volume pattern | Batch (100–1000 docs) | Streaming (~120–360 chunks/meeting) |
| Lifecycle | Permanent | Session-scoped, user decides to archive or discard |

Forcing both through a single pipeline would require either crippling live latency (batch processing) or abandoning temporal tracking (GLiNER2 for everything). Two paths is the natural architecture.

### 5.2 Why GLiNER2 for Bulk KB

- **Speed:** ~130ms/chunk on CPU vs ~29 min/doc with Graphiti
- **Cost:** $0.00 vs ~$0.50–2.00/doc
- **Sufficient for static content:** Bulk KB documents don't change — there's no fact evolution to track, no temporal ordering within a single document that requires episodic memory
- **Quality is solvable:** The entity quality problem is a schema problem, not a model problem. Richer descriptions + per-document schema induction close the gap (see Section 6.2)

### 5.3 Why Graphiti for Live Transcripts

- **Temporal episodic memory:** Each 30-sec chunk becomes an episode. When the same topic is discussed at t=2min and t=45min, the graph connects them through shared entities with temporal edges.
- **Fact evolution:** Corrections mid-meeting are tracked via `expired_at` / supersession, not overwrites.
- **Designed for streaming:** Graphiti's `add_episode()` is incremental — it processes one episode and immediately resolves it against the existing graph.
- **Latency fits:** Graphiti ingestion ~2–5s per short text chunk. With 30-sec intervals, the chunk is fully indexed before the next one arrives. Vector search via Qdrant (~200ms) provides immediate searchability while Graphiti processes in the background.

### 5.4 Why Dynamic Schema Induction for GLiNER2

GLiNER2's extraction quality is directly tied to schema description richness. The current 7 entity types miss entire domains. Two-layer solution:

**Layer 1 — Expanded base schema (always present):**
Broadened from 7 to 14 entity types and 8 to 12 relationship types covering business, legal, financial, medical, and technical domains. This is the floor — even without schema induction, extraction is reasonable across diverse documents.

**Layer 2 — Per-document LLM schema induction (additive):**
A single cheap LLM call (~$0.001 with Haiku/GPT-4o-mini) per document analyzes a text sample and proposes domain-specific types with descriptions. These are merged on top of the base schema.

**Why not per-chunk induction?** The domain doesn't change within a document. One call per document is sufficient.

**Why not pre-defined schema profiles?** Too rigid — requires manual maintenance of profiles for every possible document domain. LLM induction generalizes zero-shot.

**What about scanned documents without OCR text?** Docling OCR runs first in the pipeline (it's already the first step). By the time schema induction is needed, extracted text is available. If OCR produces too little text (< 200 chars), the base schema is used as fallback.

---

## 6. Architecture

### 6.1 System Overview

```
                 ┌────────────────────────────────┐
                 │         Shared Storage          │
                 │                                 │
                 │  Qdrant          Neo4j          │
                 │  ├─ dense_docs   ├─ GLiNER2     │
                 │  └─ multivec     │  entities    │
                 │    (A only)      └─ Graphiti    │
                 │                     episodes    │
                 └───────┬──────────────┬──────────┘
                         │              │
          ┌──────────────┴──┐     ┌─────┴───────────────┐
          │ Path A: Bulk KB │     │ Path B: Live Txn     │
          │                 │     │                      │
          │ CocoIndex       │     │ FastAPI POST         │
          │ Docling OCR     │     │ Jina embed (dense)   │
          │ Jina embed      │     │ Graphiti ingest      │
          │  (dense+ColBERT)│     │  (temporal episodes) │
          │ GLiNER2 extract │     │                      │
          │  (dynamic schema│     │ session_id tagging   │
          └─────────────────┘     └──────────────────────┘
                         │              │
                         └──────┬───────┘
                        ┌───────┴────────┐
                        │  MCP Server    │
                        │  (FastMCP)     │
                        │  4 search tools│
                        │  + session ctx │
                        └───────┬────────┘
                                │
                        ┌───────┴────────┐
                        │  LLM Agent     │
                        │  (meeting      │
                        │  recommendations)
                        └────────────────┘
```

**Storage sharing:**
- **Qdrant:** Both paths write to `documents_dense`. Path A also writes to `documents_multivec` for visual content. Live transcript chunks are tagged with `is_live: true` and `session_id` in the payload.
- **Neo4j:** Single instance. GLiNER2 entities carry `source_engine: "gliner"`. Graphiti data is partitioned by `group_id = session_id`. No cross-engine entity resolution in v1 — search tools query both independently and merge results.

### 6.2 Neo4j Namespacing

**GLiNER2 entities (bulk KB):**
```
(:Entity {
  name: "Acme Corp",
  types: ["organization"],
  description: "...",
  source_engine: "gliner",
  source: "contracts/master-agreement.pdf",
  first_seen: datetime(),
  last_seen: datetime()
})
```

**Graphiti episodes (live transcript):**
```
Graphiti manages its own node/edge schema internally.
Partitioned by group_id = session_id.
Edges carry created_at, expired_at, valid_at timestamps.
```

---

## 7. Path A: Bulk KB Pipeline

### 7.1 Pipeline Changes

The existing CocoIndex pipeline is modified minimally. The only change is injecting the schema inducer before GLiNER2 extraction:

```
file_to_pages() → Docling OCR (if scan)
    │
    ├─ Text pages:
    │   ├─ semantic_chunk()
    │   ├─ Jina embed → Qdrant dense_docs
    │   ├─ schema_inducer(sample_text) → dynamic schema  [NEW]
    │   └─ GLiNER2 extract(dynamic_schema) → Neo4j       [MODIFIED]
    │
    └─ Visual pages:
        ├─ Jina embed → Qdrant dense_docs
        └─ Jina embed_multivec → Qdrant multivec (if enabled)
```

### 7.2 Schema Inducer

**New module: `ingestion/schema_inducer.py`**

```python
class SchemaInducer:
    """Proposes domain-specific entity/relationship types for a document."""

    async def induce(self, sample_text: str) -> InducedSchema:
        """Analyze sample text, return entity + relationship types with descriptions."""
        ...

    def merge_with_base(self, induced: InducedSchema) -> MergedSchema:
        """Merge induced types with base schema from constants.py."""
        ...
```

**LLM prompt structure:**

```
You are a knowledge graph schema designer. Given the following document excerpt,
propose entity types and relationship types that would capture the key information.

Return JSON with two keys:
- "entity_types": {"type_name": "description of what this type represents"}
- "relationship_types": {"rel_name": "description: X [rel] Y means..."}

Rules:
- Propose 3-8 entity types specific to this document's domain
- Propose 3-6 relationship types
- Descriptions must be clear enough for a non-expert NER model to use
- Do not duplicate these base types (they are already included): [list base types]

Document excerpt:
---
{sample_text}
---
```

**Schema cache:**

```python
_schema_cache: dict[str, tuple[MergedSchema, float]] = {}

def _cache_key(sample_text: str) -> str:
    """Hash first 500 chars + MIME type as domain signature."""
    return hashlib.sha256(sample_text[:500].encode()).hexdigest()
```

Cache TTL is configurable via `SCHEMA_CACHE_TTL` (default 3600s). Documents with similar openings (e.g. all contracts from the same template) hit the cache and skip the LLM call.

**Scanned documents:** Docling OCR runs before schema induction in the pipeline. By the time the inducer is called, OCR text is available. If OCR produces < 200 chars (badly degraded scan), the base schema is used as fallback — no LLM call.

### 7.3 GLiNER2 Engine Changes

`GLiNEREngine` currently builds the schema once in `__init__` from `constants.py`. Modified to accept per-document schemas:

```python
class GLiNEREngine:
    def __init__(self) -> None:
        self._extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        self._driver = AsyncGraphDatabase.driver(...)
        self._base_schema = self._build_base_schema()  # from constants.py

    async def ingest(
        self,
        chunks: list[TextChunk],
        source_key: str,
        schema: MergedSchema | None = None,  # NEW parameter
    ) -> None:
        active_schema = schema or self._base_schema
        # ... extraction using active_schema
```

The `GraphEngine` protocol gains the optional `schema` parameter on `ingest()`. `GraphitiEngine` ignores it (Graphiti discovers entity types autonomously via its own LLM).

### 7.4 Expanded Base Schema

The base schema is broadened from 7 → 14 entity types and 8 → 12 relationship types, covering business, legal, financial, medical, and technical domains. This is the floor that's always present — dynamic induction adds domain-specific types on top.

**Entity types (14):**

| Type | Description |
|-|-|
| `person` | A named individual, author, speaker, or public figure |
| `organization` | A company, institution, government body, or team |
| `location` | A physical place, address, region, country, or facility |
| `date_time` | A specific date, time period, deadline, or schedule reference |
| `monetary_value` | An amount of money, price, fee, budget, or financial figure |
| `document` | A named contract, agreement, report, policy, regulation, or standard |
| `product` | A named product, service, platform, or deliverable |
| `technology` | A programming language, framework, tool, protocol, or system |
| `metric` | A quantitative measure, KPI, percentage, statistic, or benchmark |
| `event` | A named meeting, conference, milestone, incident, or release |
| `legal_term` | A clause, obligation, right, liability, warranty, or legal concept |
| `role` | A job title, department, committee, or functional responsibility |
| `concept` | An abstract idea, methodology, strategy, design pattern, or practice |
| `requirement` | A specification, condition, constraint, criterion, or deliverable requirement |

**Relationship types (12):**

| Type | Description |
|-|-|
| `created_by` | X was created, authored, or produced by Y |
| `owned_by` | X is owned, managed, or governed by Y |
| `uses` | X uses, depends on, or integrates Y |
| `part_of` | X is a component, section, or subset of Y |
| `related_to` | X is associated with or relevant to Y |
| `measured_by` | X is measured, evaluated, or quantified by Y |
| `requires` | X requires, mandates, or depends on Y |
| `applies_to` | X applies to, governs, or regulates Y |
| `succeeds` | X replaces, supersedes, or follows Y |
| `conflicts_with` | X contradicts, opposes, or is incompatible with Y |
| `valued_at` | X has a monetary value, cost, or price of Y |
| `scheduled_for` | X is planned, due, or scheduled for Y |

---

## 8. Path B: Live Transcript Ingestion

### 8.1 Why a Separate HTTP Endpoint (Not CocoIndex)

CocoIndex is designed for batch processing with incremental state tracking. It reads from S3/local filesystem, manages source lineage, and writes state to PostgreSQL. None of this is needed for live transcripts:
- Transcripts arrive via push (HTTP POST), not pull (S3/filesystem)
- There's no incremental state to track — each chunk is new
- Latency must be < 5 seconds, not minutes
- Session lifecycle (start/end/archive/discard) has no CocoIndex equivalent

A lightweight FastAPI endpoint is the right tool for this.

### 8.2 API Endpoints

**New module: `ingestion/live_ingest.py`**

#### `POST /session/start`

Creates a new meeting session.

```json
// Request
{
  "session_id": "meeting-2026-03-06-1430",
  "metadata": {
    "title": "Q1 Budget Review",
    "participants": ["Alice", "Bob"]
  }
}

// Response
{
  "session_id": "meeting-2026-03-06-1430",
  "status": "active",
  "created_at": "2026-03-06T14:30:00Z"
}
```

Internally: initializes Graphiti `group_id` for this session.

#### `POST /ingest/transcript`

Ingests a single transcript chunk.

```json
// Request
{
  "session_id": "meeting-2026-03-06-1430",
  "text": "Alice: I think we should revisit the payment terms on the Acme contract. The current net-60 is causing cash flow issues.",
  "timestamp": "2026-03-06T14:32:00Z",
  "speaker": "Alice"
}

// Response (returned after Qdrant upsert, ~200ms)
{
  "status": "accepted",
  "vector_indexed": true,
  "graph_status": "processing"
}
```

**Processing flow:**

```python
async def ingest_transcript(chunk: TranscriptChunk) -> IngestResponse:
    # 1. Embed and upsert to Qdrant (awaited — ~200ms)
    vector = await embedder.embed_text([chunk.text])
    qdrant.upsert(
        collection_name=DENSE_COLLECTION,
        points=[PointStruct(
            id=make_point_id(f"{chunk.session_id}::{chunk.timestamp}"),
            vector=vector[0],
            payload={
                "source_file": f"session:{chunk.session_id}",
                "content_type": "transcript",
                "is_live": True,
                "session_id": chunk.session_id,
                "speaker": chunk.speaker,
                "timestamp": chunk.timestamp.isoformat(),
                "text_content": chunk.text,
            },
        )],
    )

    # 2. Graphiti ingest (background — ~2-5s)
    asyncio.create_task(_graphiti_ingest(chunk))

    return IngestResponse(
        status="accepted",
        vector_indexed=True,
        graph_status="processing",
    )
```

The endpoint returns after step 1 (~200ms). Step 2 runs as a background task. The LLM can immediately retrieve the chunk via vector search while the temporal graph catches up within seconds.

```python
async def _graphiti_ingest(chunk: TranscriptChunk) -> None:
    """Background task: ingest chunk as Graphiti episode."""
    client = await get_graphiti()
    episode_name = f"{chunk.session_id}:t{chunk.timestamp.isoformat()}"
    await client.add_episode(
        name=episode_name,
        episode_body=chunk.text,
        source_description=f"Meeting transcript, speaker: {chunk.speaker or 'unknown'}",
        reference_time=chunk.timestamp,
        group_id=chunk.session_id,
    )
```

#### `POST /session/end`

Ends a session and handles lifecycle.

```json
// Request
{
  "session_id": "meeting-2026-03-06-1430",
  "archive": true
}
```

**When `archive=true`:**
- Update all Qdrant points for this session: set `is_live=false`
- Graphiti data remains as-is (permanent temporal record)
- The meeting transcript becomes part of the searchable KB

**When `archive=false`:**
- Delete all Qdrant points where `session_id` matches
- Delete Graphiti episodes and related nodes/edges for `group_id`
- Meeting data is fully purged

### 8.3 Graphiti Configuration for Live Path

Graphiti is configured once at startup, reused across sessions:

```python
# Shared Graphiti client (existing singleton from graphiti_client.py)
client = await get_graphiti()
```

**Graphiti settings for low-latency use:**

| Setting | Value | Rationale |
|-|-|-|
| LLM model | Fast model (GPT-4o-mini or Haiku) | Minimize per-episode latency. Frontier models are unnecessary for short transcript chunks |
| `group_id` | `session_id` | Partition graph per meeting. Enables clean deletion on discard |
| `reference_time` | Chunk timestamp | Preserves meeting timeline for temporal queries |

The same Graphiti client that serves `graph_search` from the MCP server handles live ingestion. No separate instance needed.

### 8.4 Why Graphiti's Latency Is Acceptable

| Operation | Latency | Within budget? |
|-|-|-|
| Jina embed → Qdrant | ~200ms | Yes — vector search immediately available |
| Graphiti `add_episode()` | ~2–5s (short text) | Yes — completes before next 30s chunk |
| Graphiti `search()` at query time | ~50ms (graph traversal, no LLM) | Yes — well under budget |

The 2–5s Graphiti ingestion is for a single short transcript chunk (1–3 sentences). This is much faster than bulk document ingestion (~29 min for a 74-chunk PDF) because each episode is small. The vector search path provides immediate results while the graph enriches in the background.

---

## 9. Search & Retrieval During Meetings

### 9.1 Session-Aware MCP Tools

The four existing MCP tools gain an optional `session_id` parameter. When provided, search behavior changes to combine live transcript context with bulk KB results.

#### `vector_search`

```python
async def vector_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
    session_id: str | None = None,     # NEW
) -> list[SearchResult]:
```

When `session_id` is set, the tool runs **two** Qdrant queries in parallel:
1. Filter `session_id == X` — transcript chunks from this meeting
2. Filter `is_live != true` OR `is_live` field absent — bulk KB results

Results are merged, with transcript results marked as `source_type: "transcript"` and sorted by timestamp (chronological order, not similarity score — the LLM needs the timeline).

#### `graph_search`

```python
async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
    session_id: str | None = None,     # NEW
) -> list[GraphFact]:
```

When `session_id` is set:
1. Graphiti search with `group_id=session_id` — temporal facts from this meeting (entities, relationships, fact evolution with timestamps)
2. GLiNER2 full-text search — entity facts from bulk KB

Both result sets are returned, clearly labeled by `source_engine`.

#### `hybrid_search`

Combines the session-aware versions of both tools. Response structure gains a `transcript_results` section:

```python
class HybridSearchResponse(BaseModel):
    vector_results: list[SearchResult]        # KB vector matches
    transcript_results: list[SearchResult]     # session transcript matches (NEW)
    graph_results: list[GraphFact]             # combined graph facts from both engines
    query: str
    session_id: str | None = None              # NEW
    strategy: str = "parallel"
    errors: list[str] | None = None
```

### 9.2 Retrieval Example: The Contract Scenario

Agent calls during a meeting where the Acme contract was discussed at t=2min, t=15min, and t=45min. At t=50min someone asks about payment terms:

```json
{
  "name": "hybrid_search",
  "arguments": {
    "query": "Acme contract payment terms",
    "session_id": "meeting-2026-03-06-1430",
    "limit": 10
  }
}
```

Response:

```json
{
  "transcript_results": [
    {
      "text": "Alice: I think we should revisit the payment terms on the Acme contract. The current net-60 is causing cash flow issues.",
      "timestamp": "14:32",
      "speaker": "Alice",
      "score": 0.89
    },
    {
      "text": "Bob: Legal reviewed it — we can push for net-30 in the renewal.",
      "timestamp": "14:45",
      "speaker": "Bob",
      "score": 0.85
    }
  ],
  "vector_results": [
    {
      "text": "Standard payment terms policy: all new contracts default to net-30...",
      "source_file": "policies/payment-terms.pdf",
      "score": 0.91
    }
  ],
  "graph_results": [
    {
      "fact": "Acme Contract valued_at €1.2M",
      "created_at": "2026-03-06T14:32:00Z",
      "source_engine": "graphiti"
    },
    {
      "fact": "Acme Contract valued_at €1.5M",
      "created_at": "2026-03-06T14:45:00Z",
      "source_engine": "graphiti",
      "note": "supersedes €1.2M"
    },
    {
      "fact": "Acme Corp (organization) part_of Technology sector",
      "source_engine": "gliner",
      "confidence": 0.92
    }
  ],
  "session_id": "meeting-2026-03-06-1430",
  "strategy": "parallel"
}
```

The LLM receives three complementary views:
1. **Transcript timeline** — what was said, by whom, when (chronological)
2. **KB context** — relevant policies and reference documents (by relevance)
3. **Graph facts** — structured entities and relationships from both Graphiti (temporal, live) and GLiNER2 (static, KB)

This is sufficient to generate a recommendation like: *"Alice raised net-60 cash flow concerns at 14:32. Bob confirmed legal supports net-30 for the renewal at 14:45. Your company policy defaults to net-30. The contract value was updated from €1.2M to €1.5M during this meeting."*

---

## 10. File Changes

### 10.1 New Files

| File | Purpose |
|-|-|
| `ingestion/live_ingest.py` | FastAPI app: `/session/start`, `/ingest/transcript`, `/session/end` |
| `ingestion/schema_inducer.py` | LLM-based per-document schema induction with caching |

### 10.2 Modified Files

| File | Change |
|-|-|
| `config/constants.py` | Expanded base schema: 14 entity types, 12 relationship types |
| `config/settings.py` | New settings (see Section 11) |
| `ingestion/graph_engine.py` | `GLiNEREngine.ingest()` accepts optional `schema` parameter; `GraphEngine` protocol updated; `GLiNEREngine.__init__` builds schema from expanded constants |
| `ingestion/pipeline.py` | Call `SchemaInducer` before graph ingestion when enabled |
| `server/mcp_server.py` | Mount live ingest FastAPI app; configure Graphiti client for both search and live ingestion |
| `server/tools/vector_search.py` | Optional `session_id` parameter, dual Qdrant query |
| `server/tools/graph_search.py` | Optional `session_id`, query both Graphiti (by group_id) and GLiNER2 (full-text) |
| `server/tools/hybrid_search.py` | Session-aware merged response with `transcript_results` section |
| `server/models.py` | New: `TranscriptChunk`, `SessionControl`, `IngestResponse`; modified: `HybridSearchResponse` |

### 10.3 Unchanged

| File | Reason |
|-|-|
| `ingestion/qdrant_setup.py` | Same collections; payload filtering is dynamic |
| `ingestion/neo4j_setup.py` | GLiNER2 uses existing Entity schema; Graphiti manages its own |
| `ingestion/embedder.py` | Reused as-is by both paths |
| `ingestion/file_processor.py` | No changes needed |
| `ingestion/cocoindex_ops.py` | CocoIndex flow definition unchanged |

---

## 11. Configuration

### 11.1 New Settings

| Variable | Default | Description |
|-|-|-|
| `LIVE_INGEST_PORT` | `8001` | Port for the live transcript ingestion API |
| `SCHEMA_INDUCTION_ENABLED` | `true` | Enable per-document LLM schema induction for GLiNER2 |
| `SCHEMA_INDUCTION_MODEL` | `claude-haiku-4-5-20251001` | Fast/cheap model for schema proposals |
| `SCHEMA_CACHE_TTL` | `3600` | Seconds to cache induced schemas before re-inducing |

### 11.2 Existing Settings Reused

| Variable | Used by |
|-|-|
| `GRAPH_ENGINE` | Path A only (bulk KB). Path B always uses Graphiti regardless of this setting |
| `LLM_API_KEY` | Schema inducer (Path A) and Graphiti (Path B) |
| `LLM_MODEL` | Graphiti episode extraction (Path B) |
| `JINA_API_KEY` | Both paths (embedding) |
| `NEO4J_*` | Both paths (shared Neo4j instance) |
| `QDRANT_URL` | Both paths (shared Qdrant instance) |

---

## 12. Session Lifecycle

```
  External transcription service
            │
            ▼
  POST /session/start
  → Creates session_id
  → Initializes Graphiti group
            │
            ▼
  POST /ingest/transcript  (repeated every ~30s)
  → Jina embed → Qdrant (immediate, ~200ms)
  → Graphiti add_episode (background, ~2-5s)
            │
    [Meeting in progress — MCP tools query
     with session_id for live context]
            │
            ▼
  POST /session/end { archive: true }
  → Qdrant: set is_live=false on session points
  → Graphiti data: kept permanently
  → Transcript becomes part of searchable KB

  POST /session/end { archive: false }
  → Qdrant: delete points by session_id
  → Graphiti: delete episodes + edges by group_id
  → Meeting data fully purged
```

**Concurrent sessions:** Not supported in v1. One active session at a time. The `session/start` endpoint rejects if an active session exists. This simplifies Graphiti client management and avoids LLM rate limit contention between concurrent meetings.

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|-|-|-|
| Graphiti LLM latency exceeds 5s per chunk | Live graph data is stale for one cycle | Vector search is always immediate; graph catches up. Monitor p95 latency and switch to faster LLM model if needed |
| Schema inducer proposes bad types | Poor GLiNER2 extraction for one document | Expanded base schema is always present as fallback; induced types are additive only, cannot remove base types |
| LLM rate limits during meeting (Graphiti + schema induction) | Ingestion failures | Live path has priority; bulk ingestion should not run concurrently with meetings. Schema induction uses a separate cheap model |
| Neo4j namespace collision between engines | Corrupted graph queries | Explicit `source_engine` property on GLiNER2 entities; Graphiti uses `group_id` partitioning. Search tools query each engine's data independently |
| Qdrant `session/end archive=false` is slow for large sessions | Endpoint blocks | Batch delete by scroll + delete pattern; 3h meeting = ~360 points, negligible |
| Graphiti background task fails silently | Missing graph data for that chunk | Log errors from background tasks. Vector search still has the data. Consider retry queue in v2 |

---

## 14. Out of Scope (v1)

- **Cross-engine entity resolution:** Merging GLiNER2 entities with Graphiti entities sharing the same canonical name. The research recommends a "Normalization Layer" using Cypher queries or GDS algorithms — deferred to v2 when real-world data shows whether this is needed.
- **Multiple concurrent meetings:** Single active session in v1. Multiple sessions would require per-session Graphiti partitioning (already designed via `group_id`) but adds complexity in rate limit management.
- **Streaming WebSocket ingestion:** External service pushes via HTTP POST. WebSocket transport can be added later if polling latency becomes a concern.
- **AutoSchemaKG-style autonomous schema induction:** Full schema construction from web-scale corpora with conceptualization layers. Too heavy for per-document use. The lightweight single-LLM-call approach is sufficient for the expected document volumes (100–1,000 docs).
- **Reranking for transcript results:** Transcript results are sorted by timestamp (chronological), not relevance score. The LLM needs the timeline, not the "most relevant" chunk. Reranking may be added for KB results if retrieval quality is insufficient.
- **Tiered extraction consolidation:** Periodically running AutoSchemaKG-style conceptualization to unify GLiNER2 and Graphiti subgraphs into a coherent enterprise ontology. Valuable at scale but premature for current volumes.
