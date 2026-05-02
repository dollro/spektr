# Data Flow

Four data paths define how data moves through Spektr: **bulk ingest** (S3 to stores), **live ingest** (streaming text to stores), **query** (agent searches), and **delete/invalidation** (removing stale data).

---

## Bulk ingest path (Path A)

A document lands in S3, triggers an SQS event, and flows through the CocoIndex pipeline into Qdrant and Neo4j. When `SCHEMA_INDUCTION_ENABLED=true` and `GRAPH_ENGINE=gliner`, a per-document LLM call proposes domain-specific entity types before GLiNER2 extraction.

```mermaid
sequenceDiagram
    participant S3 as AWS S3
    participant SQS as AWS SQS
    participant Coco as CocoIndex Pipeline
    participant FP as File Processor
    participant Jina as Jina v4 API
    participant QD as Qdrant
    participant GE as Graph Engine<br/>(Graphiti or GLiNER2)
    participant PG as PostgreSQL

    S3->>SQS: S3 event notification (create/update)
    SQS->>Coco: Deliver event
    Coco->>Coco: Check state — skip if unchanged
    Coco->>FP: Read file bytes + filename

    FP->>FP: Guess MIME type
    alt Text file (md, txt, csv, json, xml, html, yaml)
        FP->>FP: Decode to text pages
    else PDF
        FP->>FP: pdf2image → page images + text extraction
    else Image (png, jpg, webp, ...)
        FP->>FP: Raw image bytes as single page
    end

    FP-->>Coco: List of pages (text and/or image)

    loop Each text page
        Coco->>FP: Semantic chunking
        FP-->>Coco: TextChunk list

        loop Each chunk
            Coco->>Jina: embed_text(chunk)
            Jina-->>Coco: dense vector (512-d)
            Coco->>QD: Upsert to documents_dense
        end

        Coco->>GE: engine.ingest(chunks, source_key)
        Note over GE: Graphiti: LLM extraction + episodes<br/>GLiNER2: local model + Cypher MERGE
    end

    loop Each PDF page
        alt Visual content detected (smart gating)
            Coco->>Jina: embed_image(resized page, 400px) → dense
            Jina-->>Coco: dense vector (512-d)
            Coco->>QD: Upsert to documents_dense

            opt ColBERT enabled
                Coco->>Jina: embed_image_multivec(page) → ColBERT
                Jina-->>Coco: multi-vector (N × 128-d)
                Coco->>QD: Upsert to documents_multivec
            end
        else Text-only page
            Coco->>QD: Store thumbnail (200px, no embedding)
        end
    end

    loop Each standalone image
        Coco->>Jina: embed_image(image) → dense
        Jina-->>Coco: dense vector (512-d)
        Coco->>QD: Upsert to documents_dense
    end

    Coco->>PG: Update pipeline state + ingestion log
```

### Key design decisions in the bulk ingest path

- **Deterministic IDs** -- chunk and point IDs are derived from `{source_file}::p{page}::c{chunk_idx}` via UUID5, making upserts idempotent.
- **Dual embedding for visual content** -- images get both a dense single-vector (for standard NN search) and ColBERT multi-vectors (for layout-aware retrieval). Text chunks get only dense vectors.
- **Dynamic schema induction** -- when enabled, a single LLM call per document proposes domain-specific entity/relationship types for GLiNER2. Results are cached by content hash. See [Knowledge Graph](../ingestion/knowledge-graph.md).
- **Pluggable graph engine** -- text chunks are passed to `engine.ingest()`. Graphiti submits them as LLM-processed episodes; GLiNER2 extracts entities and relations locally and writes directly to Neo4j. See [Knowledge Graph](../ingestion/knowledge-graph.md).

---

## Live ingest path (Path B)

Streaming text data arrives via HTTP POST to the live ingestion FastAPI server and is indexed into Qdrant immediately. Graphiti temporal graph ingestion runs as a background task.

```mermaid
sequenceDiagram
    participant Src as External Source
    participant API as Live Ingest API<br/>(FastAPI, port 8001)
    participant Jina as Jina v4 API
    participant QD as Qdrant
    participant Graphiti as Graphiti
    participant Neo4j as Neo4j

    Src->>API: POST /session/start {session_id, metadata}
    API-->>Src: {status: "active"}

    loop Each text chunk (~every 30s)
        Src->>API: POST /ingest/chunk {session_id, text, timestamp}
        API->>Jina: embed_text(text)
        Jina-->>API: dense vector (512-d)
        API->>QD: Upsert to documents_dense<br/>(is_live=true, session_id, timestamp)
        API-->>Src: {status: "accepted", vector_indexed: true, graph_status: "processing"}

        Note over API,Graphiti: Background task (does not block response)
        API->>Graphiti: add_episode(text, group_id=session_id, reference_time=timestamp)
        Graphiti->>Neo4j: Create/update temporal entities and edges
    end

    Src->>API: POST /session/end {session_id, archive}
    alt archive = true
        API->>QD: Set is_live=false on session points
        Note over Neo4j: Graphiti data kept permanently
    else archive = false
        API->>QD: Delete points by session_id
        API->>Graphiti: Delete episodes by group_id
    end
    API-->>Src: {status: "ended"}
```

### Key design decisions in the live ingest path

- **Immediate vector availability** -- the HTTP response returns after Qdrant upsert (~200ms). Graphiti runs in the background (~2-5s per chunk).
- **Session isolation** -- all live data is tagged with `session_id` and `is_live=true`. Graphiti partitions by `group_id=session_id`.
- **Clean lifecycle** -- sessions can be archived (data becomes permanent KB) or discarded (full purge from both Qdrant and Neo4j).
- **Single active session** -- v1 supports one active session at a time to avoid LLM rate limit contention.

---

## Query path

An LLM agent calls one of the six MCP tools. The four search tools (`vector_search`, `visual_search`, `graph_search`, `hybrid_search`) embed the query and search the relevant store. The two listing/inventory tools (`list_documents`, `list_document_chunks`) enumerate what's in the corpus without running a similarity query — useful for agents that need to ground a question against an exhaustive view of the available material before searching. The sequence diagram below covers the search tools; listing tools follow a simpler path (Auth → Tool → Qdrant scroll → results).

```mermaid
sequenceDiagram
    participant Agent as LLM Agent
    participant MCP as FastMCP Server
    participant Auth as Bearer Auth Middleware
    participant Tool as Search Tool
    participant Jina as Jina v4 API
    participant QD as Qdrant
    participant GE as Graph Engine (Neo4j)

    Agent->>MCP: tools/call (MCP protocol)
    MCP->>Auth: Check Authorization header
    alt MCP_API_KEY is set
        Auth->>Auth: Validate Bearer token
        Note over Auth: Reject if missing or invalid
    end
    Auth->>Tool: Dispatch to tool function

    alt vector_search
        Tool->>Jina: embed_text_query(query)
        Jina-->>Tool: query vector (512-d)
        Tool->>QD: query_points(documents_dense, vector, filters)
        QD-->>Tool: scored results
    else visual_search
        Tool->>Jina: embed_query_multi_vector(query)
        Jina-->>Tool: query multi-vectors (N × 128-d)
        Tool->>QD: query_points(documents_multivec, "colbert")
        QD-->>Tool: scored results
        opt VLM generation enabled
            Tool->>Tool: generate_visual_answer(query, results)
        end
    else graph_search
        Tool->>GE: engine.search(query)
        GE-->>Tool: GraphFact results
    else hybrid_search
        par Parallel execution
            Tool->>QD: vector_search(query)
        and
            Tool->>GE: graph_search(query)
        end
        QD-->>Tool: vector results
        GE-->>Tool: graph results
        opt Reranking enabled
            Tool->>Tool: rerank(query, vector_results)
        end
        Tool->>Tool: Merge into combined response
    end

    Tool-->>Agent: Search results (JSON)
```

### Tool summary

| Tool | Backend | Embedding | Best for |
|-|-|-|-|
| `vector_search` | Qdrant `documents_dense` | Dense (provider-dependent) | General semantic search over text and images |
| `visual_search` | Qdrant `documents_multivec` | ColBERT 128-d | Charts, diagrams, tables, formatted content |
| `graph_search` | Neo4j via GraphEngine | Graphiti semantic or Neo4j full-text | Entity lookup, facts, relationships |
| `hybrid_search` | Both (parallel) | Dense (provider-dependent) | Comprehensive retrieval combining both stores |
| `list_documents` | Qdrant scroll | None | Enumerating distinct source files in the corpus |
| `list_document_chunks` | Qdrant scroll | None | Exhaustive paginated listing of chunks for a given document |

### Filtering and reranking

- **`vector_search`** supports optional `content_type` and `source_file` payload filters passed to Qdrant.
- **Reranking** -- when `RERANK_ENABLED=true`, `vector_search` and `hybrid_search` pass results through a reranker before returning.
- **VLM generation** -- when `VLM_GENERATION_ENABLED=true`, `visual_search` generates a natural-language answer from the top visual results and prepends it.

### Error handling

All tools catch exceptions and return a structured error object (`{"error": "...", "query": "..."}`) instead of raising, so agents always get a usable response. `hybrid_search` runs both backends in parallel and reports partial failures in an `errors` list while still returning results from the successful backend.

---

## Delete / invalidation path

When a file is deleted from S3, the SQS event reaches the CocoIndex pipeline. CocoIndex's incremental state tracking detects the deletion and removes the file from its state table.

```mermaid
sequenceDiagram
    participant S3 as AWS S3
    participant SQS as AWS SQS
    participant Coco as CocoIndex Pipeline
    participant PG as PostgreSQL

    S3->>SQS: S3 event notification (delete)
    SQS->>Coco: Deliver delete event
    Coco->>PG: Remove file from pipeline state
    Note over Coco: Qdrant points and Neo4j entities<br/>from the deleted file remain until<br/>a full re-index or manual cleanup
```

!!! warning "Stale data after deletion"
    CocoIndex currently tracks deletions in its own state, but does not propagate deletes to Qdrant or Neo4j. Vectors and graph entities from deleted files persist until a full re-index. In Graphiti, temporal metadata (`expired_at`) can mark facts as outdated, but this requires explicit invalidation via the Graphiti API.

### Re-index strategy

To fully clean stale data:

1. Clear Qdrant collections: delete and recreate `documents_dense` and `documents_multivec`
2. Reset Neo4j graph: clear Graphiti episodes or drop and recreate the database
3. Reset CocoIndex state: drop the PostgreSQL ingestion tables
4. Re-run the pipeline: `uv run python -m ingestion.pipeline`
