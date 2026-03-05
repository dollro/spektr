# Data Flow

Three data paths define how documents move through Spektr: **ingest** (S3 to stores), **query** (agent searches), and **delete/invalidation** (removing stale data).

---

## Ingest path

A document lands in S3, triggers an SQS event, and flows through the CocoIndex pipeline into Qdrant and Neo4j.

```mermaid
sequenceDiagram
    participant S3 as AWS S3
    participant SQS as AWS SQS
    participant Coco as CocoIndex Pipeline
    participant FP as File Processor
    participant Jina as Jina v4 API
    participant QD as Qdrant
    participant Gr as Graphiti (Neo4j)
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
            Jina-->>Coco: dense vector (2048-d)
            Coco->>QD: Upsert to documents_dense
        end

        Coco->>Gr: add_episode(chunk_text, source, time)
        Note over Gr: Graphiti extracts entities,<br/>discovers relationships,<br/>tracks temporal metadata
    end

    loop Each visual page (image / PDF page)
        Coco->>Jina: embed_image(page) → dense
        Jina-->>Coco: dense vector (2048-d)
        Coco->>QD: Upsert to documents_dense

        Coco->>Jina: embed_image_multivec(page) → ColBERT
        Jina-->>Coco: multi-vector (N × 128-d)
        Coco->>QD: Upsert to documents_multivec
    end

    Coco->>PG: Update pipeline state + ingestion log
```

### Key design decisions in the ingest path

- **Deterministic IDs** -- chunk and point IDs are derived from `{source_file}::p{page}::c{chunk_idx}` via UUID5, making upserts idempotent.
- **Dual embedding for visual content** -- images get both a dense single-vector (for standard NN search) and ColBERT multi-vectors (for layout-aware retrieval). Text chunks get only dense vectors.
- **Graphiti episode ingestion** -- text chunks are submitted as episodes. Graphiti's internal LLM pipeline handles entity extraction, deduplication, and relationship discovery. No separate entity extraction step in Spektr's code.

---

## Query path

An LLM agent calls one of the four MCP tools. The server embeds the query, searches the relevant store, and returns results.

```mermaid
sequenceDiagram
    participant Agent as LLM Agent
    participant MCP as FastMCP Server
    participant Auth as Bearer Auth Middleware
    participant Tool as Search Tool
    participant Jina as Jina v4 API
    participant QD as Qdrant
    participant Gr as Graphiti (Neo4j)

    Agent->>MCP: tools/call (MCP protocol)
    MCP->>Auth: Check Authorization header
    alt MCP_API_KEY is set
        Auth->>Auth: Validate Bearer token
        Note over Auth: Reject if missing or invalid
    end
    Auth->>Tool: Dispatch to tool function

    alt vector_search
        Tool->>Jina: embed_text_query(query)
        Jina-->>Tool: query vector (2048-d)
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
        Tool->>Gr: client.search(query)
        Gr-->>Tool: edges with facts, sources, timestamps
    else hybrid_search
        par Parallel execution
            Tool->>QD: vector_search(query)
        and
            Tool->>Gr: graph_search(query)
        end
        QD-->>Tool: vector results
        Gr-->>Tool: graph results
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
| `vector_search` | Qdrant `documents_dense` | Dense 2048-d | General semantic search over text and images |
| `visual_search` | Qdrant `documents_multivec` | ColBERT 128-d | Charts, diagrams, tables, formatted content |
| `graph_search` | Neo4j via Graphiti | None (Graphiti semantic) | Entity lookup, temporal facts, relationships |
| `hybrid_search` | Both (parallel) | Dense 2048-d | Comprehensive retrieval combining both stores |

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
