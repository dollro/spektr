# Search Tools

The MCP server exposes the following tools. Each returns structured JSON and handles errors gracefully by returning an error object instead of raising exceptions.

| Tool | Use case |
|-|-|
| `vector_search` | Best for **specific** questions answered by a few chunks. Vector similarity ranks and truncates. |
| `graph_search` | Best for **entity / relationship** queries. |
| `hybrid_search` | Best general-purpose tool: vector + graph in parallel. |
| `list_documents` | **Discovery** — what's in the knowledge base. |
| `list_document_chunks` | **Exhaustive enumeration** of one source document — use when vector top-k truncation hides parts of a long document. |

Three of the search tools (`vector_search`, `graph_search`, `hybrid_search`) support an optional `session_id` parameter for session-aware search. When provided, search results combine data from the active session (live-ingested chunks) with bulk KB results.

---

## `vector_search`

Dense semantic search against the Qdrant `documents_dense` collection. Embeds the query with the configured embedder (Jina v4 by default; Voyage or OpenRouter when configured), performs nearest-neighbor lookup, and optionally reranks results when `RERANK_ENABLED=true`.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Natural language search query |
| `limit` | `int` | `10` | Maximum number of results. Clamped to `[1, 100]` |
| `content_type` | `str \| None` | `None` | MIME type filter (e.g. `application/pdf`) |
| `source_file` | `str \| None` | `None` | Source file name filter |
| `session_id` | `str \| None` | `None` | When set, runs dual queries: one for session data (filtered by `session_id`), one for bulk KB (excluding live data). Session results are sorted chronologically by timestamp |

### Return Schema

Returns `list[SearchResult]`:

| Field | Type | Description |
|-|-|-|
| `score` | `float` | Similarity score |
| `text` | `str` | Matched text chunk |
| `source_file` | `str` | Original file name |
| `page_number` | `int` | Page within the source document |
| `content_type` | `str` | MIME type of the source |
| `metadata` | `dict` | Additional payload metadata |

### Example

```json
{
  "name": "vector_search",
  "arguments": {
    "query": "quarterly revenue breakdown",
    "limit": 5,
    "content_type": "application/pdf"
  }
}
```

---

## `visual_search`

ColBERT multi-vector search against the Qdrant `documents_multivec` collection. Best suited for visually rich content: charts, diagrams, tables, and formatted layouts. When `VLM_GENERATION_ENABLED=true`, a VLM-generated answer is prepended to the results.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Natural language search query |
| `limit` | `int` | `5` | Maximum number of results. Clamped to `[1, 100]` |

### Return Schema

Returns `list[VisualSearchResult]`:

| Field | Type | Description |
|-|-|-|
| `score` | `float` | ColBERT similarity score |
| `source_file` | `str` | Original file name |
| `page_number` | `int` | Page within the source document |
| `content_type` | `str` | MIME type of the source |
| `source_key` | `str` | Source-agnostic key for the page image |
| `metadata` | `dict` | Additional payload metadata |

When VLM generation is enabled, the first element may be a VLM answer:

| Field | Type | Description |
|-|-|-|
| `type` | `str` | Always `"vlm_answer"` |
| `answer` | `str` | Generated answer from the vision-language model |
| `query` | `str` | Original query |

### Example

```json
{
  "name": "visual_search",
  "arguments": {
    "query": "architecture diagram showing data flow",
    "limit": 3
  }
}
```

---

## `graph_search`

Searches the Neo4j knowledge graph for entities and relationships. The search backend depends on the `GRAPH_ENGINE` setting -- Graphiti uses its semantic search API, GLiNER2 uses a Neo4j full-text index with relationship traversal. Both return `GraphFact` results.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Search text |
| `search_type` | `str` | `"entity"` | Search mode. Currently only `"entity"` is supported; reserved for future modes |
| `limit` | `int` | `10` | Maximum number of results. Clamped to `[1, 100]` |
| `session_id` | `str \| None` | `None` | When set, also queries Graphiti with `group_ids=[session_id]` for temporal facts from the active session, in addition to standard engine search |

### Return Schema

Returns `list[GraphFact]`:

| Field | Type | Description |
|-|-|-|
| `fact` | `str` | The extracted fact or relationship |
| `source` | `str \| None` | Description of the source |
| `created_at` | `str \| None` | Timestamp when the fact was created (Graphiti only) |
| `expired_at` | `str \| None` | Timestamp when the fact was superseded (Graphiti only) |
| `entities` | `list[str] \| None` | Entity names in the fact (GLiNER only) |
| `relation_type` | `str \| None` | Typed relationship label (GLiNER only) |
| `confidence` | `float \| None` | Extraction confidence (GLiNER only) |

### Example

```json
{
  "name": "graph_search",
  "arguments": {
    "query": "Who is the CTO of Acme Corp?",
    "limit": 5
  }
}
```

---

## `hybrid_search`

Runs `vector_search` and `graph_search` in parallel using `asyncio.create_task`, combining results from both backends. Handles partial failures gracefully -- if one backend fails, results from the other are still returned.

When `RERANK_ENABLED=true`, vector results are reranked before being included in the response.

**Graph-vs-vector deduplication.** After both backends return, graph facts whose `source` matches the `source_file` of any vector hit are dropped. This avoids surfacing the same document twice (once as a chunk, once as a graph fact). Dedup is skipped when either backend errored.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Natural language search query |
| `limit` | `int` | `10` | Maximum results per backend. Clamped to `[1, 100]` |
| `session_id` | `str \| None` | `None` | When set, separates live session results into `transcript_results` and bulk KB results into `vector_results` |

### Return Schema

Returns `HybridSearchResponse`:

| Field | Type | Description |
|-|-|-|
| `vector_results` | `list[SearchResult]` | Dense vector search results (bulk KB) |
| `transcript_results` | `list[SearchResult]` | Session transcript results, sorted chronologically (only populated when `session_id` is set) |
| `graph_results` | `list[GraphFact]` | Knowledge graph facts (combined from both engines when session-aware) |
| `query` | `str` | Original query |
| `session_id` | `str \| None` | Session ID if session-aware search was used |
| `strategy` | `str` | Always `"parallel"` |
| `errors` | `list[str]` | Present only if one or both backends failed |

### Example

```json
{
  "name": "hybrid_search",
  "arguments": {
    "query": "latest compliance requirements",
    "limit": 10
  }
}
```

---

## `list_documents`

Enumerate distinct documents ingested into the knowledge base, with chunk and page counts. Scrolls the Qdrant `documents_dense` collection and excludes live transcript data (`is_live=True`).

Use this for discovery — answering "what's in the knowledge base?" before drilling in with `vector_search` or `list_document_chunks`.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `limit` | `int` | `100` | Max number of documents to return. Clamped to `[1, 1000]` |

### Return Schema

Returns `list[dict]` sorted by `source_file`:

| Field | Type | Description |
|-|-|-|
| `source_file` | `str` | Document key (file name) |
| `chunk_count` | `int` | Number of chunks ingested for this document |
| `page_count` | `int` | Number of distinct pages observed |
| `content_types` | `list[str]` | Sorted list of MIME / payload content types present |

### Example

```json
{
  "name": "list_documents",
  "arguments": {
    "limit": 50
  }
}
```

---

## `list_document_chunks`

Enumerate **all** chunks of a single document in deterministic `(page_number, chunk_index)` order. No similarity ranking, no reranker — just paginated reads.

Use this when an agent self-detects partial coverage on a long document (e.g. a 90-page portfolio matrix), or whenever exhaustive content is required. Vector top-k will always truncate the long tail; this tool does not.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `source_file` | `str` | required | Document key (as returned by `list_documents`). |
| `page_from` | `int \| None` | `None` | Inclusive lower page bound. |
| `page_to` | `int \| None` | `None` | Inclusive upper page bound. |
| `content_type` | `str \| None` | `None` | Optional payload filter (e.g. `text_chunk`). |
| `limit` | `int` | `200` | Max chunks per response. Hard cap `500`. |
| `offset` | `int` | `0` | Number of chunks to skip — paginate by adding `limit` between calls. |

### Return Schema

Returns `list[dict]` ordered by `(page_number, chunk_index)`:

| Field | Type | Description |
|-|-|-|
| `source_file` | `str` | Same as the `source_file` argument. |
| `page_number` | `int` | Page within the source. |
| `chunk_index` | `int` | Chunk position within the page. |
| `content_type` | `str` | e.g. `text_chunk`. |
| `text` | `str` | The chunk's textual content. |
| `metadata` | `dict` | mime_type, ingested_at, source_key. |

### Pagination convention

When `len(result) == limit`, more chunks remain — call again with `offset += limit`. When the response is shorter than `limit` (or empty), the document is fully enumerated.

### Example

```json
{
  "name": "list_document_chunks",
  "arguments": {
    "source_file": "portfolio_matrix.pdf",
    "page_from": 30,
    "page_to": 60,
    "limit": 200
  }
}
```

### When to choose this over `vector_search`

- Question is "give me **all** X in this document" rather than "find X most relevant to Y".
- Document is long (40+ pages) and the agent needs comprehensive coverage.
- Vector search returned hits but the agent suspects coverage is partial.

---

## Error Handling

All tools catch exceptions internally and return an error object rather than raising:

```json
{
  "error": "vector_search failed: Connection refused",
  "query": "original query",
  "partial_results": []
}
```

This ensures the MCP server remains operational and the calling agent receives a structured error it can reason about.
