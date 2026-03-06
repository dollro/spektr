# Search Tools

The MCP server exposes four search tools. Each tool returns structured JSON and handles errors gracefully by returning an error object instead of raising exceptions.

---

## `vector_search`

Dense semantic search against the Qdrant `documents_dense` collection. Embeds the query with Jina v4, performs nearest-neighbor lookup, and optionally reranks results when `RERANK_ENABLED=true`.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Natural language search query |
| `limit` | `int` | `10` | Maximum number of results |
| `content_type` | `str \| None` | `None` | MIME type filter (e.g. `application/pdf`) |
| `source_file` | `str \| None` | `None` | Source file name filter |

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
| `limit` | `int` | `5` | Maximum number of results |

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
| `limit` | `int` | `10` | Maximum number of results |

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

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Natural language search query |
| `limit` | `int` | `10` | Maximum results per backend |

### Return Schema

Returns `HybridSearchResponse`:

| Field | Type | Description |
|-|-|-|
| `vector_results` | `list[SearchResult]` | Dense vector search results |
| `graph_results` | `list[GraphFact]` | Knowledge graph facts |
| `query` | `str` | Original query |
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
