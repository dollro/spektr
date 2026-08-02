# Response Models

Pydantic models used by MCP search tools and the live ingestion API, defined in `server/models.py`.

The agent HTTP layer defines its own request/response models (`QueryRequest`,
`QueryResponse`) in `agent/api.py` — see [HTTP API](../agent/http-api.md)
for their fields and usage.

## SearchResult

Returned by the dense vector search tool.

| Field | Type | Default | Description |
|-|-|-|-|
| `score` | `float` | — | Cosine similarity score |
| `text` | `str` | — | Matched text chunk |
| `source_file` | `str` | — | Original filename |
| `page_number` | `int` | — | Page number within source file |
| `content_type` | `str` | — | MIME content type (`text/plain`, `application/pdf`, etc.) |
| `metadata` | `dict` | `{}` | Additional metadata from the document |

```json
{
  "score": 0.892,
  "text": "The transformer architecture uses self-attention mechanisms...",
  "source_file": "attention-paper.pdf",
  "page_number": 3,
  "content_type": "application/pdf",
  "metadata": {"chunk_index": 7}
}
```

## VisualSearchResult

Returned by the ColBERT multi-vector visual search tool. Unlike `SearchResult`, this model returns a source key instead of text content, since visual results reference images.

| Field | Type | Default | Description |
|-|-|-|-|
| `score` | `float` | — | ColBERT MaxSim similarity score |
| `source_file` | `str` | — | Original filename |
| `page_number` | `int` | — | Page number within source file |
| `content_type` | `str` | — | MIME content type (`image/png`, `image/jpeg`, etc.) |
| `source_key` | `str` | — | Source-agnostic key for the document (S3 key or local path) |
| `metadata` | `dict` | `{}` | Additional metadata from the document |

```json
{
  "score": 0.781,
  "source_file": "architecture-diagram.png",
  "page_number": 1,
  "content_type": "image/png",
  "source_key": "documents/architecture-diagram.png",
  "metadata": {}
}
```

## GraphFact

Returned by the knowledge graph search tool. Represents a single fact extracted from Neo4j. Fields populated depend on the active graph engine (`GRAPH_ENGINE` setting).

| Field | Type | Default | Description |
|-|-|-|-|
| `fact` | `str` | — | The extracted fact or relationship statement |
| `source` | `str \| None` | `None` | Source document the fact was extracted from |
| `created_at` | `str \| None` | `None` | ISO 8601 timestamp when the fact was created (Graphiti) |
| `expired_at` | `str \| None` | `None` | ISO 8601 timestamp when the fact was superseded (Graphiti) |
| `entities` | `list[str] \| None` | `None` | Entity names involved in the fact (GLiNER) |
| `relation_type` | `str \| None` | `None` | Typed relationship label (GLiNER) |
| `confidence` | `float \| None` | `None` | Extraction confidence score (GLiNER) |

Graphiti example:

```json
{
  "fact": "Spektr uses Qdrant as its vector store",
  "source": "architecture-overview.md",
  "created_at": "2026-01-15T10:30:00Z",
  "expired_at": null
}
```

GLiNER example:

```json
{
  "fact": "Spektr uses Qdrant",
  "entities": ["Spektr", "Qdrant"],
  "relation_type": "USES",
  "confidence": 1.0
}
```

## FusedSearchResult

One rank-fused, optionally reranked hit. Returned inside `results` and `live_results` by both `multi_search` and `hybrid_search`.

| Field | Type | Default | Description |
|-|-|-|-|
| `id` | `str` | — | Qdrant point ID |
| `text` | `str` | — | Matched text chunk |
| `source_file` | `str` | — | Original filename |
| `page_number` | `int` | `0` | Page number within source file |
| `chunk_index` | `int` | `0` | Chunk position within the page |
| `score` | `float` | — | Rerank score when reranking is enabled, otherwise equal to `fusion_score` |
| `fusion_score` | `float` | — | Reciprocal Rank Fusion score across retrieval channels |
| `channels` | `list[str]` | `[]` | Which channel(s) surfaced this hit — `dense`, `sparse`, or both |
| `metadata` | `dict` | `{}` | Additional metadata from the document |

```json
{
  "id": "3f1c9e2a-...",
  "text": "The transformer architecture uses self-attention mechanisms...",
  "source_file": "attention-paper.pdf",
  "page_number": 3,
  "chunk_index": 5,
  "score": 0.312,
  "fusion_score": 0.0323,
  "channels": ["dense", "sparse"],
  "metadata": {}
}
```

## FusedSearchResponse

BREAKING CHANGE: this replaces the old `HybridSearchResponse` shape
(`vector_results`/`graph_results`/`strategy`/`errors`). See
[Search Tools — Breaking change](../server/search-tools.md) for the
before/after migration table.

Returned by both `multi_search` (deterministic, no LLM) and `hybrid_search`
(adds query decomposition and a relevance-gated retry) — the two tools share
this exact schema. When `session_id` is provided, live session data is
separated into `live_results`.

| Field | Type | Default | Description |
|-|-|-|-|
| `results` | `list[FusedSearchResult]` | `[]` | Fused, reranked KB results, best first |
| `graph_facts` | `list[GraphFact]` | `[]` | Knowledge graph facts, queried in parallel and not fused into the ranking |
| `live_results` | `list[FusedSearchResult]` | `[]` | Live session hits (only populated with `session_id`) |
| `query` | `str` | — | The original search query |
| `session_id` | `str \| None` | `None` | Session ID if session-aware search was used |
| `sub_queries` | `list[str] \| None` | `None` | `hybrid_search` only — the decomposed sub-queries actually used |
| `retried` | `bool \| None` | `None` | `hybrid_search` only — `true` if the relevance gate fired a retry |
| `degraded` | `list[str] \| None` | `None` | **Omitted when healthy.** Present only on partial failure — the channel names that failed (`dense`, `sparse`, `rerank`, `graph`) |

Not part of the `FusedSearchResponse` model, but present on the raw dict a tool returns: an `error: str` key appears only when **both** `dense` and `sparse` failed, signalling total retrieval outage rather than a query that matched nothing (see `server/tools/multi_search.py::shape_response`).

```json
{
  "results": [
    {
      "id": "3f1c9e2a-...",
      "text": "The transformer architecture uses self-attention mechanisms...",
      "source_file": "attention-paper.pdf",
      "page_number": 3,
      "chunk_index": 5,
      "score": 0.312,
      "fusion_score": 0.0323,
      "channels": ["dense", "sparse"],
      "metadata": {}
    }
  ],
  "live_results": [],
  "graph_facts": [
    {
      "fact": "Transformers replaced recurrent architectures for NLP tasks",
      "source": "attention-paper.pdf",
      "created_at": "2026-01-15T10:30:00Z",
      "expired_at": null
    }
  ],
  "query": "transformer architecture attention mechanism",
  "session_id": null
}
```

## LiveChunk

Request model for live ingestion (`POST /ingest/chunk`).

| Field | Type | Default | Description |
|-|-|-|-|
| `session_id` | `str` | — | Active session identifier |
| `text` | `str` | — | Text content of the chunk |
| `timestamp` | `datetime` | — | When the chunk was produced |

## SessionStartRequest

Request model for starting a live session (`POST /session/start`).

| Field | Type | Default | Description |
|-|-|-|-|
| `session_id` | `str` | — | Unique session identifier |
| `metadata` | `dict` | `{}` | Arbitrary session metadata |

## SessionEndRequest

Request model for ending a live session (`POST /session/end`).

| Field | Type | Default | Description |
|-|-|-|-|
| `session_id` | `str` | — | Session to end |
| `archive` | `bool` | `false` | `true` keeps data permanently; `false` purges all session data |

## IngestResponse

Response from live text ingestion.

| Field | Type | Default | Description |
|-|-|-|-|
| `status` | `str` | — | `"accepted"` on success |
| `vector_indexed` | `bool` | — | Whether the chunk was indexed in Qdrant |
| `graph_status` | `str` | — | `"processing"` (Graphiti runs as background task) |
