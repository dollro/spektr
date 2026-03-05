# Response Models

Pydantic models used by MCP search tools, defined in `server/models.py`.

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

Returned by the knowledge graph search tool. Represents a single fact extracted from the Neo4j temporal knowledge graph via Graphiti.

| Field | Type | Default | Description |
|-|-|-|-|
| `fact` | `str` | — | The extracted fact or relationship statement |
| `source` | `str \| None` | `None` | Source document the fact was extracted from |
| `created_at` | `str \| None` | `None` | ISO 8601 timestamp when the fact was created |
| `expired_at` | `str \| None` | `None` | ISO 8601 timestamp when the fact was superseded (if applicable) |

```json
{
  "fact": "Spektr uses Qdrant as its vector store",
  "source": "architecture-overview.md",
  "created_at": "2026-01-15T10:30:00Z",
  "expired_at": null
}
```

## HybridSearchResponse

Returned by the hybrid search tool, which executes vector and graph searches in parallel and fuses the results.

| Field | Type | Default | Description |
|-|-|-|-|
| `vector_results` | `list[SearchResult]` | `[]` | Dense vector search results |
| `graph_results` | `list[GraphFact]` | `[]` | Knowledge graph facts |
| `query` | `str` | — | The original search query |
| `strategy` | `str` | `"parallel"` | Fusion strategy used |

```json
{
  "vector_results": [
    {
      "score": 0.892,
      "text": "The transformer architecture uses self-attention mechanisms...",
      "source_file": "attention-paper.pdf",
      "page_number": 3,
      "content_type": "application/pdf",
      "metadata": {}
    }
  ],
  "graph_results": [
    {
      "fact": "Transformers replaced recurrent architectures for NLP tasks",
      "source": "attention-paper.pdf",
      "created_at": "2026-01-15T10:30:00Z",
      "expired_at": null
    }
  ],
  "query": "transformer architecture attention mechanism",
  "strategy": "parallel"
}
```
