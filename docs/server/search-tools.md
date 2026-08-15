# Search Tools

!!! warning "Breaking change — hybrid_search response shape"
    As of the retrieval upgrade, `hybrid_search` no longer returns
    `{vector_results, graph_results, live_results, strategy, errors}`. It now
    returns a single ranked `results` list plus `graph_facts`, and a new
    `multi_search` tool exists alongside it.

    | Before | After |
    |-|-|
    | `vector_results` | `results` |
    | `graph_results` | `graph_facts` |
    | `live_results` | `live_results` (unchanged) |
    | `strategy` | removed |
    | `errors` | `degraded` (channel names, not messages) |

    Each result gains `id`, `chunk_index`, `fusion_score`, and `channels`.
    `channels` records which retrieval channel(s) surfaced the hit — `dense`,
    `sparse`, or both.

    **Migration:** replace `response["vector_results"]` with
    `response["results"]` and `response["graph_results"]` with
    `response["graph_facts"]`. Ranking now happens server-side (RRF fusion +
    reranking), so clients that re-sorted results themselves should stop.

    `degraded` is **omitted entirely** on a healthy run — its presence always
    means partial failure. An `error` key appears only when both the dense
    and sparse channels failed, distinguishing "retrieval is down" from
    "query matched nothing."

The MCP server exposes the following tools. Each returns structured JSON and handles errors gracefully by returning an error object instead of raising exceptions.

| Tool | Use case |
|-|-|
| `vector_search` | Best for **specific** questions answered by a few chunks. Vector similarity ranks and truncates. |
| `graph_search` | Best for **entity / relationship** queries. |
| `multi_search` | Fast, deterministic dense + sparse fusion with reranking. No LLM calls. The default general-purpose tool. |
| `hybrid_search` | Same fused pipeline as `multi_search`, plus query decomposition and a relevance-gated retry. Costs one extra LLM call. |
| `list_documents` | **Discovery** — what's in the knowledge base. |
| `list_document_chunks` | **Exhaustive enumeration** of one source document — use when vector top-k truncation hides parts of a long document. |

Four of the search tools (`vector_search`, `graph_search`, `multi_search`, `hybrid_search`) support an optional `session_id` parameter for session-aware search. When provided, search results combine data from the active live session with bulk KB results.

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
| `session_id` | `str \| None` | `None` | When set, runs dual queries: one for live session data (filtered by `session_id`), one for bulk KB (excluding live data). Session results are sorted chronologically by timestamp |

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

!!! warning "This tool requires `jina-v4` + `native`"
    It queries `documents_multivec` with `using="colbert"` and has **no dense fallback**, so it returns nothing unless `MULTIVEC_ENABLED=true` — which only `jina-v4` on the `native` route supports.

    **Searching images does not require this tool.** With `IMAGE_EMBED_STRATEGY=smart|all` on a pair that can embed images (including the default `gemini-2` + `openrouter`), page images become points in `documents_dense` alongside text, in the same vector space. A plain text query therefore retrieves them through `vector_search`, `multi_search`, and `hybrid_search`. `visual_search` adds late-interaction *precision* on top; it is not the only path to image content.

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

## `multi_search`

Deterministic fused search: dense + sparse retrieval, merged with Reciprocal Rank Fusion, then reranked. No LLM calls anywhere in this path — use it when latency and cost matter more than recall on hard multi-part questions.

Runs two Qdrant channels concurrently against `documents_dense`'s named vectors: `dense` (semantic similarity) and `sparse` (miniCOIL lexical matching, see [Embeddings](../ingestion/embeddings.md)). The channels never compare scores directly — cosine similarity and miniCOIL scores are not on the same scale. Instead they're fused by rank: [Reciprocal Rank Fusion](https://en.wikipedia.org/wiki/Reciprocal_rank_fusion) with `k=RRF_K` (default 60) scores each hit as `sum(1 / (k + rank))` across every channel that returned it. When `RERANK_ENABLED=true` (default), the top `RERANK_CANDIDATES` (default 50) fused results are rescored by `jina-reranker-v3.5`, a listwise reranker — see the note on its score scale under `hybrid_search` below.

Graph facts are queried in parallel via `graph_search` and returned as separate supporting context — they are not fused into the ranking.

Any channel that fails (dense, sparse, rerank, or graph) is recorded in `degraded` rather than aborting the whole call; the tool still returns whatever channels succeeded.

### Parameters

| Name | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | required | Natural language search query |
| `limit` | `int` | `10` | Maximum fused results returned. Clamped to `[1, 100]` |
| `content_type` | `str \| None` | `None` | MIME type filter (e.g. `application/pdf`) |
| `source_file` | `str \| None` | `None` | Source file name filter |
| `session_id` | `str \| None` | `None` | When set, live-session hits are separated into `live_results`; bulk KB hits stay in `results` |

### Return Schema

Returns a dict:

| Field | Type | Description |
|-|-|-|
| `results` | `list[FusedResult]` | Fused, reranked KB results, best first |
| `graph_facts` | `list[GraphFact]` | Knowledge graph facts (see [`graph_search`](#graph_search)); not fused into the ranking |
| `live_results` | `list[FusedResult]` | Live session hits, separated out when `session_id` is set |
| `query` | `str` | Original query |
| `session_id` | `str \| None` | Session ID if provided |
| `degraded` | `list[str] \| None` | **Omitted when healthy.** Present only on partial failure — channel names that failed (`dense`, `sparse`, `rerank`, `graph`) |
| `error` | `str \| None` | Present only when **both** `dense` and `sparse` failed — signals total retrieval outage, distinct from a query that simply matched nothing |

Each `FusedResult`:

| Field | Type | Description |
|-|-|-|
| `id` | `str` | Qdrant point ID |
| `text` | `str` | Matched text chunk |
| `source_file` | `str` | Original file name |
| `page_number` | `int` | Page within the source document |
| `chunk_index` | `int` | Chunk position within the page |
| `score` | `float` | Rerank score when `RERANK_ENABLED=true`, else equal to `fusion_score` |
| `fusion_score` | `float` | RRF fusion score across channels |
| `channels` | `list[str]` | Which retrieval channel(s) surfaced this hit — `["dense"]`, `["sparse"]`, or both |
| `metadata` | `dict` | Additional payload metadata |

### Example

```json
{
  "name": "multi_search",
  "arguments": {
    "query": "latest compliance requirements",
    "limit": 10
  }
}
```

---

## `hybrid_search`

Same fused retrieval core as `multi_search` — dense + sparse -> RRF -> rerank — wrapped in two extra stages: query decomposition before retrieval, and a relevance-gated single retry after reranking. Returns the identical schema, plus `sub_queries` and `retried`. Costs one cheap LLM call for decomposition; use `multi_search` when that cost or latency is unwelcome.

**Decomposition.** When `DECOMPOSE_ENABLED=true` (default), the query is split into up to `DECOMPOSE_MAX_SUBQUERIES` (default 4) sub-queries by one LLM call (`DECOMPOSE_MODEL`, falling back to `LLM_MODEL` when unset). Each sub-query becomes its own dense + sparse channel pair, all fused together in one RRF pass. If decomposition fails or the query is a single ask, it falls back to `[query]` unchanged — a decomposition outage degrades to single-query retrieval rather than an error.

**Relevance-gated retry.** The fused, reranked results are checked against `RERANK_SCORE_FLOOR` (default `0.0`). `jina-reranker-v3.5` is listwise and its scores are unbounded and logit-like, not the pointwise v2 reranker's bounded `[0, 1]` — a strong match scores around `+0.39`, and irrelevant text scores negative. A floor of `0.0` therefore means "retry only when the best candidate was judged actively irrelevant," not "retry on anything less than a great match." When `RETRY_ENABLED=true` (default) and the top score is below the floor, the candidate pool is widened by `RETRY_LIMIT_MULTIPLIER` (default `3`x `limit`) and the whole retrieve-and-rank step runs once more. This targets the failure mode where the right chunk was ranked outside the initial pool.

Every gate evaluation is logged, fired or not, so the retry rate is measurable rather than guessed — see [Relevance-gate telemetry](../operations/observability.md#relevance-gate-telemetry) and `task retry-stats`. The rate is the standing evidence for whether first-stage recall needs investment: retries that fire *and improve* the top-1 score mean the reranker never saw the right candidate.

### Parameters

Identical to [`multi_search`](#multi_search): `query`, `limit`, `content_type`, `source_file`, `session_id`.

### Return Schema

Identical to `multi_search`, plus:

| Field | Type | Description |
|-|-|-|
| `sub_queries` | `list[str]` | The sub-queries actually used for retrieval (`[query]` if decomposition was skipped or failed) |
| `retried` | `bool` | `true` if the relevance gate fired and the candidate pool was widened once |

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

Enumerate distinct documents ingested into the knowledge base, with chunk and page counts. Scrolls the Qdrant `documents_dense` collection and excludes live session data (`is_live=True`).

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

`vector_search`, `visual_search`, and `graph_search` catch exceptions internally and return an error object rather than raising:

```json
{
  "error": "vector_search failed: Connection refused",
  "query": "original query",
  "partial_results": []
}
```

`multi_search` and `hybrid_search` use a different, more granular scheme — see their Return Schema sections above. A failed channel is recorded by name in `degraded` and the tool still returns whatever succeeded; a top-level `error` key only appears when both retrieval channels are down. There is no `partial_results` field on these two tools.

Either way, the MCP server remains operational and the calling agent receives a structured signal it can reason about instead of an exception.
