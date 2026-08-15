# MCP Server Contract

**Status:** current as of branch `chore/architecture-upgrade` (retrieval upgrade + CocoIndex v1 migration).
**Audience:** operators and owners of MCP clients already connected to the previous Spektr release.

This page is the executive summary of the server's wire contract: what is guaranteed to stay
the same, what changed, and the minimum work an existing client must do to keep working.
It supersedes nothing — [Search Tools](search-tools.md) and [Response Models](../api/models.md)
remain the detailed reference — it just states the contract in one place.

---

## 1. Bottom line

| Question | Answer |
|-|-|
| Do existing clients still **connect**? | **Yes.** Transport, URL, server name, auth and handshake are unchanged. No client config edit is required to establish a session. |
| Do existing clients still **work**? | **Only if they don't parse `hybrid_search` output.** That one tool's response body changed shape. Everything else is backward compatible. |
| Is anything required on the **server side**? | **Yes — a one-time re-index.** `documents_dense` moved to named vectors. An un-migrated collection makes every search tool fail. |
| Was the MCP protocol version bumped? | No. Standard MCP; capability negotiation is unchanged. |

The connection contract and the data contract changed independently. A client can reconnect
with zero changes and will discover one new tool; it will only break if it reads
`hybrid_search`'s old keys.

---

## 2. Connection contract — unchanged

Nothing in this section changed. Existing `.mcp.json` files, Pydantic AI transports and
reverse-proxy routes keep working as-is.

| Element | Value |
|-|-|
| Server name | `rag-knowledge-base` |
| Transports | `http` (streamable-http, default), `sse` (legacy), `stdio` — selected by `MCP_TRANSPORT` |
| Endpoint | `{scheme}://{MCP_HOST}:{MCP_PORT}{MCP_PATH}` — default `http://localhost:8080/mcp` |
| Auth | Optional Bearer token. `Authorization: Bearer <MCP_API_KEY>`; empty `MCP_API_KEY` disables auth |
| Auth scope | `tools/call` and `tools/list` are intercepted. `initialize`, `ping` and other methods pass through unauthenticated |
| Failure mode | `PermissionError` → MCP error response ("Authentication required" / "Invalid token") |

See [Authentication](authentication.md) and [Client Setup](client-setup.md) for the full detail.

---

## 3. Tool surface

Seven tools. Six are always registered; `visual_search` appears only when
`MULTIVEC_ENABLED=true`.

| Tool | Status | Compatibility |
|-|-|-|
| `vector_search` | unchanged signature & schema | ✅ compatible (one behavioural fix, §5) |
| `graph_search` | unchanged | ✅ compatible |
| `visual_search` | unchanged (still gated by `MULTIVEC_ENABLED`) | ✅ compatible |
| `list_documents` | unchanged | ✅ compatible |
| `list_document_chunks` | unchanged | ✅ compatible |
| `multi_search` | **new** | ➕ additive — old clients simply won't call it |
| `hybrid_search` | **response shape changed**, two new optional params | ⚠️ **breaking** — see §4 |

`multi_search` and `hybrid_search` are session-aware alongside `vector_search` and
`graph_search` (four session-aware tools in total).

A client that re-runs `tools/list` will see `multi_search` appear. A client that caches the
tool list from the previous version keeps working — it just won't have access to the new tool.

---

## 4. The one breaking change: `hybrid_search`

`hybrid_search` was "vector search + graph search in parallel". It is now the fused retrieval
pipeline (dense + sparse → RRF → rerank) plus query decomposition and a relevance-gated retry.
Its response body changed accordingly.

### Field mapping

| Before (`HybridSearchResponse`) | After (`FusedSearchResponse`) | Notes |
|-|-|-|
| `vector_results` | `results` | Rename. Item schema also changed — see below |
| `graph_results` | `graph_facts` | Rename. Item schema (`GraphFact`) unchanged |
| `live_results` | `live_results` | Key unchanged; item schema changed, ordering changed (§5) |
| `query` | `query` | Unchanged |
| `session_id` | `session_id` | Unchanged |
| `strategy` (always `"parallel"`) | — | **Removed** |
| `errors` (list of messages) | `degraded` (list of channel names) | Replaced; different semantics (§6) |
| — | `sub_queries` | **New** — `hybrid_search` only |
| — | `retried` | **New** — `hybrid_search` only |
| — | `error` | **New** — only on total retrieval outage (§6) |

### Result item mapping

Items in `results` / `live_results` are now `FusedSearchResult`, not `SearchResult`:

| Field | Before | After |
|-|-|-|
| `text`, `source_file`, `page_number`, `metadata` | ✅ | ✅ unchanged |
| `score` | cosine similarity, or reranker score when reranking was on | reranker score when `RERANK_ENABLED=true`, otherwise equal to `fusion_score` |
| `content_type` | present | **removed** — filter with the `content_type` *parameter* instead, or read `metadata.mime_type` |
| `original_score` | present after reranking | **removed** — use `fusion_score` for the pre-rerank ordering signal |
| `id` | — | **new** — Qdrant point ID, stable across calls |
| `chunk_index` | — | **new** — chunk position within the page |
| `fusion_score` | — | **new** — RRF score across channels |
| `channels` | — | **new** — `["dense"]`, `["sparse"]`, or both |

### Parameters

`hybrid_search` gained two optional parameters and lost none:

```
hybrid_search(query, limit=10, content_type=None, source_file=None, session_id=None)
```

Previously `(query, limit=10, session_id=None)`. MCP passes arguments by name, so existing
callers sending `{"query": ..., "limit": ..., "session_id": ...}` are unaffected.

### Minimum client migration

```diff
- for hit in response["vector_results"]:
+ for hit in response["results"]:
      ...
- for fact in response["graph_results"]:
+ for fact in response["graph_facts"]:
      ...
- if response.get("errors"):
+ if response.get("degraded"):
      ...
- hit["content_type"]          # no longer present on fused results
+ hit["metadata"].get("mime_type")
```

Also: **stop re-sorting results client-side.** Ranking is now server-side (RRF fusion plus a
listwise reranker); re-sorting by `score` or `fusion_score` in the client discards that work.

### If you would rather not migrate now

`multi_search` returns the *identical* schema to `hybrid_search`, so migrating to the new shape
is a prerequisite for either tool — there is no "old shape" tool left. Clients that cannot
migrate immediately should call `vector_search` + `graph_search` directly, which return exactly
what they always did, and move to `multi_search`/`hybrid_search` when convenient.

---

## 5. Non-breaking behaviour changes

These change results, not schemas. No client code edit needed, but they are visible.

- **KB filter correctness (`vector_search`, session mode).** Bulk KB points never write an
  `is_live` key, and Qdrant's `IsNullCondition` matches an explicit JSON `null`, not a missing
  key — so the old `is_live == False OR is_null(is_live)` filter matched *zero* real KB points.
  It is now `must_not: is_live == True`. Effect: session-scoped `vector_search` now actually
  returns KB results. Clients that worked around the empty KB half should remove the workaround.
- **`live_results` ordering.** Previously sorted chronologically by timestamp. Now ranked by
  relevance like every other channel. Clients that relied on chronological order must sort by
  `metadata` timestamp themselves.
- **No graph/vector dedup.** The old `hybrid_search` dropped graph facts whose `source` matched
  a vector hit's `source_file`. The new pipeline does not — `graph_facts` is independent
  supporting context and may reference documents already present in `results`.
- **Reranker model.** `jina-reranker-v2-base-multilingual` → `jina-reranker-v3.5`, listwise.
  Scores are **unbounded and logit-like**, not the old bounded `[0, 1]`: a strong match is around
  `+0.39` and irrelevant text scores negative. **Any client threshold tuned against `[0, 1]`
  scores is now wrong** and must be recalibrated or removed.
- **Sparse channel.** Retrieval now includes miniCOIL lexical matching alongside dense
  similarity. Recall improves; the ordering of familiar queries will differ from the previous
  release.
- **Live (Path B) points** are written with both `dense` and `sparse` named vectors, so
  session data is reachable through both channels.

---

## 6. Error and degradation contract

Two distinct schemes, deliberately:

**`vector_search`, `visual_search`, `graph_search`** — unchanged. On failure they return an
error object rather than raising:

```json
{ "error": "vector_search failed: Connection refused", "query": "...", "partial_results": [] }
```

**`multi_search`, `hybrid_search`** — per-channel degradation. No `partial_results` field.

| Key | Meaning |
|-|-|
| `degraded` | **Omitted entirely when healthy.** Its presence always means partial failure. Values are channel names: `dense`, `sparse`, `rerank`, `graph` |
| `error` | Present **only** when both `dense` and `sparse` failed — total retrieval outage, as distinct from a query that matched nothing |

Every stage degrades rather than raising: a decomposition outage falls back to the original
query, a reranker outage returns fusion-ordered results, a graph outage returns empty
`graph_facts`. A call returning `results: []` with no `error` key means "nothing matched", not
"retrieval is down" — clients should distinguish the two.

---

## 7. Operator prerequisite: re-index before serving

**This is the one action that can break connected clients regardless of their code.**

`documents_dense` moved from a single unnamed vector to named `dense` + `sparse` vectors.
Named and unnamed vectors cannot coexist in a Qdrant collection, and Qdrant cannot migrate
vector configuration in place. Against an un-migrated collection **every search tool fails** —
the server stays up and returns structured errors, but returns no results.

Required sequence:

1. `task backup`
2. Drop and rebuild `documents_dense` — full procedure in [Re-indexing](../operations/reindex.md)
3. Re-ingest (`task ingest`); verify with `task doctor` and `task smoke`

Search is unavailable between the drop and the first successful re-ingest, and re-embedding
costs real API spend proportional to corpus size. **Schedule a maintenance window and tell
client owners.** For a new environment rather than a rebuild, see
[First Ingest](../operations/first-ingest.md).

---

## 8. Server-side configuration deltas

Client-visible defaults introduced by this branch. All have working defaults; none are
required to bring the server up.

| Variable | Default | Effect on responses |
|-|-|-|
| `SPARSE_ENABLED` | `true` | Adds the sparse channel to `multi_search` / `hybrid_search` |
| `SPARSE_MODEL` | `Qdrant/minicoil-v1` | Local CPU (fastembed), no API cost |
| `RRF_K` | `60` | Rank-fusion constant behind `fusion_score` |
| `RERANK_MODEL` | `jina-reranker-v3.5` | Sets the `score` scale (see §5) |
| `RERANK_CANDIDATES` | `50` | Fused candidates sent to the reranker |
| `RERANK_SCORE_FLOOR` | `0.0` | Retry threshold; `<0` means judged actively irrelevant |
| `RETRY_ENABLED` / `RETRY_LIMIT_MULTIPLIER` | `true` / `3` | Drives the `retried` flag on `hybrid_search` |
| `DECOMPOSE_ENABLED` / `DECOMPOSE_MAX_SUBQUERIES` | `true` / `4` | Drives `sub_queries` on `hybrid_search` |
| `DECOMPOSE_MODEL` | `""` (falls back to `LLM_MODEL`) | The one LLM call in `hybrid_search` |

Unrelated to the MCP surface but part of the same upgrade: `DATABASE_URL` is **gone** —
CocoIndex v1 keeps its ledger in a local LMDB directory (`COCOINDEX_DB_PATH`, default
`state/cocoindex.db`). There is no PostgreSQL in the stack. `S3_SQS_QUEUE_URL` is now optional.

---

## 9. Client upgrade checklist

1. **Operator:** back up, re-index `documents_dense`, re-ingest, verify with `task doctor`.
2. **Client:** no endpoint change needed — same URL, transport and Bearer token.
3. **Client:** send the Bearer token on `tools/list`, not just `tools/call`. `tools/list` is now
   gated when `MCP_API_KEY` is set, so a client that discovered tools unauthenticated will fail
   at startup with "Authentication required". Standard MCP clients that authenticate the whole
   session are unaffected.
4. **Client:** if you call `hybrid_search`, apply the §4 rename (`vector_results` → `results`,
   `graph_results` → `graph_facts`, `errors` → `degraded`) and drop `content_type` /
   `original_score` reads.
5. **Client:** remove client-side re-sorting of search results.
6. **Client:** recalibrate or remove any score threshold — the reranker scale changed.
7. **Client:** treat a missing `degraded` key as healthy; treat `error` as a retrieval outage.
8. **Client (optional):** refresh the tool list to pick up `multi_search`, and make it the
   default general-purpose tool — it is faster and cheaper than `hybrid_search` and returns the
   same schema.
9. **Client (optional):** if you relied on chronologically ordered `live_results`, sort them
   yourself.

---

## 10. Reference

- [Search Tools](search-tools.md) — per-tool parameters and schemas
- [Response Models](../api/models.md) — `FusedSearchResult`, `FusedSearchResponse`, `GraphFact`
- [Authentication](authentication.md) · [Client Setup](client-setup.md)
- [Re-indexing](../operations/reindex.md) · [First Ingest](../operations/first-ingest.md)
