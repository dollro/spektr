# Retrieval Architecture Upgrade — Design

**Branch:** `chore/architecture-upgrade`
**Date:** 2026-07-31
**Status:** Approved design, pending implementation plan

---

## Goal

Bring Spektr's retrieval path to the 2026 production baseline: a lexical channel fused with dense via RRF, a listwise reranker, query decomposition, and a relevance-gated retry. Today the pipeline is dense-only with a pointwise reranker and no fusion, rewriting, or feedback loop.

## Motivation

An audit of the current code (not docs) found:

- **No lexical/sparse channel at all.** No BM25, SPLADE, or miniCOIL. Dense-only retrieval fails on exact identifiers — part numbers, error codes, section references, proper nouns — which is a large share of technical-document queries.
- **`hybrid_search` is not hybrid.** It runs vector and graph search in parallel and concatenates the results into `{vector_results, graph_results, live_results}`. There is no rank or score fusion. Ranking is effectively delegated to the consuming LLM, which is expensive in context and unreliable.
- **Reranker two generations behind.** `jina-reranker-v2-base-multilingual` (pointwise) versus `jina-reranker-v3.5` (listwise, 63.20 nDCG@10 on BEIR).
- **No query transformation.** Multi-part questions are issued verbatim against a chunk index.
- **No retrieval feedback loop.** Single-shot; a poor result set is returned as-is.

The published 2026 consensus pipeline is `rewrite → sparse + dense → RRF → rerank → generate → gate → retry`. Spektr implements two of those six stages.

`tests/eval/thresholds.yaml` already names the symptom: *"context_precision is the weakest… tighten after adding a reranker scoring cutoff or chunking improvements."*

## Scope

**In scope:** miniCOIL sparse channel, client-side RRF fusion, reranker v3.5, query decomposition, relevance-gated retry, a `multi_search` tool, LLM model bump, and a labeled retrieval eval set.

**Explicitly out of scope** (candidates for later branches):

- MUVERA for the multivector visual collection
- GraphRAG community summarization (`search_type` beyond `"entity"`)
- HyDE and step-back query rewriting
- DeepEval adoption
- Anthropic-style contextual retrieval (LLM-generated per-chunk context)

### Ruled out — Jina v5

Not a deferred candidate. Investigated and rejected:

- **v5 drops late chunking.** Jina maintainer, on the v5-text model card: *"This model does not support late chunking because of the different pooling method"* (last-token pooling). This would kill the Docling `HybridChunker` → `late_chunking=True` path and make `contextualized_text` dead weight.
- **v5-omni does not document multi-vector output.** `visual_search` depends on `embed_multi_vector()` producing ColBERT-style 128d-per-patch vectors for MaxSim over `documents_multivec`. v5-omni reaches 79.08 on ViDoRe MIEB via *single-vector* multimodal embeddings — a different retrieval architecture, not a drop-in swap.
- **Hosting risk.** v5-omni points at Elastic Inference Service rather than the hosted Jina API that `jina_api_url` targets. Jina is now "Jina AI by Elastic."

v4 is the multimodal/visually-rich retrieval model; v5-text is a compressed text base layer. Spektr uses v4 for what it is for. Revisit only if Jina ships late chunking plus documented multi-vector output in a v5 multimodal variant, if `documents_multivec` is retired, or if the v4 endpoint is deprecated.

Path B (live ingest), the graph engine layer, `visual_search`, `graph_search`, `list_documents`, and `list_document_chunks` are untouched.

`vector_search` requires one **mandatory minimal change**: it currently queries `documents_dense` via its unnamed vector. Once the collection uses named vectors, that query must target `using="dense"` or it breaks outright. Its behavior, signature, and return shape are otherwise unchanged. `list_documents` and `list_document_chunks` are payload/scroll-based and unaffected by the vector rename.

## Decisions

| Decision | Choice | Rationale |
|-|-|-|
| Scope | Retrieval path only | One coherent spec; no embedding migration blast radius |
| MCP contract | Breaking change, documented | Split-list shape is the context-cost problem being fixed |
| Migration | Full re-ingest from source | User's call; clean state, accepts embedding spend |
| LLM stages | Server-side, inside MCP tools | Works for any MCP client, not just `agent/` |
| Cheap path | Preserved as `multi_search` | LLM-free, deterministic, low-latency option retained |
| Tool surface | Two tools, identical return schema | `hybrid_search` wraps `multi_search`; one core path |
| Graph channel | Kept out of RRF | Facts and chunks are different granularity; rank-based fusion would over-weight a short graph channel |
| Query transform | Decomposition only, Haiku 4.5 | Highest value per unit of complexity; composes naturally with RRF |
| Fusion | Client-side, not Qdrant native | Per-channel ranks must stay inspectable for eval attribution |
| Code structure | New `retrieval/` package | File/function size caps; stage-level testability |

## Architecture

### New package: `retrieval/`

Pure retrieval logic. No FastMCP or transport imports.

| Module | Responsibility | Depends on |
|-|-|-|
| `models.py` | `Candidate`, `FusedResult` dataclasses | pydantic |
| `channels.py` | `dense_channel()`, `sparse_channel()` → ranked `list[Candidate]` | Qdrant, embedder |
| `fusion.py` | `rrf(channels, k=60)` → fused list. Pure function, no I/O | — |
| `rerank.py` | Jina reranker v3.5 listwise call | Jina API |
| `decompose.py` | `decompose(query) -> list[str]` | Haiku via `LLM_API_TYPE` |
| `gate.py` | `should_retry(results, floor) -> bool`. Pure function | — |
| `pipeline.py` | Composes stages into `fast_pipeline()` / `smart_pipeline()` | all of the above |

`server/tools/multi_search.py` and `server/tools/hybrid_search.py` become thin adapters (target < 80 lines each) that call a pipeline and shape the MCP response.

`server/tools/reranker.py` moves to `retrieval/rerank.py`. Its current callers (`vector_search.py`, `hybrid_search.py`) update their imports.

### Ingestion changes

- **`ingestion/sparse_embedder.py`** — miniCOIL encoding, mirroring the `embedder.py` dispatcher shape.
- **`documents_dense` schema change** — named vectors replace the current unnamed dense vector:
  - `dense`: 512d, cosine (unchanged values, now named)
  - `sparse`: miniCOIL, `Modifier.IDF`

Qdrant requires sparse vectors to be named, and named and unnamed vectors cannot coexist in one collection. This is what forces the re-index.

## Data flow

### `multi_search` — no LLM

1. Embed query: dense (Jina, `task=query`) and sparse (miniCOIL) concurrently
2. Two Qdrant queries against the named vectors, concurrently → two ranked candidate lists
3. `rrf(channels, k=60)` → one fused list
4. Rerank the top `rerank_candidates` (default 50) fused results against the query → final ordering
5. `graph_search` runs in parallel with steps 1–4 → `graph_facts`

Latency is `max(dense, sparse)` rather than their sum.

### `hybrid_search` — adds two stages around the identical core

1. `decompose(query)` → sub-queries, or `[query]` if not multi-part
2. Run `multi_search` steps 1–3 **per sub-query**; fuse every resulting channel in a single RRF pass (N sub-queries × 2 channels, all equal citizens)
3. Rerank against the **original** query, never a sub-query — sub-queries are a retrieval device; the user's intent is the ranking target
4. Gate: if top-1 rerank score < `rerank_score_floor`, retry once with `limit × retry_limit_multiplier` candidates, re-fuse, re-rerank

The retry makes no second LLM call. It targets the "right chunk was at rank 60, we fetched 20" failure, which one extra Qdrant round trip fixes.

### Return schema — identical for both tools

Illustrative example showing every possible key. In a healthy `multi_search` call, `sub_queries`, `retried`, and `degraded` are all absent.

```json
{
  "results": [{
    "text": "...",
    "source_file": "...",
    "page_number": 3,
    "chunk_index": 7,
    "score": 0.82,
    "fusion_score": 0.031,
    "channels": ["dense", "sparse"],
    "metadata": {}
  }],
  "graph_facts": [],
  "live_results": [],
  "query": "...",
  "session_id": null,
  "sub_queries": ["..."],
  "retried": false,
  "degraded": ["sparse"]
}
```

- `score` — rerank score, primary sort key
- `fusion_score` — pre-rerank RRF score, retained for debugging
- `channels` — provenance; makes the sparse investment auditable
- `sub_queries`, `retried` — `hybrid_search` only
- `degraded` — omitted entirely when all stages are healthy
- `live_results` — populated when `session_id` is set and `metadata.source_type == "live"`; kept separate from `results` because session and KB chunks are not competing for the same slot

### Breaking change

`hybrid_search` no longer returns `{vector_results, graph_results, live_results}`. Callers move to `results` + `graph_facts`. Requires updating `agent/` and a migration note in `docs/server/search-tools.md`.

## Error handling

Every stage has a defined fallback. Tools always return a shaped response rather than raising, matching the existing pattern in `hybrid_search.py`.

| Stage fails | Behavior | `degraded` entry |
|-|-|-|
| Sparse channel | RRF over dense alone | `sparse` |
| Dense channel | RRF over sparse alone | `dense` |
| Both channels | `results: []` plus `error` key | `dense`, `sparse` |
| Rerank | Sort by `fusion_score` | `rerank` |
| Decompose | Identity — `sub_queries = [query]` | `decompose` |
| Graph | `graph_facts: []` | `graph` |

Key property: a sparse-channel outage degrades to exactly today's behavior, not an outage. There is no configuration in which this ships worse than the current system.

## Configuration

New settings in `config/settings.py`:

```python
sparse_enabled: bool = True
sparse_model: str = "Qdrant/minicoil-v1"
rrf_k: int = 60
rerank_model: str = "jina-reranker-v3.5"
rerank_candidates: int = 50           # fused candidates sent to the reranker
rerank_score_floor: float = 0.3
retry_enabled: bool = True
retry_limit_multiplier: int = 3       # candidate-pool widening on gated retry
decompose_enabled: bool = True
decompose_model: str = "claude-haiku-4-5-20251001"
decompose_max_subqueries: int = 4
llm_model: str = "claude-sonnet-5"   # bumped from claude-sonnet-4-20250514
```

Every new stage is independently killable by flag. These same flags drive the eval ablation matrix.

## New dependencies

- **`fastembed` (≥ 0.7.0)** — required for `Qdrant/minicoil-v1`. Local CPU model: no API cost, but adds Docker image weight and a first-call warm-up.
- **LLM dependency in `server/`** — the MCP server currently never calls an LLM. Decomposition changes that, making `LLM_API_TYPE` load-bearing for the search path. `multi_search` remains LLM-free, so the fast path is unaffected, and decomposition degrades to identity on failure.

## Migration

1. `constants.py`: add `DENSE_VECTOR_NAME = "dense"`, `SPARSE_VECTOR_NAME = "sparse"`
2. `qdrant_setup.py`: `documents_dense` uses named-vector config; sparse with `Modifier.IDF`
3. Drop `documents_dense`, drop CocoIndex tracking rows, `task ingest` from source
4. `scripts/doctor.py`: extend to flag points missing a sparse vector (currently checks only `embedder_model` / `embedder_dim`)

### miniCOIL `avg_len`

miniCOIL takes an `avg_len` option (corpus average document length) for BM25-style length normalization at **index** time only, not query time. The pipeline does not compute corpus-level statistics.

Resolution: pin a fixed value in `constants.py` derived from the current chunking (~512 chars ≈ 80 tokens), documented as needing revisit if chunk sizing changes. A two-pass corpus scan for the exact value is not worth the pipeline complexity.

## Testing

### Unit tests — no infrastructure required

| Module | Approach |
|-|-|
| `fusion.py` | Pure function, fixture lists. Known ranks → asserted RRF scores. Edge cases: empty channel, single channel, total overlap, zero overlap |
| `gate.py` | Pure function over scores. Threshold boundaries, empty results |
| `decompose.py` | Mocked LLM. Multi-part → N sub-queries; simple → identity; failure → identity + `degraded` |
| `channels.py` | Mocked Qdrant. Named-vector targeting, limit handling |
| `rerank.py` | Mocked HTTP. Response parsing, retry/backoff, failure → `fusion_score` fallback |
| `pipeline.py` | All stages mocked. Composition plus every degradation path in the table above |

### Integration tests (`test_integration`, needs Docker)

Named-vector collection creation, sparse upsert and query round trip, one end-to-end `multi_search` against a seeded corpus.

### Eval

The existing eval gate cannot validate this work:

1. **The golden set is 11 questions over a single-document corpus.** Sparse retrieval's value is exact-term matching and cross-document discrimination; an 11-question single-doc set yields noise, not signal. Decomposition has nothing to decompose against one document.
2. **RAGAS measures generation quality, not retrieval quality.** `faithfulness` and `answer_relevancy` sit downstream of retrieval and are confounded by the LLM. Attributing a win to RRF versus rerank versus decomposition requires retrieval metrics.

Therefore, in scope:

- **`tests/eval/retrieval_set.yaml`** — 40–60 queries with labeled relevant `(source_file, chunk_index)` pairs over a multi-document corpus. Deliberately includes exact-identifier and multi-hop queries.
- **Retrieval-only metrics** — recall@10, nDCG@10, MRR computed directly from tool output. No LLM in the loop: fast, free, deterministic enough to gate on.
- **Ablation matrix** — the Section "Configuration" flags allow `dense-only / +sparse / +rerank / +decompose` runs producing a per-stage attribution table.

RAGAS remains for end-to-end generation quality; the retrieval eval sits alongside it.

Building the labeled retrieval set requires domain judgment on what "relevant" means for this corpus and is estimated at roughly one day of work.

## Documentation to update

- `docs/server/search-tools.md` — `multi_search`, new schema, breaking-change migration note
- `docs/ingestion/embeddings.md` — sparse channel
- `docs/operations/` — re-ingest runbook
- `docs/eval/golden-set.md` — retrieval set and ablation matrix
- `CLAUDE.md` — payload-schema production contract now includes sparse vectors

## Open verification item

`jina-reranker-v2-base-multilingual` is pointwise; `jina-reranker-v3.5` is listwise. Sources confirm the `/v1/rerank` request schema is unchanged from v3 → v3.5, but the v2 → v3 response shape has not been confirmed against the live API. **First implementation task:** call `/v1/rerank` with both model strings and diff the response shapes before the swap lands.

## References

- [jina-reranker-v3.5](https://arxiv.org/abs/2607.18152) — listwise, hybrid attention, self-distillation
- [jina-reranker-v3](https://arxiv.org/pdf/2509.25085) — last-but-not-late interaction
- [miniCOIL](https://qdrant.tech/articles/minicoil/) — sparse neural retrieval
- [Working with miniCOIL](https://qdrant.tech/documentation/fastembed/fastembed-minicoil/) — fastembed usage, `Modifier.IDF`
- [MUVERA](https://arxiv.org/html/2405.19504) — out of scope, relevant to the visual collection later
- [Hybrid Search: BM25, Vector & Reranking Reference 2026](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
- [Reasoning Agentic RAG survey](https://arxiv.org/pdf/2506.10408)
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/engineering/contextual-retrieval) — out of scope, relevant later
