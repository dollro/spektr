# Capability Slots — Implementation Plan

**Goal:** Make the reranker, the late-interaction model, and the dense embedder independently selectable via a shared model×route registry, so a fully self-hosted configuration becomes expressible and no single vendor is load-bearing for the retrieval path.

**Architecture:** `config/embedding_models.py` gains a sibling `config/rerank_models.py`; `retrieval/rerankers/` mirrors `ingestion/embedders/` with one adapter per wire protocol; a new `local` route serves OpenAI-compatible self-hosted endpoints (TEI / vLLM / Infinity). The dense slot keeps its existing keys. `MULTIVEC_ENABLED` becomes `MULTIVEC_MODEL`.

**Tech stack:** Python 3.13, uv, Ruff (95 cols), mypy strict, pytest-asyncio, httpx, tenacity, Qdrant, pydantic-settings.

**Design doc:** [`2026-08-16-capability-slots-design.md`](./2026-08-16-capability-slots-design.md)
**Audit that motivated it:** findings D1–D8, referenced inline by id.

**Branch:** `chore/architecture-rewrite` (forked from `develop` @ `f0c47c9`)

---

## Conventions for this plan

- Every phase ends green: `task check` (lint + typecheck + test) passes before the next phase starts.
- Phases 1–3 each land as their own commit. Do not batch them.
- No phase lowers a threshold in `tests/eval/thresholds.yaml`. A drop needs a reason in the commit message (`CLAUDE.md`, Eval gate).
- Any change to a `.env` key must be applied to **all three** of `.env`, `.env.example`, `.env.prod`. They are currently out of sync — see Phase 4.

### ✅ `task typecheck` — was red on `develop`, now green

`mypy .` used to abort **fatally** before checking most of the tree: `scripts/` has no `__init__.py`, so each script was reachable under two module names ("Source file found twice"). That abort had been masking **295 errors in 37 files**.

Resolved rather than deferred:

- `mypy_path = "."` + `explicit_package_bases = true` clears the abort.
- `ignore_missing_imports` for genuinely untyped third-party libs (`boto3`, `botocore`, `gliner2`) and for the optional `eval` extra (`ragas`, `datasets`, `langchain_*`, `yaml`) that a default checkout does not install. Imports only — no code checks relaxed.
- `untyped_calls_exclude = ["pymupdf", "graphiti_core"]`: both ship class annotations but leave the called methods bare.
- **Production packages stay fully `strict`.** Five latent bugs were fixed at the source rather than suppressed:
  - `ingestion/graph_writer.py` — `rs.outcome.exception()` on `Future | None` would `AttributeError` *inside* the retry path, masking the original Neo4j error.
  - `server/tools/vector_search.py` — an explicit null `text` payload returned `None` into `SearchResult.text: str`, dropping the whole search into its error branch. `_dual_query` already guarded this; the two paths disagreed.
  - `ingestion/entity_extractor.py`, `server/tools/vlm_generator.py` — `.text` read off an Anthropic content block union without narrowing; a thinking/tool-use block first would raise.
  - `server/mcp_server.py` — unguarded `context.fastmcp_context` (`Context | None`) in middleware.
- `tests/` runs non-strict (`disallow_untyped_defs` etc. off). Strict typing there was never actually enforced, so enabling it is a new requirement, not a regression fix.
- `disable_error_code` for `tests.*` is **four** codes — `union-attr` (67, qdrant's `Filter.must` union), `attr-defined` (23, `LogRecord` extras and monkeypatched globals), `call-arg` (10, `Settings(_env_file=...)`), `type-arg` (5). The other 15 errors, across six codes, were **fixed at the source** — including a real one this exposed: a test assigning to the now read-only `multivec_slot_filled` property.

`task check` (lint + typecheck + test) is green. Phase gates use it.

---

## Phase 0 — Make the default config a tested artifact

> Rationale: D1, D2 and D5 are one systemic finding — the fresh-deployment path is unexercised. This phase makes that path assertable before anything else changes.

### 0.1 Boot test for the shipped example config

- [x] Add `tests/test_default_config.py`.
- [x] `test_env_example_loads()` — construct `Settings(_env_file=".env.example")` and assert it does not raise. This alone catches any validator that the example config violates.
- [x] `test_env_example_declares_every_settings_field()` — for each field on `Settings` that is not a computed property, assert the uppercase key appears in `.env.example`. Expect this to fail initially: `RERANK_MODEL`, `RERANK_SCORE_FLOOR`, `RRF_K`, `SPARSE_ENABLED`, `SPARSE_MODEL`, `DECOMPOSE_*` and `RETRY_*` are all absent despite `RERANK_MODEL` being named in `.env.example:322`'s own comment. Mark the known gaps with an explicit allowlist that Phase 4 empties.
- [x] `test_rerank_credentials_present_when_enabled()` — assert that a config with `RERANK_ENABLED=true` resolves a non-empty API key. **This is the D1 regression test.** It must fail against the current tree.

### 0.2 Degradation smoke test

- [x] Add `tests/test_integration_default_path.py`, marked `integration`.
- [x] Seed two points into the test collections, run `multi_search("...")`, and assert `"degraded" not in response`. This is the end-to-end form of the same guarantee — it catches a broken rerank, a broken channel, or a broken graph engine as one assertion.
- [x] Confirm it uses the redirected test collections from `tests/conftest.py` and the ephemeral Neo4j container. **Do not** point the fixtures at the dev instance — per `CLAUDE.md` that destroys the knowledge graph.

### 0.3 Fix D2 — Graphiti ignores `LLM_API_TYPE`

- [x] `ingestion/graphiti_client.py:79-92` constructs `OpenAIGenericClient` and `OpenAIRerankerClient` unconditionally, while `ingestion/entity_extractor.py:120-129` dispatches on `settings.llm_api_type` correctly.
- [x] With shipped defaults (`LLM_API_TYPE=anthropic`, `LLM_MODEL=claude-haiku-4.5`, `LLM_BASE_URL=""`, `GRAPH_ENGINE=graphiti`) this sends a Claude model id to `api.openai.com` with an Anthropic key.
- [x] Choose the narrow fix now, since the broad fix (routing Graphiti through `get_llm_client()`) requires an adapter to Graphiti's `LLMClient` interface and belongs with the cross-encoder work in Phase 1.5: add a `Settings` validator rejecting `GRAPH_ENGINE=graphiti` + `GRAPH_ENABLED=true` when `LLM_API_TYPE != "openai"` and `LLM_BASE_URL` is empty, with a message naming both escape hatches.
- [x] Add `test_graphiti_requires_openai_compatible_llm()` to `tests/test_settings.py`.

**Phase 0 exit:** `task check` green; the three new tests in 0.1 and the validator test pass; 0.2 passes under `task test-integration`.

---

## Phase 1 — Rerank slot

> The user-visible win with zero blast radius: reranking is query-time only, so nothing re-ingests.

### 1.1 Rerank registry

- [x] Create `config/rerank_models.py`, mirroring `config/embedding_models.py`.

```python
RerankProtocol = Literal["jina", "cohere", "voyage", "tei"]
RerankRoute = Literal["native", "local"]

@dataclass(frozen=True)
class RerankSpec:
    protocol: RerankProtocol
    routes: tuple[RerankRoute, ...]
    default_route: RerankRoute
    wire_id: str                    # model id sent on the wire
    # Scores are NOT comparable across rerankers. jina-reranker-v3.5 emits
    # unbounded logits; TEI/Cohere cross-encoders emit [0,1]. The relevance
    # gate compares top-1 against a floor, so a floor tuned for one scale
    # silently disables the gate on the other.
    score_scale: Literal["logit", "unit"]
    default_score_floor: float
```

- [x] Seed the registry with `jina-reranker-v3.5` (`protocol="jina"`, `score_scale="logit"`, `default_score_floor=0.0`) so today's behaviour is the default and reproduces exactly.
- [x] Add `qwen3-reranker` (`protocol="tei"`, `routes=("local",)`, `score_scale="unit"`, `default_score_floor≈0.5`) as the first self-hosted entry. Treat the floor as provisional — Phase 1.6 calibrates it.
- [x] Add `spec(model)` raising `ValueError` on an unknown name, matching `config/embedding_models.py:106-113`.

### 1.2 Settings

- [x] Add `rerank_route: RerankRoute | Literal[""] = ""` — empty resolves to the model's `default_route`, mirroring `embedding_route` (`config/settings.py:32,242-243`).
- [x] Add `rerank_api_key: str = ""` and `rerank_api_url: str = ""`. Both empty fall back per route: `native` + `protocol="jina"` → `jina_api_key` / `jina_api_url`, preserving current behaviour for existing `.env` files.
- [x] **Change `rerank_score_floor` from `float = 0.0` to `float | None = None`.** `0.0` cannot be the unset sentinel because it is a legitimate floor for a logit-scaled model. Add a property `resolved_score_floor` returning the override or the spec's default.
- [x] Extend `_validate_embedding_selection` (or add a sibling validator) to reject: an unknown `RERANK_MODEL`; a `RERANK_ROUTE` not in the spec's `routes`; and `RERANK_ENABLED=true` with no resolvable API key **or** no resolvable base URL. That last clause is the D1 fix.
- [x] Keep `RERANK_ENABLED` — it is a genuine off switch with documented gate interaction (`.env.example:322-328`).

### 1.3 Reranker adapters

- [x] Create `retrieval/rerankers/__init__.py`.
- [x] Define the protocol in `retrieval/rerank.py` (keep it next to its only consumers; `retrieval/` must not import from `server/` or `fastmcp`, per `CLAUDE.md`):

```python
@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]: ...
    async def close(self) -> None: ...
```

- [x] `retrieval/rerankers/jina.py` — `POST {base}/v1/rerank`, body `{model, query, documents, top_n}`, read `resp["results"][i]["index"]` and `["relevance_score"]`. This is a lift of the existing `_rerank_request` (`retrieval/rerank.py:39-72`) with the response mapped to tuples.
- [x] `retrieval/rerankers/tei.py` — `POST {base}/rerank`, body `{query, texts, raw_scores: false}`, response is a bare list of `{index, score}`.
- [x] Defer `cohere.py` and `voyage.py`. The registry makes them a ~30-line addition each; adding them now means shipping untested wire code. Note the deferral in the registry docstring so it does not read as an oversight.
- [x] Carry the retry shape across unchanged: `wait_exponential(multiplier=1, min=2, max=15)`, `stop_after_attempt(settings.max_retries)`. **Correct one thing while moving it:** the current retry predicate is `retry_if_exception_type((httpx.HTTPStatusError,))`, which retries 4xx as eagerly as 5xx. Use the `_is_retryable` shape already proven in `ingestion/embedders/openrouter.py:52-69` (429 + 5xx + transient network, and note that `httpx.TimeoutException` is a *sibling* of `NetworkError`, not a subclass of `ConnectError`).

### 1.4 Factory and call-site rewiring

- [x] Add `create_reranker() -> Reranker` to `retrieval/rerank.py`, dispatching on `spec(settings.rerank_model).protocol`. Lazy-import each adapter, matching `create_embedder()` (`ingestion/embedder.py:115-143`).
- [x] **Delete the module-level `RERANK_URL`** (`retrieval/rerank.py:32`). It is computed from `settings.jina_api_url` at import time, which is precisely what makes the reranker unswappable. `tests/test_rerank.py:13` imports it — update that test.
- [x] Rewrite `rerank()` and `rerank_dicts()` to call the factory. Their public signatures and their `RerankError` / degradation semantics do not change — `retrieval/pipeline.py:113-117` and `server/tools/vector_search.py:119-121` must need no edit.
- [x] Cache the reranker instance at module level behind a `_get_reranker()`, mirroring `_get_embedder()` in `retrieval/channels.py:34-38`. Add a `reset_reranker()` test hook, mirroring `ingestion/sparse_embedder.py:52-55`.

> **Loop affinity.** `retrieval/rerank.py` currently opens a fresh `httpx.AsyncClient` per call (`:50`), so it has never had the cross-loop problem the embedders solved with `WeakKeyDictionary` (`ingestion/embedders/jina.py:94-142`). If an adapter caches its client, it inherits that problem. Either keep per-call clients (simplest; reranking is one call per query, not per chunk) or copy the `_ensure_loop_resources` pattern wholesale. **Recommend per-call** — do not invent a third approach.

### 1.5 Graphiti cross-encoder

- [x] `ingestion/graphiti_client.py:91` passes `OpenAIRerankerClient` — a second hosted reranker in the ingestion path, unaffected by `RERANK_MODEL`.
- [x] Add a `_SlotCrossEncoder` adapting `create_reranker()` to Graphiti's `CrossEncoderClient` interface, mirroring `_JinaGraphitiEmbedder` (`ingestion/graphiti_client.py:39-67`).
- [x] Verify the installed `graphiti_core`'s cross-encoder method signature before writing the adapter — read it from the installed wheel, do not assume it matches the embedder's shape.

### 1.6 Tests

- [x] `tests/test_rerank.py` — keep every existing assertion; update the `RERANK_URL` import. Parametrise the response-parsing tests across the `jina` and `tei` adapters.
- [x] New: `test_score_floor_follows_model()` — assert an unset `RERANK_SCORE_FLOOR` yields the spec default, and that switching model changes it.
- [x] New: `test_unit_scale_gate_fires()` in `tests/test_gate.py` — a unit-scaled reranker with its own default floor must still be able to fire the gate. This is the regression test for the silent-gate-disable failure the design doc identifies.
- [x] New: `test_rerank_rejects_4xx_without_retry()` — pins the retry-predicate correction from 1.3.
- [~] Run `task eval-retrieval` and record recall@10 / nDCG@10 / MRR in the commit message. Reranking is unchanged in behaviour at this phase, so **these numbers must not move**. Any movement means the Jina adapter is not a faithful lift. **DEFERRED — needs an ingested corpus + running services; not runnable in this environment.**

**Phase 1 exit:** `task check` green; `task eval-retrieval` unchanged from baseline; `RERANK_MODEL=qwen3-reranker` + `RERANK_ROUTE=local` boots and serves against a local TEI container.

---

## Phase 2 — `local` route for the dense slot

> Unlocks sovereignty. Re-ingest required, but only for those who opt in.

### 2.1 Registry

- [x] Add `"local"` to `EmbeddingRoute` (`config/embedding_models.py:24`).
- [x] Extend `ModelSpec` with the fields the open models need, which the hosted ones did not:

```python
    local_model_id: str = ""     # HF id served by TEI/vLLM/Infinity
    # Open encoders (e5, bge, Qwen3) take the retrieval role as an
    # INSTRUCTION PREFIX in the text, not as an `input_type` field. This is
    # exactly why OpenRouterEmbedder's docstring excludes that family
    # (ingestion/embedders/openrouter.py:76-80) — the contract differs.
    query_prefix: str = ""
    passage_prefix: str = ""
```

- [x] Register `qwen3-embedding-8b`: `default_dimensions=4096`, `recommended_dimensions=1024`, `routes=("local",)`, `allowed_dimensions=()` (Matryoshka), `image_routes=()`, `query_prefix` set to Qwen3's instruct template, `passage_prefix=""`.
- [x] **Verify `default_dimensions` and the instruct template against the Qwen3-Embedding model card before committing.** Do not carry these from this plan — they are the two values most likely to have shifted.
- [x] Leave `image_routes` empty. Qwen3-Embedding is text-only; advertising otherwise is the exact failure `config/embedding_models.py:11-15` was written to prevent.

### 2.2 Client

- [x] Create `ingestion/embedders/local.py` with `LocalEmbedder`.
- [x] Start from `OpenRouterEmbedder` (`ingestion/embedders/openrouter.py`) — same OpenAI-compatible `/v1/embeddings`, same batching, same `sorted(data["data"], key=index)` ordering guard at `:206`.
- [x] Replace `input_type` with prefix application: prepend `query_prefix` / `passage_prefix` per `task`. Raise on an unmapped task, matching `_input_type` (`:42-49`) — the same silent-typo hazard applies.
- [x] **Do not send `dimensions`.** TEI ignores it and vLLM's support varies by model. Truncate Matryoshka-style client-side and renormalise, then assert `len(vector) == settings.dense_dimensions` before returning. A silently full-width vector against a truncated collection is a runtime Qdrant rejection per point.
- [x] `embed_image`, `embed_multi_vector`, `embed_query_multi_vector` raise `NotImplementedError` with a message naming the multivec slot as the alternative.
- [x] Keep the `WeakKeyDictionary` loop-affinity pattern verbatim from `openrouter.py:104-154` — CocoIndex runs file components on multiple worker threads, each with its own loop, sharing one embedder instance.
- [x] Rate limiting: local endpoints have no RPM quota, but they do have finite GPU memory. Use `asyncio.Semaphore(settings.local_max_concurrent)` and **no** `TokenBucket`. Do not copy the RPM limiter — an arbitrary rate cap on a local server is throughput lost for nothing.

### 2.3 Settings and factory

- [x] Add `local_embed_api_url` (e.g. `http://localhost:8080`), `local_embed_api_key` (optional — TEI usually has none), `local_max_concurrent: int = 8`, `local_batch_size: int = 32`.
- [x] Add the `route == "local"` branch to `create_embedder()` (`ingestion/embedder.py:125-143`).
- [x] Extend `Settings.embedding_model_id` (`config/settings.py:199-206`) — it currently indexes a two-key dict that raises `KeyError` for anything but `jina-v4` / `voyage-4` on the native route. Add the `local` branch returning `spec.local_model_id`.
- [x] Confirm the existing image validator (`config/settings.py:278-284`) already forces `IMAGE_EMBED_STRATEGY=none` for a text-only dense model. It should — verify with a test rather than by reading.

### 2.4 Tests

- [x] `tests/test_local_embedder.py`, modelled on `tests/test_openrouter_embedder.py`.
- [x] `test_prefixes_applied_per_task()` — passage and query inputs receive different prefixes.
- [x] `test_unknown_task_raises()`.
- [x] `test_truncates_and_renormalises()` — a 4096-d response with `EMBEDDING_DIMENSIONS=1024` returns 1024 floats of unit norm.
- [x] `test_batch_order_preserved_by_index()` — feed a deliberately shuffled `data` array and assert reordering, matching the guard at `openrouter.py:206`.
- [x] `test_image_embedding_rejected_for_text_only_model()` in `tests/test_settings.py`.

**Phase 2 exit:** `task check` green; a documented `docker run` of TEI serving Qwen3-Embedding, ingesting a sample doc end-to-end and answering `task smoke`.

---

## Phase 3 — Multivec slot

> Fixes D5. Partial re-ingest: `documents_multivec` only.

### 3.1 Registry and settings

- [x] Add `config/multivec_models.py` (or a `MULTIVEC_REGISTRY` in `config/embedding_models.py` — prefer a separate module; the specs share no fields with `ModelSpec` beyond a route).
- [x] Fields: `routes`, `default_route`, `wire_id` / `local_model_id`, and **`dimensions`** — `MULTIVEC_DIM = 128` is currently a module constant (`config/constants.py:11`) consumed by `create_multivec_collection` (`ingestion/qdrant_setup.py:74`). ColPali variants differ; make it a spec field and have `qdrant_setup` read it from the active spec.
- [x] Register `jina-v4-colbert` (`route="native"`, dim 128) reproducing today's behaviour, and `colqwen2.5` (`route="local"`) as the self-hosted entry.
- [x] Add `multivec_model: str = ""` and `multivec_route`. Empty model = slot unfilled = `visual_search` unavailable.
- [x] Declare `multivec_enabled: bool = False` on `Settings` **solely to raise a migration error**, exactly as `embedding_provider` does (`config/settings.py:41,228-238`). Message: `MULTIVEC_ENABLED has been replaced by MULTIVEC_MODEL; set MULTIVEC_MODEL=jina-v4-colbert to keep current behaviour, or leave empty to disable.`
- [x] Replace the `multivec_enabled` reads at `config/settings.py:267`, `ingestion/qdrant_setup.py:93`, and `ingestion/page_processor.py:320` with a `multivec_slot_filled` property.
- [x] Drop the old cross-slot validator (`config/settings.py:267-274`) that rejected `MULTIVEC_ENABLED=true` unless the *dense* model listed the capability. That coupling is the defect; the slot's own route validation replaces it.

### 3.2 Client and wiring

- [x] `ingestion/embedders/colqwen_local.py` implementing only `embed_multi_vector` and `embed_query_multi_vector`.
- [x] `create_multivec_embedder()` in `ingestion/embedder.py`, separate from `create_embedder()`.
- [x] `ingestion/page_processor.py:320-351` currently calls `embedder.embed_multi_vector` on the *dense* embedder. Repoint it at the multivec embedder. Keep the raise-don't-swallow semantics — a point not declared is a point CocoIndex deletes (`page_processor.py:8-16`).
- [x] `server/tools/visual_search.py:61` likewise calls `_get_embedder().embed_query_multi_vector`. Repoint.
- [x] Confirm `visual_search` degrades with a clear "multivec slot not configured" message rather than raising, when the slot is empty.

### 3.3 Tests

- [x] `test_multivec_enabled_raises_migration_error()` in `tests/test_settings.py`.
- [x] `test_multivec_slot_independent_of_dense_model()` — `gemini-2` dense + `jina-v4-colbert` multivec validates cleanly. This combination is rejected today and is the point of the phase.
- [x] Update `tests/test_qdrant_setup.py:145` — it asserts against the `DENSE_DIM` constant; the multivec assertions must move to the spec.
- [x] Update `tests/test_pipeline_dual_embed.py` for the split embedder.

**Phase 3 exit:** `task check` green; `MULTIVEC_MODEL=` empty and `=jina-v4-colbert` both boot; the mixed `gemini-2` + `jina-v4-colbert` config validates.

---

## Phase 4 — Audit cleanup

> Small, independent, same files. One commit.

- [x] **D9 (found during Phase 0)** — FIXED in commit following Phase 1. — `server/tools/graph_search.py:72-80` catches every exception and *returns* `[{"error": ...}]` instead of raising. `run_graph` (`server/tools/multi_search.py:79-83`) only marks the channel failed when the call raises, so `"graph"` can never reach `degraded`, and the error dict is handed back inside `graph_facts` as though it were a fact. This breaks the contract in `CLAUDE.md` — "every pipeline stage degrades rather than raising… a stage failure appends its name to `degraded`" — for the one stage that cannot. It also means the Phase 0.2 smoke test does **not** catch a broken graph engine, contrary to what 0.2 claims above. Fix by raising from the internal path while keeping a caught, dict-shaped response for direct MCP callers; the `run_graph` side already handles a raise correctly.

- [x] **D3** — `IMAGE_EMBED_MAX_PX=400` (`.env.example:317`) has no `Settings` field, is swallowed by `extra="ignore"`, and nothing resizes. Pages render at 150 DPI (`ingestion/file_processor.py:214`) → ~1240×1754 px for A4. **Decide and do one:** implement the downscale (a `PIL` thumbnail before `embed_image`) or delete the key. If deleting, also rewrite `ingestion/embedders/openrouter.py:34-39`, whose `_IMAGE_TOKEN_ESTIMATE = 258.0` is justified by that non-existent cap.
- [x] **D3b** — re-derive `_TOKENS_PER_TILE = 10` (`ingestion/embedders/jina.py:47`). Qwen2.5-VL merges 2×2 patches into one visual token, so a 28×28-px tile is closer to 1 token than 10. At 10, an A4 page estimates 28,350 tokens against a default `JINA_TPM=100000` — ~3.5 pages/minute. Verify against a real invoice before changing; an under-estimate causes 429 storms, so do not "fix" this from theory alone.
- [x] **D4** — `config/embedding_models.py:11-15` claims gemini-2's `image_routes` is empty because `embed_image` raises. Line 100 sets `image_routes=("openrouter",)` and `openrouter.py:214-253` implements it. Rewrite the docstring.
- [x] **D6** — `DENSE_DIM = 512` (`config/constants.py:10`) is stale and leaks into eval fixtures via `scripts/make_eval_fixture.py:27,86`. Point the fixture at `settings.dense_dimensions`; keep a test-local constant in `tests/` if the suite still needs one.
- [x] **D7** — `_JinaGraphitiEmbedder` (`ingestion/graphiti_client.py:45-64`) passes `task="query"` in both `create()` and `create_batch()`, and `create_batch` is the *indexing* path. Not broken (it is internally symmetric) but undocumented and contrary to the asymmetry this codebase treats as load-bearing. Either switch `create_batch` to `passage` — **which changes the graph vector space and requires re-embedding entity names** — or add a comment recording the decision. **Recommend the comment**; the re-embed is not worth it without evidence.
- [x] **D7b** — same file: `create()` embeds an entire list and returns only `vectors[0]`, silently discarding the rest. Log a warning, or embed only `texts[0]`.
- [x] **D8** — `docs/ingestion/embeddings.md:260` says the client sets `EMBEDDING_DIM=512`; the code sets `str(settings.dense_dimensions)` (`ingestion/graphiti_client.py:24`). Update the doc.
- [x] **Env trio sync** — `.env` has the retrieval keys, `.env.example` has almost none, `.env.prod` has none. Add `RERANK_MODEL`, `RERANK_ROUTE`, `RERANK_SCORE_FLOOR`, `RRF_K`, `SPARSE_ENABLED`, `SPARSE_MODEL`, `DECOMPOSE_*`, `RETRY_*` plus every key added by Phases 1–3, to all three, with comments. Then empty the allowlist in `test_env_example_declares_every_settings_field` from Phase 0.1.

---

## Phase 5 — Documentation

- [x] `docs/ingestion/embeddings.md` — restructure around the three slots. The provider comparison table becomes per-slot. Add the local route, its prefix contract, and the client-side truncation note.
- [x] New `docs/ingestion/reranking.md` — the rerank registry, the wire-protocol table, and the score-scale trap. The gate interaction currently lives only in an `.env.example` comment; it belongs here.
- [x] `docs/operations/reindex.md` — add the per-slot cost table: rerank = free, multivec = partial, dense = full.
- [x] `docs/configuration/environment.md` — the new keys.
- [x] `CLAUDE.md` — line 42 describes the vector store as "dense + optional ColBERT multi-vector, **Jina-only**". That is exactly the coupling Phase 3 removes; rewrite it around the slot model. Add a new entry under "Production contracts": **a reranker's score scale is part of its registry entry — `RERANK_SCORE_FLOOR` is meaningless without it, and a floor tuned for a logit-scaled model silently disables the gate on a unit-scaled one.**
- [x] Add a self-hosting runbook: `docker run` lines for TEI (dense + rerank) and the ColQwen server, with the matching `.env` block.

---

## Risks

| Risk | Mitigation |
|---|---|
| Score-scale mismatch silently disables the relevance gate | `score_scale` + `default_score_floor` in the registry; `test_unit_scale_gate_fires` |
| Jina adapter is not a faithful lift, changing retrieval quality invisibly | `task eval-retrieval` must be **unchanged** at Phase 1 exit, not merely above threshold |
| Client-side Matryoshka truncation produces non-unit vectors against a COSINE collection | Renormalise and assert length before returning; `test_truncates_and_renormalises` |
| Caching an `httpx.AsyncClient` in a reranker adapter reintroduces the cross-loop lock bug | Keep per-call clients (1.3) |
| `MULTIVEC_ENABLED` removal silently disables `visual_search` on existing deployments | Declared-and-raises migration error, following the `EMBEDDING_PROVIDER` precedent |
| Phase 2 tempts a model swap on the dense slot mid-plan | Model selection is explicitly out of scope; it is gated by `task eval-retrieval` in a separate change |

---

## Out of scope

- **Graph fusion into RRF**, and the chunk-level provenance it requires. Separate plan — `GLiNEREngine.ingest` stores only `e.source = source_key` (`ingestion/graph_engine.py:284-299`), so no fact can be joined to its supporting chunk today.
- **Choosing models.** This plan makes them selectable. `jina-v4` → v5-omni and ColBERT → ColQwen are eval-gated changes.
- **Cohere and Voyage rerank adapters.** The registry makes each a ~30-line addition; adding untested wire code now is not free.
