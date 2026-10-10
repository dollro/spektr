# Reranking

Reranking is the last stage of the retrieval pipeline: fused candidates are re-scored against the original query, listwise, and the top `limit` are returned. It is selected **independently of `EMBEDDING_MODEL`**, on the same model×route axes the embedding registry uses.

**Source:** `retrieval/rerank.py` (protocol + factory), `retrieval/rerankers/` (protocol adapters), `config/rerank_models.py` (registry)

## Why a registry and not a base URL

Rerank APIs never converged the way `/v1/embeddings` did. The vendors disagree on the path, the request field names **and** the response envelope, so the model string has to carry which dialect to speak:

|Protocol|Path|Documents key|Response|Score key|
|-|-|-|-|-|
|`jina`|`/v1/rerank`|`documents`|`{"results": [...]}`|`relevance_score`|
|`cohere`|`/v2/rerank`|`documents`|`{"results": [...]}`|`relevance_score`|
|`voyage`|`/v1/rerank`|`documents`|`{"data": [...]}`|`relevance_score`|
|`tei`|`/rerank`|`texts`|bare JSON array|`score`|

Every one of those differences is a **silent wrong answer** rather than an error if an adapter gets it wrong — reading `results` off a bare list yields nothing, not an exception.

`cohere` and `voyage` are members of `RerankProtocol` but have no registry entry and no adapter yet. That is a deliberate deferral, not an oversight: the registry makes each a small addition, but shipping wire code that has never run against the vendor is not free.

## Selecting a reranker

|Setting|Meaning|
|-|-|
|`RERANK_ENABLED`|Master switch. `false` skips the stage entirely.|
|`RERANK_MODEL`|Registry key — `jina-reranker-v3.5` or `qwen3-reranker`.|
|`RERANK_ROUTE`|Empty for the model's default. `native` (vendor API) or `local` (your host).|
|`RERANK_API_KEY`|Falls back to `JINA_API_KEY` on non-local routes, so existing configs keep working.|
|`RERANK_API_URL`|Required on `local`. **No fallback** — silently pointing a self-hosted slot at `api.jina.ai` would be worse than failing.|
|`RERANK_SCORE_FLOOR`|Leave empty. See below.|

|`RERANK_MODEL`|protocol|routes|score scale|default floor|
|-|-|-|-|-|
|`jina-reranker-v3.5`|`jina`|`native`|logit (unbounded)|`0.0`|
|`qwen3-reranker`|`tei`|`local`|unit `[0, 1]`|`0.5` *(provisional)*|

`qwen3-reranker`'s floor is a plausible sigmoid midpoint, **not a measured one**. Calibrate it against `task eval-retrieval` before treating it as a real threshold.

**Changing the reranker costs no re-ingest.** It is query-time only — nothing is stored, no collection changes. That is what makes this the cheapest slot to experiment with.

## ⚠ Score scales are not comparable — and the gate depends on them

This is the trap the registry exists for.

`hybrid_search`'s relevance gate (`retrieval/gate.py`) compares the top-1 reranked score against a floor and widens the candidate pool when it falls short:

```python
return results[0].score < floor
```

- `jina-reranker-v3.5` is **listwise** and emits **unbounded logits** centred near zero. A floor of `0.0` means "worse than neutral" — a meaningful boundary.
- A TEI or Cohere cross-encoder emits a **sigmoid score in `[0, 1]`**. Nothing can ever fall below `0.0`.

So carrying a floor across a model switch does not merely mistune the gate — **it disables it**, with no exception, no log line, and no visible symptom beyond retrieval quietly getting worse on hard queries.

Two guards, because the mistake has two shapes:

1. **Unset floor.** `RERANK_SCORE_FLOOR=` adopts the active model's `default_score_floor`. The sentinel is `None`, not `0.0`, because `0.0` is a legitimate logit floor and a falsy check could not tell them apart.
2. **Explicitly wrong floor.** A pinned value left over from a previous model is caught at startup:

```
RERANK_SCORE_FLOOR=0.0 is unreachable for qwen3-reranker, which emits scores
in [0, 1]: the relevance gate would never fire. Leave RERANK_SCORE_FLOOR empty
to adopt the model's default of 0.5, or set a value strictly between 0 and 1.
```

This mirrors how `EMBEDDING_DIMENSIONS` is validated against the active model's `allowed_dimensions` — the same class of hazard, the same shape of fix.

### One more scale coupling

`score_scale` is a property of the model **as this codebase calls it**, not of the weights. The Qwen3-Reranker checkpoint emits raw logit differences; it reads as `unit` only because `retrieval/rerankers/tei.py` sends `raw_scores: false`, which applies the sigmoid server-side. Flipping that flag invalidates the registry entry.

## Degradation

`rerank()` **raises** `RerankError`; the caller decides how to degrade, because it is the only layer that knows whether it has a `degraded` channel to report into.

- `retrieval/pipeline.py` catches it, appends `"rerank"` to `degraded`, and returns the RRF-fused order.
- `rerank_dicts()` (used by `vector_search`, which predates the typed pipeline) swallows it and returns the input order — that caller has nowhere to report.

**When reranking is degraded, `FusedResult.score` carries the RRF fusion score instead**, which is always positive and so never falls below a floor of `0.0`. Those runs cannot fire the gate by construction, which is why `log_gate_decision` records `gate_reranked` — a retry rate without that flag has a meaningless denominator.

Retries are 429/5xx and transient network errors only. A 4xx fails on the first attempt: the previous implementation retried a `401` three times with exponential backoff before degrading, which cost three round trips to learn nothing.

## Graphiti's cross-encoder

Graphiti used `OpenAIRerankerClient` — a **second, independent** hosted reranker in the ingestion path, pointed at OpenAI regardless of `RERANK_MODEL`. It now goes through the same slot via `_SlotCrossEncoder` (`ingestion/graphiti_client.py`), mirroring how `_JinaGraphitiEmbedder` wraps the configured embedder.

That adapter degrades rather than raising: entity resolution is a ranking *refinement*, and failing a whole ingest because a reranker was unreachable is far worse than falling back to input order.

## Self-hosting

```bash
docker run --gpus all -p 8081:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Qwen/Qwen3-Reranker-4B
```

```bash
RERANK_ENABLED=true
RERANK_MODEL=qwen3-reranker
RERANK_ROUTE=local
RERANK_API_URL=http://localhost:8081
RERANK_SCORE_FLOOR=
```

TEI serves exactly one model, chosen at container start, and its `/rerank` body has **no `model` field**. The registry's `wire_id` therefore records which checkpoint the container must serve — a mismatch produces no error, just scores from a different model, which is why the adapter surfaces it as `model_name`.

TEI also has no `top_n` parameter: it scores every text, and the adapter truncates client-side so it honours the same contract as the hosted protocols.

## Adding a protocol

1. Create `retrieval/rerankers/<protocol>.py` implementing `rerank(query, documents, top_n) -> list[tuple[int, float]]` and `close()`. `(index, score)` is the entire contract — keeping it that narrow is what let the Jina v2 → v3.5 change of the unused `document` field pass unnoticed.
2. Add the entry to `config/rerank_models.py`, **including `score_scale` and a `default_score_floor` on that scale**.
3. Add a branch to `create_reranker()` in `retrieval/rerank.py`.
4. Add wire-contract tests to `tests/test_rerankers.py` — assert the request key names and the response envelope, since those are what differ.

Keep HTTP clients **per-call**. The embedder clients need the `WeakKeyDictionary` loop-affinity dance because CocoIndex shares one instance across worker threads with separate event loops; reranking is one call per *query*, not per chunk, so the pooling that dance protects is worth less than the deadlock it avoids.

See also: [Embeddings](embeddings.md) | [Search Tools](../server/search-tools.md) | [Re-indexing](../operations/reindex.md)
