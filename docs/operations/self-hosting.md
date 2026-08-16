# Running Spektr without a US vendor API

Every model in the retrieval path can be replaced with one you host. This page is the operational recipe; [Embeddings](../ingestion/embeddings.md) explains the slot model and [Reranking](../ingestion/reranking.md) the score-scale trap.

## What actually leaves your network

Start by knowing what you are removing. On the shipped defaults:

|Component|Default|Where it runs|
|-|-|-|
|Dense embeddings|`gemini-2` via OpenRouter|Google, through a US gateway that fans out to several undisclosed upstream endpoints|
|Sparse channel|miniCOIL via fastembed|**Local CPU already**|
|Reranker|`jina-reranker-v3.5`|Jina (EU vendor, now under Elastic)|
|Late interaction|slot unfilled|—|
|Graph extraction|Graphiti + LLM|Whatever `LLM_BASE_URL` points at|

The sparse channel and the `gliner` graph engine were always local. It is the dense, rerank and graph-LLM paths that carry data off-box.

## The two honest options

**There is no single self-hostable model that covers text, image and ColBERT the way `jina-v4` does.** Sovereignty means splitting one model into two or three. That is what the slots are for, and it has a real consequence you should decide on before starting.

### Option A — fully self-hosted, images via visual search

```bash
# dense: text only, Apache 2.0 weights
EMBEDDING_MODEL=qwen3-embedding-8b
EMBEDDING_ROUTE=local
LOCAL_EMBED_API_URL=http://tei-embed:80
EMBEDDING_DIMENSIONS=1024        # Matryoshka; 4096 is the full width

# images leave documents_dense entirely
IMAGE_EMBED_STRATEGY=none
MULTIVEC_MODEL=colqwen2.5

# rerank
RERANK_ENABLED=true
RERANK_MODEL=qwen3-reranker
RERANK_ROUTE=local
RERANK_API_URL=http://tei-rerank:80
RERANK_SCORE_FLOOR=              # leave empty — adopts the model's own

# graph: local CPU extraction, no LLM call at all
GRAPH_ENGINE=gliner
```

**The cost, stated plainly:** cross-modal retrieval from one shared vector space is a `gemini-2` privilege. With a text-only dense model, a plain text query no longer retrieves page images through the dense channel — they are reachable through `visual_search` instead. `Settings` enforces this (`IMAGE_EMBED_STRATEGY` must be `none`) so the switch fails at startup rather than silently dropping every page.

### Option B — keep the shared space, accept a licence

`jina-embeddings-v5-omni` puts text, image, audio and video in one space and can be self-hosted. Its weights are **CC BY-NC 4.0** — commercial self-hosting needs a licence from Elastic. That is a procurement decision, not an engineering one, and it is the only way to keep single-space cross-modal retrieval off a vendor API.

## Serving the models

```bash
docker run --gpus all -p 8080:80 \
  -v $PWD/hf-cache:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Qwen/Qwen3-Embedding-8B

docker run --gpus all -p 8081:80 \
  -v $PWD/hf-cache:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Qwen/Qwen3-Reranker-4B
```

ColQwen is served through [`colpali-engine`](https://github.com/illuin-tech/colpali) rather than TEI, because late interaction returns a grid of patch vectors rather than one vector per input.

Point `LOCAL_EMBED_API_URL` and `RERANK_API_URL` at these. Neither falls back to a vendor host when unset — a self-hosted slot silently pointed at `api.jina.ai` would be worse than a startup failure.

### Sizing

`LOCAL_MAX_CONCURRENT` (default 8) is the only limiter on the local route. There is deliberately **no request-per-minute bucket**: a server you own has no quota, it has finite GPU memory, and an arbitrary rate cap is throughput lost for nothing. Lower it if you see OOMs; raise it if the GPU is idle.

`LOCAL_BATCH_SIZE` (default 32) is texts per call. An 8B encoder on a batch of 32 is slower than a fanned-out managed endpoint, which is why the client's timeout is 120s rather than the 60s used for vendor routes.

## Migration cost

Not uniform, and worth knowing before you start:

|Slot changed|Cost|
|-|-|
|**rerank**|**Nothing.** Query-time only — no re-ingest, no collection change. Try it first.|
|**multivec**|Re-index of `documents_multivec` only. `documents_dense` is untouched.|
|**dense**|**Full re-ingest**, plus recreating `documents_dense` — the vector space and its width both change. See [Re-indexing](reindex.md).|

Because the rerank slot is free, adopt it first and measure with `task eval-retrieval` before touching the dense slot.

## Verifying

```bash
task doctor          # flags mixed embedder_model / embedder_dim across points
task smoke           # direct vector_search, no MCP or LLM
task eval-retrieval  # recall@10, nDCG@10, MRR — no LLM in the loop
```

Every point carries `embedder_model` and `embedder_dim` in its payload, so `task doctor` will tell you if a partial re-ingest left two models' vectors in one collection. That mixture is not detectable by search quality alone — it just quietly degrades.

**Calibrate the score floor.** `qwen3-reranker`'s registered `default_score_floor` of `0.5` is a plausible sigmoid midpoint, not a measured one. Run `task eval-retrieval` and tune it before treating it as a real threshold.

## What remains hosted

Even on Option A, check these before claiming an air gap:

- `LLM_API_TYPE` / `LLM_BASE_URL` — used by query decomposition (`hybrid_search`), schema induction and VLM captioning. `GRAPH_ENGINE=gliner` removes the LLM from *ingestion*, not from the whole system. Set `DECOMPOSE_ENABLED=false` and `SCHEMA_INDUCTION_ENABLED=false`, or point `LLM_BASE_URL` at a local vLLM.
- `DOCUMENT_SOURCE=s3` — AWS, obviously.
- `LOGFIRE_TOKEN` — leave `OBSERVABILITY_LOCAL_ONLY=true` (the default) and nothing ships.

See also: [Embeddings](../ingestion/embeddings.md) | [Reranking](../ingestion/reranking.md) | [Re-indexing](reindex.md)
