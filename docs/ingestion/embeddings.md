# Embeddings

Spektr selects embeddings on **two orthogonal axes**, because they genuinely are orthogonal — the same model can be reachable through more than one endpoint:

- `EMBEDDING_MODEL` — **which model**: `jina-v4`, `voyage-4`, or `gemini-2`. Determines capabilities and dimensionality.
- `EMBEDDING_ROUTE` — **which endpoint serves it**: `native` (the vendor's own API) or `openrouter` (the gateway). **Leave empty** to take the model's default route.
- `EMBEDDING_DIMENSIONS` — **leave at `0`** to take the model's recommended size.

Switching model is therefore a one-line change: set `EMBEDDING_MODEL` and leave the other two unset. Route and dimensions follow from the registry, so there is nothing to retune by hand and no stale value to carry across a switch.

|`EMBEDDING_MODEL`|resolves to route|resolves to dimensions|
|-|-|-|
|`gemini-2`|`openrouter`|768|
|`voyage-4`|`native`|1024|
|`jina-v4`|`native`|512|

Pin either value only to override the registry deliberately.

|Model|`native`|`openrouter`|
|-|-|-|
|`jina-v4`|`api.jina.ai`|not served — Qwen Research licence forbids third-party hosting|
|`voyage-4`|`api.voyageai.com`|`voyageai/voyage-4-large`|
|`gemini-2`|not supported (deliberate)|`google/gemini-embedding-2`|

`gemini-2` has no native route on purpose: Google direct returns 429s citing Vertex quota on AI Studio keys, whereas OpenRouter fronts three endpoints for this model and routes around a degraded one. Illegal pairs are rejected by a `Settings` validator at startup, not discovered at first embed.

The predecessor setting `EMBEDDING_PROVIDER` conflated these axes and could not express "voyage-4 via OpenRouter" at all. It is now rejected with a migration message rather than silently ignored — `extra="ignore"` would otherwise boot a production box on the default model against an index built by a different one.

**Source:** `ingestion/embedder.py` (protocol + factory), `ingestion/embedders/` (provider implementations)

## Switching Providers

1. Set `EMBEDDING_MODEL` (and `EMBEDDING_ROUTE`) in `.env`
2. Fill in the API key for the chosen provider
3. Restart the ingestion pipeline and MCP server

**Important:** Vectors from different providers are not compatible (different dimensionalities and embedding spaces). Switching providers on an existing deployment requires re-ingesting all documents and recreating Qdrant collections. Every Qdrant point carries `embedder_model` and `embedder_dim` in its payload — `scripts/doctor.py` flags mixed values.

## Provider Comparison

|Feature|Jina v4|Voyage AI|OpenRouter (Gemini Embedding 2)|
|-|-|-|-|
|Text embeddings|Yes|Yes|Yes|
|Image embeddings|Yes|Yes (native route)|Yes|
|ColBERT multi-vector|Yes|No|No (single-vector model)|
|Document/query asymmetry|Yes (`task`)|Yes (`input_type`)|Yes (`input_type`)|
|Models|Single for all|Separate text + multimodal|Any OpenRouter embedding model|
|Default dense dimensions|2048 (MRL-truncatable)|1024|3072 (MRL-truncatable)|
|Valid sizes|any 1..2048 (Matryoshka)|256 / 512 / 1024 / 2048|any 1..3072 (Matryoshka)|
|Override|`EMBEDDING_DIMENSIONS` (one key for all models; `0` = model default). Validated against the active model's valid sizes at startup|←|←|
|Late chunking|Yes (`late_chunking=True`)|No (parameter ignored)|No (parameter ignored)|
|Text endpoint|`/v1/embeddings`|`/v1/embeddings`|`/v1/embeddings`|
|Image endpoint|Same|`/v1/multimodalembeddings`|Same (`input` carries an `image_url` content block)|

The Qdrant `documents_dense` collection is sized at provisioning time using `settings.dense_dimensions`, which returns the active provider's configured value. Switching providers therefore requires recreating the collection.

## Embedder Protocol

All providers implement the `Embedder` protocol defined in `ingestion/embedder.py`:

|Method|Input|Output|Description|
|-|-|-|-|
|`embed_text(texts, task, dimensions, late_chunking)`|`list[str]`|`list[list[float]]`|Batch text embedding. `late_chunking` only honoured by Jina.|
|`embed_text_query(query, dimensions)`|`str`|`list[float]`|Single query embedding|
|`embed_image(image_bytes, media_type)`|`bytes`|`list[float]`|Single image embedding. Raises `NotImplementedError` when the active model+route pair has no image support|
|`embed_multi_vector(image_bytes, media_type)`|`bytes`|`list[list[float]]`|ColBERT token vectors (`jina-v4` + `native` only)|
|`embed_query_multi_vector(query)`|`str`|`list[list[float]]`|ColBERT query vectors (`jina-v4` + `native` only)|
|`close()`|—|—|Close HTTP client|

Every embedder also exposes `model_name: str`, `dim: int`, `tokens_used: float`, and `reset_token_counter()`. The pipeline writes `embedder_model` and `embedder_dim` into every Qdrant point payload, both in the bulk and the live ingest paths.

The `task` parameter uses generic values: `"passage"` (for documents) and `"query"` (for search queries). Each provider maps these to its own API terminology internally.

## Factory

Use `create_embedder()` to get the configured provider:

```python
from ingestion.embedder import Embedder, create_embedder

embedder: Embedder = create_embedder()
vectors = await embedder.embed_text(["hello world"])
```

`create_embedder()` dispatches on the **route**, since that determines the wire protocol; the route's client resolves its own model id from the registry in `config/embedding_models.py`. Implementations are imported lazily, so unused routes need no configuration.

Dimensionality has a single source of truth: `settings.dense_dimensions` returns `EMBEDDING_DIMENSIONS` when set, else the model's registry default. The old per-provider `*_DENSE_DIMENSIONS` keys are gone.

### Why one dimension key, not one per model

Per-model *values* live in the registry (`recommended_dimensions`), not in `.env`, and `EMBEDDING_DIMENSIONS=0` adopts them. So you get the per-model behaviour without per-model keys — those would be dead config for the two inactive models, the sprawl the model/route split removed.

The single key remains as an *override*, for when the recommended size is not what you want.

The risk that argues for per-model keys is real, though: a size tuned for one model silently carrying over when you switch. `768` suits `gemini-2` but is not one of `voyage-4`'s four sizes. That is handled by validating the single key against the active model's `allowed_dimensions` at startup, so the switch fails loudly instead of requesting a size the API will reject:

```
EMBEDDING_DIMENSIONS=768 is not supported by voyage-4; it emits 256, 512, 1024, 2048.
```

An empty `allowed_dimensions` marks a Matryoshka model, where any size from 1 up to the default is valid.

## Jina v4

**Selected by:** `EMBEDDING_MODEL=jina-v4`, `EMBEDDING_ROUTE=native` (the only route — see the licensing note above).

**Config:** `JINA_API_KEY`, `JINA_API_URL`, `JINA_MODEL`, `JINA_RPM` (default 500), `JINA_TPM` (default 100,000), `JINA_MAX_CONCURRENT`, `JINA_BATCH_SIZE`. Dimensions come from `EMBEDDING_DIMENSIONS` (default 2048).

All requests go to `${JINA_API_URL}/v1/embeddings` with:

- `model`: `jina-embeddings-v4` (configurable)
- `task`: maps `passage` → `retrieval.passage`, `query` → `retrieval.query`
- `dimensions`: per-call override or `EMBEDDING_DIMENSIONS` (Matryoshka — any size up to 2048 is valid)
- `normalized: true`, `embedding_type: "float"`
- Images base64-encoded as `data:{media_type};base64,{b64}`
- Image methods use a 120s timeout (vs 60s default)
- ColBERT mode adds `embedding_type_params: {"output_type": "colbert"}` and forces `dimensions=128`
- `late_chunking=True` sends the whole list as a single batch (no `JINA_BATCH_SIZE` slicing)

## Voyage AI

**Selected by:** `EMBEDDING_MODEL=voyage-4` with either route. On `openrouter` the ids are `voyageai/voyage-4-large` (text) and `voyageai/voyage-multimodal-3.5` (image, not yet implemented on that route).

**Config:** `VOYAGE_API_KEY`, `VOYAGE_API_URL`, `VOYAGE_TEXT_MODEL`, `VOYAGE_MULTIMODAL_MODEL`, `VOYAGE_RPM`, `VOYAGE_MAX_CONCURRENT`. Dimensions come from `EMBEDDING_DIMENSIONS` (default 1024).

Uses two separate endpoints:

- Text: `${VOYAGE_API_URL}/v1/embeddings` with `voyage-4-large`
- Images: `${VOYAGE_API_URL}/v1/multimodalembeddings` with `voyage-multimodal-3.5`

ColBERT multi-vector methods raise `NotImplementedError`. The settings validator rejects `MULTIVEC_ENABLED=true` unless the active model+route pair lists that capability — only `jina-v4` via `native` does.

## OpenRouter

**Selected by:** `EMBEDDING_ROUTE=openrouter`, with `EMBEDDING_MODEL` of `gemini-2` or `voyage-4`.

**Config:** `OPENROUTER_API_KEY`, `OPENROUTER_API_URL`, `OPENROUTER_RPM` (default 300), `OPENROUTER_MAX_CONCURRENT` (default 10), `OPENROUTER_HTTP_REFERER` (optional), `OPENROUTER_X_TITLE` (optional). Dimensions come from `EMBEDDING_DIMENSIONS`; the model id is resolved from the registry, not configured.

OpenAI-compatible `/v1/embeddings` gateway. This is **not** a general client for all 32 models OpenRouter lists — the `input_type` contract below holds for the Gemini and Voyage families but not for the e5/bge/sentence-transformers ones, which expect instruction prefixes in the text instead. The registry is what keeps that honest: only registered model+route pairs are selectable. The optional `HTTP-Referer` / `X-Title` headers are forwarded for OpenRouter ranking attribution and can be left empty.

The GA `google/gemini-embedding-2` returns **bit-identical vectors** to the retired `-preview` id, so that rename needed no re-ingest.

### Gateway parameter quirks

Verified live against the endpoint — these are not guessable from the docs:

|Parameter|Behaviour|
|-|-|
|`dimensions`|**Honoured.** `dimensions=768` returns 768 floats|
|`output_dimensionality`|**Silently ignored.** Gemini's native spelling; the gateway drops it and returns the full 3072|
|`input_type`|**Honoured.** `"query"` / `"document"`|
|`task_type`, `task`|**Silently ignored.** Gemini's and Jina's spellings respectively|

The endpoint is deterministic: identical requests return bit-identical vectors.

**Unrecognised `input_type` values are accepted and silently ignored**, falling back to the symmetric default with no error. A typo therefore costs the query/document asymmetry invisibly, which is why `_input_type()` raises on an unmapped task rather than forwarding it.

### Document/query asymmetry

`embed_text(task=...)` maps onto `input_type`: `passage` → `document`, `query` → `query`. This matters — embeddings of the *same text* differ by cos 0.82 between the two modes.

Both sides of the index must use the same mapping. Changing it changes the vector space and requires a full re-ingest; it is not a drop-in fix for an existing index.

### Image embedding

Supported for `gemini-2` on this route. Three properties, all verified against the live endpoint, make it work without a second index:

|Property|Behaviour|
|-|-|
|Model id|The same id embeds text and images — gemini-2 is natively multimodal, so there is no separate image model|
|`dimensions`|**Honoured for image input**, so image vectors match the collection's size|
|`input_type`|Accepted but has **no effect** on images (byte-identical output), so it is deliberately not sent|
|Media types|`image/png` and `image/jpeg` both accepted|

Because text and images land in one 768-d space in `documents_dense`, a plain text query retrieves page images through the ordinary dense channel — no dedicated tool required. Measured on synthetic pages: the query *"a bar chart of quarterly revenue"* scores 0.729 against a chart image versus 0.504 against an unrelated one.

Images bill on a separate meter (\$0.45/M for gemini-2) and use a 120s timeout rather than the 60s text default.

**ColBERT multi-vector remains unavailable**, and not for want of implementing: gemini-2 emits a single vector. The `visual_search` tool queries `documents_multivec` with `using="colbert"` and has no dense fallback, so it stays `jina-v4` + `native` only. Image *retrieval* on this route happens through `vector_search` / `multi_search` / `hybrid_search`, not `visual_search`.

The settings validator rejects `IMAGE_EMBED_STRATEGY` other than `none` when the active pair has no image implementation, so a misconfiguration fails at startup instead of silently dropping pages.

The settings validator rejects `MULTIVEC_ENABLED=true` on this route.

### Rate limits

OpenRouter uses a single `TokenBucket` (RPM only) plus `asyncio.Semaphore(OPENROUTER_MAX_CONCURRENT)`. Token usage is estimated as `sum(len(t) for t in texts) / 4`. There is **no** TPM bucket — OpenRouter's per-minute limits are looser than Jina's, and per-key throttling is enforced server-side via 429s, which back-pressure-pause the limiter for `Retry-After` seconds.

## Retry Logic

All providers use `tenacity` with the same retry shape (retry on `httpx.HTTPStatusError` for 429/5xx, plus transient connection errors for Jina/OpenRouter):

|Parameter|Value|
|-|-|
|Wait|Exponential: Jina min 5s/max 60s (multiplier=2); Voyage/OpenRouter min 2s/max 30s (multiplier=1)|
|Max attempts|`settings.max_retries` (default: 3)|
|Retryable errors|HTTP 429, 5xx, `httpx.ReadError`, `httpx.ConnectError`, `httpx.RemoteProtocolError` (Jina, OpenRouter); 429/5xx only for Voyage|
|Non-retryable|4xx errors (except 429) raise immediately|

On 429, the response's `Retry-After` header is honoured by calling `TokenBucket.pause(seconds)` on all rate limiters owned by that provider, so concurrent requests back off together.

## Rate Limiting

Jina uses a **dual `TokenBucket` rate limiter** system (defined in `ingestion/embedder.py`):

- **RPM limiter:** `TokenBucket(tokens_per_sec=jina_rpm/60, burst=jina_max_concurrent)` — consumes 1 token per request
- **TPM limiter:** `TokenBucket(tokens_per_sec=jina_tpm/60, burst=jina_tpm)` — consumes variable tokens per request based on payload estimation

Jina token estimation:

- Text inputs: `len(text) / 4` per string (standard heuristic)
- Image inputs: tile-based estimator. The image is decoded and tiled into 28×28 patches; each tile costs ~10 tokens (matching Jina v4's Qwen2.5-VL vision encoder). If decoding fails the estimator falls back to `_FALLBACK_TOKENS = 2000` (see `_estimate_image_tokens` in `embedders/jina.py`).

`TokenBucket.acquire(n)` supports variable token counts (not just 1). On HTTP 429, both RPM and TPM limiters are paused for `Retry-After` seconds.

Voyage and OpenRouter both use a single RPM `TokenBucket` plus an `asyncio.Semaphore` for concurrency. Configure via `*_RPM` and `*_MAX_CONCURRENT` settings.

## Smart Image Embedding (Path A)

PDF pages are selectively image-embedded based on a fast heuristic to avoid expensive vision-model calls on text-only pages.

**Strategy** (`IMAGE_EMBED_STRATEGY`):

|Value|Behavior|
|-|-|
|`smart` (default)|Embed a PDF page only when `Page.has_visual_content == True` (i.e. the PDF page has at least one embedded raster image as detected by `fitz_page.get_images()`)|
|`all`|Embed every PDF page as an image (slow, expensive)|

Pure image inputs (`.png`, `.jpg`, etc.) are always embedded as images regardless of `IMAGE_EMBED_STRATEGY`.

> **Note:** The strategy is only consulted for `content_type == "pdf"` pages. There is no current "skip everything" branch — text-only PDF pages with `IMAGE_EMBED_STRATEGY=smart` simply have their image step skipped (no Qdrant point written for that page's image).

## Pipeline Execution Strategy

Text and image embedding tasks use different concurrency strategies to stay within API limits:

- **Text tasks** run concurrently via `asyncio.gather` with a `Semaphore(2)`, allowing parallel inflight requests
- **Image tasks** run sequentially — image inputs cost orders of magnitude more tokens per request than text, so concurrent image requests would immediately blow the TPM bucket (see Jina tile-based estimator above)

This split prevents TPM exhaustion while still maximising throughput for text-heavy workloads.

## Standalone Embedding Helpers

`ingestion/cocoindex_ops.py` exposes the embedder as plain synchronous functions using `from ingestion._utils import run_async`.

|Helper|Wraps|Description|
|-|-|-|
|`op_embed_text(text)`|`embed_text([text])`|Text -> dense vector|
|`op_embed_image(image_bytes)`|`embed_image(image_bytes)`|Image -> dense vector|
|`op_embed_image_multivec(image_bytes)`|`embed_multi_vector(image_bytes)`|Image -> ColBERT token vectors|

The embedder instance is lazily initialized via `create_embedder()` on first use. None of these are wired into the ingestion app — the bulk pipeline embeds through `ingestion/page_processor.py` — they exist for callers that want a one-shot embedding without an event loop.

## Graphiti Integration

When Graphiti is used for the knowledge graph, it requires its own embedder interface. A `_JinaGraphitiEmbedder` adapter (defined in `ingestion/graphiti_client.py`) wraps the configured project embedder via `create_embedder()` and delegates all calls to it. This means:

- Graph embeddings share the same rate limiters as document ingestion
- No duplicate HTTP clients or separate API quota consumption
- `graphiti_client.py` sets `EMBEDDING_DIM=512` in `os.environ` at import time so Graphiti's vector index sizing matches its internal expectations

## Sparse Channel (miniCOIL)

Retrieval also has a lexical channel, independent of the dense embedding provider above. It exists to catch exact-match queries — part numbers, error codes, proper nouns — that a semantic embedding can miss.

**Model:** `Qdrant/minicoil-v1` (`SPARSE_MODEL`), via [fastembed](https://github.com/qdrant/fastembed). miniCOIL behaves like BM25 that understands word sense — it keeps exact keyword matching while disambiguating by context. It runs locally on CPU; there is no API call and no per-token cost. The first encode call pays a one-time model load penalty.

**Source:** `ingestion/sparse_embedder.py`, consumed by `retrieval/channels.py::sparse_channel`.

Toggle with `SPARSE_ENABLED` (default `true`). When disabled, `multi_search`/`hybrid_search` retrieve on the dense channel only.

### Document vs. query encoding

`encode_documents()` (indexing) and `encode_query()` (search) are asymmetric:

- **Indexing** — `SparseTextEmbedding.embed()` applies BM25-style length normalisation, controlled by `avg_len` (`MINICOIL_AVG_LEN = 80` in `config/constants.py`).
- **Querying** — `SparseTextEmbedding.query_embed()` applies **no** length normalisation.

This is not a design choice made per call — fastembed's API doesn't expose `avg_len` as a per-call keyword on `embed()`/`query_embed()` at all. It's a **constructor** argument of `SparseTextEmbedding`, so it's fixed once at model load time and fastembed itself decides internally, based on which method is called, whether to apply it. Passing `avg_len` anywhere else (e.g. as a per-call `embed(options=...)` kwarg) is silently ignored and does nothing — there's no error, the value just doesn't take effect. If sparse search results look off after touching this code, check that the normalisation is still happening at `SparseTextEmbedding(...)` construction in `ingestion/sparse_embedder.py::_load_model`, not somewhere per-call.

### Named vectors on `documents_dense`

The dense and sparse channels don't live in separate collections — they're two **named vectors** on the same `documents_dense` point, set up by `ingestion/qdrant_setup.py::create_dense_collection`:

- `dense` — the active provider's dense embedding, `COSINE` distance
- `sparse` — miniCOIL, with `Modifier.IDF` (Qdrant applies IDF weighting server-side at query time)

Every text-chunk point written by both ingestion paths carries both vectors (`ingestion/pipeline.py` for Path A, `ingestion/live_ingest.py` for Path B). Image and VLM-caption points carry `dense` only — there's no text to run through miniCOIL. `scripts/doctor.py` flags text-chunk points missing a `sparse` vector.

**Named and unnamed vectors cannot coexist in the same Qdrant collection.** Before this channel existed, `documents_dense` used a single unnamed vector; adding `sparse` required switching `dense` to a named vector too, which is a breaking schema change requiring a full re-index. See [Re-indexing Runbook](../operations/reindex.md).

Retrieval reads these named vectors directly with `query_points(..., using="dense")` / `using="sparse"` — see `retrieval/channels.py`.

## Adding a New Provider

1. Create `ingestion/embedders/<provider>.py` implementing all `Embedder` protocol methods (including `model_name`/`dim`)
2. Add provider settings to `config/settings.py` and `.env.example`, and extend `Settings.dense_dimensions`
3. Add a branch to `create_embedder()` in `ingestion/embedder.py`
4. Update the model validator in `Settings` if the provider can't support `MULTIVEC_ENABLED=true`
5. Add tests in `tests/test_<provider>_embedder.py`

See also: [Pipeline Overview](overview.md) | [CocoIndex Pipeline](cocoindex.md) | [Architecture Data Flow](../architecture/data-flow.md)
