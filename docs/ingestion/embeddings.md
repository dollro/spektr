# Embeddings

Spektr uses a provider-agnostic embedding abstraction. The active provider is selected via `EMBEDDING_PROVIDER` in `.env`. Currently supported: **Jina v4**, **Voyage AI**, and **OpenRouter** (any OpenAI-compatible embedding model — defaults to Google Gemini Embedding 2).

**Source:** `ingestion/embedder.py` (protocol + factory), `ingestion/embedders/` (provider implementations)

## Switching Providers

1. Set `EMBEDDING_PROVIDER` to `"jina"`, `"voyage"`, or `"openrouter"` in `.env`
2. Fill in the API key for the chosen provider
3. Restart the ingestion pipeline and MCP server

**Important:** Vectors from different providers are not compatible (different dimensionalities and embedding spaces). Switching providers on an existing deployment requires re-ingesting all documents and recreating Qdrant collections. Every Qdrant point carries `embedder_model` and `embedder_dim` in its payload — `scripts/doctor.py` flags mixed values.

## Provider Comparison

|Feature|Jina v4|Voyage AI|OpenRouter (Gemini Embedding 2)|
|-|-|-|-|
|Text embeddings|Yes|Yes|Yes|
|Image embeddings|Yes|Yes|No (text-only)|
|ColBERT multi-vector|Yes|No|No|
|Models|Single for all|Separate text + multimodal|Any OpenRouter embedding model|
|Default dense dimensions|2048 (`JINA_DENSE_DIMENSIONS`, MRL-truncatable)|1024 (`VOYAGE_DENSE_DIMENSIONS`)|3072 by default for Gemini Embedding 2 (`OPENROUTER_DENSE_DIMENSIONS`)|
|Late chunking|Yes (`late_chunking=True`)|No (parameter ignored)|No (parameter ignored)|
|Text endpoint|`/v1/embeddings`|`/v1/embeddings`|`/v1/embeddings`|
|Image endpoint|Same|`/v1/multimodalembeddings`|Not supported|

The Qdrant `documents_dense` collection is sized at provisioning time using `settings.dense_dimensions`, which returns the active provider's configured value. Switching providers therefore requires recreating the collection.

## Embedder Protocol

All providers implement the `Embedder` protocol defined in `ingestion/embedder.py`:

|Method|Input|Output|Description|
|-|-|-|-|
|`embed_text(texts, task, dimensions, late_chunking)`|`list[str]`|`list[list[float]]`|Batch text embedding. `late_chunking` only honoured by Jina.|
|`embed_text_query(query, dimensions)`|`str`|`list[float]`|Single query embedding|
|`embed_image(image_bytes, media_type)`|`bytes`|`list[float]`|Single image embedding (Jina/Voyage only)|
|`embed_multi_vector(image_bytes, media_type)`|`bytes`|`list[list[float]]`|ColBERT token vectors (Jina only)|
|`embed_query_multi_vector(query)`|`str`|`list[list[float]]`|ColBERT query vectors (Jina only)|
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

The factory reads `settings.embedding_provider` and lazily imports the corresponding implementation, so unused providers don't require their configuration to be set.

## Jina v4

**Config:** `JINA_API_KEY`, `JINA_API_URL`, `JINA_MODEL`, `JINA_DENSE_DIMENSIONS` (default 2048), `JINA_RPM` (default 500), `JINA_TPM` (default 100,000), `JINA_MAX_CONCURRENT`, `JINA_BATCH_SIZE`

All requests go to `${JINA_API_URL}/v1/embeddings` with:

- `model`: `jina-embeddings-v4` (configurable)
- `task`: maps `passage` → `retrieval.passage`, `query` → `retrieval.query`
- `dimensions`: per-call override or `JINA_DENSE_DIMENSIONS` (Matryoshka — any size up to 2048 is valid)
- `normalized: true`, `embedding_type: "float"`
- Images base64-encoded as `data:{media_type};base64,{b64}`
- Image methods use a 120s timeout (vs 60s default)
- ColBERT mode adds `embedding_type_params: {"output_type": "colbert"}` and forces `dimensions=128`
- `late_chunking=True` sends the whole list as a single batch (no `JINA_BATCH_SIZE` slicing)

## Voyage AI

**Config:** `VOYAGE_API_KEY`, `VOYAGE_API_URL`, `VOYAGE_TEXT_MODEL`, `VOYAGE_MULTIMODAL_MODEL`, `VOYAGE_DENSE_DIMENSIONS` (default 1024), `VOYAGE_RPM`, `VOYAGE_MAX_CONCURRENT`

Uses two separate endpoints:

- Text: `${VOYAGE_API_URL}/v1/embeddings` with `voyage-4-large`
- Images: `${VOYAGE_API_URL}/v1/multimodalembeddings` with `voyage-multimodal-3.5`

ColBERT multi-vector methods raise `NotImplementedError`. The settings model validator rejects `MULTIVEC_ENABLED=true` when `embedding_provider=voyage`.

## OpenRouter

**Config:** `OPENROUTER_API_KEY`, `OPENROUTER_API_URL`, `OPENROUTER_MODEL`, `OPENROUTER_DENSE_DIMENSIONS` (default 3072 for Gemini Embedding 2), `OPENROUTER_RPM` (default 300), `OPENROUTER_MAX_CONCURRENT` (default 10), `OPENROUTER_HTTP_REFERER` (optional), `OPENROUTER_X_TITLE` (optional)

OpenAI-compatible `/v1/embeddings` gateway. Default model is `google/gemini-embedding-2-preview` (3072d, MRL truncation). Any OpenRouter-served embedding model can be selected via `OPENROUTER_MODEL`; set `OPENROUTER_DENSE_DIMENSIONS` to that model's output size (or to a smaller MRL-truncated value). The optional `HTTP-Referer` / `X-Title` headers are forwarded for OpenRouter ranking attribution and can be left empty.

**Text-only.** `embed_image`, `embed_multi_vector`, and `embed_query_multi_vector` raise `NotImplementedError`. For text-only corpora set `IMAGE_EMBED_STRATEGY=smart` and accept that PDF pages without embedded images won't be image-embedded; mixing PDFs with images while using OpenRouter currently means image pages will fail their embedding step (caught by the page-level try/except). For multimodal coverage use `embedding_provider=jina|voyage`.

The settings model validator rejects `MULTIVEC_ENABLED=true` with `embedding_provider=openrouter`.

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

## CocoIndex Ops Wrapper

`ingestion/cocoindex_ops.py` exposes the embedder as synchronous CocoIndex operations using `from ingestion._utils import run_async`.

|CocoIndex Op|Wraps|Description|
|-|-|-|
|`op_embed_text(text)`|`embed_text([text])`|Text -> dense vector|
|`op_embed_image(image_bytes)`|`embed_image(image_bytes)`|Image -> dense vector|
|`op_embed_image_multivec(image_bytes)`|`embed_multi_vector(image_bytes)`|Image -> ColBERT token vectors|

The embedder instance is lazily initialized via `create_embedder()` on first use.

## Graphiti Integration

When Graphiti is used for the knowledge graph, it requires its own embedder interface. A `_JinaGraphitiEmbedder` adapter (defined in `ingestion/graphiti_client.py`) wraps the configured project embedder via `create_embedder()` and delegates all calls to it. This means:

- Graph embeddings share the same rate limiters as document ingestion
- No duplicate HTTP clients or separate API quota consumption
- `graphiti_client.py` sets `EMBEDDING_DIM=512` in `os.environ` at import time so Graphiti's vector index sizing matches its internal expectations

## Adding a New Provider

1. Create `ingestion/embedders/<provider>.py` implementing all `Embedder` protocol methods (including `model_name`/`dim`)
2. Add provider settings to `config/settings.py` and `.env.example`, and extend `Settings.dense_dimensions`
3. Add a branch to `create_embedder()` in `ingestion/embedder.py`
4. Update the model validator in `Settings` if the provider can't support `MULTIVEC_ENABLED=true`
5. Add tests in `tests/test_<provider>_embedder.py`

See also: [Pipeline Overview](overview.md) | [CocoIndex Pipeline](cocoindex.md) | [Architecture Data Flow](../architecture/data-flow.md)
