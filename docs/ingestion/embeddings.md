# Embeddings

Spektr uses a provider-agnostic embedding abstraction. The active provider is selected via `EMBEDDING_PROVIDER` in `.env`. Currently supported: **Jina v4** and **Voyage AI**.

**Source:** `ingestion/embedder.py` (protocol + factory), `ingestion/embedders/` (provider implementations)

## Switching Providers

1. Set `EMBEDDING_PROVIDER` to `"jina"` or `"voyage"` in `.env`
2. Fill in the API key for the chosen provider
3. Restart the ingestion pipeline and MCP server

**Important:** Vectors from different providers are not compatible. Switching providers on an existing deployment requires re-ingesting all documents and recreating Qdrant collections.

## Provider Comparison

| Feature | Jina v4 | Voyage AI |
|-|-|-|
| Text embeddings | Yes | Yes |
| Image embeddings | Yes | Yes |
| ColBERT multi-vector | Yes | No |
| Models | Single model for all | Separate text + multimodal models |
| Default dimensions | 2048 | 1024 |
| Text endpoint | `/v1/embeddings` | `/v1/embeddings` |
| Image endpoint | Same | `/v1/multimodalembeddings` |

## Embedder Protocol

All providers implement the `Embedder` protocol defined in `ingestion/embedder.py`:

| Method | Input | Output | Description |
|-|-|-|-|
| `embed_text(texts, task, dimensions)` | `list[str]` | `list[list[float]]` | Batch text embedding |
| `embed_text_query(query, dimensions)` | `str` | `list[float]` | Single query embedding |
| `embed_image(image_bytes, media_type)` | `bytes` | `list[float]` | Single image embedding |
| `embed_multi_vector(image_bytes, media_type)` | `bytes` | `list[list[float]]` | ColBERT token vectors |
| `embed_query_multi_vector(query)` | `str` | `list[list[float]]` | ColBERT query vectors |
| `close()` | — | — | Close HTTP client |

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

**Config:** `JINA_API_KEY`, `JINA_MODEL`, `JINA_DENSE_DIMENSIONS`, `JINA_RPM` (default: 500), `JINA_TPM` (default: 100,000 for free tier), `JINA_MAX_CONCURRENT`, `JINA_BATCH_SIZE`

All requests go to `https://api.jina.ai/v1/embeddings` with:

- `model`: `jina-embeddings-v4` (configurable)
- `normalized`: `true`
- `embedding_type`: `float`
- Images are base64-encoded as `data:{media_type};base64,{b64}`
- Image methods use a 120s timeout (vs 60s default)
- ColBERT mode uses `embedding_type_params: {"output_type": "colbert"}`

## Voyage AI

**Config:** `VOYAGE_API_KEY`, `VOYAGE_TEXT_MODEL`, `VOYAGE_MULTIMODAL_MODEL`, `VOYAGE_DENSE_DIMENSIONS`, `VOYAGE_RPM`, `VOYAGE_MAX_CONCURRENT`

Uses two separate endpoints:

- Text: `https://api.voyageai.com/v1/embeddings` with `voyage-4-large`
- Images: `https://api.voyageai.com/v1/multimodalembeddings` with `voyage-multimodal-3.5`

ColBERT multi-vector methods raise `NotImplementedError`. If `MULTIVEC_ENABLED=true`, use Jina instead.

## Retry Logic

Both providers use `tenacity` with the same retry strategy:

| Parameter | Value |
|-|-|
| Wait | Exponential, 5s min, 60s max (multiplier=2) |
| Max attempts | `settings.max_retries` (default: 3) |
| Retryable errors | HTTP 429, 5xx, `httpx.ReadError`, `httpx.ConnectError`, `httpx.RemoteProtocolError` |
| Non-retryable | 4xx errors (except 429) raise immediately |

## Rate Limiting

The Jina provider uses a **dual `TokenBucket` rate limiter** system (defined in `ingestion/embedder.py`):

- **RPM limiter:** `TokenBucket(tokens_per_sec=jina_rpm/60, burst=jina_max_concurrent)` — consumes 1 token per request
- **TPM limiter:** `TokenBucket(tokens_per_sec=jina_tpm/60, burst=jina_tpm)` — consumes variable tokens per request based on payload estimation

Token estimation:
- Text inputs: `len(text) / 4` tokens per string
- Image inputs: `len(base64_data) / 4` tokens

`TokenBucket.acquire(n)` supports variable token counts (not just 1). On HTTP 429, both RPM and TPM limiters are paused.

The Voyage provider uses a single `TokenBucket` + `asyncio.Semaphore` for concurrency. Configure via `*_RPM` and `*_MAX_CONCURRENT` settings.

## Pipeline Execution Strategy

Text and image embedding tasks use different concurrency strategies to stay within API limits:

- **Text tasks** run concurrently via `asyncio.gather` with a `Semaphore(2)`, allowing parallel inflight requests
- **Image tasks** run sequentially — each image encodes to 50,000–100,000+ tokens, so concurrent image requests would immediately blow the TPM limit

This split prevents TPM exhaustion while still maximising throughput for text-heavy workloads.

## CocoIndex Ops Wrapper

`ingestion/cocoindex_ops.py` exposes the embedder as synchronous CocoIndex operations using a `run_async` helper.

| CocoIndex Op | Wraps | Description |
|-|-|-|
| `op_embed_text(text)` | `embed_text([text])` | Text -> dense vector |
| `op_embed_image(image_bytes)` | `embed_image(image_bytes)` | Image -> dense vector |
| `op_embed_image_multivec(image_bytes)` | `embed_multi_vector(image_bytes)` | Image -> ColBERT token vectors |

The embedder instance is lazily initialized via `create_embedder()` on first use.

## Graphiti Integration

When Graphiti is used for the knowledge graph, it requires its own embedder interface. A `_JinaGraphitiEmbedder` adapter (defined in `ingestion/embedders/jina.py`) wraps the same `JinaV4Embedder` instance and delegates all calls to it. This means:

- Graph embeddings share the same rate limiters as document ingestion
- No duplicate HTTP clients or separate API quota consumption
- `EMBEDDING_DIM=2048` is set automatically on the Graphiti client via the adapter

## Adding a New Provider

1. Create `ingestion/embedders/<provider>.py` implementing all `Embedder` protocol methods
2. Add provider settings to `config/settings.py` and `.env.example`
3. Add a branch to `create_embedder()` in `ingestion/embedder.py`
4. Add tests in `tests/test_<provider>_embedder.py`

See also: [Pipeline Overview](overview.md) | [CocoIndex Pipeline](cocoindex.md) | [Architecture Data Flow](../architecture/data-flow.md)
