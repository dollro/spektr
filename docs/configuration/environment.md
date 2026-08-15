# Environment Variables

All configuration is loaded from a `.env` file (or real environment variables) via Pydantic Settings in `config/settings.py`. Copy `.env.example` to `.env` and fill in the required values.

```bash
cp .env.example .env
```

!!! warning "Empty values must not carry an inline comment"
    Docker Compose's `env_file:` parser only strips an inline comment when a value precedes it. `FOO=  # note` therefore sets `FOO` to the literal string `# note` inside the container — not to empty. Put the comment on the line *above* any variable you intend to leave empty:

    ```ini
    # optional key prefix to restrict the scan
    S3_PREFIX=
    ```

    Inline comments are fine on lines that do have a value. This only affects the containerised path; local runs read `.env` through python-dotenv, which strips them correctly either way. A garbage `S3_PREFIX` makes the bucket scan match nothing and ingest zero documents with no error, so the distinction matters.

## Migrating an older `.env`

`scripts/migrate_env.py` upgrades an env file written for the pre-CocoIndex-v1 stack (the one with PostgreSQL) to the current schema. See [Production Deployment](../deployment/production.md#migrating-an-existing-envprod).

## Embedding Provider

| Variable | Default | Required | Description |
|-|-|-|-|
| `EMBEDDING_MODEL` | `gemini-2` | No | `jina-v4`, `voyage-4`, or `gemini-2` — determines capabilities and default dimensions |
| `EMBEDDING_ROUTE` | `openrouter` | No | `native` (vendor API) or `openrouter` (gateway). Illegal pairs are rejected at startup |
| `EMBEDDING_DIMENSIONS` | `0` | No | `0` = the model's default; MRL models accept smaller |

Set this to choose the active provider, then configure the corresponding section below. See [Embeddings](../ingestion/embeddings.md) for details on switching providers.

## Jina v4 (when `EMBEDDING_MODEL=jina-v4`)

| Variable | Default | Required | Description |
|-|-|-|-|
| `JINA_API_KEY` | — | If jina | API key for Jina v4 embedding service |
| `JINA_API_URL` | `https://api.jina.ai` | No | Base URL (change for proxy/self-hosted) |
| `JINA_MODEL` | `jina-embeddings-v4` | No | Jina embedding model name |
| `JINA_RPM` | `500` | No | Jina requests per minute (free=500, tier1=500, tier2=5000) |
| `JINA_TPM` | `100000` | No | Jina tokens per minute (free=100k, tier1=10M, tier2=100M) |
| `JINA_BATCH_SIZE` | `10` | No | Max texts per `embed_text` API call (lower to avoid TPM caps on long docs) |
| `JINA_MAX_CONCURRENT` | `5` | No | Max concurrent requests to Jina API |

## Voyage AI (when `EMBEDDING_MODEL=voyage-4`)

| Variable | Default | Required | Description |
|-|-|-|-|
| `VOYAGE_API_KEY` | — | If voyage | API key for Voyage AI |
| `VOYAGE_API_URL` | `https://api.voyageai.com` | No | Base URL |
| `VOYAGE_TEXT_MODEL` | `voyage-4-large` | No | Text embedding model |
| `VOYAGE_MULTIMODAL_MODEL` | `voyage-multimodal-3.5` | No | Image embedding model |
| `VOYAGE_RPM` | `300` | No | Voyage requests per minute |
| `VOYAGE_MAX_CONCURRENT` | `10` | No | Max concurrent requests to Voyage API |

ColBERT multi-vector is not supported by Voyage; `MULTIVEC_ENABLED=true` raises a validation error at startup unless the active model+route pair supports it (only `jina-v4` + `native` does).

## OpenRouter (when `EMBEDDING_ROUTE=openrouter`)

OpenAI-compatible `/v1/embeddings` endpoint, serving `gemini-2` and `voyage-4`; the wire model id comes from the registry, not from a setting. Documents and queries are embedded asymmetrically via `input_type`. **Image embedding is supported for `gemini-2`** — text and images share one vector space, so `IMAGE_EMBED_STRATEGY=smart` works here. ColBERT multi-vector is not available, so `visual_search` still needs `jina-v4` + `native`.

| Variable | Default | Required | Description |
|-|-|-|-|
| `OPENROUTER_API_KEY` | — | If openrouter | API key from <https://openrouter.ai/keys> |
| `OPENROUTER_API_URL` | `https://openrouter.ai/api` | No | Base URL |
| `OPENROUTER_BATCH_SIZE` | `100` | No | Max inputs per call. Gemini hard-rejects more with a non-retryable 400 |
| `OPENROUTER_RPM` | `300` | No | Requests per minute |
| `OPENROUTER_MAX_CONCURRENT` | `10` | No | Max concurrent requests |
| `OPENROUTER_HTTP_REFERER` | — | No | Optional `HTTP-Referer` header for openrouter.ai rankings |
| `OPENROUTER_X_TITLE` | — | No | Optional `X-Title` header for openrouter.ai rankings |

ColBERT multi-vector is not available on this route (gemini-2 emits a single vector); `MULTIVEC_ENABLED=true` raises a validation error at startup, and `visual_search` requires `jina-v4` + `native`. `IMAGE_EMBED_STRATEGY=smart|all` *is* supported here — image points go to `documents_dense` and are retrieved by the ordinary dense channel.

## Qdrant

| Variable | Default | Required | Description |
|-|-|-|-|
| `QDRANT_URL` | `http://localhost:6333` | No | Qdrant server URL |
| `QDRANT_DENSE_COLLECTION` | `documents_dense` | No | Collection name for dense vectors |
| `QDRANT_MULTIVEC_COLLECTION` | `documents_multivec` | No | Collection name for ColBERT multi-vectors |

## Neo4j

| Variable | Default | Required | Description |
|-|-|-|-|
| `NEO4J_URI` | `bolt://localhost:7687` | No | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | No | Neo4j username |
| `NEO4J_PASSWORD` | — | Yes | Neo4j password |

## CocoIndex Pipeline State

CocoIndex keeps its target-state ledger, memoization cache and component tree in a local LMDB directory — no database service is involved.

| Variable | Default | Required | Description |
|-|-|-|-|
| `COCOINDEX_DB_PATH` | `state/cocoindex.db` | No | LMDB state *directory*. Lives under `state/` so the production `ingest_state` volume covers it |
| `COCOINDEX_LMDB_MAP_SIZE` | 4 GiB (`4294967296`) | No | Read by CocoIndex itself, not by `config/settings.py`. Raise if the ledger outgrows the default map size |

## Document Source

The ingestion pipeline reads files from exactly one source. Settings validation rejects `s3` without `S3_BUCKET_NAME`, and rejects `sharepoint` without all `SHAREPOINT_*` required fields populated.

| Variable | Default | Required | Description |
|-|-|-|-|
| `DOCUMENT_SOURCE` | `local` | No | Source selector: `local`, `s3`, or `sharepoint` |
| `LOCAL_DOCUMENTS_PATH` | `documents` | No | Filesystem path for `local` and `sharepoint` modes |

## AWS (when `DOCUMENT_SOURCE=s3`)

Credentials can be set explicitly here or left empty to use the default boto3 credential chain (`~/.aws/credentials`, IAM role, etc.). `AWS_ENDPOINT_URL` is for LocalStack / MinIO / R2; leave empty for real AWS.

| Variable | Default | Required | Description |
|-|-|-|-|
| `S3_BUCKET_NAME` | — | When `DOCUMENT_SOURCE=s3` | S3 bucket containing source documents |
| `S3_PREFIX` | — | No | Optional key prefix restricting the scan to part of the bucket |
| `S3_SQS_QUEUE_URL` | — | No | SQS queue used as a *trigger* for a catch-up scan in live mode. Empty falls back to interval-only sweeps |
| `S3_SQS_DEBOUNCE_SECONDS` | `5` | No | Wait after the first event so a burst coalesces into a single update |
| `S3_FULL_SCAN_INTERVAL_HOURS` | `24` | No | Safety-net sweep interval for missed or expired events (and the only trigger when no queue is configured) |
| `AWS_REGION` | `us-east-1` | No | AWS region for S3 and SQS |
| `AWS_ACCESS_KEY_ID` | — | No | Optional explicit access key (else default credential chain) |
| `AWS_SECRET_ACCESS_KEY` | — | No | Optional explicit secret key |
| `AWS_ENDPOINT_URL` | — | No | S3-compatible endpoint (e.g. `http://localhost:4566` for LocalStack) |

CocoIndex v1's `amazon_s3` connector is scan-only — there is no built-in SQS push trigger. `ingestion/sqs_trigger.py` supplies the trigger externally; see [CocoIndex Pipeline](../ingestion/cocoindex.md#s3-sqs-as-a-trigger).

See [AWS Setup](../deployment/aws-setup.md) for IAM permissions.

## SharePoint (when `DOCUMENT_SOURCE=sharepoint`)

All six required fields below must be populated when `DOCUMENT_SOURCE=sharepoint`. See [SharePoint Setup](../ingestion/sharepoint-setup.md) for the full walkthrough.

| Variable | Default | Required | Description |
|-|-|-|-|
| `SHAREPOINT_TENANT_ID` | — | If sharepoint | Azure AD directory (tenant) id |
| `SHAREPOINT_CLIENT_ID` | — | If sharepoint | Azure AD app (client) id |
| `SHAREPOINT_CLIENT_SECRET` | — | If sharepoint | Client secret value |
| `SHAREPOINT_SITE_ID` | — | If sharepoint | e.g. `contoso.sharepoint.com,abc-1234,def-5678` |
| `SHAREPOINT_DRIVE_ID` | — | If sharepoint | The document library's drive id |
| `SHAREPOINT_ROOT_FOLDER_PATH` | — | If sharepoint | In-scope folder path, e.g. `/Engineering/Specs` |
| `SHAREPOINT_LOCAL_SUBDIR` | `sharepoint` | No | Subdir under `LOCAL_DOCUMENTS_PATH` for synced files |
| `SHAREPOINT_SYNC_INTERVAL_SECONDS` | `180` | No | Polling interval for the sync service |
| `SHAREPOINT_STATE_DIR` | `state/sharepoint` | No | Where the delta token + index live |

## LLM

| Variable | Default | Required | Description |
|-|-|-|-|
| `LLM_API_TYPE` | `anthropic` | No | LLM API type (`anthropic` or `openai`) |
| `LLM_MODEL` | `claude-sonnet-5` | No | Model identifier for entity extraction and generation |
| `LLM_API_KEY` | — | Yes | API key for the configured LLM provider |
| `LLM_BASE_URL` | — | No | Custom base URL for OpenAI-compatible servers (OpenRouter, Ollama, vLLM, LiteLLM) |

When `LLM_BASE_URL` is set, the **agent** uses an OpenAI-compatible client regardless of `LLM_API_TYPE`. The ingestion and VLM components still use `LLM_API_TYPE` to select the Anthropic or OpenAI SDK, but pass `LLM_BASE_URL` to the client constructor.

## MCP Server

| Variable | Default | Required | Description |
|-|-|-|-|
| `MCP_TRANSPORT` | `http` | No | Transport: `http` (streamable-http, recommended), `sse` (legacy), or `stdio` |
| `MCP_HOST` | `0.0.0.0` | No | Bind address. Use `127.0.0.1` for local-only; keep `0.0.0.0` for Docker / LAN / remote clients |
| `MCP_PORT` | `8080` | No | Port for `http`/`sse` transports |
| `MCP_PATH` | `/mcp` | No | URL path for `http`/`sse`. Full endpoint is `{scheme}://{MCP_HOST}:{MCP_PORT}{MCP_PATH}` |
| `MCP_API_KEY` | — | No | Bearer token for MCP server authentication. Empty disables auth |
| `MCP_PUBLIC_DOMAIN` | — | Prod only | Public domain for Traefik routing (e.g. `mcp.example.com`) |

## Live Ingestion (Path B)

`INGEST_API_KEY` enables a two-layer auth scheme: it gates `/session/start`, which then returns an ephemeral per-session token used for `/ingest/chunk` and `/session/end`. Leave empty to disable auth. See [Authentication](../server/authentication.md).

| Variable | Default | Required | Description |
|-|-|-|-|
| `LIVE_INGEST_PORT` | `8001` | No | Port for the live ingestion FastAPI server |
| `INGEST_API_KEY` | — | No | Bearer token gating `/session/start` (two-layer auth) |

## Resilience

| Variable | Default | Required | Description |
|-|-|-|-|
| `PIPELINE_TIMEOUT` | `3600` | No | Per-file processing timeout in seconds (increase for large PDFs with graph ingestion) |
| `PIPELINE_MAX_RETRIES` | `3` | No | Failures per file before the poison-pill swallows the error and lets CocoIndex write the file's memoization entry |
| `PIPELINE_MAX_CONCURRENT_FILES` | `4` | No | Files processed in parallel (CocoIndex's `max_inflight_components`, whose own default of 1024 would blow past embedding rate limits) |
| `GRAPHITI_CONCURRENCY` | `3` | No | Max concurrent Graphiti episode ingestions per page (bounded by LLM rate limits) |
| `EXTRACTION_TIMEOUT` | `30` | No | Timeout in seconds for entity extraction |
| `TOOL_TIMEOUT` | `30` | No | Timeout in seconds for MCP tool execution |
| `MAX_RETRIES` | `3` | No | Max retry attempts for transient failures |

## Image Embedding

| Variable | Default | Required | Description |
|-|-|-|-|
| `IMAGE_EMBED_STRATEGY` | `smart` | No | `smart` (Docling-gated, embeds only pages with figures/tables/formulas) or `all` (embeds every page). Note: `none` is documented in `.env.example` but is not a supported value in `config/settings.py` |
| `IMAGE_EMBED_MAX_PX` | `400` | No | Max pixels on longest side before embedding (reduces token cost) |

## Schema Induction

| Variable | Default | Required | Description |
|-|-|-|-|
| `SCHEMA_INDUCTION_ENABLED` | `true` | No | Enable per-document LLM schema induction for GLiNER2 (Path A only) |
| `SCHEMA_INDUCTION_MODEL` | — | No | Model used for schema proposals. Empty falls back to `LLM_MODEL`; set it to a cheaper/faster model to split the two. Must use the naming of whatever endpoint `LLM_BASE_URL` points at |
| `SCHEMA_CACHE_TTL` | `3600` | No | Seconds to cache induced schemas before re-inducing |

## Feature Flags

| Variable | Default | Required | Description |
|-|-|-|-|
| `RERANK_ENABLED` | `true` | No | Enable cross-encoder reranking (precision win, ~20ms) |
| `VLM_GENERATION_ENABLED` | `false` | No | Enable VLM-based generation for visual search |
| `MULTIVEC_ENABLED` | `false` | No | Enable ColBERT multi-vector embeddings (requires `jina-colbert-v2`, Jina only) |
| `GRAPH_ENABLED` | `true` | No | Enable Neo4j knowledge graph ingestion and search |
| `GRAPH_ENGINE` | `graphiti` | No | Graph extraction engine: `graphiti` (LLM-based) or `gliner` (local CPU model, zero API cost). See [Knowledge Graph](../ingestion/knowledge-graph.md) |
| `GRAPH_SEMAPHORE_LIMIT` | `10` | No | Max concurrent LLM calls within Graphiti's internal pipeline (Graphiti engine only) |
| `GRAPH_EPISODE_TARGET_SIZE` | `1500` | No | Target chars per Graphiti episode (groups small chunks before sending) |

## Observability

| Variable | Default | Required | Description |
|-|-|-|-|
| `LOG_LEVEL` | `INFO` | No | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `LOG_FORMAT` | `json` | No | Log output format (`json` or `text`) |
| `LOGFIRE_TOKEN` | — | No | Logfire Cloud token. Required to ship traces off-host |
| `OBSERVABILITY_LOCAL_ONLY` | `true` | No | When `true`, spans are emitted in-process but not shipped anywhere. Set to `false` (with `LOGFIRE_TOKEN`) to ship to Logfire Cloud |
| `SERVICE_NAME` | `spektr` | No | Service name tag attached to all spans |
