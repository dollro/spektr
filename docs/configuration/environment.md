# Environment Variables

All configuration is loaded from a `.env` file (or real environment variables) via Pydantic Settings in `config/settings.py`. Copy `.env.example` to `.env` and fill in the required values.

```bash
cp .env.example .env
```

## Embedding Provider

| Variable | Default | Required | Description |
|-|-|-|-|
| `EMBEDDING_PROVIDER` | `jina` | No | Active embedding provider: `jina`, `voyage`, or `openrouter` |

Set this to choose the active provider, then configure the corresponding section below. See [Embeddings](../ingestion/embeddings.md) for details on switching providers.

## Jina v4 (when `EMBEDDING_PROVIDER=jina`)

| Variable | Default | Required | Description |
|-|-|-|-|
| `JINA_API_KEY` | — | If jina | API key for Jina v4 embedding service |
| `JINA_API_URL` | `https://api.jina.ai` | No | Base URL (change for proxy/self-hosted) |
| `JINA_MODEL` | `jina-embeddings-v4` | No | Jina embedding model name |
| `JINA_DENSE_DIMENSIONS` | `2048` | No | Dimensionality of dense embeddings (Matryoshka: 128/256/512/1024/2048) |
| `JINA_RPM` | `500` | No | Jina requests per minute (free=500, tier1=500, tier2=5000) |
| `JINA_TPM` | `100000` | No | Jina tokens per minute (free=100k, tier1=10M, tier2=100M) |
| `JINA_BATCH_SIZE` | `10` | No | Max texts per `embed_text` API call (lower to avoid TPM caps on long docs) |
| `JINA_MAX_CONCURRENT` | `5` | No | Max concurrent requests to Jina API |

## Voyage AI (when `EMBEDDING_PROVIDER=voyage`)

| Variable | Default | Required | Description |
|-|-|-|-|
| `VOYAGE_API_KEY` | — | If voyage | API key for Voyage AI |
| `VOYAGE_API_URL` | `https://api.voyageai.com` | No | Base URL |
| `VOYAGE_TEXT_MODEL` | `voyage-4-large` | No | Text embedding model |
| `VOYAGE_MULTIMODAL_MODEL` | `voyage-multimodal-3.5` | No | Image embedding model |
| `VOYAGE_DENSE_DIMENSIONS` | `1024` | No | Output dimensions (256, 512, 1024, 2048) |
| `VOYAGE_RPM` | `300` | No | Voyage requests per minute |
| `VOYAGE_MAX_CONCURRENT` | `10` | No | Max concurrent requests to Voyage API |

ColBERT multi-vector embeddings are not supported by Voyage; setting `MULTIVEC_ENABLED=true` with `EMBEDDING_PROVIDER=voyage` raises a validation error at startup.

## OpenRouter (when `EMBEDDING_PROVIDER=openrouter`)

OpenAI-compatible `/v1/embeddings` endpoint. The default targets Google Gemini Embedding 2 (text-only, 3072d, MRL truncation). Any OpenRouter-served embedding model can be selected. **Text-only**: image embedding raises `NotImplementedError`, so use `jina` or `voyage` if your pipeline needs PDF page or image embeddings.

| Variable | Default | Required | Description |
|-|-|-|-|
| `OPENROUTER_API_KEY` | — | If openrouter | API key from <https://openrouter.ai/keys> |
| `OPENROUTER_API_URL` | `https://openrouter.ai/api` | No | Base URL |
| `OPENROUTER_MODEL` | `google/gemini-embedding-2-preview` | No | OpenRouter embedding model id |
| `OPENROUTER_DENSE_DIMENSIONS` | `3072` | No | Output dimensions (Gemini Embedding 2 supports MRL: 768/1536/3072) |
| `OPENROUTER_RPM` | `300` | No | Requests per minute |
| `OPENROUTER_MAX_CONCURRENT` | `10` | No | Max concurrent requests |
| `OPENROUTER_HTTP_REFERER` | — | No | Optional `HTTP-Referer` header for openrouter.ai rankings |
| `OPENROUTER_X_TITLE` | — | No | Optional `X-Title` header for openrouter.ai rankings |

ColBERT multi-vector embeddings are not supported by OpenRouter; setting `MULTIVEC_ENABLED=true` with `EMBEDDING_PROVIDER=openrouter` raises a validation error at startup.

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

## PostgreSQL

| Variable | Default | Required | Description |
|-|-|-|-|
| `DATABASE_URL` | `postgresql://cocoindex:cocoindex@localhost:5432/cocoindex` | No | PostgreSQL connection URL for CocoIndex pipeline state |

## Document Source

The ingestion pipeline reads files from exactly one source. Settings validation rejects `s3` without `S3_BUCKET_NAME`+`S3_SQS_QUEUE_URL`, and rejects `sharepoint` without all `SHAREPOINT_*` required fields populated.

| Variable | Default | Required | Description |
|-|-|-|-|
| `DOCUMENT_SOURCE` | `local` | No | Source selector: `local`, `s3`, or `sharepoint` |
| `LOCAL_DOCUMENTS_PATH` | `documents` | No | Filesystem path for `local` and `sharepoint` modes |

## AWS (when `DOCUMENT_SOURCE=s3`)

Credentials can be set explicitly here or left empty to use the default boto3 credential chain (`~/.aws/credentials`, IAM role, etc.). `AWS_ENDPOINT_URL` is for LocalStack / MinIO / R2; leave empty for real AWS.

| Variable | Default | Required | Description |
|-|-|-|-|
| `S3_BUCKET_NAME` | — | When `DOCUMENT_SOURCE=s3` | S3 bucket containing source documents |
| `S3_SQS_QUEUE_URL` | — | When `DOCUMENT_SOURCE=s3` | SQS queue URL for S3 event notifications |
| `AWS_REGION` | `us-east-1` | No | AWS region for S3 and SQS |
| `AWS_ACCESS_KEY_ID` | — | No | Optional explicit access key (else default credential chain) |
| `AWS_SECRET_ACCESS_KEY` | — | No | Optional explicit secret key |
| `AWS_ENDPOINT_URL` | — | No | S3-compatible endpoint (e.g. `http://localhost:4566` for LocalStack) |

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
| `LLM_MODEL` | `claude-sonnet-4-20250514` | No | Model identifier for entity extraction and generation |
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

`INGEST_API_KEY` enables a two-layer auth scheme: it gates `/session/start`, which then returns an ephemeral per-session token used for `/ingest/transcript` and `/session/end`. Leave empty to disable auth. See [Authentication](../server/authentication.md).

| Variable | Default | Required | Description |
|-|-|-|-|
| `LIVE_INGEST_PORT` | `8001` | No | Port for the live ingestion FastAPI server |
| `INGEST_API_KEY` | — | No | Bearer token gating `/session/start` (two-layer auth) |

## Resilience

| Variable | Default | Required | Description |
|-|-|-|-|
| `PIPELINE_TIMEOUT` | `3600` | No | Per-file processing timeout in seconds (increase for large PDFs with graph ingestion) |
| `PIPELINE_MAX_RETRIES` | `3` | No | Failures per file before the poison-pill swallows the error and lets CocoIndex mark it processed |
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
| `SCHEMA_INDUCTION_MODEL` | `claude-haiku-4-5-20251001` | No | Fast/cheap model used for schema proposals |
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
