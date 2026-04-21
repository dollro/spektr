# Environment Variables

All configuration is loaded from a `.env` file (or real environment variables) via Pydantic Settings in `config/settings.py`. Copy `.env.example` to `.env` and fill in the required values.

```bash
cp .env.example .env
```

## Embedding Provider

| Variable | Default | Required | Description |
|-|-|-|-|
| `EMBEDDING_PROVIDER` | `jina` | No | Which embedding provider to use (`jina` or `voyage`) |

Set this to choose the active provider, then configure the corresponding section below. See [Embeddings](../ingestion/embeddings.md) for details on switching providers.

## Jina v4 (when `EMBEDDING_PROVIDER=jina`)

| Variable | Default | Required | Description |
|-|-|-|-|
| `JINA_API_KEY` | — | If jina | API key for Jina v4 embedding service |
| `JINA_API_URL` | `https://api.jina.ai` | No | Base URL (change for proxy/self-hosted) |
| `JINA_MODEL` | `jina-embeddings-v4` | No | Jina embedding model name |
| `JINA_DENSE_DIMENSIONS` | `512` | No | Dimensionality of dense embeddings (Matryoshka: 128/256/512/1024/2048) |

## Voyage AI (when `EMBEDDING_PROVIDER=voyage`)

| Variable | Default | Required | Description |
|-|-|-|-|
| `VOYAGE_API_KEY` | — | If voyage | API key for Voyage AI |
| `VOYAGE_API_URL` | `https://api.voyageai.com` | No | Base URL |
| `VOYAGE_TEXT_MODEL` | `voyage-4-large` | No | Text embedding model |
| `VOYAGE_MULTIMODAL_MODEL` | `voyage-multimodal-3.5` | No | Image embedding model |
| `VOYAGE_DENSE_DIMENSIONS` | `1024` | No | Output dimensions (256, 512, 1024, 2048) |

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

## AWS

| Variable | Default | Required | Description |
|-|-|-|-|
| `S3_BUCKET_NAME` | — | Yes | S3 bucket containing source documents |
| `S3_SQS_QUEUE_URL` | — | Yes | SQS queue URL for S3 event notifications |
| `AWS_REGION` | `us-east-1` | No | AWS region for S3 and SQS |

AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) are loaded from the standard AWS credential chain. See [AWS Setup](../deployment/aws-setup.md) for IAM permissions.

## LLM

| Variable | Default | Required | Description |
|-|-|-|-|
| `LLM_API_TYPE` | `anthropic` | No | LLM API type (`anthropic`, `openai`) |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | No | Model identifier for entity extraction and generation |
| `LLM_API_KEY` | — | Yes | API key for the configured LLM provider |
| `LLM_BASE_URL` | — | No | Custom base URL for OpenAI-compatible servers (OpenRouter, Ollama, vLLM, LiteLLM) |

When `LLM_BASE_URL` is set, the **agent** uses an OpenAI-compatible client regardless of `LLM_API_TYPE`. The ingestion and VLM components still use `LLM_API_TYPE` to select the Anthropic or OpenAI SDK, but pass `LLM_BASE_URL` to the client constructor.

## MCP

| Variable | Default | Required | Description |
|-|-|-|-|
| `MCP_TRANSPORT` | `sse` | No | MCP transport protocol (`sse` or `stdio`) |
| `MCP_HOST` | `0.0.0.0` | No | Bind address for the SSE server. Use `127.0.0.1` for local-only; keep `0.0.0.0` for Docker / LAN / remote clients |
| `MCP_PORT` | `8080` | No | Port for the MCP SSE server |

## Auth

| Variable | Default | Required | Description |
|-|-|-|-|
| `MCP_API_KEY` | — | Yes | Bearer token for MCP server authentication |

## Resilience

| Variable | Default | Required | Description |
|-|-|-|-|
| `PIPELINE_TIMEOUT` | `3600` | No | Per-file processing timeout in seconds (increase for large PDFs with graph ingestion) |
| `GRAPHITI_CONCURRENCY` | `3` | No | Max concurrent Graphiti episode ingestions per page (bounded by LLM rate limits) |
| `JINA_MAX_CONCURRENT` | `5` | No | Max concurrent requests to Jina API |
| `JINA_RPM` | `500` | No | Jina requests per minute limit |
| `VOYAGE_MAX_CONCURRENT` | `10` | No | Max concurrent requests to Voyage API |
| `VOYAGE_RPM` | `300` | No | Voyage requests per minute limit |
| `EXTRACTION_TIMEOUT` | `30` | No | Timeout in seconds for entity extraction |
| `TOOL_TIMEOUT` | `30` | No | Timeout in seconds for MCP tool execution |
| `MAX_RETRIES` | `3` | No | Max retry attempts for transient failures |

## Image Embedding

| Variable | Default | Required | Description |
|-|-|-|-|
| `IMAGE_EMBED_MAX_PX` | `400` | No | Max pixels on longest side before embedding (reduces token cost) |
| `IMAGE_EMBED_STRATEGY` | `smart` | No | `smart` (Docling-gated), `all` (every page), `none` (skip images) |

## Live Ingestion

| Variable | Default | Required | Description |
|-|-|-|-|
| `LIVE_INGEST_PORT` | `8001` | No | Port for the live ingestion FastAPI server |
| `INGEST_API_KEY` | `""` | No | Bearer token for `/session/start`. When set, enables per-session token auth on all live ingest endpoints. See [Authentication](../server/authentication.md). |

## Schema Induction

| Variable | Default | Required | Description |
|-|-|-|-|
| `SCHEMA_INDUCTION_ENABLED` | `true` | No | Enable per-document LLM schema induction for GLiNER2 (Path A only) |
| `SCHEMA_INDUCTION_MODEL` | `claude-haiku-4-5-20251001` | No | Fast/cheap model used for schema proposals |
| `SCHEMA_CACHE_TTL` | `3600` | No | Seconds to cache induced schemas before re-inducing |

## Feature Flags

| Variable | Default | Required | Description |
|-|-|-|-|
| `RERANK_ENABLED` | `false` | No | Enable Jina reranker for search results |
| `VLM_GENERATION_ENABLED` | `false` | No | Enable VLM-based generation for visual search |
| `MULTIVEC_ENABLED` | `false` | No | Enable ColBERT multi-vector embeddings (requires `jina-colbert-v2`, Jina only) |
| `GRAPH_ENABLED` | `true` | No | Enable Neo4j knowledge graph ingestion and search |
| `GRAPH_ENGINE` | `graphiti` | No | Graph extraction engine: `graphiti` (LLM-based) or `gliner` (local CPU model, zero API cost). See [Knowledge Graph](../ingestion/knowledge-graph.md) |
| `GRAPH_SEMAPHORE_LIMIT` | `10` | No | Max concurrent LLM calls within Graphiti's internal pipeline (Graphiti engine only) |

## Observability

| Variable | Default | Required | Description |
|-|-|-|-|
| `LOG_LEVEL` | `INFO` | No | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `json` | No | Log output format (`json` or `text`) |
