# Spektr

RAG-as-MCP-Server. Automatically syncs documents from AWS S3 (or local dir) into a dual knowledge store — Qdrant vector DB + Neo4j temporal knowledge graph via Graphiti — and exposes search tools to LLM agents via the MCP protocol. Primary consumers are LLM agents (Pydantic AI, Claude Code, custom frameworks). No human-facing UI.

## Architecture

```
S3 / Local ──► CocoIndex Pipeline
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼              ▼
 Jina / Voyage   Graphiti       PostgreSQL
 (embeddings)   (KG extraction)  (state)
      │             │
      ▼             ▼
   Qdrant         Neo4j
  (vectors)    (knowledge graph)
      │             │
      └──────┬──────┘
             ▼
      FastMCP Server ◄── Bearer Auth
             │
      LLM Agent (MCP)
```

- **Pipeline:** CocoIndex (S3/SQS source or local dir → classify → chunk → embed → store)
- **Embeddings:** Provider-agnostic. Jina v4 (default): dense 2048d. ColBERT 128d multi-vectors available but disabled by default (`MULTIVEC_ENABLED=true` to enable). Voyage AI: dense 1024d.
- **Vector Store:** Qdrant (`documents_dense`, `documents_multivec`)
- **Knowledge Graph:** Neo4j 5 + Graphiti (temporal entity/relationship tracking). Graphiti uses a Jina embedder adapter and an OpenAI-compatible LLM (via OpenRouter or direct) for entity extraction.
- **State Tracking:** PostgreSQL 17 (CocoIndex pipeline state)
- **MCP Server:** FastMCP (SSE + stdio transport)
- **Agent:** Pydantic AI with MCP tool bindings
- **Cloud:** AWS S3 + SQS

## Search Tools

| Tool | Purpose |
|-|-|
| `vector_search` | Semantic similarity search (Qdrant dense vectors) |
| `visual_search` | Visual/layout search (Qdrant ColBERT multi-vectors, requires `MULTIVEC_ENABLED=true`) |
| `graph_search` | Entity and relationship search (Graphiti temporal KG) |
| `hybrid_search` | Parallel vector + graph fusion |

## Quick Start

```bash
# 1. Infrastructure
docker compose up -d
./scripts/wait-for-services.sh

# 2. Configure
cp .env.example .env
# Edit .env: set JINA_API_KEY, NEO4J_PASSWORD, LLM_API_KEY
# Optional: set EMBEDDING_PROVIDER=voyage, LLM_BASE_URL for OpenRouter

# 3. Install
uv sync

# 4. Run
uv run python -m ingestion.pipeline     # Ingest documents
uv run python -m server.mcp_server      # Start MCP server
uv run python -m agent.api              # Agent HTTP endpoint (optional)
```

## Document Processing Pipeline

- **PDF pages** are rasterized to 150 DPI PNG and embedded as images. If OCR text is available, chunks are also embedded as text.
- **Text tasks run concurrently**, **image tasks run sequentially** to avoid exceeding Jina's TPM limit (image embeddings consume 50-100k+ tokens each).
- Text chunks are ingested into Neo4j via Graphiti for entity/relationship extraction.
- **TPM-aware rate limiting:** TokenBucket with variable token counts estimates payload tokens before each request.
- **Retry** on HTTP 429/5xx and transient network errors (ReadError, ConnectError).

## Embedding Providers

| Feature | Jina v4 | Voyage AI |
|-|-|-|
| Text | Yes | Yes |
| Images | Yes | Yes |
| ColBERT multi-vector | Yes | No |
| Default dimensions | 2048 | 1024 |

Switching providers requires re-ingesting all documents.

## Connecting an Agent

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

agent = Agent(
    "openai:gpt-4o",
    mcp_servers=[MCPServerSSE(url="http://localhost:8000/sse")],
)
```

Or use the built-in agent HTTP endpoint:

```bash
uv run python -m agent.api   # POST /query {"query": "..."}
```

## Stack

| Component | Technology |
|-|-|
| Language | Python 3.13 |
| Package Manager | uv |
| Pipeline | CocoIndex |
| Embeddings | Jina v4 / Voyage AI |
| Vector Store | Qdrant |
| Knowledge Graph | Neo4j 5 + Graphiti |
| State DB | PostgreSQL 17 |
| MCP Server | FastMCP |
| Agent | Pydantic AI |
| Cloud | AWS S3 + SQS |

## Authentication

Set `MCP_API_KEY` in `.env` to enable Bearer token authentication on the MCP server. Leave empty to disable.

## Testing

```bash
uv run pytest                   # Unit tests (integration skipped by default)
uv run pytest -m integration    # Integration tests (requires Docker services)
```

## Documentation

```bash
make docs-serve                 # or: uv run mkdocs serve
```

## License

See LICENSE for details.
