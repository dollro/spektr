# Spektr

RAG-as-MCP-Server with dual-path ingestion. Batch-processes documents from S3 or local filesystem into a dual knowledge store (Qdrant + Neo4j), and accepts streaming text data in real time via HTTP. Exposes session-aware search tools to LLM agents via the MCP protocol.

## Architecture

```
      Path A: Bulk KB                         Path B: Live
  S3 / Local Directory                       HTTP POST
          |                                      |
  CocoIndex Pipeline                         FastAPI
      |            |                         |          |
  Jina/Voyage   GLiNER2 or Graphiti*     Jina/Voyage   Graphiti (always)
   (embed)      (GRAPH_ENGINE setting)    (embed)      (temporal KG)
      |            |                         |          |
      v            v                         v          v
    Qdrant       Neo4j                     Qdrant     Neo4j

  * GRAPH_ENGINE only controls Path A. Path B always uses Graphiti.

                       LLM Agent
                           |
                   FastMCP (streamable-http)
                           |
        +----------+-------+-------+-----------+
        |          |               |            |
  vector_search  graph_search  hybrid_search  visual_search
   (Qdrant)       (Neo4j)      (both)         (ColBERT)
```

## Setup — Local Development

Prerequisites: Python 3.13, [uv](https://docs.astral.sh/uv/), Docker, [go-task](https://taskfile.dev) (optional but recommended).

```bash
# 1. Clone and configure
git clone <repo-url> && cd spektr
cp .env.example .env              # fill in API keys (JINA_API_KEY, LLM_API_KEY, NEO4J_PASSWORD)

# 2. Start infrastructure
task up                            # Qdrant, Neo4j, PostgreSQL via docker compose

# 3. Install dependencies
task setup                         # installs uv + project deps

# 4. Ingest documents
task ingest                        # one-shot bulk ingest from documents/ or S3

# 5. Start MCP server
task serve                         # streamable-http on http://localhost:8080/mcp
```

Connect an MCP client (Claude Code, Claude Desktop, etc.):

```json
{
  "mcpServers": {
    "spektr": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

## Setup — Production (Docker Compose + Traefik)

For a single Linux VM where Traefik already handles TLS and reverse proxying. Everything runs containerized; only the MCP service is exposed via Traefik labels.

Prerequisites: Docker Engine 24+, Docker Compose v2+, an external Traefik instance with a `proxy` network, DNS pointing at the VM.

```bash
# 1. Configure
cp .env.example .env.prod
# Edit .env.prod:
#   - Set container hostnames: QDRANT_URL=http://qdrant:6333, NEO4J_URI=bolt://neo4j:7687, etc.
#   - Set strong passwords: NEO4J_PASSWORD, POSTGRES_PASSWORD
#   - Set API keys: JINA_API_KEY, LLM_API_KEY
#   - Set MCP_API_KEY (generate: python3 -c "import secrets; print(secrets.token_urlsafe(48))")
#   - Set MCP_PUBLIC_DOMAIN=mcp.example.com
#   - See "Production Overrides" section at bottom of .env.example

# 2. Build and start
task prod:build                    # build the app image
task prod:up                       # start all services

# 3. Ingest (first time or on-demand)
task prod:ingest                   # one-shot bulk ingest

# 4. Verify
curl https://mcp.example.com/mcp   # should respond (Traefik routes + auto-TLS)
```

The `ingest-live` service runs as a long-lived daemon that polls S3 via SQS for new files — no cron needed if using S3 as the document source.

Connect an MCP client with bearer auth:

```json
{
  "mcpServers": {
    "spektr": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_API_KEY>"
      }
    }
  }
}
```

Without Traefik, see [Production Deployment docs](docs/deployment/production.md#without-traefik).

## Ingestion Paths

**Path A — Bulk documents** from S3 or local `documents/` directory. CocoIndex manages incremental state. Supports PDF (Docling OCR + layout analysis), images, and text files. With `--live` flag, runs as a daemon polling S3/SQS for changes; without it, runs one-shot and exits.

**Path B — Streaming text** via HTTP POST (`live_ingest.py`, port 8001). Data is vector-indexed in Qdrant immediately (~200ms) and ingested into Graphiti's temporal knowledge graph as a background task (~2-5s). Session lifecycle: start → ingest chunks → end (archive or discard).

## Search Tools

The MCP server exposes four tools. Three support an optional `session_id` for combining live session data with bulk KB results.

| Tool | Backend | `session_id` | Description |
|-|-|-|-|
| `vector_search` | Qdrant dense | Yes | Semantic similarity search |
| `visual_search` | Qdrant ColBERT | No | Layout-aware search for charts, tables, diagrams |
| `graph_search` | Neo4j | Yes | Entity and relationship lookup |
| `hybrid_search` | Both (parallel) | Yes | Combined vector + graph results |

## Configuration

### Required

| Variable | Description |
|-|-|
| `JINA_API_KEY` or `VOYAGE_API_KEY` | Embedding provider API key |
| `NEO4J_PASSWORD` | Neo4j password (must match docker-compose) |
| `LLM_API_KEY` | LLM provider API key (for Graphiti / schema induction) |

### Key Options

| Variable | Default | Description |
|-|-|-|
| `EMBEDDING_PROVIDER` | `jina` | `jina` or `voyage` |
| `MCP_TRANSPORT` | `http` | `http` (streamable-http), `sse` (legacy), `stdio` |
| `MCP_API_KEY` | — | Bearer token for MCP auth (empty = no auth) |
| `MCP_PUBLIC_DOMAIN` | — | Public domain for Traefik routing (prod only) |
| `GRAPH_ENGINE` | `graphiti` | `graphiti` (LLM) or `gliner` (local CPU) |
| `GRAPH_ENABLED` | `true` | Disable Neo4j entirely |
| `DOCUMENT_SOURCE` | `local` | `local` or `s3` |
| `MULTIVEC_ENABLED` | `false` | ColBERT multi-vector embeddings (Jina only) |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking |

Full reference: `.env.example` and [docs/configuration/environment.md](docs/configuration/environment.md)

## Daily Drivers

```bash
task smoke                 # vector_search smoke test (no MCP/LLM needed)
task smoke-graph           # graph_search smoke test
task ask -- "question"     # end-to-end: agent + MCP + LLM (needs task serve)
task doctor                # diff CocoIndex tracking vs Qdrant; flags drift
task backup                # snapshot Qdrant + Neo4j + Postgres
```

## Testing

```bash
task test                  # unit tests
task test-integration      # integration tests (needs Docker services)
task lint                  # Ruff
task typecheck             # mypy
task check                 # lint + typecheck + test
task eval                  # RAGAS retrieval-quality eval
```

## Documentation

```bash
task docs-serve            # MkDocs dev server at http://localhost:8000
```

## Stack

| Component | Technology |
|-|-|
| Language | Python 3.13 |
| Package manager | uv |
| Bulk ingestion | CocoIndex |
| Live ingestion | FastAPI |
| Embeddings | Jina v4 / Voyage AI |
| Vector store | Qdrant |
| Knowledge graph | Neo4j + Graphiti / GLiNER2 |
| State DB | PostgreSQL |
| MCP server | FastMCP (streamable-http) |
| Agent | Pydantic AI |
| Cloud | AWS S3 + SQS |

## License

See LICENSE for details.
