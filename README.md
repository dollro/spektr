# Spektr

RAG-as-MCP-Server with dual-path ingestion. Batch-processes documents from S3 or local filesystem into a dual knowledge store (Qdrant + Neo4j), and accepts streaming text data in real time via HTTP. Exposes session-aware search tools to LLM agents via the MCP protocol.

**Ingestion**
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
```

**Search**
```
                       LLM Agent
                           |
                   FastMCP (SSE/stdio)
                           |
        +----------+-------+-------+-----------+
        |          |               |            |
  vector_search  graph_search  hybrid_search  visual_search
   (Qdrant)       (Neo4j)      (both)         (ColBERT)
        |          |               |            |
        v          v               v            v
      Qdrant     Neo4j       Qdrant+Neo4j    Qdrant
                              (parallel)

  All tools except visual_search support session_id
  for combining bulk KB with live session data.
```

## Quick Start

```bash
docker compose up -d              # Qdrant, Neo4j, PostgreSQL
cp .env.example .env              # Configure (see below)
uv sync                           # Install dependencies

uv run python -m ingestion.pipeline    # Ingest documents (bulk)
uv run python -m server.mcp_server     # MCP server (port 8000)
```

**Optional services:**

```bash
uv run uvicorn ingestion.live_ingest:app --port 8001   # Live ingestion
uv run python -m agent.api                              # Agent HTTP API
```

## Usage Options

### Ingestion: Two Independent Paths

**Path A — Bulk documents** from S3 or local `documents/` directory. CocoIndex manages incremental state. Supports PDF (Docling OCR + layout analysis), images, and text files. Entity extraction via GLiNER2 (local CPU, zero API cost) with optional per-document schema induction.

```bash
uv run python -m ingestion.pipeline
```

**Path B — Streaming text** via HTTP POST. Data is vector-indexed in Qdrant immediately (~200ms) and ingested into Graphiti's temporal knowledge graph as a background task (~2-5s). Session lifecycle: start → ingest chunks → end (archive or discard).

```bash
# Start the live ingestion server
uv run uvicorn ingestion.live_ingest:app --port 8001

# Start a session
curl -X POST http://localhost:8001/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-001", "metadata": {"title": "Example"}}'

# Ingest a text chunk
curl -X POST http://localhost:8001/ingest/transcript \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-001", "text": "Discussion about Q1 results...", "timestamp": "2026-03-07T10:00:00Z", "speaker": "Alice"}'

# End session (archive=true keeps data, false purges it)
curl -X POST http://localhost:8001/session/end \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-001", "archive": true}'
```

### Graph Engine: Two Modes

Set `GRAPH_ENGINE` in `.env`:

| Engine | Setting | Use case | Speed | Cost |
|-|-|-|-|-|
| GLiNER2 | `gliner` | Bulk documents (Path A) | ~15 sec / doc | $0.00 |
| Graphiti | `graphiti` | Bulk documents (Path A, default) | ~29 min / doc | ~$0.001 / chunk |

GLiNER2 runs a 205MB local model — no API calls, deterministic. Graphiti uses an LLM for temporal entity extraction with fact evolution tracking (bi-temporal model: `created_at` / `expired_at`).

> **Note:** `GRAPH_ENGINE` only controls the **bulk ingestion** path (Path A). Live streaming (Path B) **always uses Graphiti** regardless of this setting — it requires a working LLM API key and Graphiti service even when `GRAPH_ENGINE=gliner`.

**Dynamic schema induction** (GLiNER2 only): when `SCHEMA_INDUCTION_ENABLED=true`, a cheap LLM call per document proposes domain-specific entity types, improving extraction quality for specialized content (legal, financial, medical, etc.). Results are cached.

### Embedding Provider: Switchable

Set `EMBEDDING_PROVIDER` in `.env`:

| Feature | Jina v4 (`jina`) | Voyage AI (`voyage`) |
|-|-|-|
| Text + image embedding | Yes | Yes |
| ColBERT multi-vector | Yes | No |
| Default dimensions | 512 (Matryoshka) | 1024 |

Switching providers requires re-ingesting all documents.

### Search Tools: Four + Session Awareness

The MCP server exposes four tools. Three support an optional `session_id` for combining live session data with bulk KB results.

| Tool | Backend | `session_id` | Description |
|-|-|-|-|
| `vector_search` | Qdrant dense | Yes | Semantic similarity search |
| `visual_search` | Qdrant ColBERT | No | Layout-aware search for charts, tables, diagrams |
| `graph_search` | Neo4j | Yes | Entity and relationship lookup |
| `hybrid_search` | Both (parallel) | Yes | Combined vector + graph results |

When `session_id` is provided, `hybrid_search` returns three result sets: `vector_results` (bulk KB), `transcript_results` (session data, chronological), and `graph_results` (combined from both engines).

### MCP Transport: SSE or stdio

| Variable | Default | Description |
|-|-|-|
| `MCP_TRANSPORT` | `sse` | `sse` for network clients, `stdio` for subprocess (Claude Code) |
| `MCP_PORT` | `8000` | Port for SSE transport |

### Agent Integration

Connect any MCP-compatible agent:

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

## Configuration

### Required

| Variable | Description |
|-|-|
| `JINA_API_KEY` or `VOYAGE_API_KEY` | Embedding provider API key |
| `NEO4J_PASSWORD` | Neo4j password (must match docker-compose.yml) |
| `LLM_API_KEY` | LLM provider API key (for Graphiti / schema induction) |
| `MCP_API_KEY` | Bearer token for MCP server auth (leave empty to disable) |

### Key Options

| Variable | Default | Description |
|-|-|-|
| `EMBEDDING_PROVIDER` | `jina` | `jina` or `voyage` |
| `GRAPH_ENGINE` | `graphiti` | `graphiti` or `gliner` |
| `GRAPH_ENABLED` | `true` | Disable Neo4j entirely |
| `SCHEMA_INDUCTION_ENABLED` | `true` | Per-document LLM schema induction (GLiNER2 only) |
| `SCHEMA_INDUCTION_MODEL` | `claude-haiku-4-5-20251001` | Model for schema proposals |
| `LIVE_INGEST_PORT` | `8001` | Live ingestion server port |
| `MULTIVEC_ENABLED` | `false` | ColBERT multi-vector embeddings (Jina only) |
| `RERANK_ENABLED` | `false` | Reranker for search results |
| `VLM_GENERATION_ENABLED` | `false` | VLM answers for visual search |
| `IMAGE_EMBED_STRATEGY` | `smart` | `smart` (Docling-gated), `all`, `none` |
| `LLM_BASE_URL` | — | Custom OpenAI-compatible endpoint (OpenRouter, Ollama, etc.) |

Full reference: [Environment Variables](docs/configuration/environment.md)

## Testing

```bash
uv run pytest                   # Unit tests
uv run pytest -m integration    # Integration tests (needs Docker services)
uv run ruff check . && uv run ruff format --check .   # Lint
uv run mypy .                   # Type check
```

## Documentation

```bash
make docs-serve                 # MkDocs dev server
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
| MCP server | FastMCP (SSE + stdio) |
| Agent | Pydantic AI |
| Cloud | AWS S3 + SQS |

## License

See LICENSE for details.
