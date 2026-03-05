# Spektr

RAG-as-MCP-Server — automatically syncs documents from AWS S3 into a dual knowledge store (Qdrant + Neo4j/Graphiti) and exposes search tools to LLM agents via MCP.

## What It Does

1. **Ingests** documents from S3 (PDF, images, text, markdown) via CocoIndex pipeline
2. **Embeds** content using Jina v4 (dense 2048d + ColBERT 128d multi-vectors)
3. **Stores** vectors in Qdrant and builds a temporal knowledge graph in Neo4j via Graphiti
4. **Serves** four search tools over MCP for any LLM agent to use

## Search Tools

| Tool | Purpose |
|-|-|
| `vector_search` | Semantic similarity search (Qdrant dense vectors) |
| `visual_search` | Visual/layout search (Qdrant ColBERT multi-vectors) |
| `graph_search` | Entity and relationship search (Graphiti temporal KG) |
| `hybrid_search` | Parallel vector + graph fusion |

## Quick Start

```bash
# 1. Infrastructure
docker compose up -d
./scripts/wait-for-services.sh

# 2. Configure
cp .env.example .env
# Edit .env with your API keys (Jina, LLM provider, Neo4j password)

# 3. Install
uv sync

# 4. Run
uv run python -m ingestion.pipeline     # Ingest documents
uv run python -m server.mcp_server      # Start MCP server
```

## Connecting an Agent

Any MCP-compatible agent can connect. With Pydantic AI:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE("http://localhost:8000/sse")
agent = Agent("anthropic:claude-sonnet-4-20250514", toolsets=[server])

result = await agent.run("What documents mention machine learning?")
```

An optional HTTP endpoint wraps the agent for non-MCP clients:

```bash
uv run python -m agent.api              # POST /query
```

## Architecture

```
S3 Bucket ──► SQS Queue ──► CocoIndex Pipeline
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
               Jina v4 API   Graphiti       PostgreSQL
               (embeddings)  (KG extraction) (pipeline state)
                    │             │
                    ▼             ▼
                 Qdrant        Neo4j
              (vectors)    (knowledge graph)
                    │             │
                    └──────┬──────┘
                           ▼
                    FastMCP Server ◄── Bearer Auth
                           │
                    LLM Agent (MCP client)
```

## Stack

| Component | Technology |
|-|-|
| Language | Python 3.13 |
| Package Manager | uv |
| Pipeline | CocoIndex |
| Embeddings | Jina v4 API |
| Vector Store | Qdrant |
| Knowledge Graph | Neo4j 5 + Graphiti |
| MCP Server | FastMCP |
| Agent | Pydantic AI |
| Cloud | AWS S3 + SQS |

## Authentication

Set `MCP_API_KEY` in `.env` to enable Bearer token auth on the MCP server. Clients must include `Authorization: Bearer <key>` in requests. Leave empty to disable auth.

## Testing

```bash
uv run pytest                           # All tests
uv run pytest -m "not integration"      # Unit tests only (no Docker needed)
uv run pytest -m integration            # Integration tests (requires Docker services)
```

## Documentation

```bash
uv run mkdocs serve                     # Serve docs locally at http://127.0.0.1:8000
```

## License

See [LICENSE](LICENSE) for details.
