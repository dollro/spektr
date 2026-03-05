# Spektr

**RAG-as-MCP-Server** -- automatically syncs documents from AWS S3 into a dual knowledge store (Qdrant vector DB + Neo4j temporal knowledge graph via Graphiti) and exposes search tools to LLM agents through the [Model Context Protocol](https://modelcontextprotocol.io/). No human-facing UI; primary consumers are Pydantic AI agents, Claude Code, and custom LLM frameworks.

## Features

- **Dual retrieval** -- vector similarity (Qdrant) and temporal knowledge graph (Neo4j/Graphiti) in a single server
- **Multimodal embeddings** -- Jina v4 produces dense 2048-d vectors for text and images, plus ColBERT 128-d multi-vectors for layout-aware visual search
- **Automatic sync** -- CocoIndex pipeline watches S3 via SQS event notifications; new, updated, and deleted files are processed incrementally
- **Four MCP search tools** -- `vector_search`, `visual_search`, `graph_search`, `hybrid_search`
- **Bearer auth middleware** -- optional token-based protection on `tools/call` requests
- **Temporal awareness** -- Graphiti tracks when facts were created and expired, so agents can reason about time

## Architecture

```mermaid
graph LR
    S3[AWS S3] -->|SQS events| Pipeline[CocoIndex Pipeline]
    Pipeline -->|dense + ColBERT vectors| Qdrant[(Qdrant)]
    Pipeline -->|episodes / entities| Neo4j[(Neo4j + Graphiti)]
    Pipeline -->|pipeline state| PG[(PostgreSQL)]
    Qdrant --- MCP[FastMCP Server]
    Neo4j --- MCP
    MCP -->|MCP protocol| Agent[LLM Agents]
```

## Quick start

```bash
# Infrastructure
docker compose up -d
./scripts/wait-for-services.sh

# Copy and fill environment variables
cp .env.example .env

# Ingest documents
uv run python -m ingestion.pipeline

# Start the MCP server
uv run python -m server.mcp_server
```

See [Local Development](deployment/local-development.md) for the full setup guide.

## Documentation map

| Section | Description |
|-|-|
| [Architecture Overview](architecture/overview.md) | System diagram, component roles, technology rationale |
| [Data Flow](architecture/data-flow.md) | Ingest, query, and delete paths with sequence diagrams |
| [Ingestion Pipeline](ingestion/overview.md) | CocoIndex, Jina v4, Graphiti integration |
| [MCP Server](server/overview.md) | Tool registration, transport, authentication |
| [Search Tools](server/search-tools.md) | Per-tool reference for all four search endpoints |
| [Agent](agent/overview.md) | Pydantic AI agent and HTTP API |
| [Configuration](configuration/environment.md) | Environment variables and infrastructure setup |
| [AWS Setup](deployment/aws-setup.md) | S3 event notifications, SQS, IAM |

## Stack

| Component | Technology |
|-|-|
| Language | Python 3.13 |
| Package manager | uv |
| Ingestion pipeline | CocoIndex |
| Embeddings | Jina v4 API (dense 2048-d + ColBERT 128-d) |
| Vector store | Qdrant v1.13 |
| Knowledge graph | Neo4j 5.26 + Graphiti |
| State DB | PostgreSQL 17.2 |
| MCP server | FastMCP (SSE + stdio) |
| Agent framework | Pydantic AI |
| Cloud | AWS S3 + SQS |
