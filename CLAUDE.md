# CLAUDE.md — Spektr

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Spektr** — a RAG-as-MCP-Server that automatically syncs documents from AWS S3 into a dual knowledge store (Qdrant vector DB + Neo4j temporal knowledge graph via Graphiti) and exposes search tools to LLM agents via the MCP protocol.

Primary consumers are LLM agents (Pydantic AI, Claude Code, custom frameworks). No human-facing UI.

## Architecture

- **Pipeline:** CocoIndex (S3/SQS source → classify → chunk → embed → store)
- **Embeddings:** Jina v4 API (dense 2048d single-vector + ColBERT 128d multi-vector)
- **Vector Store:** Qdrant (two collections: `documents_dense`, `documents_multivec`)
- **Knowledge Graph:** Neo4j 5 + Graphiti (temporal entity/relationship tracking)
- **State Tracking:** PostgreSQL 17 (CocoIndex pipeline state)
- **MCP Server:** FastMCP (SSE + stdio transport)
- **Agent:** Pydantic AI with MCP tool bindings
- **Cloud:** AWS S3 + SQS (document source + event notifications)

## Project Layout

```
├── ingestion/                   # Document processing pipeline
│   ├── embedder.py              # Jina v4 API wrapper (JinaV4Embedder)
│   ├── file_processor.py        # MIME classify, PDF-to-images, chunking
│   ├── entity_extractor.py      # LLM-based entity + relationship extraction
│   ├── graph_writer.py          # Neo4j upserts (GraphWriter)
│   ├── jina_cocoindex_ops.py    # Custom CocoIndex ops wrapping Jina v4
│   ├── pipeline.py              # CocoIndex pipeline definition
│   ├── qdrant_setup.py          # Qdrant collection provisioning
│   └── neo4j_setup.py           # Neo4j schema provisioning
├── server/                      # MCP server
│   ├── mcp_server.py            # FastMCP server setup + tool registration
│   ├── models.py                # Pydantic response models
│   ├── providers.py             # LLM provider abstraction
│   └── tools/                   # MCP tool implementations
│       ├── vector_search.py     # Dense vector search (Qdrant)
│       ├── visual_search.py     # ColBERT multi-vector search (Qdrant)
│       ├── graph_search.py      # Knowledge graph search (Neo4j)
│       └── hybrid_search.py     # Parallel vector + graph fusion
├── agent/                       # Pydantic AI agent
│   ├── agent.py                 # Agent with MCP tool bindings
│   └── api.py                   # Optional FastAPI HTTP endpoint
├── config/                      # Configuration
│   ├── settings.py              # Pydantic Settings (.env loading)
│   └── constants.py             # Collection names, dimensions, entity types
├── tests/                       # Test suite
│   ├── conftest.py              # Shared fixtures
│   ├── fixtures/                # Sample files (PDF, PNG, TXT)
│   └── test_*.py                # Unit + integration tests
├── scripts/                     # Helper scripts
│   └── wait-for-services.sh     # Health check polling for Docker services
├── docs/                        # Documentation (MkDocs)
├── docker-compose.yml           # Neo4j, Qdrant, PostgreSQL
├── pyproject.toml               # Python deps + tool config (single source of truth)
├── .env.example                 # All required environment variables
└── .gitignore
```

## Quick Start

### Infrastructure

```bash
docker compose up -d                    # Start Neo4j, Qdrant, PostgreSQL
./scripts/wait-for-services.sh          # Wait for all services to be healthy
```

### Running

```bash
uv run python -m ingestion.pipeline     # Run ingestion pipeline
uv run python -m server.mcp_server      # Start MCP server
uv run python -m agent.api              # Start agent HTTP endpoint (optional)
```

### Testing

```bash
uv run pytest                           # All tests
uv run pytest -m "not integration"      # Unit tests only
uv run pytest -m integration            # Integration tests (requires Docker services)
```

### Documentation

```bash
make docs-serve                         # Serve MkDocs locally
```

## Stack

| Component | Technology |
|-|-|
| Language | Python 3.13 |
| Package Manager | uv (via pyproject.toml) |
| Pipeline | CocoIndex |
| Embeddings | Jina v4 API |
| Vector Store | Qdrant |
| Knowledge Graph | Neo4j 5 + Graphiti |
| State DB | PostgreSQL 17 |
| MCP Server | FastMCP |
| Agent | Pydantic AI |
| Cloud | AWS S3 + SQS |
| Testing | pytest + pytest-asyncio |
| Linting | Ruff + mypy |
| Infrastructure | Docker Compose |

## Dependencies

- Python dependencies managed via `pyproject.toml` (single source of truth)
- Always use `uv` for package management, never pip directly
- Add new deps to the appropriate group: `dependencies` (base), `dev`, `test`
- Infrastructure containers pinned: `qdrant/qdrant:v1.13.2`, `neo4j:5.26-community`, `postgres:17.2`

## Code Organization Rules

- **Max file size**: 600 lines. Refactor into modules if exceeded.
- **Max function length**: 60 lines. Extract helpers.
- **Max class length**: 100 lines. Single concept per class.
- **Single responsibility**: Each module/file does ONE thing.
- **Prefer composition**: Break complex logic into small, composable functions.

## Code Style

- **Python:** Ruff formatting + linting, mypy — configured in `pyproject.toml`
- **Line length:** 95 characters

## Core Principles

- **KISS**: Straightforward solutions over complex ones
- **YAGNI**: Implement only what's needed
- **Single Responsibility**: One clear purpose per function/class
- **Fail Fast**: Raise exceptions immediately on errors

## Environment

- **Env file:** `.env` (gitignored), documented in `.env.example`
- **Key variables:** Jina API key, Neo4j credentials, Qdrant URL, PostgreSQL URL, AWS credentials, LLM provider config, MCP transport/port

## Git Workflow

Branches: `main` (production), `develop` (ongoing development), `feat/*`, `fix/*`, `chore/*`

Merge flow: `feat/xyz` → `develop` → `main`

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore
```

**Never include "claude code" or "written by claude" in commit messages.**

## Planning

All plan files go into `.claude/plans/` in branch-specific subdirectories. The subdirectory name is the branch name with `/` replaced by `-`.

Example: on branch `feat/ingestion` → plans go in `.claude/plans/feat-ingestion/`

Use the `/branch` skill to create a new branch — it automatically creates the corresponding plan directory.

## Security

- Never commit secrets → use environment variables
- MCP server requires Bearer token authentication
- All credentials in `.env`, never hardcoded

## Key Files

| File | Purpose |
|-|-|
| `pyproject.toml` | Python deps + tool config (single source of truth) |
| `docker-compose.yml` | Infrastructure services (Neo4j, Qdrant, PostgreSQL) |
| `.env.example` | All required environment variables |
| `spec.md` | Technical specification |
| `PLAN.md` | Implementation plan with task registry |
| `rag-mcp-architecture-blueprint.md` | Architecture blueprint |
| `docs/` | Technical documentation (MkDocs) |

## Documentation

Architecture docs live in `docs/`. Key topics:

- Architecture overview and data flow
- Ingestion pipeline (CocoIndex, Jina v4, Graphiti)
- MCP server and search tools
- Knowledge graph schema and temporal awareness
- Deployment and infrastructure
- Configuration reference
- AWS setup (S3 event notifications, SQS, IAM)
