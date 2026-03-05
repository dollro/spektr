# CLAUDE.md — Spektr

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Spektr** — a RAG-as-MCP-Server pipeline. Ingests documents (PDF, images) from local filesystem or S3, builds vector embeddings (Qdrant) and a knowledge graph (Neo4j), then exposes search tools via an MCP server for AI agents.

## Architecture

- **Ingestion:** CocoIndex pipeline + Docling/PyMuPDF for file processing, Jina/Voyage for embeddings
- **Vector Store:** Qdrant (dense + optional ColBERT multi-vector)
- **Knowledge Graph:** Neo4j via Graphiti for entity/relationship extraction
- **MCP Server:** FastMCP (SSE or stdio transport) exposing search tools
- **Agent:** Pydantic AI agent with MCP tool access
- **LLM:** Anthropic or OpenAI-compatible (configurable via `LLM_API_TYPE`)
- **Config:** Pydantic Settings from `.env`

## Project Layout

```
├── agent/                  # Pydantic AI agent
│   ├── agent.py            # Agent definition
│   └── api.py              # Agent API endpoints
├── config/                 # Configuration
│   ├── settings.py         # Pydantic Settings (single source of truth)
│   ├── constants.py        # Shared constants (collections, dimensions, entity types)
│   └── logging.py          # Logging configuration
├── ingestion/              # Document ingestion pipeline
│   ├── pipeline.py         # Main ingestion orchestrator
│   ├── file_processor.py   # PDF/image processing (Docling + PyMuPDF fallback)
│   ├── embedder.py         # Embedding dispatcher
│   ├── embedders/          # Provider implementations (jina.py, voyage.py)
│   ├── entity_extractor.py # LLM-based entity extraction
│   ├── graph_writer.py     # Neo4j graph writer
│   ├── graphiti_client.py  # Graphiti adapter for knowledge graph
│   ├── cocoindex_ops.py    # CocoIndex operations
│   ├── qdrant_setup.py     # Qdrant collection setup
│   └── neo4j_setup.py      # Neo4j schema setup
├── server/                 # MCP server
│   ├── mcp_server.py       # FastMCP server entry point
│   ├── models.py           # Shared Pydantic models
│   ├── providers.py        # Service provider initialization
│   └── tools/              # MCP tool implementations
│       ├── vector_search.py
│       ├── graph_search.py
│       ├── hybrid_search.py
│       ├── visual_search.py
│       ├── reranker.py
│       └── vlm_generator.py
├── tests/                  # Test suite
├── docs/                   # MkDocs documentation
├── plans/                       # Files for brainstorming and further implemenation planning
├── scripts/                # Utility scripts
├── docker-compose.yml      # Qdrant + Neo4j + PostgreSQL
├── pyproject.toml          # Python config (single source of truth)
├── Makefile                # docs-serve, docs-build
└── mkdocs.yml              # Documentation site config
```

## Quick Start

### Infrastructure (Docker)

```bash
docker compose up -d                  # Start Qdrant, Neo4j, PostgreSQL
docker compose down                   # Stop services
```

### Python (local venv via uv)

```bash
uv sync                               # Install dependencies
uv run python -m ingestion.pipeline   # Run ingestion pipeline
uv run python -m server.mcp_server    # Start MCP server
```

### Testing and linting

```bash
uv run pytest                          # Unit tests (excludes integration)
uv run pytest -m integration           # Integration tests (needs Docker services)
uv run ruff check .                    # Lint
uv run ruff format .                   # Format
uv run mypy .                          # Type check
```

### Documentation

```bash
make docs-serve                        # Serve MkDocs locally
make docs-build                        # Build docs
```

**Access Points:** Qdrant http://localhost:6333 | Neo4j http://localhost:7474 | MCP http://localhost:8080

## Code Style

- **Python:** Ruff (95 chars), mypy strict — configured in `pyproject.toml`
- **Target:** Python 3.13
- **Package manager:** uv (always use `uv`, never pip directly)
- **Dependencies:** `pyproject.toml` with `[dependency-groups]` dev

## Code Organization Rules

- **Max file size**: Keep files under 600 lines. Refactor into modules if exceeded.
- **Max function length**: 60 lines. Extract helpers.
- **Max class length**: 100 lines. Single concept per class.
- **Single responsibility**: Each module/file should do ONE thing well.
- **Prefer composition**: Break complex logic into small, composable functions.

## Core Principles

- **KISS**: Choose straightforward solutions over complex ones
- **YAGNI**: Implement features only when needed
- **Single Responsibility**: Each function/class has one clear purpose
- **Fail Fast**: Raise exceptions immediately when issues occur

## Environment

- **Env file:** `.env` (gitignored), copy from `.env.example`
- **Python deps:** `pyproject.toml` with dev group
- **Config:** `config/settings.py` — Pydantic Settings, reads from `.env`

## Git Workflow

Branches: `main` (production), `develop` (ongoing development), `feat/*`, `fix/*`, `chore/*`

Merge flow: `chore-xyz` → `develop` → `main`

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore
```

**Never include "claude code" or "written by claude" in commit messages.**


## Planning

All plan files go into `./plans/` in branch-specific subdirectories. The subdirectory name is the branch name with `/` replaced by `-`.

Example: on branch `chore/pydantic` → plans go in `./plans/chore-pydantic/`

Use the `/branch` skill to create a new branch — it automatically creates the corresponding plan directory.


## Security

- Never commit secrets → use environment variables
- Validate all user input
- Use parameterized queries

## Key Files

| File | Purpose |
|-|-|
| `pyproject.toml` | Python deps + tool config (single source of truth) |
| `config/settings.py` | All runtime settings (Pydantic Settings from `.env`) |
| `config/constants.py` | Shared constants (collection names, dimensions, entity types) |
| `docker-compose.yml` | Infrastructure services (Qdrant, Neo4j, PostgreSQL) |
| `.env.example` | Environment variable template with documentation |
| `mkdocs.yml` | Documentation site config |
| `docs/` | Full technical documentation (MkDocs Material) |

## Detailed Documentation

Architecture docs live in `docs/`. Run `make docs-serve` to browse locally.

- Ingestion pipeline, file processing, embeddings, knowledge graph
- MCP server, search tools, agent integration
- Configuration and infrastructure setup
