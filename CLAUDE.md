# CLAUDE.md — Spektr

## Project Overview

**Spektr** — a RAG-as-MCP-Server pipeline. Ingests documents (PDF, images) from local filesystem or S3, builds vector embeddings (Qdrant) and a knowledge graph (Neo4j), then exposes search tools via an MCP server for AI agents.

## Architecture

- **Ingestion:** CocoIndex pipeline + Docling/PyMuPDF for file processing, Jina/Voyage for embeddings
- **Vector Store:** Qdrant (dense + optional ColBERT multi-vector)
- **Knowledge Graph:** Neo4j with pluggable engine — Graphiti (LLM) or GLiNER2 (local CPU) via `GRAPH_ENGINE`
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
│   ├── graph_engine.py     # GraphEngine protocol + factory (Graphiti/GLiNER2)
│   ├── entity_extractor.py # LLM-based entity extraction
│   ├── graph_writer.py     # Graphiti-based graph writer
│   ├── graphiti_client.py  # Graphiti client singleton
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
├── docs/                   # MkDocs documentation (architecture, guides, API)
├── plans/                  # Disposable brainstorming & implementation plans (NOT source of truth)
├── scripts/                # Utility scripts
├── docker-compose.yml      # Qdrant + Neo4j + PostgreSQL
├── pyproject.toml          # Python config (single source of truth)
├── Makefile                # docs-serve, docs-build
└── mkdocs.yml              # Documentation site config
```

**Start here:** When exploring an unfamiliar area, read the relevant `docs/` page first for the "why", then explore the code for the "how".

## Quick Start

```bash
cp .env.example .env                  # Configure environment (gitignored)
docker compose up -d                  # Start Qdrant, Neo4j, PostgreSQL
uv sync                               # Install dependencies
uv run python -m ingestion.pipeline   # Run ingestion pipeline
uv run python -m server.mcp_server    # Start MCP server
```

```bash
uv run pytest                          # Unit tests (excludes integration)
uv run pytest -m integration           # Integration tests (needs Docker services)
uv run ruff check .                    # Lint
uv run ruff format .                   # Format
uv run mypy .                          # Type check
make docs-serve                        # Serve MkDocs locally
make docs-build                        # Build docs
```

**Access Points:** Qdrant http://localhost:6333 | Neo4j http://localhost:7474 | MCP http://localhost:8080

## Code Standards

**Style & tooling:**
- Python 3.13, Ruff (95 chars), mypy strict — configured in `pyproject.toml`
- Package manager: uv (always use `uv`, never pip directly)
- Dependencies: `pyproject.toml` with `[dependency-groups]` dev

**Organization:**
- Max file size: 600 lines. Max function: 60 lines. Max class: 100 lines.
- Single responsibility: each module/file does ONE thing well
- Prefer composition: break complex logic into small, composable functions

**Principles:** KISS, YAGNI, fail fast. Each function/class has one clear purpose.

**Security:** Never commit secrets (use env vars). Validate all user input. Use parameterized queries.

## Git Workflow

Branches: `main` (production), `develop` (ongoing development), `feat/*`, `fix/*`, `chore/*`

Merge flow: `chore-xyz` → `develop` → `main`

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore
```

**Never include "claude code" or "written by claude" in commit messages.**


## Planning

`plans/` contains disposable brainstorming notes and implementation plans. These are working documents used during development — they are **not** source-of-truth documentation. The authoritative docs live in `docs/`. Do not update plan files when updating actual documentation or code; they can go stale without harm.

All plan files go into `./plans/` in branch-specific subdirectories. The subdirectory name is the branch name with `/` replaced by `-`.

Example: on branch `chore/pydantic` → plans go in `./plans/chore-pydantic/`

Use the `/branch` skill to create a new branch — it automatically creates the corresponding plan directory.
