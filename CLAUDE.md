# CLAUDE.md — Spektr

## ⚠️ Mandatory: Read Docs Before Code

Before modifying or exploring ANY module, you MUST first read the corresponding page(s) in `docs/`. Do NOT start by browsing source files or running `find`/`grep` across the codebase.

**Workflow for every task:**

1. Identify which area of the project the task touches
2. Read the matching `docs/` page(s) below — they contain the "why" and architectural decisions not visible in code
3. Only then explore the relevant source files for the "how"
4. Never skip steps 1–2

**Doc → Code mapping:**

| Area | Read first | Then explore |
|---|---|---|
| Ingestion (bulk, Path A) | `docs/ingestion/` | `ingestion/pipeline.py`, `file_processor.py`, `embedder.py`, `graph_engine.py` |
| Ingestion (live, Path B) | `docs/ingestion/` | `ingestion/live_ingest.py`, `graphiti_client.py` |
| MCP server & tools | `docs/mcp-server/` | `server/mcp_server.py`, `server/tools/` |
| Vector store (Qdrant) | `docs/infrastructure/` | `ingestion/qdrant_setup.py`, `ingestion/embedders/` |
| Knowledge graph (Neo4j) | `docs/infrastructure/` | `ingestion/neo4j_setup.py`, `ingestion/graph_writer.py` |
| Entity extraction & schema | `docs/ingestion/` | `ingestion/entity_extractor.py`, `schema_inducer.py` |
| Agent | `docs/agent/` | `agent/` |
| Configuration | `docs/configuration/` | `config/settings.py`, `config/constants.py` |

> Adjust paths above if your `docs/` structure differs — the principle stands: always docs first.

**`plans/` is disposable brainstorming, NOT authoritative documentation. Do not treat plan files as source of truth. Do not update them when changing code or docs.**

---

## Project Overview

**Spektr** — a RAG-as-MCP-Server pipeline. Dual-path ingestion: batch documents (PDF, images) from local filesystem or S3, and real-time streaming text via HTTP. Builds vector embeddings (Qdrant) and a knowledge graph (Neo4j), then exposes session-aware search tools via an MCP server for AI agents.

## Architecture

- **Ingestion (Path A — Bulk):** CocoIndex pipeline + Docling/PyMuPDF for file processing, Jina/Voyage/OpenRouter for embeddings (selected by `EMBEDDING_PROVIDER`), pluggable `GraphEngine` for entity/relation extraction. Document source is selected by `DOCUMENT_SOURCE` (`local` default, or `s3` / `sharepoint`). Triggered by S3→SQS events (long-running `--live` daemon polls SQS), SharePoint sync (`services/sharepoint_sync/`), or local filesystem watch. Without `--live`, runs as one-shot batch. The `ingest-live` prod service is this SQS daemon — not an HTTP endpoint.
- **Ingestion (Path B — Live):** FastAPI HTTP endpoint (`live_ingest.py`, port 8001) for streaming text, configured embedding provider, Graphiti for temporal episodic memory
- **Vector Store:** Qdrant (dense + optional ColBERT multi-vector, Jina-only). Both paths write to `documents_dense`; live data tagged with `session_id` and `is_live`
- **Knowledge Graph:** Neo4j with two pluggable engines for Path A — **Graphiti is the default** (`GRAPH_ENGINE=graphiti`, LLM-based temporal episodes); **GLiNER2 is opt-in** (`GRAPH_ENGINE=gliner`, schema-driven CPU extraction). Path B always uses Graphiti directly, regardless of `GRAPH_ENGINE`. Both engines coexist in the same Neo4j instance.
- **MCP Server:** FastMCP (streamable-http default, SSE legacy, stdio) exposing six tools: four search (`vector_search`, `visual_search`, `graph_search`, `hybrid_search`) and two listing (`list_documents`, `list_document_chunks`). Bearer auth via `MCP_API_KEY`.
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
│   ├── constants.py        # Shared constants (collections, dimensions, 14 entity types, 12 relationship types)
│   ├── logging.py          # Logging configuration
│   └── observability.py    # Logfire/OTel setup (setup_observability, instrument_fastapi)
├── ingestion/              # Document ingestion pipeline
│   ├── pipeline.py         # Main ingestion orchestrator
│   ├── file_processor.py   # PDF/image processing (Docling + PyMuPDF fallback)
│   ├── embedder.py         # Embedding dispatcher
│   ├── embedders/          # Provider implementations (jina.py, voyage.py, openrouter.py)
│   ├── graph_engine.py     # GraphEngine protocol + factory (Graphiti/GLiNER2)
│   ├── entity_extractor.py # LLM-based entity extraction
│   ├── graph_writer.py     # Graphiti-based graph writer
│   ├── graphiti_client.py  # Graphiti client singleton
│   ├── schema_inducer.py   # LLM-based per-document schema induction
│   ├── live_ingest.py      # Live streaming ingestion (FastAPI, Path B)
│   ├── cocoindex_ops.py    # CocoIndex operations
│   ├── target_connector.py # Qdrant/Neo4j target connectors for CocoIndex flow
│   ├── _failure_tracker.py # SQLite-backed per-file retry counter (poison-pill)
│   ├── _utils.py           # Internal ingestion helpers
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
│       ├── list_documents.py
│       ├── list_document_chunks.py
│       ├── reranker.py
│       └── vlm_generator.py
├── services/               # Long-running auxiliary services
│   └── sharepoint_sync/    # SharePoint → local-volume sync daemon (Path A source)
├── tests/                  # Test suite
├── docs/                   # MkDocs documentation — READ FIRST (see top of file)
├── plans/                  # Disposable brainstorming — NOT source of truth
├── scripts/                # Utility scripts (backup.py, restore.py, doctor.py, …)
├── docker-compose.yml      # Qdrant + Neo4j + PostgreSQL (local dev)
├── docker-compose.prod.yml # Full production stack (app + data services + Traefik labels on mcp)
├── Dockerfile              # Multi-stage Python 3.13 + uv image (shared by all app services)
├── Caddyfile               # Sample reverse-proxy config (alternative to external Traefik)
├── Taskfile.yml            # go-task entry points (task --list to enumerate)
├── pyproject.toml          # Python config (single source of truth)
└── mkdocs.yml              # Documentation site config
```

## Quick Start

Use [go-task](https://taskfile.dev) as the task runner. `task --list` shows all tasks.

```bash
cp .env.example .env     # Configure environment (gitignored)
task setup               # Install uv and project dependencies
task up                  # Start Qdrant, Neo4j, PostgreSQL
task ingest              # Run bulk ingestion pipeline
task serve               # Start MCP server
```

```bash
task test                # Unit tests (excludes integration)
task test-integration    # Integration tests (needs Docker services)
task eval                # RAGAS retrieval-quality eval (needs `uv sync --extra eval`)
task lint                # Ruff lint
task format              # Ruff format
task typecheck           # mypy
task check               # lint + typecheck + test
task docs-serve          # Serve MkDocs locally
task docs-build          # Build docs
```

**Daily drivers:**

```bash
task smoke               # Direct vector_search smoke test (no MCP/LLM)
task smoke-graph         # Direct graph_search smoke test
task ask -- "question"   # End-to-end through agent + MCP + LLM (needs `task serve`)
task doctor              # Diff CocoIndex tracking vs Qdrant; flags drift
task doctor-fix          # Repair drift (deletes orphan tracking rows)
task backup              # Snapshot Qdrant + Neo4j + Postgres to ./backups/<ts>/
task restore -- --from backups/<ts> --target all --yes-i-know-this-wipes-things
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

## Production contracts

These are load-bearing invariants introduced by the `feat/prod-hardening` work. Don't break them silently.

**Ingestion failure semantics** (`ingestion/pipeline.py` + `ingestion/_failure_tracker.py`):
- A failing `ingest_file` re-raises so CocoIndex leaves the tracking row out and retries next run.
- After `PIPELINE_MAX_RETRIES` (default 3) failures for the same file, the poison-pill kicks in: log CRITICAL, swallow, let CocoIndex mark it processed so the rest of the batch proceeds.
- Failure counts live in `state/ingestion_failures.db` (sqlite, gitignored). Successful ingest resets the count.

**Qdrant payload schema:**
- Every dense point carries `embedder_model` (str) and `embedder_dim` (int). Never write points without them — `scripts/doctor.py` flags mixed values.
- `source_file` is the logical key; `metadata.source_key` duplicates it for compatibility.

**Eval gate** (`tests/eval/`, `task eval`):
- PRs that touch `ingestion/`, `server/tools/`, `config/`, or the agent must not drop any metric below `tests/eval/thresholds.yaml`.
- Thresholds are raised over time, never silently lowered. A drop needs a reason in the commit message.

**Observability** (`config/observability.py`):
- Every process entrypoint calls `setup_observability()` (idempotent). FastAPI apps additionally call `instrument_fastapi(app)`.
- Local-only by default. Set `LOGFIRE_TOKEN` + `OBSERVABILITY_LOCAL_ONLY=false` to ship to Logfire Cloud.
- JSON log records carry `trace_id`/`span_id` when emitted inside an active span.

**Backup cadence** (`scripts/backup.py`):
- `task backup` captures Qdrant + Neo4j + Postgres + a `manifest.json`. Recommended nightly via cron.
- Neo4j dump requires ~10-30s downtime (Community Edition has no online backup).
- `task restore` refuses without `--yes-i-know-this-wipes-things`.

See `docs/operations/` for runbooks, `docs/eval/golden-set.md` for eval details.

## Git Workflow

Branches: `main` (production), `develop` (ongoing development), `feat/*`, `fix/*`, `chore/*`

Merge flow: `feature-branch` → `develop` → `main`

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore
```

**Never include "claude code" or "written by claude" in commit messages.**

## Planning

All plan files go into `./plans/` in branch-specific subdirectories. The subdirectory name is the branch name with `/` replaced by `-`.

Example: on branch `chore/pydantic` → plans go in `./plans/chore-pydantic/`

Use the `/branch` skill to create a new branch — it automatically creates the corresponding plan directory.

