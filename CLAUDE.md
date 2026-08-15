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
| Ingestion (bulk, Path A) | `docs/ingestion/` | `ingestion/app.py`, `runner.py`, `pipeline.py`, `page_processor.py`, `file_processor.py`, `embedder.py`, `graph_engine.py` |
| Ingestion (live, Path B) | `docs/ingestion/` | `ingestion/live_ingest.py`, `graphiti_client.py` |
| MCP server & tools | `docs/server/` | `server/mcp_server.py`, `server/tools/` |
| Vector store (Qdrant) | `docs/configuration/infrastructure.md`, `docs/ingestion/embeddings.md` | `ingestion/qdrant_setup.py`, `ingestion/qdrant_target.py`, `ingestion/embedders/` |
| Knowledge graph (Neo4j) | `docs/configuration/infrastructure.md`, `docs/ingestion/knowledge-graph.md` | `ingestion/neo4j_setup.py`, `ingestion/graph_writer.py` |
| Entity extraction & schema | `docs/ingestion/` | `ingestion/entity_extractor.py`, `schema_inducer.py` |
| Agent | `docs/agent/` | `agent/` |
| Configuration | `docs/configuration/` | `config/settings.py`, `config/constants.py` |

> Adjust paths above if your `docs/` structure differs — the principle stands: always docs first.

**`plans/` is disposable brainstorming, NOT authoritative documentation. Do not treat plan files as source of truth. Do not update them when changing code or docs.**

---

## Project Overview

**Spektr** — a RAG-as-MCP-Server pipeline. Dual-path ingestion: batch documents (PDF, images) from local filesystem or S3, and real-time streaming text via HTTP. Builds vector embeddings (Qdrant) and a knowledge graph (Neo4j), then exposes session-aware search tools via an MCP server for AI agents.

## Architecture

- **Ingestion (Path A — Bulk):** CocoIndex v1 app (`ingestion/app.py` — one memoized component per file, Qdrant points *declared* on native collection targets) + Docling/PyMuPDF for file processing, embeddings selected by `EMBEDDING_MODEL` x `EMBEDDING_ROUTE` (see `config/embedding_models.py`), pluggable `GraphEngine` for entity/relation extraction. Document source is selected by `DOCUMENT_SOURCE` (`local` default, or `s3` / `sharepoint`). Live mode: `localfs` is a real watcher; S3 is scan-only in v1, so `ingestion/sqs_trigger.py` uses SQS as a *trigger* for a debounced catch-up scan (plus an interval and a startup sweep). Without `--live`, runs as one-shot batch. The `ingest-live` prod service is this daemon — not an HTTP endpoint.
- **Pipeline state:** CocoIndex v1 keeps its target-state ledger, memoization cache and component tree in a local **LMDB directory** (`COCOINDEX_DB_PATH`, default `state/cocoindex.db`). No PostgreSQL anywhere in the stack.
- **Ingestion (Path B — Live):** FastAPI HTTP endpoint (`live_ingest.py`, port 8001) for streaming text, configured embedding provider, Graphiti for temporal episodic memory
- **Vector Store:** Qdrant (dense + optional ColBERT multi-vector, Jina-only). Both paths write to `documents_dense`; live data tagged with `session_id` and `is_live`
- **Knowledge Graph:** Neo4j with two pluggable engines for Path A — **Graphiti is the default** (`GRAPH_ENGINE=graphiti`, LLM-based temporal episodes); **GLiNER2 is opt-in** (`GRAPH_ENGINE=gliner`, schema-driven CPU extraction). Path B always uses Graphiti directly, regardless of `GRAPH_ENGINE`. Both engines coexist in the same Neo4j instance.
- **MCP Server:** FastMCP (streamable-http default, SSE legacy, stdio) exposing seven tools: five search (`vector_search`, `visual_search`, `graph_search`, `multi_search`, `hybrid_search`) and two listing (`list_documents`, `list_document_chunks`). `multi_search` and `hybrid_search` share a fused dense+sparse retrieval core (`retrieval/`); `hybrid_search` adds query decomposition and a relevance-gated retry. Bearer auth via `MCP_API_KEY`.
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
│   ├── app.py              # CocoIndex v1 App: lifespan, source selection, source_key()
│   ├── runner.py           # Process entrypoint: run_pipeline, mode dispatch, error reporting
│   ├── sqs_trigger.py      # SQS-as-trigger daemon for the S3 source (live mode)
│   ├── pipeline.py         # Per-file processing (process_file_impl) + CLI shim
│   ├── page_processor.py   # Per-page chunk/embed/declare-point
│   ├── vlm_caption.py      # VLM captioning of visual pages -> graph
│   ├── qdrant_target.py    # Native Qdrant connector wiring (managed_by=USER)
│   ├── graph_target.py     # Custom TargetHandler: graph cleanup on source deletion
│   ├── file_processor.py   # PDF/image processing (Docling + PyMuPDF fallback)
│   ├── embedder.py         # Embedding dispatcher
│   ├── embedders/          # Provider implementations (jina.py, voyage.py, openrouter.py)
│   ├── graph_engine.py     # GraphEngine protocol + factory (Graphiti/GLiNER2)
│   ├── entity_extractor.py # LLM-based entity extraction
│   ├── graph_writer.py     # Graphiti-based graph writer
│   ├── graphiti_client.py  # Graphiti client singleton
│   ├── schema_inducer.py   # LLM-based per-document schema induction
│   ├── live_ingest.py      # Live streaming ingestion (FastAPI, Path B)
│   ├── cocoindex_ops.py    # Standalone one-shot embedding helpers (not wired into the app)
│   ├── _failure_tracker.py # SQLite-backed per-file retry counter (poison-pill)
│   ├── _utils.py           # Internal ingestion helpers
│   ├── qdrant_setup.py     # Qdrant collection setup (sole provisioning authority)
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
├── docker-compose.yml      # Qdrant + Neo4j (local dev)
├── docker-compose.prod.yml # Full production stack (app + data services + Traefik labels on mcp)
├── Dockerfile              # Multi-stage Python 3.13 + uv image (shared by all app services)
├── Caddyfile               # Sample reverse-proxy config (alternative to external Traefik)
├── Taskfile.yml            # go-task entry points (bare `task` for a grouped overview)
├── pyproject.toml          # Python config (single source of truth)
└── mkdocs.yml              # Documentation site config
```

## Quick Start

Use [go-task](https://taskfile.dev) as the task runner. A bare `task` prints a grouped
overview of the common commands; `task --list` shows all tasks alphabetically, and
`task --summary <name>` explains one task — what it does, its args, and its gotchas.

```bash
cp .env.example .env     # Configure environment (gitignored)
task setup               # Install uv and project dependencies
task up                  # Start Qdrant, Neo4j
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

**Test isolation — integration tests never touch dev data.** Qdrant is
redirected to `test_documents_dense` / `test_documents_multivec`
(`tests/conftest.py`, before `config` is imported). Neo4j Community has no
equivalent namespace, so the suite starts an **ephemeral
`neo4j:5.26-community` container** (Testcontainers) and repoints
`settings.neo4j_uri` at it for the session — that is what makes the per-test
`MATCH (n) DETACH DELETE n` safe. The container is autouse and only starts when
integration tests were collected, so `task test` needs no Docker. Do not point
the Neo4j fixtures back at the dev instance: it destroys the knowledge graph.

**Daily drivers:**

```bash
task smoke               # Direct vector_search smoke test (no MCP/LLM)
task smoke-graph         # Direct graph_search smoke test
task ask -- "question"   # End-to-end through agent + MCP + LLM (needs `task serve`)
task doctor              # Diff CocoIndex's LMDB ledger vs Qdrant; flags drift
task doctor-fix          # Repair drift (deletes Qdrant points no CocoIndex run declared)
task ingest -- --full-reprocess   # Reprocess everything, invalidating the memo cache
task backup              # Snapshot Qdrant + Neo4j + CocoIndex state to ./backups/<ts>/
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
- A failing `process_file_impl` re-raises, so CocoIndex writes no memoization entry and re-processes the file next run.
- After `PIPELINE_MAX_RETRIES` (default 3) failures for the same file, the poison-pill kicks in: log CRITICAL, swallow, let the memo entry be written so the file is not retried. CocoIndex separately logs-and-swallows a failing component, so the rest of the batch proceeds regardless.
- `app.update()` does **not** raise on per-file failures. `ingestion/runner.py` reads `stats().total.num_errors` and the exit code reflects it — don't "fix" that by assuming a raise.
- Failure counts live in `state/ingestion_failures.db` (sqlite, gitignored). Successful ingest resets the count.

**CocoIndex targets** (`ingestion/qdrant_target.py` + `ingestion/graph_target.py`):
- Qdrant collection targets are mounted `managed_by=ManagedBy.USER`. CocoIndex must never create, replace or drop a collection: a *replace* would drop Path B's live-session points, which share `documents_dense`. `ingestion/qdrant_setup.ensure_collections` is the sole provisioning authority.
- Deletion is per point id; there is no orphan sweep, so points CocoIndex never declared (live sessions) are invisible to Path A's reconciliation.
- Points are declared, not upserted — nothing reaches Qdrant until a file's component fully succeeds.
- Graph writes stay side effects (Graphiti is episodic and doesn't fit declared target state). Only *cleanup on source deletion* goes through the custom `TargetHandler` in `graph_target.py`.
- `QDRANT_DB` / `_PROVIDER_NAME` context-key strings are part of persistent tracking keys. Renaming them orphans the ledger and re-declares everything.

**Pipeline state:**
- CocoIndex v1 stores its ledger, memo cache and component tree in an LMDB directory (`COCOINDEX_DB_PATH`, default `state/cocoindex.db`). No PostgreSQL, no `DATABASE_URL`, no `ragingestion__cocoindex_tracking` table.
- `PIPELINE_MAX_CONCURRENT_FILES` (default 4) caps `max_inflight_components`; CocoIndex's own default of 1024 would blow past embedding rate limits.
- Forcing reprocessing = `task ingest -- --full-reprocess`, or deleting the state directory. Losing that directory costs a full reprocess, not data loss (point ids are deterministic uuid5).

**Qdrant payload schema:**
- `documents_dense` uses **named vectors**: `dense` (embedding, cosine) and
  `sparse` (miniCOIL, `Modifier.IDF`). Never write points with an unnamed
  vector — the collection will reject them.
- Every text-chunk point carries both vectors. Image and VLM-caption points
  carry `dense` only. `scripts/doctor.py` flags text chunks missing `sparse`.
- Every dense point carries `embedder_model` (str) and `embedder_dim` (int).
- `source_file` is the logical key; `metadata.source_key` duplicates it.

**Eval gate** (`tests/eval/`, `task eval`):
- PRs that touch `ingestion/`, `server/tools/`, `config/`, or the agent must not drop any metric below `tests/eval/thresholds.yaml`.
- Thresholds are raised over time, never silently lowered. A drop needs a reason in the commit message.
- Retrieval changes must also pass `task eval-retrieval` (recall@10, nDCG@10,
  MRR against `tests/eval/retrieval_set.yaml`). These metrics have no LLM in
  the loop and are the primary gate for retrieval work; RAGAS remains the
  gate for generation quality.

**Retrieval pipeline** (`retrieval/`):
- `retrieval/` must not import from `server/` or `fastmcp`. It is composed by
  MCP adapters and knows nothing about transport.
- `hybrid_search` must delegate its retrieval core to the same code path as
  `multi_search`. If the two ever diverge, the "hybrid is multi plus two
  stages" contract is broken and the ablation matrix stops being meaningful.
- Every pipeline stage degrades rather than raising. A stage failure appends
  its name to `degraded` and the tool still returns results.

**Observability** (`config/observability.py`):
- Every process entrypoint calls `setup_observability()` (idempotent). FastAPI apps additionally call `instrument_fastapi(app)`.
- Local-only by default. Set `LOGFIRE_TOKEN` + `OBSERVABILITY_LOCAL_ONLY=false` to ship to Logfire Cloud.
- JSON log records carry `trace_id`/`span_id` when emitted inside an active span.

**Backup cadence** (`scripts/backup.py`):
- `task backup` captures Qdrant + Neo4j + the CocoIndex LMDB state + a `manifest.json`. Recommended nightly via cron.
- Neo4j dump requires ~10-30s downtime (Community Edition has no online backup).
- LMDB has no safe hot-copy: stop the ingest process, or take the backup between ingests.
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

