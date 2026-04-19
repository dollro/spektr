# Plan: Commit Series for Remediation + Documentation Work

## Context

This session produced two bodies of work: (1) code remediation from the review (bug fixes, Graphiti integration, MCP auth, test improvements) and (2) full MkDocs documentation. The untracked files also include server/, agent/, and test files from prior sessions that were never committed. Commits should be logical, ordered, and follow the project's `<type>(<scope>): <subject>` convention.

## Commit Strategy

Seven commits, ordered so each builds on the previous cleanly.

### Commit 1 — MCP server, agent, and tool modules (previously uncommitted)
```
feat(server): add MCP server, search tools, agent, and response models
```
**Files:**
- `server/mcp_server.py` (includes BearerAuthMiddleware)
- `server/models.py`
- `server/providers.py`
- `server/tools/vector_search.py`
- `server/tools/visual_search.py`
- `server/tools/graph_search.py`
- `server/tools/hybrid_search.py`
- `server/tools/reranker.py`
- `server/tools/vlm_generator.py`
- `agent/agent.py`
- `agent/api.py`
- `config/logging.py`
- `config/settings.py` (mcp_api_key addition)

**Why first:** These are new modules that other changes reference. Tests import from them.

### Commit 2 — Bug fixes (Phase 1 of remediation)
```
fix(ingestion): fix _run_async, page_number tracking, embedder retry logic
```
**Files:**
- `ingestion/jina_cocoindex_ops.py` (pool.submit fix)
- `ingestion/file_processor.py` (page_number param)
- `ingestion/embedder.py` (wire _is_retryable_status into retry decorator)
- `ingestion/entity_extractor.py` (.strip() instead of .title())

**Why grouped:** All four are independent bug fixes from the code review.

### Commit 3 — Graphiti integration (Phase 2 core feature)
```
feat(graphiti): replace raw Neo4j with Graphiti temporal knowledge graph
```
**Files:**
- `pyproject.toml` (graphiti-core dependency)
- `ingestion/graphiti_client.py` (new — singleton lifecycle)
- `ingestion/graph_writer.py` (GraphitiWriter + legacy rename)
- `ingestion/pipeline.py` (switch to GraphitiWriter, add char_count)
- `config/constants.py` (revert hardcoded entity types)

**Why separate:** This is the largest architectural change — deserves its own commit.

### Commit 4 — Test fixes and new edge case tests
```
test: fix async fixtures, update graph tests for Graphiti, add edge cases
```
**Files:**
- `tests/conftest.py` (async rag_agent fixture)
- `tests/test_integration_ingestion.py` (async/sync fix, GraphitiWriter mocks)
- `tests/test_tools.py` (Graphiti-based graph tests, auth tests, edge cases)
- `tests/test_e2e.py` (Graphiti mocks, remove raw Cypher seeding)
- `tests/test_embedder.py` (retry exhaustion test)
- `tests/test_entity_extractor.py` (updated normalization assertions)
- `tests/test_agent.py` (new — already existed untracked)

### Commit 5 — Configuration and environment
```
chore(config): add MCP auth key, update .env.example, add .gitignore entries
```
**Files:**
- `.env.example` (auth, resilience, feature flags, observability vars)
- `.gitignore` (site/)

### Commit 6 — Documentation (MkDocs site)
```
docs: add full MkDocs documentation site with 17 pages
```
**Files:**
- `mkdocs.yml`
- `Makefile`
- `docs/index.md`
- `docs/architecture/overview.md`
- `docs/architecture/data-flow.md`
- `docs/ingestion/overview.md`
- `docs/ingestion/file-processing.md`
- `docs/ingestion/embeddings.md`
- `docs/ingestion/knowledge-graph.md`
- `docs/ingestion/cocoindex.md`
- `docs/server/overview.md`
- `docs/server/search-tools.md`
- `docs/server/authentication.md`
- `docs/agent/overview.md`
- `docs/agent/http-api.md`
- `docs/configuration/environment.md`
- `docs/configuration/infrastructure.md`
- `docs/deployment/aws-setup.md`
- `docs/deployment/local-development.md`
- `docs/api/models.md`
- `docs/AWS_SETUP.md` (deleted — moved to deployment/)

### Commit 7 — Project meta
```
docs: update CLAUDE.md and add README
```
**Files:**
- `CLAUDE.md`
- `README.md`
- `uv.lock`
- `.claude/contracts.md`

## Verification

After all commits:
- `git log --oneline` shows 7 clean commits
- `uv run ruff check .` passes
- `uv run pytest -m "not integration"` passes
- `uv run mkdocs build` succeeds without warnings
