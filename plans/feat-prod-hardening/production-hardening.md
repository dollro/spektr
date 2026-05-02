# Spektr — Production Hardening Plan

## Context

Spektr is a dual-path RAG-as-MCP-server project (CocoIndex bulk ingest + FastAPI live ingest → Qdrant + Neo4j/Graphiti → FastMCP + Pydantic AI agent). Today's work uncovered a real ingestion-atomicity bug (pipeline exceptions silently caught, CocoIndex marks failed files as "processed," retries skip them) and several production gaps common to GraphRAG stacks: no embedder versioning, no automated RAG evaluation, no trace-level observability, and no backup tooling.

This plan sequences five steps: a preparatory branch/commit step, then four incremental production-hardening phases. Each phase is independently shippable, testable, and leaves the system in a known-good state. Total effort ~3 days.

---

## Phase 0 — Branch setup (~15 min)

### Goal
Ship today's accumulated tooling work to `develop` as clean commits, then cut a dedicated feature branch for the production-hardening work so it lands as one reviewable PR.

### Today's uncommitted work on `develop` (to be committed first)

Grouped into two logical commits:

**Commit 1 — `fix(tests): repair test rot from dual-path and auth refactors`**
- `tests/conftest.py` — `_make_mock_extraction` entity type `"CONCEPT"` → `"concept"`
- `tests/test_graph_writer.py` — lowercase entity types, `USES_TECHNOLOGY`/`PRODUCES` → `uses`/`created_by`
- `tests/test_integration_ingestion.py` — `graphiti_writer=` → `graph_engine=`, removed stale `GraphitiWriter` patch, added `_configure_mock_pipeline_settings` helper with `graph_enabled=False`, `pipeline_timeout=3600`
- `tests/test_e2e.py` — moved `get_graphiti` patch target to `ingestion.graphiti_client`, added `get_graph_engine` mocks, simplified hybrid test
- `tests/test_integration_live_ingest.py` + `tests/test_integration_live_e2e.py` — `settings.ingest_api_key=""` override

**Commit 2 — `chore: taskfile runner + smoke/ask/doctor helpers + list_documents MCP tool`**
- `Makefile` deleted; `Taskfile.yml` (new) with: `setup`, `up`, `down`, `ingest`, `serve`, `test`, `test-integration`, `lint`, `format`, `typecheck`, `check`, `docs-serve`, `docs-build`, `smoke`, `smoke-graph`, `ask`, `doctor`
- `scripts/smoke_search.py`, `scripts/smoke_graph.py`, `scripts/ask.py`, `scripts/doctor.py` (new)
- `server/tools/list_documents.py` (new) + `server/mcp_server.py` registration
- `agent/agent.py` — pass `Authorization: Bearer <mcp_api_key>` header to `MCPServerSSE` when configured
- `docs/resources/spektr-learning-material.md` (new) + `mkdocs.yml` nav update
- `CLAUDE.md` + `README.md` — updated commands (`task <name>` instead of `make`)

**Left uncommitted / to review first:**
- `.env.example` — inspect diff; if it's just documentation/whitespace, fold into commit 2; if it adds new env vars, commit separately.
- `exp.txt` — untracked scratch file, leave in place.
- `tests/test_live_ingest.py` — was already modified at session start (pre-today); check with `git diff` and decide separately.

### Steps

1. `git diff .env.example tests/test_live_ingest.py` — review pre-existing and new modifications.
2. Stage + commit the two groups above using `git add <paths>` per commit (not `git add -A`).
3. Push `develop`.
4. Create branch: `git checkout -b feat/prod-hardening develop`.
5. Create branch plan directory: `mkdir -p plans/feat-prod-hardening` (project convention per `CLAUDE.md`).
6. Copy this plan into the branch: `cp /home/rodo/.claude/plans/make-a-detailed-plan-tidy-squid.md plans/feat-prod-hardening/production-hardening.md` so it's versioned with the work.

### Verification
- `git log --oneline develop -5` shows the two new commits.
- `git branch --show-current` returns `feat/prod-hardening`.
- `task test` + `task test-integration` still green on the new branch.
- `plans/feat-prod-hardening/production-hardening.md` exists.

---

## Phase A — Quick wins (~1 day)

### A1. Ingestion atomicity: re-raise + poison-pill

**Why:** `ingest_file` in `ingestion/pipeline.py` lines 626-638 catches `TimeoutError` and `Exception` silently — CocoIndex sees success, writes tracking row, next run skips the file. We hit this with a `gliner2` import error. Re-raising restores CocoIndex's retry semantics; a poison-pill prevents one bad file from permanently blocking a batch.

**Files to change:**
- `ingestion/pipeline.py` (~lines 622-638): replace silent `except TimeoutError` / `except Exception` blocks with a block that records failure, then re-raises for the first N attempts, swallows + logs CRITICAL beyond N.
- New `ingestion/_failure_tracker.py` (~80 LOC): `record_failure(source_file) -> int`, `reset_failure(source_file)`, `should_poison(source_file, max_retries) -> bool`. Persist counts in a new Postgres table `spektr_ingestion_failures` on the existing CocoIndex DB (reuse `settings.database_url`). Use `psycopg` (or whatever CocoIndex already depends on — verify before adding a dep).
- `config/settings.py`: add `pipeline_max_retries: int = 3`.
- `.env.example`: add the new setting under the `# Resilience` section that already exists.

**Reuse:** existing `ingestion/_utils.py::run_async` (timeout wrapper, already used). Structured JSON log from `config/logging.py` for the CRITICAL poison log.

**Tests (flat-style, matching existing convention):**
- `tests/test_pipeline_atomicity.py` — mock an `_process_all_pages` that raises; assert `ingest_file` re-raises under the threshold and swallows after. Unit-level, no Docker.
- Extend `tests/test_integration_ingestion.py` with a "force failure" test that runs `ingest_file` with a deliberately broken embedder mock and confirms no point lands in Qdrant AND no tracking row is written.

**Verify:** `task test`, `task test-integration`, then simulate: rename a `.pdf` over a text file, `task ingest` fails 3× noisily, `task doctor` flags drift, delete the bad file, `task ingest` proceeds. Confirm failure counts reset on success.

### A2. Embedder versioning in Qdrant payload

**Why:** `ingestion/pipeline.py` lines 229-241 write point payloads with no `embedder_model` / `embedder_dim`. A silent provider swap (Jina→Voyage) or dimension change produces garbage retrieval; impossible to audit.

**Files to change:**
- `ingestion/embedders/jina.py` and `ingestion/embedders/voyage.py`: expose `model_name: str` and `dim: int` properties (probably already known internally — surface them). Keep the `Embedder` protocol in `ingestion/embedder.py` requiring both.
- `ingestion/pipeline.py`: in the dense-point payload construction (lines ~229-241) and in `ingestion/live_ingest.py` (lines ~137-147 for live-chunk payloads), add `embedder_model`, `embedder_dim` keys.
- `scripts/doctor.py`: scroll a sample of points, detect mixed `embedder_model`/`embedder_dim` across the collection, warn if inconsistent with current `settings.embedding_provider`.

**Tests:**
- Extend `tests/test_pipeline_dual_embed.py` with payload assertions for the new keys.
- Extend `tests/test_integration_live_ingest.py` similarly.
- Small unit test for the doctor version-check path.

**Verify:** `task ingest` on a fresh file → `curl -s http://localhost:6333/collections/documents_dense/points/<id>` shows the new keys. `task doctor` still exits 0.

### A3. `task doctor --fix`

**Why:** `scripts/doctor.py` detects drift but only reports. Orphan CocoIndex rows (tracked but zero points in Qdrant) need manual SQL today. Closes the loop.

**Files to change:**
- `scripts/doctor.py`: add `argparse` with `--fix` and `--yes` flags. When `--fix` and `only_cocoindex` is non-empty, issue `DELETE FROM ragingestion__cocoindex_tracking WHERE source_key::text = ANY(%s)` via the existing `docker compose exec psql` pattern already in the script. Prompt for confirmation unless `--yes`.
- `Taskfile.yml`: add `task doctor-fix` target that calls the script with `--fix`.

**Tests:** `tests/test_doctor.py` — mock subprocess + `list_documents`, assert DELETE is issued only in `--fix` mode.

**Verify:** drop a Qdrant collection manually → `task doctor` flags drift → `task doctor-fix --yes` → `task ingest` reprocesses.

### A4. Flip `RERANK_ENABLED=true` default

**Why:** cross-encoder reranking is the single biggest precision lift for ~20ms cost; currently off by default (`.env.example` and `config/settings.py`).

**Files to change:**
- `config/settings.py`: `rerank_enabled: bool = True`.
- `.env.example`: flip the default and the comment.
- Confirm `server/tools/reranker.py` works cold (downloads / loads cross-encoder model) under `task serve`. If startup cost is painful, gate the model-load on first use (lazy init pattern).

**Tests:** verify `tests/test_tools.py` reranker tests still green.

**Verify:** `task smoke` — compare top-3 scores before/after. Should be visibly reordered for ambiguous queries.

---

## Phase B — RAG eval harness (~1 day)

### Why

No automated retrieval-quality test exists. Risk: silent regressions on prompt/model/chunker changes. Industry bar from 2026 RAG reviews: faithfulness ≥ 0.80, context precision ≥ 0.70, answer relevance ≥ 0.75 in production.

### Layout

- `tests/eval/` (new top-level test dir, pytest-marked `eval`).
- `tests/eval/fixtures/golden_set.yaml` — 15 curated Q&A pairs against whatever docs are in `documents/` at eval time. Schema per item: `id, question, expected_context_substrings: list[str], expected_answer_hint: str, tags: list[str]`.
- `tests/eval/conftest.py` — fixtures: load golden set, spin up agent (or retriever only for retrieval-focused runs).
- `tests/eval/test_retrieval_quality.py` — runs RAGAS over the golden set, asserts metrics ≥ thresholds.
- `tests/eval/thresholds.yaml` — `faithfulness: 0.80`, `context_precision: 0.70`, `answer_relevance: 0.75`.
- `eval-reports/` (gitignored) — dated JSON runs.

### Deps

Add `ragas>=0.2` and `datasets` as a pyproject optional-dep group `eval` (avoids slowing down `uv sync` for everyone). RAGAS can use Anthropic via its langchain wrapper; no OpenAI required. Configure RAGAS's `RunConfig` to use the project's existing Anthropic client settings.

### Task + CI wiring

- `Taskfile.yml`: `task eval` → `uv run pytest tests/eval -m eval`.
- In `pyproject.toml` pytest config: register `eval` marker, exclude from default `addopts`.
- `.gitlab-ci.yml` already exists (per earlier ls of sidesupport, check ours) — add an `eval` job that runs on merge to `develop`/`main`. Cache Qdrant state using a small seeded collection snapshot to keep runs <5 min.

### Reuse

The existing `scripts/ask.py` already wires up the agent identically to production — reuse `create_rag_agent` from `agent/agent.py` inside the eval fixture.

### Verify

- `task eval` passes on current ingested corpus.
- Deliberately degrade a prompt (e.g. tell agent to always answer "I don't know") → eval fails with a metric breakdown.

---

## Phase C — Observability (~half day)

### Why

`config/logging.py` gives structured JSON logs with extras (duration_ms, tool, etc.), but no trace IDs link ingest → embed → qdrant → tool call → LLM response. Debugging a slow query or a wrong answer requires stitching logs manually.

### Pick: Logfire with local-only default

Pydantic AI instruments natively (`logfire.instrument_pydantic_ai()`). FastAPI, httpx have built-in instrumentors. `logfire.configure(send_to_logfire=False)` emits OTEL locally — no vendor lock-in, no paid tier required. Cloud mode opt-in via `LOGFIRE_TOKEN`.

### Files to change

- New `config/observability.py`: `setup_observability()` reads `settings.logfire_token`, `settings.observability_local_only: bool = True`, `settings.service_name = "spektr"`. Calls `logfire.configure(...)`, then `logfire.instrument_pydantic_ai()`, `logfire.instrument_fastapi(app)`, `logfire.instrument_httpx()`. Idempotent (guard with module-level flag).
- `config/settings.py`: add the three settings.
- `.env.example`: `LOGFIRE_TOKEN=` (optional), `OBSERVABILITY_LOCAL_ONLY=true`, `SERVICE_NAME=spektr`.
- Call `setup_observability()` from entrypoints:
  - `ingestion/pipeline.py::run_pipeline` (before `cocoindex.init()`)
  - `ingestion/live_ingest.py` (right after `app = FastAPI(...)`)
  - `server/mcp_server.py` (before `mcp.run(...)`)
  - `agent/api.py` (in `lifespan`)
  - `scripts/ask.py` (before `create_rag_agent()`)
- `config/logging.py`: add a logging filter that pulls OTEL trace_id/span_id from context and merges them into the JSON log record. Existing JSONFormatter stays backward compatible — just adds two more keys when present.

### Optional: Jaeger in docker-compose

Add a gated `jaeger` service to `docker-compose.yml` behind an `-f docker-compose.obs.yml` profile so dev doesn't pay for it unless wanted. OTLP endpoint `http://jaeger:4318`.

### Deps

`logfire>=3.0` (pulls in `opentelemetry-sdk` transitively). Single line in `pyproject.toml`.

### Tests

- `tests/test_observability.py` — assert `setup_observability()` is idempotent, assert no network egress when `OBSERVABILITY_LOCAL_ONLY=true` (mock the httpx transport).

### Verify

- `task ingest` → trace appears in `logfire` dashboard if token set, or in Jaeger if local profile enabled.
- `task ask -- "..."` → trace shows LLM call, MCP tool call, Qdrant query, all linked by trace_id that also appears in JSON logs.

---

## Phase D — Backups (~half day)

### Why

Qdrant/Neo4j/Postgres are Docker volumes today; no snapshot strategy, no documented restore. Re-ingesting ~100GB of docs after a disk loss is not acceptable.

### Files to create

- `scripts/backup.py` (~120 LOC): argparse subcommands `qdrant`, `neo4j`, `postgres`, `all`. Output dir `./backups/{YYYYMMDD-HHMMSS}/{qdrant,neo4j,postgres}/`. Writes a `manifest.json` at the top level (service versions, collection list + point counts, row counts).
  - Qdrant: `POST /collections/{name}/snapshots` via httpx, stream the resulting tarball to disk. Loop over all collections.
  - Neo4j: `docker compose exec neo4j neo4j-admin database backup neo4j --to-path=/tmp` then copy out. Graphiti + GLiNER paths both store here.
  - Postgres: `docker compose exec -T postgres pg_dump -U cocoindex -Fc cocoindex` → file.
- `scripts/restore.py` (~100 LOC): inverse of backup, refuses to run without `--yes-i-know-this-wipes-things`. Reads manifest first to confirm target versions match.
- `Taskfile.yml`: `task backup`, `task backup:qdrant`, `task restore -- --from backups/<ts> --target all`.
- `.gitignore`: add `backups/` and `eval-reports/`.

### Retention

Optional `--prune-older-than 30d` flag on `backup.py` — removes old dated dirs. Cron-friendly.

### Tests

- `tests/test_backup.py` with mocked httpx + subprocess.
- Round-trip integration test gated behind `SPEKTR_RUN_BACKUP_IT=1` env (not in default CI): seed tiny collection, backup, wipe, restore, verify identical.

### Verify

Manually: `task backup` → inspect `backups/<ts>/manifest.json`, confirm file sizes non-zero. Then `docker compose down -v`, `docker compose up -d`, `task restore -- --from backups/<ts> --target all`, `task smoke` still returns the same results.

---

## Cross-cutting

### Docs updates
- `docs/operations/atomicity.md` (new) — failure semantics, how to reset failure counts, poison-pill threshold.
- `docs/operations/observability.md` (new) — Logfire setup, local-only mode, Jaeger profile.
- `docs/operations/backup-restore.md` (new) — runbook with recovery-time expectations.
- `docs/eval/golden-set.md` (new) — how to add fixtures, metric meanings, threshold rationale.
- `mkdocs.yml` nav: add an "Operations" section.

### CLAUDE.md updates
Four short additions:
- "Ingestion failure semantics" — failures re-raise; retries bounded; poison-pill on persistent failure.
- "Qdrant payload schema includes embedder_model and embedder_dim."
- "Eval gate — PRs that touch retrieval-path code must not regress thresholds in `tests/eval/thresholds.yaml`."
- "Backup cadence: `task backup` nightly (recommended cron)."

### Interaction with live-ingest Path B
- Atomicity: live ingest should return HTTP 5xx on ingestion failure rather than returning `status=accepted` silently. A middleware increments the same `_failure_tracker` and returns 503 with `Retry-After` when over threshold.
- Versioning: live-chunk payloads get the same `embedder_model`/`embedder_dim` keys.
- Observability: `setup_observability()` called once per process — FastAPI instrumentation covers Path B automatically.
- Backups: path-agnostic.

### Sequencing
A1 → A2 → A3 → A4 → B → C → D. A is the only phase with runtime semantic changes (failures now surface); release-note it. Others are additive.

---

## Critical files

Modified:
- `ingestion/pipeline.py`
- `ingestion/live_ingest.py`
- `ingestion/embedder.py`, `ingestion/embedders/jina.py`, `ingestion/embedders/voyage.py`
- `config/settings.py`, `config/logging.py`
- `.env.example`
- `scripts/doctor.py`
- `Taskfile.yml`
- `pyproject.toml`
- `CLAUDE.md`
- `mkdocs.yml`

Created:
- `ingestion/_failure_tracker.py`
- `config/observability.py`
- `scripts/backup.py`, `scripts/restore.py`
- `tests/test_pipeline_atomicity.py`, `tests/test_doctor.py`, `tests/test_observability.py`, `tests/test_backup.py`
- `tests/eval/{fixtures/golden_set.yaml, conftest.py, test_retrieval_quality.py, thresholds.yaml}`
- `docs/operations/{atomicity,observability,backup-restore}.md`
- `docs/eval/golden-set.md`

---

## End-to-end verification

After each phase:
1. `task lint && task typecheck && task test` must be green.
2. `task test-integration` must be green (services must be up via `task up`).
3. `task doctor` must exit 0 after a fresh `task ingest`.
4. `task smoke` + `task smoke-graph` must return reasonable results.

After phase B:
5. `task eval` must pass thresholds on current corpus.

After phase C:
6. A `task ask -- "..."` invocation must produce a single trace spanning: agent → MCP → tool → Qdrant, with trace_id appearing in JSON logs.

After phase D:
7. Round-trip drill: `task backup`, stop services, wipe volumes, restart, `task restore ...`, re-run smoke + doctor — identical results.
