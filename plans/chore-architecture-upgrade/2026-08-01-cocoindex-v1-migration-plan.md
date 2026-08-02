# CocoIndex v0 → v1 Migration — Implementation Plan

Date: 2026-08-01
Branch: `chore/architecture-upgrade`
Follows: [`2026-08-01-cocoindex-v1-vs-v0.md`](./2026-08-01-cocoindex-v1-vs-v0.md),
[`2026-08-01-cocoindex-v1-sqs-and-voice-research.md`](./2026-08-01-cocoindex-v1-sqs-and-voice-research.md)

**Verification basis.** The research docs cite a local upstream clone at `/home/rodo/Coding/cocoindex`
which **does not exist in this environment**. This plan is instead verified against the real
**cocoindex 1.0.18 wheel**, installed and read at
`…/scratchpad/ccv1/cocoindex/`. All API claims below carry `file:line` citations into that tree.
Three findings correct the research docs — see [§0](#0-corrections-to-the-research-docs).

---

## 0. Corrections to the research docs

| Research claim | Verified reality | Impact |
|-|-|-|
| Check `handle.stats().total.num_errored` | The field is **`num_errors`**; `UpdateStats.total` is a property summing `by_component` (`_internal/update_stats.py:62,84-96`) | Trigger loop uses `num_errors` |
| §4b: "embedders must implement `__coco_vector_schema__`" | **Not required.** `VectorSchema` satisfies `VectorSchemaProvider` itself (`resources/schema.py:26-30`), so `QdrantVectorDef(schema=VectorSchema(dtype=np.dtype(np.float32), size=dim))` works directly | Embedders are **untouched**. One work item deleted. |
| `managed_by=USER` "prevents `delete_collection`" | Stronger: `resolve_system_transition` returns `None` for a user-managed desired state (`connectorkits/statediff.py:114-146`), so CocoIndex issues **no collection action at all** — no create, no replace, no drop | `ensure_collections()` remains the sole provisioning authority; confirmed mandatory |

Three further facts the research docs did not have:

1. **`cocoindex.inspect` exists** (`inspect.py` → `_internal/inspect_api.py`) and exposes
   `iter_stable_paths_by_name(env, app_name)` / `iter_target_states_by_name(...)` — read access to
   the LMDB ledger **without importing the app**. This is the direct replacement for `doctor.py`'s
   `psql` query, and it is a *better* one. `scripts/doctor.py`'s rewrite is smaller than feared.
2. **Mounted-component failures are swallowed by default** (`_internal/component_ctx.py:163-167`) —
   logged at ERROR, never propagated. This is *exactly* the poison-pill's "keep the batch moving"
   half, now free. But it also means `app.update()` returns success on a wholly-failed run;
   `num_errors` must be checked explicitly.
3. **`declare_point` performs no schema validation** (`connectors/qdrant/_target.py:549-564`) —
   only the point *id* is checked (u64 or UUID; `_target.py:737-765`). A `{"dense": [...],
   "sparse": SparseVector(...)}` dict passes through untouched to `client.upsert`. Spektr's
   existing uuid5 string ids are already valid.

---

## 1. Scope

**In scope — Track 1 (Path A, bulk ingestion) only.**

Per the research recommendation, Path B (`ingestion/live_ingest.py`) is **not touched**. It imports
no CocoIndex, its requirement is low latency, and CocoIndex adds latency. `LiteLLMTranscriber`
adoption is a separate, independent product decision and is **not** part of this migration.

Out of scope, explicitly: audio/STT ingestion, the native Neo4j connector (Graphiti stays), Kafka,
any change to retrieval (`retrieval/`, `server/`).

---

## 2. Target architecture

```
                      ┌─────────────────────────────────────────┐
  DOCUMENT_SOURCE     │           ingestion/app.py              │
  ─────────────       │  coco.App("RagIngestion", app_main)     │
  local / sharepoint ─┤  @coco.lifespan: db_path, qdrant,       │
    localfs.walk_dir  │    embedder, s3 client (ContextKeys)    │
    (live=True)       │                                          │
  s3 ────────────────►│  app_main:                               │
    amazon_s3.        │    dense = await qdrant.mount_collection │
    list_objects      │             _target(..., managed_by=USER)│
                      │    multivec = … (if enabled)             │
                      │    await coco.mount_each(process_file,   │
                      │              source.items(), dense, mv)  │
                      └──────────────────┬──────────────────────┘
                                         │
             ┌───────────────────────────▼──────────────────────────┐
             │  @coco.fn(memo=True) process_file(file, dense, mv)   │
             │   → file_to_pages → chunk → embed                    │
             │   → dense.declare_point(...)   ← reconciled per id   │
             │   → mv.declare_point(...)                            │
             │   → graph_engine.ingest(...)   ← side effect         │
             │   → declare_graph_state(source_key, fp) ← cleanup   │
             └──────────────────────────────────────────────────────┘

  live triggering:
    local/sharepoint → app.update(live=True)         (native watcher)
    s3               → ingestion/sqs_trigger.py       (Option E)
                       long-poll SQS → debounce → app.update()
                       → check stats().total.num_errors → delete msgs
                       + 24h interval sweep + startup run
```

### Module layout (new/changed)

| File | Status | Purpose |
|-|-|-|
| `ingestion/app.py` | **new** | `coco.App`, lifespan, ContextKeys, `app_main`, source selection |
| `ingestion/qdrant_target.py` | **new** | `CollectionSchema` construction, point declaration helpers |
| `ingestion/graph_target.py` | **new** | Custom `TargetHandler` for graph cleanup — replaces `target_connector.py` |
| `ingestion/sqs_trigger.py` | **new** | Option E trigger daemon |
| `ingestion/runner.py` | **new** | `run_pipeline(live=...)` — provisioning + mode selection. Keeps `pipeline.py` under the 600-line cap. |
| `ingestion/pipeline.py` | **rewritten (partly)** | Per-file processing only: `process_file` impl, page/chunk helpers. Flow def, `run_pipeline`, `_get_qdrant_client` move out. |
| `ingestion/target_connector.py` | **deleted** | Qdrant half → native connector; graph half → `graph_target.py` |
| `ingestion/cocoindex_ops.py` | **changed** | Drop `@cocoindex.op.function()`; plain async helpers (were never wired into the flow) |
| `ingestion/_utils.py` `run_async` | **retained** | Still used by `runner.py` for provisioning; no longer used per file |
| `scripts/doctor.py` | **changed** | `psql` → `cocoindex.inspect`; `--fix` semantics change (§7) |
| `scripts/backup.py` / `restore.py` | **changed** | `postgres` target → `cocoindex` target (LMDB dir tar) |
| `config/settings.py` | **changed** | `database_url` removed; `cocoindex_db_path`, SQS trigger settings added |
| `docker-compose*.yml` | **changed** | `postgres` service removed |

---

## 3. Design decisions

### D1 — Points are **declared**, not upserted (native Qdrant connector)

`_process_text_page` / `_process_visual_page` stop calling `qdrant.upsert()` and instead call
`collection.declare_point(PointStruct(...))`. Consequences:

- **Deletion comes free and correct.** Per-point reconciliation is keyed on the point id and emits
  `client.delete(points_selector=PointIdsList(points=[explicit ids]))` only
  (`connectors/qdrant/_target.py:254-284`). There is **no orphan sweep** — verified by reading
  `_apply_actions` and `reconcile` in full. Path B's live-session points are structurally invisible.
- **Upserts are batched by the framework** — one `client.upsert` per flush across many points
  (`_target.py:269-274`), replacing today's per-page upsert calls.
- **No partial writes.** Points land only after the whole component's processing succeeds.
- `_delete_qdrant_points` in `target_connector.py` is deleted.

**`managed_by=target.ManagedBy.USER` is mandatory**, not optional. `ensure_collections()` in
`runner.py` stays the provisioning authority, exactly as today, and CocoIndex never issues a
collection-level action. This is what keeps a future schema change from dropping the collection
(and Path B's live session points with it).

The `schema=` argument is still required even under `USER` (the tracking record carries it,
`_target.py:474-482`), so we build a real `CollectionSchema`:

```python
await qdrant.CollectionSchema.create(vectors={
    DENSE_VECTOR_NAME: qdrant.QdrantVectorDef(
        schema=VectorSchema(dtype=np.dtype(np.float32), size=settings.dense_dimensions),
        distance="cosine",
    ),
    SPARSE_VECTOR_NAME: qdrant.QdrantSparseVectorDef(modifier="idf"),
})
```

Dense and sparse share one dict — verified at `_target.py:181-224`, split into
`vectors_config`/`sparse_vectors_config` only at create time (`_target.py:429-456`), which
`managed_by=USER` never reaches.

### D2 — Graph cleanup via a custom `TargetHandler`

Graphiti stays (decided in the research doc), so graph writes remain side effects inside
`process_file`. Cleanup-on-delete needs an explicit mechanism; v1's only one is
`TargetHandler.reconcile(key, NON_EXISTENCE, …)` — verified: there is no `on_delete` callback on a
plain component (`_internal/live_component.py:377-403`, `_internal/component_ctx.py:46-48`).

`ingestion/graph_target.py` registers a root provider once at import:

```python
_provider = coco.register_root_target_states_provider("spektr/graph", _GraphHandler())
```

`process_file` declares `_provider.target_state(source_key, fingerprint)`. On existence with an
unchanged fingerprint → `reconcile` returns `None` (no action, matching today's no-op upsert). On
`NON_EXISTENCE` → emits a delete action whose async sink dispatches to `_remove_graphiti_episodes`
or `_remove_gliner_entities` — **both functions move over unchanged**, including their log-only
error handling.

One behaviour fix carried across: `_remove_graphiti_episodes` currently calls `close_graphiti()`
after every single delete, tearing down the shared singleton mid-run. Since the sink now receives a
**batch** of deletes, close once after the batch instead.

### D3 — `process_file` is async; embedder shared via `ContextKey`

v0 forced a sync op, so `ingest_file` bridged through `run_async` (a fresh `asyncio.run` per file),
which is why each embedder rebinds its `httpx` client and semaphore to a new loop
(`embedders/jina.py:87-104`) and why a **new embedder is constructed per file**
(`pipeline.py:603`). v1 is async throughout: one loop, one embedder provided in the lifespan via
`EMBEDDER = coco.ContextKey[Embedder]("spektr/embedder")`, closed in the lifespan teardown.

`settings.pipeline_timeout` moves from `run_async(..., timeout=)` to `asyncio.timeout(...)` inside
the impl. Per-file text-page concurrency (`asyncio.Semaphore(2)`) is unchanged.

**Testability.** `@coco.fn`-decorated functions require an active component context, which would
break the existing direct-call unit tests. So the decorator wraps a plain impl:

```python
async def process_file_impl(...) -> None:   # all logic; unit-testable, no coco context
    ...

@coco.fn(memo=True)
async def process_file(file: FileLike, dense, multivec) -> None:
    await process_file_impl(...)
```

`logic_tracking` defaults to `"full"` (`_internal/function.py:1990-2003`), so edits to the impl
still invalidate the memo.

### D4 — File-level concurrency

v1 defaults `max_inflight_components` to **1024** (`_internal/app.py:48-49`), which would fan out
1024 concurrent files against rate-limited embedding APIs. New setting
`pipeline_max_concurrent_files: int = 4`, passed as `AppConfig(max_inflight_components=…)`.

### D5 — Poison-pill semantics preserved 1:1

| Behaviour | v0 | v1 |
|-|-|-|
| Retry next run | re-raise → tracking row not written | re-raise → **no memo entry written** for the call |
| Give up after `PIPELINE_MAX_RETRIES` | swallow → row written | swallow + return → memo entry written, file not retried |
| Batch keeps moving | implicit | framework guarantee (`component_ctx.py:163-167`) |
| Counter | `state/ingestion_failures.db` | unchanged |

The `try/except/else` block moves verbatim into `process_file_impl`. The POISON PILL log message's
remediation text changes (`delete the tracking row` → `cocoindex drop` / `--full-reprocess`).

### D6 — SQS as trigger (Option E), SQS now optional

`ingestion/sqs_trigger.py`, driven by three triggers all calling the same `app.update()`:

```python
await _run_update(app)                                # 1. startup sweep
while True:
    msgs = await _receive(sqs, wait=20)               # 2. long poll (free while idle)
    due = (loop.time() - last_full) > interval        # 3. interval sweep (default 24h)
    if not msgs and not due:
        continue
    if msgs:
        await asyncio.sleep(settings.s3_sqs_debounce_seconds)
        msgs += await _drain(sqs)
    errors = await _run_update(app)                   # returns stats().total.num_errors
    last_full = loop.time()
    if msgs and errors == 0:
        await _delete_batch(sqs, msgs)                # only on a clean run
```

Deliberate: messages are deleted **only** when `num_errors == 0`, so a failed run replays rather
than silently dropping. `app.update()` is re-entrant across calls in one process — the `core.App`
is cached under a lock and each call builds a fresh processor
(`_internal/app.py:248-273,275-324`) — but is not safe *concurrently*, so the loop awaits each
handle's result before the next iteration.

Client: `aiobotocore`, which arrives with the `cocoindex[amazon-s3]` extra
(`METADATA:60`) — no new dependency.

**Settings validator relaxed**: `DOCUMENT_SOURCE=s3` now requires only `S3_BUCKET_NAME`.
`S3_SQS_QUEUE_URL` becomes optional — without it, live mode degrades to interval-only sweeps
(research doc Option B, "defensible on the same arithmetic"). Prod behaviour is unchanged because
prod sets both.

### D7 — State: LMDB replaces Postgres

- New setting `cocoindex_db_path: str = "state/cocoindex.db"` (a **directory**), set on
  `builder.settings.db_path` in the lifespan (`_internal/setting.py`, `environment.py:388-397`).
- It lands inside `/app/state`, which the `ingest_state` volume **already** mounts on exactly the
  three services that run the pipeline — no new volume needed.
- `settings.database_url` deleted; `os.environ.setdefault("COCOINDEX_DATABASE_URL", …)` deleted;
  `postgres` service and `postgres_data` volume removed from both compose files, along with the
  now-dangling `depends_on: postgres` entries (including `sharepoint-sync`'s, which was already
  vestigial — that service imports neither cocoindex nor postgres).
- `model_config` is `extra="ignore"`, so a stale `DATABASE_URL` in someone's `.env` is harmless.

---

## 4. Implementation phases

Each phase leaves the tree lint-clean and type-clean; tests are written alongside, not after.

### Phase 1 — Dependency bump
1. `pyproject.toml`: `cocoindex>=0.3.39,<1.0` → `cocoindex[amazon-s3,qdrant]>=1.0.18,<2.0`.
2. `uv lock && uv sync`; inspect the resolution for conflicts against `qdrant-client`,
   `graphiti-core`, `fastapi`, `pydantic-ai`.
3. Gate: `uv run python -c "import cocoindex; print(cocoindex.__version__)"`.

### Phase 2 — Settings & config
`config/settings.py`: remove `database_url`; add `cocoindex_db_path`,
`pipeline_max_concurrent_files`, `s3_sqs_debounce_seconds` (default 5),
`s3_full_scan_interval_hours` (default 24), `s3_prefix` (default `""`); relax the s3 validator.
Update `.env.example`.

### Phase 3 — Qdrant target module
`ingestion/qdrant_target.py`: `build_dense_schema()`, `build_multivec_schema()`, and thin
`declare_*_point` helpers so `pipeline.py` never imports the connector directly.

### Phase 4 — Graph target module
`ingestion/graph_target.py`: `_GraphHandler(coco.TargetHandler)`, module-level provider
registration, `declare_graph_state(source_key, fingerprint)`. Move `_remove_graphiti_episodes` /
`_remove_gliner_entities` across. Delete `ingestion/target_connector.py`.
Keep `handle_file_delete` as a public re-export so `tests/test_pipeline_delete.py` and
`tests/test_integration_delete.py` keep a supported entry point.

### Phase 5 — Pipeline rewrite
`ingestion/pipeline.py`: `ingest_file` → `async def process_file_impl`; page/visual processors take
target handles instead of a `QdrantClient`; batch the miniCOIL encode (research §4f) by hoisting
`encode_documents(all_texts)` out of `_build_chunk_point`; remove the flow def, `run_pipeline`,
`_get_qdrant_client`.

### Phase 6 — App & runner
`ingestion/app.py` (lifespan, ContextKeys, `app_main`, source selection, `@coco.fn process_file`)
and `ingestion/runner.py` (`run_pipeline`: observability → `ensure_collections` → Neo4j schema →
one-shot / live / SQS-trigger dispatch → `num_errors` reporting → graph engine close).
`ingestion/pipeline.py`'s `__main__` delegates to `runner`, so `python -m ingestion.pipeline
[--live]` and every Taskfile/compose command keep working unchanged.

### Phase 7 — SQS trigger
`ingestion/sqs_trigger.py` per D6.

### Phase 8 — Ops scripts
- `scripts/doctor.py`: `_tracked_files()` via `cocoindex.inspect`; `--fix` per §7.
- `scripts/backup.py` / `restore.py`: `postgres` → `cocoindex` target (tar the LMDB dir; require the
  writer stopped — LMDB has no safe hot-copy).

### Phase 9 — Infra & docs
Compose files, then the 12+ docs pages inventoried in the ops report
(`docs/ingestion/cocoindex.md`, `docs/operations/{atomicity,backup-restore,reindex}.md`,
`docs/configuration/{infrastructure,environment}.md`, `docs/deployment/production.md`,
`docs/ingestion/{s3-sqs-setup,sharepoint-setup}.md`,
`docs/architecture/{overview,data-flow}.md`), plus `CLAUDE.md`'s production-contracts section.

### Phase 10 — Tests & review
§6 below, then a code review pass.

---

## 5. Risks

| Risk | Severity | Mitigation |
|-|-|-|
| **Memoized components must replay their declared target states**, or unchanged files' points get reconciled to non-existence on the next run | **High** — silent data loss | This is CocoIndex's core contract, but it is not verified by reading Python (the ledger is Rust-side). **Must be verified empirically** by an integration smoke test: ingest → re-run → assert point count unchanged and `num_adds == 0`. Recorded as a blocking gate, not an assumption. |
| S3 source has thinner production mileage (issue #2111, memo-state crash) | Medium | Interval sweep + startup run make a missed/failed event self-healing; failures are visible via `num_errors` |
| `documents_multivec` uses a *multi*-vector; `MultiVectorSchema` path less exercised | Low | `multivec_enabled` defaults to `False`; the target is only mounted when enabled |
| LMDB 4 GiB default map size | Low | `COCOINDEX_LMDB_MAP_SIZE` env var exists (`_internal/setting.py`); document it |
| Concurrent LMDB access from two containers | Medium | Only `ingest`/`ingest-live` run the app, and they are not run simultaneously; document the constraint |

---

## 6. Test plan

**Unit (runs under `task test`, no services):**
- `test_graph_target.py` — `reconcile` returns `None` for unchanged fingerprint; emits a delete
  action for `NON_EXISTENCE` with prior records; returns `None` for `NON_EXISTENCE` with no prior
  records; sink dispatches to graphiti vs gliner per `settings.graph_engine`.
- `test_sqs_trigger.py` — startup sweep runs once; no messages + not due → no update; messages →
  debounce + update + delete; `num_errors > 0` → **messages not deleted**; interval elapsed with no
  messages → update.
- `test_qdrant_target.py` — dense schema carries both named vectors and the IDF modifier;
  multivec schema only built when enabled.
- `test_pipeline_atomicity.py` — **updated** for the async impl; contract assertions unchanged.
- `test_pipeline_*.py` (chunking, dual_embed, bulk_graphiti, vlm_graphiti) — **updated** to assert
  against declared points instead of `qdrant.upsert` calls.
- `test_doctor.py` — **updated**: `cocoindex.inspect` mocked in place of `subprocess`/psql.
- `test_backup.py` — **updated** for the `cocoindex` target.
- `test_document_source_validation.py` — **updated** for the relaxed s3 validator.
- `test_target_connector.py` — **replaced** by `test_graph_target.py`.

**Integration (`task test-integration`, needs Docker):**
- `test_integration_cocoindex_v1.py` — **new**: real app over a temp dir with a stub embedder into
  a real Qdrant. Asserts (a) points appear, (b) re-run is a no-op (`num_adds == 0`, count stable) —
  *the D-risk gate*, (c) deleting the source file removes exactly its points and leaves a
  hand-inserted `session:*` point untouched.

**Known environment limits (recorded honestly, not worked around):**
- `huggingface.co` is blocked by network policy, so `fastembed` cannot fetch `Qdrant/minicoil-v1`.
  **4 tests already fail on `main` for this reason** — `test_pipeline_bulk_graphiti.py::…`,
  `test_pipeline_chunking.py::…` ×3. Baseline: **4 failed, 413 passed**.
- The eval gate (`task eval-retrieval`) needs a live Docker stack, real embedding-provider keys,
  and a freshly ingested corpus; its recall/nDCG/MRR floors are currently **commented out** in
  `tests/eval/thresholds.yaml` and non-gating. It cannot be run here. The commands to run it
  post-merge are documented in the final report.

---

## 7. `scripts/doctor.py` after the migration

The drift doctor's premise changes. Today, `tracked − indexed` drift is possible precisely
*because* upserts are side effects CocoIndex knows nothing about. Once points are declared, that
class of drift is structurally eliminated — CocoIndex owns the points.

What remains meaningful:

| Check | Post-migration |
|-|-|
| Tracked source files | via `cocoindex.inspect.iter_stable_paths_by_name` — informational |
| `tracked − indexed` | should be empty by construction; still reported (a non-empty result means processing errored) |
| `indexed − tracked` | **still real** — v0 leftovers, manual data, test fixtures. Excludes `is_live` session points, which are legitimately untracked. |
| Mixed embedder model/dim, missing sparse | unchanged, Qdrant-side, source-agnostic |

`--fix` changes meaning: it no longer deletes tracking rows (there is no SQL to delete). It deletes
**orphan Qdrant points** (`indexed − tracked`, excluding live-session points), which is the drift
CocoIndex cannot self-heal, still behind the existing `--yes` gate. For `tracked − indexed` it
prints the v1 remediation (`task ingest -- --full-reprocess`). The introspection call is written
defensively: if it yields nothing, the diff is skipped with a clear message rather than crashing.

---

## 7b. Found during implementation (not anticipated by §3)

Three defects the plan did not foresee, all discovered by running the code rather than reading it.

**1. `source_file` came out as an absolute path.** `localfs` builds `FileLike.file_path.path`
from the walked root, so `str(file.file_path.path)` yields `/app/documents/arxiv.pdf`, not
`arxiv.pdf`. v0's `filename` field was relative. Every Qdrant payload, `list_documents`, the
delete path and `tests/eval/retrieval_set.yaml` are keyed on it — this would have invalidated the
entire corpus and the eval fixtures. Fixed by `ingestion/app.py::source_key()`, which strips the
configured root (`amazon_s3` already yields prefix-stripped relative paths).

**2. A swallowed page-embedding failure now *deletes* data.** Under v0 these points were upserted
directly, so `_process_text_page`'s `except: log; return` merely skipped an upsert and previously
written points survived. Under v1 a page that is not *declared* is reconciled to non-existence —
and returning normally memoizes the file, so it would never retry. A transient 429 outliving the
embedder's retries would silently delete a page's chunks and mark the file done. The three
embedding handlers in `ingestion/page_processor.py` now re-raise. Verified empirically
(`test_failing_file_keeps_its_previous_points`): a failing component **keeps** its previously
declared points and surfaces via `num_errors`, so raising is both safe and visible.

**3. `doctor --fix` could wipe the corpus.** With `--fix` repurposed to delete orphan Qdrant
points (§7), an empty-but-*readable* ledger makes every indexed document look orphaned — which is
exactly the state after the `rm -rf state/cocoindex.db` reindex the docs now recommend.
`task doctor-fix` would have deleted everything. `scripts/doctor.py` now refuses to fix when
`tracked` is empty. The pre-existing `None` guard only covered an *unreadable* ledger.

Also worth recording: §5's headline risk — "memoized components must replay their declared target
states" — is **confirmed**, not assumed. `tests/test_integration_cocoindex_v1.py` shows run 2 over
unchanged files reports `num_adds == 0` with the point count unchanged.

## 8. Rollback

`git revert` of the migration commit(s) restores v0 code, but **not** v0 state: the Postgres
tracking table is untouched by this work, so a revert plus `docker compose up postgres` resumes
where v0 left off. The LMDB directory is disposable — deleting `state/cocoindex.db` and re-running
is always safe (it costs a full reprocess, not data loss, since Qdrant points are keyed
deterministically by uuid5).
