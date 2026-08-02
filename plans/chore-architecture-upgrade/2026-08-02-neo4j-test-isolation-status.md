# Neo4j test isolation — implementation status

Companion to `2026-08-02-neo4j-test-isolation.md`. Records what landed, what is
still in flight, and what could not be checked in this sandbox.

Date: 2026-08-02 · Branch: `chore/architecture-upgrade`

---

## Working tree state — resolved

The baseline experiment described below is **finished** and `tests/conftest.py`
has been **restored to the implemented version**. `git status` shows it as
modified; `grep -c neo4j_container tests/conftest.py` returns 2. Unit tests
re-run green (445 passed) and ruff is clean against the restored file. Nothing
outstanding here.

---

## What was implemented

All seven sections of the plan are complete.

| § | Change | File | Notes |
|---|---|---|---|
| 1 | `testcontainers[neo4j]>=4.8` dev dep | `pyproject.toml`, `uv.lock` | Import path resolved to **`testcontainers.community.neo4j`**. The `testcontainers.neo4j` shim the plan mentions still exists but now emits a `DeprecationWarning`, so the code imports the real module directly. |
| 2 | Session-scoped `neo4j_container` fixture | `tests/conftest.py` | `neo4j:5.26-community`, password from `settings.neo4j_password`, `NEO4J_PLUGINS='["apoc"]'` + unrestricted `apoc.*`. Startup failure is caught and re-raised via `pytest.fail(..., pytrace=False)` naming the image and explaining why a throwaway container is used. |
| 3 | Autouse redirect + cache reset | `tests/conftest.py` | `_use_ephemeral_neo4j` returns early unless the session collected `integration` items. Mutates `settings.neo4j_uri`, then nulls `ingestion.graph_engine._engine`, `ingestion.graphiti_client._client` and `._graphiti_embedder`. |
| 4 | Schema provisioned once | `tests/conftest.py` | `_provision_test_schema()` calls `ingestion.neo4j_setup.create_neo4j_schema`, driven from the sync fixture with `asyncio.run` so no session-scoped event loop is needed. |
| 5 | Wipes left alone | — | `MATCH (n) DETACH DELETE n` and the `test_graph_writer.py` delete are unchanged, as specified. |
| 6 | Abandoned artifact removed | `tests/test_neo4j_scope.py` | Deleted. It was untracked, so this produces no git diff. |
| 7 | Docs | `docs/deployment/local-development.md`, `CLAUDE.md` | Testing sections now state that integration tests use an ephemeral Neo4j and never touch the dev graph, alongside the existing `test_*` Qdrant note. |

### Two decisions that differed from the plan text

1. **Import path** — `testcontainers.community.neo4j`, not the deprecated shim.
   The plan explicitly asked to verify which one resolves; this is the answer.
2. **`settings.neo4j_uri` assignment works.** The plan's fallback to a manually
   managed `pytest.MonkeyPatch` was not needed — `Settings` does not set
   `validate_assignment`, so plain attribute assignment is fine.

### One addition beyond the plan

A last-line-of-defence assertion was added to the `neo4j_driver` fixture,
mirroring the one already in `qdrant_client`:

```python
assert _EPHEMERAL_NEO4J_URI is not None and settings.neo4j_uri == _EPHEMERAL_NEO4J_URI
```

**Why:** without it there is a live hole. A Neo4j test that forgets
`@pytest.mark.integration` never triggers the autouse container, and the
per-test wipe then lands on the developer's real graph — precisely the
regression this work exists to prevent. The Qdrant path already has this guard
plus `tests/test_collection_isolation.py`; this restores the symmetry.

**Caveat:** the guard only covers the `neo4j_driver` fixture. The separate
`writer` fixture in `tests/test_graph_writer.py` builds its own `GraphWriter`
and wipes `Document`/`Chunk`/`Entity` with no such check. Worth extending, but
it was not in scope here.

---

## Verification — what passed

| # | Check | Result |
|---|---|---|
| 1 | `task test` | ✅ **445 passed**, 53 deselected, 18.8s. **No container started** — matches the plan's expected count exactly. |
| 2 | `task test-integration` runs | ✅ Ephemeral container + `testcontainers/ryuk` reaper started alongside the dev stack and cleaned themselves up afterwards. 12 failures, all pre-existing — see below. |
| 3 | **Isolation proof** | ✅ **The point of the change, and it holds.** |
| 6 | `uv run ruff check .` | ✅ Clean. `ruff format --check tests/conftest.py` also clean. |
| 6 | `mypy tests/conftest.py` | ✅ No new errors. The only error in the file is the pre-existing line-107 `_make_mock_extraction` list-item complaint; the repo's other 32 errors are unrelated and unchanged. |
| 7 | Docker-unavailable path | ⚠️ Message written and reviewed, but **not exercised** — Docker was available throughout. Untested code path. |

### Detail on the isolation proof

This sandbox had no ingested corpus (Qdrant had zero collections; Neo4j was
empty), so the plan's before/after count comparison would have been vacuous.
Sentinel data was seeded into the dev Neo4j instead:

```
Document {source_key: 'dev/arxiv.pdf'} -[:HAS_CHUNK]-> Chunk {id: 'dev-sentinel-chunk'}
                                                        -[:MENTIONS]-> Entity {name: 'dev-sentinel-entity'}
```

The baseline experiment turned this into a direct A/B, since it ran the same
suite against the pre-change conftest with the same sentinel data present:

| Run | Conftest | Dev Neo4j label counts afterwards |
|---|---|---|
| Post-change | implemented | `Chunk: 1, Document: 1, Entity: 1` — **unchanged** |
| Baseline | `HEAD` | `{}` — **all three destroyed** |

So the regression is reproduced and the fix is demonstrated against it, rather
than merely asserted.

---

## Integration failure count — resolved, pre-existing

The integration run produced **12 failed, 38 passed, 2 skipped**, not the
plan's expected "50 passed, 2 skipped".

**A baseline run against `HEAD`'s conftest produced exactly the same result:
12 failed, 38 passed, 2 skipped, same twelve test IDs.** The failures predate
this change and are environmental. (Baseline 1069s, post-change 1083s — the
~14s delta is container startup plus schema provisioning.)

Every failure traces to the sandbox environment rather than to this change:

| Failing tests | Cause |
|---|---|
| `test_embedder.py` ×3 | `403 Blocked by network policy: domain api.jina.ai:443` |
| `test_entity_extractor.py::test_real_llm_extraction` | `openai.PermissionDeniedError` — same policy |
| `test_integration_ingestion.py` ×3 | fastembed cannot download the sparse model (blocked) |
| `test_integration_live_e2e.py` ×1, `test_integration_live_ingest.py` ×3 | require embeddings — same root cause |
| `test_graph_engine_loops.py::test_engine_search_works_across_event_loops` | `ModuleNotFoundError: No module named 'gliner2'` — this `.env` sets `GRAPH_ENGINE=gliner` and the optional dep is not installed |

**None are Neo4j-connectivity or schema failures**, which is what a broken
ephemeral container would produce.

On a machine with network access and `gliner2` installed, expect the plan's
50 passed, 2 skipped.

---

## Not verifiable in this sandbox

These plan verification steps need the real corpus and outbound API access,
both unavailable here. They should be run on a normal dev machine before merge.

- **Step 4** — `task smoke-graph -- 'robot communication'` still returns facts
  after an integration run. This is the regression that started the work; it
  returned `(no results)` before. **This is the single most valuable
  outstanding check.**
- **Step 5** — `task doctor` reports both documents in sync.
- **Step 7** — stop Docker, run `task test-integration`, confirm the failure
  message names the missing container rather than surfacing a raw
  `ServiceUnavailable` from the Neo4j driver.

Also unrun: `task eval` / `task eval-retrieval` (need an LLM key and the
ingested corpus).

---

## Known follow-ups — flagged, deliberately not fixed

1. **`tests/eval/test_retrieval_metrics.py` now scores an empty corpus.** It is
   `@pytest.mark.integration`, so it runs in the integration suite against
   isolated Qdrant collections *and* — new as of this change — an empty
   ephemeral Neo4j, while scoring recall/nDCG/MRR against a labelled set
   referencing `arxiv.pdf`. It passes today only because `recall_at_10`,
   `ndcg_at_10` and `mrr` are commented out in `tests/eval/thresholds.yaml` as
   "NON-GATING until the retrieval set grows". **When those bars are enabled
   they will measure nothing and fail.** It should either become `eval`-marked
   only, or seed its own fixture corpus. This was already flagged in the plan
   as out of scope; the ephemeral Neo4j makes it slightly more acute, since
   previously the graph half of the score came from real dev data.

   Note `task eval-retrieval` runs `pytest ... -m integration`, so it now gets
   the ephemeral database too.

2. **`tests/test_graph_writer.py`'s `writer` fixture has no isolation guard**
   (see "One addition beyond the plan").

3. **Integration suite wall time is ~18 minutes** in this sandbox, most of it
   spent in network timeouts and retries against blocked hosts. Not
   representative of a connected machine, but worth watching.

4. **Container startup adds ~10s** to every integration run. Acceptable, and
   unit runs are unaffected.

---

## Files touched

```
pyproject.toml                              # + testcontainers[neo4j]>=4.8
uv.lock                                     # regenerated
tests/conftest.py                           # container + autouse redirect + guard  ⚠️ SEE TOP
tests/test_neo4j_scope.py                   # deleted (was untracked)
docs/deployment/local-development.md        # testing section note
CLAUDE.md                                   # test-isolation paragraph
```

`git status` will show more modified files than this — `.env.example`,
`Taskfile.yml`, `config/constants.py`, `config/settings.py`,
`ingestion/graph_engine.py`, `tests/test_backup.py`,
`tests/test_qdrant_setup.py`, `tests/test_collection_isolation.py`,
`tests/test_graph_engine_loops.py` — all of which were already modified or
untracked before this work started and are **not** part of this change.
