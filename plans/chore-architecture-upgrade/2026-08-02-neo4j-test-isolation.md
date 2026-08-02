# Isolate integration tests from the dev Neo4j via an ephemeral container

## Context

`task test-integration` currently destroys the developer's knowledge graph. Two
fixtures wipe the shared Neo4j instance:

- `tests/conftest.py:198,205` — `MATCH (n) DETACH DELETE n`, before *and* after
  every test using the `neo4j_driver` fixture.
- `tests/test_graph_writer.py:20` — deletes every `Document`/`Chunk`/`Entity`.

This was observed live: after an integration run, `task smoke-graph` returned
nothing and the graph had to be rebuilt with `task ingest -- --full-reprocess`.
The same class of bug was already fixed for Qdrant by pointing tests at `test_*`
collections (`config/constants.py`, `tests/conftest.py`).

Neo4j has no equivalent cheap namespace: the stack runs `neo4j:5.26-community`,
and multi-database is Enterprise-only (verified — only `neo4j` and `system`
exist). Scoping the delete to test-created nodes was considered and rejected: it
stops the destruction but not the *interference* — tests `MERGE` on
`Entity {name}` and would keep mutating real nodes, and the constraint collision
already seen (`Node already exists with label Entity and property name='team'`)
happens before any cleanup.

**Chosen approach:** an ephemeral Neo4j started and destroyed by the test run
itself (Testcontainers). Nothing permanent is added to `docker-compose.yml` and
there is nothing to remember to start. The existing full-wipe fixtures become
correct as written, because the database then genuinely belongs to the tests.

## Changes

### 1. Dev dependency — `pyproject.toml`

Add to `[dependency-groups] dev`:

```toml
"testcontainers[neo4j]>=4.8",
```

Import path is `testcontainers.neo4j` (a shim); the implementation now lives at
`testcontainers.community.neo4j`. Verify which import resolves for the pinned
version and use that.

### 2. Container fixture — `tests/conftest.py`

Session-scoped, mirroring production config:

```python
@pytest.fixture(scope="session")
def neo4j_container():
    from testcontainers.neo4j import Neo4jContainer

    container = (
        Neo4jContainer("neo4j:5.26-community", password=os.environ["NEO4J_PASSWORD"])
        .with_env("NEO4J_PLUGINS", '["apoc"]')
        .with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*")
    )
    with container as c:
        yield c
```

APOC is **mandatory**, not optional: `ingestion/neo4j_setup.py:47` raises
`RuntimeError("APOC plugin not available")`, and both `graph_engine.py:327` and
`graph_writer.py:327` call `apoc.merge.relationship`. It needs no network —
`apoc-5.26.28-core.jar` ships inside the image under `/var/lib/neo4j/labs/` and
the entrypoint copies it into `plugins/` when `NEO4J_PLUGINS` is set (verified
against the running container).

`Neo4jContainer` handles readiness itself (log wait on "Remote interface
available at", then `verify_connectivity()`), so no manual polling is needed.

### 3. Redirect settings + reset cached clients — `tests/conftest.py`

A session-scoped **autouse** fixture that only starts the container when the
session actually collected integration tests, so `task test` stays fast:

```python
@pytest.fixture(scope="session", autouse=True)
def _use_ephemeral_neo4j(request):
    if not any(i.get_closest_marker("integration") for i in request.session.items):
        yield
        return
    container = request.getfixturevalue("neo4j_container")
    from config.settings import settings
    settings.neo4j_uri = container.get_connection_url()
    _reset_neo4j_singletons()
    # provision schema once — see section 4
    yield
```

Autouse is required, not optional. The graph engine is reachable *transitively*
from tests that request no Neo4j fixture — `tests/eval/test_retrieval_metrics.py`
reaches it via `multi_search` → `graph_search`, which is exactly how it poisoned
the event loop earlier. A non-autouse fixture would leave that path pointed at
the real database.

Two caches must be reset after the URI changes, because both capture it:
- `ingestion.graph_engine._engine` — `GLiNEREngine.__init__` copies
  `settings.neo4j_uri` into `self._neo4j_uri` (`graph_engine.py:168`).
- `ingestion.graphiti_client._client` (and `_graphiti_embedder`).

All other consumers read `settings.neo4j_uri` at call time (`neo4j_setup.py:15`,
`graph_writer.py:249`, `test_integration_live_e2e.py:117,196`), so mutating the
settings singleton is sufficient — no pinned port or pre-import env needed. If
the pydantic `Settings` instance turns out to reject assignment, use a manually
managed `pytest.MonkeyPatch` (session fixtures cannot take the function-scoped
`monkeypatch`).

### 4. Provision schema once against the fresh container

A brand-new container has no constraints and, critically, no `entity_fulltext`
index — `GLiNEREngine.search` runs
`db.index.fulltext.queryNodes('entity_fulltext', …)` and fails without it. Call
the existing `ingestion.neo4j_setup.create_neo4j_schema(driver)` once in the
session fixture. The per-test `neo4j_driver` fixture already calls it, but tests
that never request that fixture (graph search paths) would otherwise run against
an unindexed database.

### 5. Leave the wipes alone

`MATCH (n) DETACH DELETE n` in `tests/conftest.py` and the
`Document`/`Chunk`/`Entity` delete in `tests/test_graph_writer.py:20` stay as
they are — they are correct once the database is the tests' own, and per-test
wipes are still wanted for isolation *between* tests.

### 6. Remove abandoned artifact

Delete `tests/test_neo4j_scope.py`. It was written for the scoped-delete
approach and imports `tests.neo4j_scope`, a module that will now never exist.

### 7. Docs

Brief note in `docs/deployment/local-development.md` (and the testing section of
`CLAUDE.md`) that integration tests run against an ephemeral Neo4j and never
touch the dev graph, alongside the existing `test_*` Qdrant collection note.

## Known follow-up (flagging, not fixing here)

`tests/eval/test_retrieval_metrics.py` is `@pytest.mark.integration`, so it runs
in the integration suite — now against empty isolated Qdrant collections and an
empty ephemeral Neo4j — while scoring recall/nDCG/MRR against a labelled set that
references `arxiv.pdf`. It passes today only because `recall_at_10`, `ndcg_at_10`
and `mrr` are commented out in `tests/eval/thresholds.yaml` as "NON-GATING until
the retrieval set grows". When those bars are enabled they will measure an empty
corpus and fail. It should either be `eval`-marked only (run against the real
corpus via `task eval`) or seed its own fixture corpus. Worth a decision, but out
of scope for this change.

## Verification

1. `task test` — 445 unit tests pass, and **no container starts** (no
   integration items collected).
2. `task test-integration` — 50 passed, 2 skipped, same as now.
3. **Isolation proof** (the point of the change): record Neo4j node counts by
   label and Qdrant `documents_dense` point count before the run; re-check after.
   Both must be unchanged.
   ```
   MATCH (n) RETURN labels(n)[0] AS label, count(n) ORDER BY label
   curl -s localhost:6333/collections/documents_dense | jq .result.points_count
   ```
4. `task smoke-graph -- 'robot communication'` still returns facts after the
   integration run — this is the regression that started this work; it returned
   `(no results)` before.
5. `task doctor` — both documents still in sync.
6. `uv run ruff check .` clean, and `mypy` on the touched files shows no new
   errors (the repo is already red for unrelated reasons — compare before/after).
7. Docker-unavailable path: confirm the failure message names the missing
   container rather than surfacing a raw connection error.
