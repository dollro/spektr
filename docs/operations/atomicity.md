# Ingestion Failure Semantics

Spektr's ingestion pipeline crosses three stores (Qdrant, Neo4j, Postgres) that don't share a transaction. We use idempotent writes + a poison-pill pattern to make partial failures safe to retry.

## The contract

Every call to `ingest_file` in `ingestion/pipeline.py` has exactly three outcomes:

1. **Success.** Points land in Qdrant, entities in Neo4j, CocoIndex marks the row processed. The failure counter (if any) for this file is reset.
2. **Transient failure.** The exception propagates out of `ingest_file`. CocoIndex does not write a tracking row — the next `task ingest` retries. Deterministic chunk UUIDs (UUIDv5 on `source_file::pX::cY`) mean re-runs upsert over prior partial writes; no duplicates.
3. **Poison pill.** After `PIPELINE_MAX_RETRIES` (default 3) consecutive failures for the same file, the exception is swallowed, a CRITICAL log is emitted (`"POISON PILL: <file> failed N times"`), and CocoIndex is allowed to mark the row processed so the rest of the batch proceeds.

## Where the counter lives

`ingestion/_failure_tracker.py` persists counts to `state/ingestion_failures.db` (stdlib sqlite3, gitignored, per-host). Schema:

```sql
CREATE TABLE ingestion_failures (
    source_file TEXT PRIMARY KEY,
    fail_count  INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    last_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The table is thread-safe via a module lock. Counts survive process restart.

## Responding to a CRITICAL log

A `POISON PILL` log means one file has permanently failed and was skipped to unblock the batch. Investigate the `last_error`:

```bash
docker compose exec -T postgres psql -U cocoindex -d cocoindex -c \
  "SELECT source_key FROM ragingestion__cocoindex_tracking;"

sqlite3 state/ingestion_failures.db \
  "SELECT source_file, fail_count, last_error FROM ingestion_failures;"
```

Typical fixes:

| Cause | Action |
|-|-|
| Corrupt PDF / unsupported file | Remove the source file. |
| Missing dependency (e.g. `gliner2`) | `uv sync --extra gliner`, then reset counter. |
| LLM / embedder API outage | Wait for the provider, then reset counter. |
| Schema drift in downstream store | Fix the schema, then reset counter. |

## Resetting a file to retry

After fixing the root cause:

```bash
# 1. Clear the failure counter
sqlite3 state/ingestion_failures.db \
  "DELETE FROM ingestion_failures WHERE source_file = 'bad.pdf';"

# 2. Run ingest
task ingest
```

`task doctor-fix` only deletes **orphan tracking rows** — rows tracked by CocoIndex but missing from Qdrant. After a poison-pill, the file is *marked processed* in CocoIndex's tracking table but has zero (or partial) Qdrant points; `doctor-fix` won't necessarily reprocess it. To force reprocessing, manually drop the tracking row:

```bash
docker compose exec -T postgres psql -U postgres -d cocoindex -c \
  "DELETE FROM ragingestion__cocoindex_tracking WHERE source_key='\"bad.pdf\"';"
```

Note the jsonb-quoted `source_key` (the inner double-quotes are part of the value).

`task doctor` alone (without `--fix`) is safe to run any time — it reports drift but doesn't mutate state.

## When the invariant is violated

If you ever see a CocoIndex tracking row for a file that has zero points in Qdrant and `task doctor` doesn't report it as drift, something in the pipeline has grown a new silent-catch branch. Grep for `except` blocks in `ingestion/pipeline.py` — they should all either re-raise or call `_failure_tracker.record_failure` explicitly.

See the unit tests in `tests/test_pipeline_atomicity.py` for the exact assertions.
