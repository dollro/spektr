# Ingestion Failure Semantics

Spektr's ingestion pipeline crosses three stores (Qdrant, Neo4j, and CocoIndex's own LMDB ledger) that don't share a transaction. We use idempotent writes + a poison-pill pattern to make partial failures safe to retry.

To exercise these semantics against a known-clean environment, start from [First Ingest](first-ingest.md).

## The contract

Every call to `process_file_impl` in `ingestion/pipeline.py` has exactly three outcomes:

1. **Success.** Declared points are flushed to Qdrant, entities land in Neo4j, CocoIndex writes the file's memoization entry. The failure counter (if any) for this file is reset.
2. **Transient failure.** The exception propagates out of `process_file_impl`. CocoIndex writes **no memoization entry** for a call that raised — the next `task ingest` re-processes the file. Deterministic chunk UUIDs (UUIDv5 on `source_file::pX::cY`) mean re-runs reuse the same point ids; no duplicates.
3. **Poison pill.** After `PIPELINE_MAX_RETRIES` (default 3) consecutive failures for the same file, the exception is swallowed and a CRITICAL log is emitted (`"POISON PILL: <file> failed N times"`). Returning normally writes the memoization entry, so CocoIndex will not retry the file.

Two things the framework gives us for free on top of that: a component that raises is logged and swallowed by CocoIndex itself, so one bad file never aborts the batch; and because points are *declared* rather than upserted mid-flight, nothing reaches Qdrant until the whole file has processed successfully — a partially-written document is structurally impossible.

Since `app.update()` does not raise for per-file failures, `ingestion/runner.py` reads `handle.stats().total.num_errors` and the process exit code reflects it. A non-zero exit from `task ingest` means at least one file errored.

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
sqlite3 state/ingestion_failures.db \
  "SELECT source_file, fail_count, last_error FROM ingestion_failures;"

task doctor          # what CocoIndex tracks vs what Qdrant actually holds
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

# 2. Force CocoIndex to reprocess (the poison-pilled file has a memo entry,
#    so a plain `task ingest` would skip it)
task ingest -- --full-reprocess
```

`--full-reprocess` invalidates the memoization cache for the whole app, so every file is re-read and re-embedded. There is no per-file equivalent; the nuclear option is deleting the state directory (`COCOINDEX_DB_PATH`, default `state/cocoindex.db`), which forces a full reprocess of everything.

`task doctor-fix` does **not** touch the ledger — it deletes Qdrant points that no CocoIndex run declared. It will not reprocess a poison-pilled file.

`task doctor` alone (without `--fix`) is safe to run any time — it reports drift but doesn't mutate state.

## When the invariant is violated

Under the declared-target model, "tracked by CocoIndex but missing from Qdrant" should be structurally impossible: CocoIndex owns the points. If `task doctor` reports files in that column, it means those files errored during processing — not that a write was lost. If you find such a file and `task doctor` *doesn't* report it as drift, something in the pipeline has grown a new silent-catch branch. Grep for `except` blocks in `ingestion/pipeline.py` and `ingestion/page_processor.py` — they should all either re-raise or call `_failure_tracker.record_failure` explicitly.

See the unit tests in `tests/test_pipeline_atomicity.py` for the exact assertions.
