# fix/openrouter-embedder-loop-race

## Symptom
First prod ingest (S3 source, `EMBEDDING_PROVIDER=openrouter`, Graphiti engine)
produced 0 Qdrant points. Logs showed, under `PIPELINE_MAX_CONCURRENT_FILES=4`:

- `ingestion.page_processor | Text embedding failed for <file> page N`
- root cause exceptions: `RuntimeError: the current task is not holding this lock`
  and `ValueError: <Token ContextVar 'current_context'> was created in a different Context`
- `ingestion.pipeline | Pipeline failed for file: <file> (attempt 1/3)`

Because `process_file_impl` re-raises on failure, no component succeeded, so
nothing was declared to Qdrant (points are declared, not upserted).

## Root cause
`OpenRouterEmbedder` kept loop-affine resources (`httpx.AsyncClient`,
`asyncio.Semaphore`) in single shared slots and recreated them "when the loop
changes" — with no synchronization. CocoIndex runs file components concurrently
on multiple worker threads, each with its own event loop, all sharing the one
embedder instance created at `app.py` build. Concurrent loops clobbered the
shared `self._client`; an in-flight request then awaited a connection-pool lock
owned by a different loop → the RuntimeError above. `TokenBucket` is documented
as single-loop-only, confirming the design never anticipated concurrent loops.

## Fix
Isolate the loop-affine resources per event loop:
- `weakref.WeakKeyDictionary[loop -> (client, semaphore)]` guarded by a
  `threading.Lock`; entries drop when a worker loop is GC'd.
- `_ensure_loop_resources()` get-or-creates the current loop's pair and returns
  it; `_request` uses the returned locals (never the shared `self._client`), so
  a concurrent overwrite is harmless.
- `self._client`/`self._semaphore` retained only for introspection/back-compat.
- `close()` best-effort-closes every per-loop client.

Connection pooling is preserved within a loop (reused across a worker thread's
components); no client is ever shared across loops.

## Test
`tests/test_openrouter_embedder.py::TestEventLoopIsolation` — two threads each
run `asyncio.run`, a barrier holds both loops open simultaneously, and it
asserts (a) each loop keeps its own client across calls and (b) the two loops
get distinct clients. Fails on the old shared-slot code, passes on the fix.

## Prod interim mitigation (not committed; `.env.prod` is gitignored)
Set `PIPELINE_MAX_CONCURRENT_FILES=1` and restarted `ingest-live` to serialize
components (one live loop at a time) so ingestion could proceed before this fix
ships. After that change: points started landing, 0 lock-errors. Once this fix
is deployed, concurrency can be raised back to 4.

## Follow-ups
- `ingestion/embedders/jina.py` and `voyage.py` almost certainly share the same
  shared-slot pattern (`_ensure_loop_resources`) — same latent bug for those
  providers. Worth the same treatment in a separate change.
- Pre-existing: this checkout has ~21 uncommitted test-file deletions (incl.
  `tests/__init__.py`, which breaks bare `pytest` path resolution — use
  `python -m pytest`). Unrelated to this fix; left untouched.
