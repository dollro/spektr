# fix/jina-voyage-embedder-loop-race

Follow-up to `fix/openrouter-embedder-loop-race`. The OpenRouter embedder's
per-loop resource fix is applied identically to the two remaining providers,
which carried the same latent bug.

## Bug (same as OpenRouter)
`JinaV4Embedder` and `VoyageEmbedder` kept their loop-affine resources
(`httpx.AsyncClient`, `asyncio.Semaphore`) in single shared slots, recreated
"when the loop changes" with no synchronization. CocoIndex runs file components
concurrently on multiple worker threads, each with its own event loop, all
sharing the one embedder instance. Concurrent loops clobber the shared
`self._client`; an in-flight request then awaits a connection-pool lock owned by
a different loop → `RuntimeError: the current task is not holding this lock`.

Only OpenRouter had hit this in prod (it is the configured provider for the HTGF
deployment); jina/voyage were latent.

## Fix
Same shape as the OpenRouter fix:
- per-loop `(client, semaphore)` in a `weakref.WeakKeyDictionary` keyed by the
  running loop, guarded by a `threading.Lock`;
- `_ensure_loop_resources()` get-or-creates and returns the current loop's pair;
- `_request` uses the returned locals; `_request_with_retry` takes the client as
  a parameter instead of reading `self._client`;
- `self._client`/`_semaphore` kept only for introspection/back-compat;
- `close()` best-effort-closes every per-loop client.

Jina keeps its dual RPM+TPM `TokenBucket` limiters shared (unchanged); Voyage
keeps its single RPM limiter shared. Those are pre-existing and non-crashing;
only the httpx client + semaphore were loop-affine.

## Tests
`TestEventLoopIsolation` added to `tests/test_embedder.py` (Jina) and
`tests/test_voyage_embedder.py` (Voyage): two threads each run `asyncio.run`
with a barrier holding both loops open, asserting each loop keeps its own client
and the two loops get distinct clients. Both fail on the old shared-slot code,
pass on the fix. Existing mocked suites still pass (29 Jina mocked, full Voyage).
