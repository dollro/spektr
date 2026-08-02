# First Ingest (from an empty stack)

How to take Spektr from nothing to a searchable corpus, and how to tell — at each layer — whether it actually worked.

Use this when you are:

- standing up a new environment for the first time,
- validating a change to the ingestion pipeline end to end,
- reproducing a bug from a known-clean baseline.

!!! danger "Step 1 destroys data"
    The clean-slate step deletes every vector, every graph node and all pipeline state. It is meant for a fresh or disposable environment. If there is anything you care about in this stack, run [`task backup`](backup-restore.md) first — or skip to [Step 2](#2-ingest) and ingest on top of what is already there.

For rebuilding an *existing* corpus after a config change (new embedding provider, new dimensionality), use [Re-indexing](reindex.md) instead — it is the same machinery with a narrower blast radius.

---

## 0. Prerequisites

```bash
task setup     # uv + dependencies
task up        # Qdrant + Neo4j
```

`.env` must have, at minimum, an API key for the configured `EMBEDDING_PROVIDER` and `NEO4J_PASSWORD`. See [Environment](../configuration/environment.md).

Put at least one supported document in `LOCAL_DOCUMENTS_PATH` (default `documents/`). Supported extensions are listed in `ingestion/pipeline.py::SUPPORTED_PATTERNS` — pdf, common image formats, and md/txt/csv/json/xml/html/yaml.

---

## 1. Clean slate

Four things hold state. All four must go, or you will not be starting from empty.

| What | Where | If you skip it |
|---|---|---|
| Vectors | Qdrant `documents_dense` (+ `documents_multivec`) | Stale points survive and `doctor` reports them as orphans |
| Graph | Neo4j | Old episodes/entities linger and pollute `graph_search` |
| CocoIndex ledger + memo cache | `state/cocoindex.db` | **Every file is skipped as unchanged** and nothing is re-embedded |
| Poison-pill counters | `state/ingestion_failures.db` | A previously poisoned file stays poisoned and is never retried |

**Drop the Qdrant collections.**

```bash
uv run python -c "
from qdrant_client import QdrantClient
from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.settings import settings
c = QdrantClient(url=settings.qdrant_url)
for name in (DENSE_COLLECTION, MULTIVEC_COLLECTION):
    if c.collection_exists(name):
        c.delete_collection(name); print('dropped', name)
    else:
        print('absent', name)
"
```

**Wipe the graph.**

```bash
uv run python -c "
import asyncio
from ingestion.neo4j_setup import get_driver
async def main():
    d = get_driver()
    try:
        async with d.session() as s:
            await s.run('MATCH (n) DETACH DELETE n')
    finally:
        await d.close()
    print('graph wiped')
asyncio.run(main())
"
```

**Clear local pipeline state.**

```bash
rm -rf state/cocoindex.db state/ingestion_failures.db
```

`state/cocoindex.db` is a *directory* (LMDB), not a file — `rm -rf`, not `rm`. If you have set `COCOINDEX_DB_PATH`, delete that path instead.

**Confirm you are actually empty.**

```bash
task doctor
```

```text
Collection 'documents_dense' does not exist yet.
  Qdrant: http://localhost:6333
  Nothing has been ingested. Run `task ingest` to create it.
```

Exit code 0. Anything else means something survived the wipe.

---

## 2. Ingest

```bash
task ingest
```

Logs are JSON by default. For a readable first run:

```bash
LOG_FORMAT=text task ingest
```

The collections are created here, before any document is processed, by `ingestion/qdrant_setup.py::ensure_collections` — which is the *only* code that provisions them. The CocoIndex Qdrant targets are mounted with `managed_by=USER`, so the engine never creates, replaces or drops a collection behind your back.

The run ends with two lines that are the primary signal:

```text
ingestion.runner INFO Update finished: 12 added, 0 reprocessed, 0 unchanged, 0 deleted, 0 errors
ingestion.runner INFO Pipeline completed in 84213ms (0 errored files)
```

!!! warning "`errors` is the number that matters — and it does not raise"
    CocoIndex logs and swallows per-file failures, so `task ingest` can complete "successfully" with files that failed. The runner reads `stats().total.num_errors` explicitly and **exits non-zero** when it is greater than zero. Check `echo $?` in scripts; do not infer success from the absence of a traceback.

    Note also that `added` counts *target states* (which includes the mounted collection targets), not documents or points. It is a progress indicator, not a success criterion. Use the point count in Step 3.

---

## 3. Verify

Work outwards from storage to search. Each layer can be healthy while the next is not.

### 3a. Points landed in Qdrant

```bash
curl -s -X POST http://localhost:6333/collections/documents_dense/points/count \
  -H 'content-type: application/json' -d '{"exact":true}'
```

```json
{"result":{"count":68},"status":"ok","time":0.0005}
```

A count of `0` after a run reporting `0 errors` means no document produced any chunk — check that your files are actually in `LOCAL_DOCUMENTS_PATH` and match a supported extension.

### 3b. The collection has both named vectors

A text chunk must carry `dense` *and* `sparse`; hybrid retrieval silently degrades to dense-only otherwise.

```bash
curl -s http://localhost:6333/collections/documents_dense | python3 -m json.tool | head -20
```

Expect `params.vectors.dense` (size = your provider's dimensionality, `Cosine`) **and** `params.sparse_vectors.sparse` with `"modifier": "idf"`.

### 3c. Payloads look right

```bash
uv run python -c "
from qdrant_client import QdrantClient
from config.constants import DENSE_COLLECTION
from config.settings import settings
pts, _ = QdrantClient(url=settings.qdrant_url).scroll(
    DENSE_COLLECTION, limit=3, with_payload=True, with_vectors=True)
for p in pts:
    pl = p.payload
    print(pl['source_file'], '| p', pl.get('page_number'), '|',
          pl.get('content_type'), '| vectors:', sorted((p.vector or {}).keys()))
"
```

`source_file` must be a **relative** key — `arxiv.pdf`, `specs/api.md` — never an absolute path. Every payload, `list_documents`, the delete path and the eval fixtures are keyed on it.

### 3d. The graph was written

Skip if `GRAPH_ENABLED=false`.

```bash
uv run python -c "
import asyncio
from ingestion.neo4j_setup import get_driver
async def main():
    d = get_driver()
    try:
        async with d.session() as s:
            r = await s.run('MATCH (n) RETURN labels(n) AS labels, count(*) AS n ORDER BY n DESC')
            rows = [rec async for rec in r]
            print([(rec['labels'], rec['n']) for rec in rows] or '(graph empty)')
    finally:
        await d.close()
asyncio.run(main())
"
```

With the default `GRAPH_ENGINE=graphiti` you should see Graphiti's episodic and entity labels; with `gliner`, `:Entity` nodes. An empty graph after a clean text ingest means graph writes failed — they are side effects and are logged, not fatal.

### 3e. Ledger and vectors agree

```bash
task doctor
```

```text
Tracked by CocoIndex : 2
Present in Qdrant    : 2
In sync              : 2

✓ Healthy:
  - arxiv.pdf  (66 chunks, 12 pages)
  - sample.pdf  (2 chunks, 2 pages)

✓ All in sync.
```

### 3f. Search actually returns something

```bash
task smoke -- "what is this document about?"
task smoke-graph -- "who is mentioned?"
```

This is the first check that exercises query embedding, so a failure here with healthy storage points at the query path, not ingestion.

---

## 4. Confirm incrementality

The single most valuable post-ingest check, and the cheapest. Run it again with no changes:

```bash
task ingest
```

```text
Update finished: 0 added, 0 reprocessed, 2 unchanged, 0 deleted, 0 errors
```

`unchanged` should equal your document count, `added` and `reprocessed` should be `0`, and the Qdrant point count from Step 3a must be **identical**. This proves memoized components are replaying their declared target states rather than re-embedding — the property the whole incremental pipeline rests on.

Then confirm deletion propagates:

```bash
mv documents/sample.pdf /tmp/    # remove a source file
task ingest
```

`deleted` becomes non-zero and that file's points disappear from Qdrant. Points that CocoIndex never declared — Path B's live-session data (`is_live=true`) sharing the same collection — are untouched, because reconciliation deletes by explicit point id and never sweeps the collection.

Move the file back and re-ingest to restore it.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Pipeline completed in Nms (K errored files)`, exit 1 | K files raised. The per-file traceback is above it in the log. | Fix the cause and re-run — a raising file writes no memo entry, so it retries automatically |
| `POISON PILL: <file> failed 3 times, giving up` | The same file failed `PIPELINE_MAX_RETRIES` times; it is now memoized and will not retry | Fix the file/cause, then `rm state/ingestion_failures.db` and `task ingest -- --full-reprocess` |
| `0 unchanged` on a repeat run, everything re-embedded | The state directory was deleted, or the code inside a memoized function changed (memoization keys on function code as well as inputs) | Expected after either; no action needed |
| Everything skipped, nothing written | A previous run already processed these files | `task ingest -- --full-reprocess`, or delete `state/cocoindex.db` |
| Points exist but `task smoke` returns nothing | Query-time embedding uses a different model than ingestion did | `task doctor` flags mixed `embedder_model`/`embedder_dim`; a provider switch needs a full [re-index](reindex.md) |
| Text chunks missing `sparse` vectors | The miniCOIL encoder could not load (it downloads `Qdrant/minicoil-v1` from HuggingFace on first use) | Ensure outbound access to `huggingface.co`, then re-ingest |
| `403 ... Blocked by network policy` from the embedding provider | Egress to the provider's API is blocked | Allow the provider's domain (`api.jina.ai`, `api.voyageai.com`, `openrouter.ai`) |
| Run reports errors but Qdrant has 0 points | Working as designed | Processing is two-phase: target states are submitted only after a file's processing succeeds, so a failed file writes nothing at all rather than half a document |

---

## Related

- [Re-indexing](reindex.md) — rebuilding an existing corpus after a config change
- [Ingestion Failure Semantics](atomicity.md) — the retry/poison-pill contract in detail
- [Backup & Restore](backup-restore.md) — capturing state before a destructive operation
- [CocoIndex Pipeline](../ingestion/cocoindex.md) — how declaration and reconciliation work
