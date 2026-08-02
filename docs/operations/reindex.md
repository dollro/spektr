# Re-indexing

Some changes to `documents_dense` are not additive — a new embedding provider, a new dense dimensionality, or (as of the retrieval upgrade) a schema change to how vectors are named. Qdrant cannot migrate a collection's vector configuration in place, so these changes require dropping and rebuilding `documents_dense` from source documents.

!!! warning "This is disruptive"
    A full re-index re-embeds every document, which costs real embedding API spend (or CPU time for local models) proportional to corpus size. **Search is unavailable for the duration** — `documents_dense` does not exist between the drop and the first successful re-ingest. Plan a maintenance window; don't run this against a live production MCP server without warning.

## When this applies

- Switching `EMBEDDING_PROVIDER` (different dimensionality, different embedding space — see [Embeddings](../ingestion/embeddings.md#switching-providers))
- Changing `JINA_DENSE_DIMENSIONS` / `VOYAGE_DENSE_DIMENSIONS` / `OPENROUTER_DENSE_DIMENSIONS`
- A vector-naming or vector-config change to `documents_dense` itself — the retrieval upgrade's move from a single unnamed vector to named `dense` + `sparse` vectors is the reference case this runbook was written for (named and unnamed vectors cannot coexist in one collection)

Not needed for provider-internal changes that don't touch dimensionality or vector config (e.g. swapping the reranker model), and not needed for `documents_multivec` alone unless you're also changing `MULTIVEC_DIM`.

If you are standing up a new environment rather than rebuilding an existing corpus — or you want to verify ingestion end to end from nothing — see [First Ingest](first-ingest.md), which also wipes Neo4j and the failure-tracker DB and walks through per-layer verification.

## Procedure

**1. Back up first.** A re-index is destructive to the collection you're about to drop — if the migration goes wrong partway through, you want a way back.

```bash
task backup
```

See [Backup and Restore](backup-restore.md).

**2. Drop the collection.**

```bash
task up   # make sure Qdrant is reachable
uv run python -c "
from qdrant_client import QdrantClient
from config.constants import DENSE_COLLECTION
from config.settings import settings
c = QdrantClient(url=settings.qdrant_url)
if c.collection_exists(DENSE_COLLECTION):
    c.delete_collection(DENSE_COLLECTION)
    print('dropped', DENSE_COLLECTION)
"
```

`documents_dense` is recreated with the new vector config on the next ingestion run, by `ingestion/qdrant_setup.py::create_dense_collection` (idempotent — it's a no-op if the collection already exists, which is why the drop above is required first). `ensure_collections` is the *only* code that provisions collections: the CocoIndex Qdrant targets are mounted with `managed_by=USER`, so the engine never creates or replaces them behind your back.

**3. Re-ingest, invalidating CocoIndex's state.**

Dropping Qdrant alone isn't enough — CocoIndex's memoization cache still holds a result for every source file, so a plain `task ingest` would skip them all. `--full-reprocess` reprocesses everything and invalidates existing caches:

```bash
task ingest -- --full-reprocess
```

If you'd rather start from a clean slate entirely, delete the state directory instead and run a plain `task ingest`:

```bash
rm -rf state/cocoindex.db      # or whatever COCOINDEX_DB_PATH points at
task ingest
```

This re-reads every source document, re-embeds it with the current provider/config, and writes fresh points into the recreated `documents_dense`. Duration scales with corpus size and the embedding provider's rate limits (see [Embeddings](../ingestion/embeddings.md) for per-provider RPM/TPM limits).

**4. Verify.**

```bash
task doctor
```

Expected: no drift between CocoIndex tracking and Qdrant contents, and no text-chunk points missing a `sparse` vector (see [Embeddings — Named vectors](../ingestion/embeddings.md#named-vectors-on-documents_dense)). If `task doctor` reports drift, do not treat the re-index as complete.

Run a smoke check before declaring victory:

```bash
task smoke              # direct vector_search, no MCP/LLM
task eval-retrieval     # recall@10 / nDCG@10 / MRR against the labelled set
```

## What re-indexing does not touch

Neo4j and the knowledge graph are untouched by this procedure — `documents_dense` and the graph are independent stores. If you're also changing graph extraction behavior, that's a separate operation; see [Knowledge Graph](../ingestion/knowledge-graph.md).

`documents_multivec` (ColBERT) is also untouched unless you separately drop and recreate it — it has its own vector config (`colbert`, 128-d, MaxSim) and its own migration triggers.
