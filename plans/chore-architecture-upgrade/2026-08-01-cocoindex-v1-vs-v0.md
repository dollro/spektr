# CocoIndex v1 vs. Spektr's v0 Usage

Date: 2026-08-01
Status: research notes — no migration decision made
Source: `.claude/skills/cocoindex/` (project-scoped skill, vendored from the upstream cocoindex repo)

## TL;DR

Spektr is pinned to `cocoindex>=0.3.39,<1.0` (installed: 0.3.39). CocoIndex v1 is a
**full API rewrite**, not an incremental release — the skill states it outright:
"It uses a completely different API from v0." Every CocoIndex symbol Spektr currently
imports has been removed. A migration is a rewrite of `ingestion/pipeline.py`'s flow
definition, `ingestion/cocoindex_ops.py`, and `ingestion/target_connector.py`, plus
knock-on changes in `scripts/doctor.py`, `scripts/backup.py`, and `docker-compose*.yml`.

Two blockers must be resolved before any migration is scoped:

1. **S3 + SQS event-driven ingestion is not documented in v1** (see Blockers).
2. **Internal state moves off PostgreSQL** onto a local db file, which breaks the
   drift-detection and backup tooling that currently reads the Postgres tracking table.

## What Spektr uses today (v0)

| Symbol | Where |
|-|-|
| `@cocoindex.flow_def(name="RagIngestion")` | `ingestion/pipeline.py:729` |
| `cocoindex.FlowBuilder`, `cocoindex.DataScope` | `ingestion/pipeline.py:731-732` |
| `flow_builder.add_source(...)` | `ingestion/pipeline.py:741,750` |
| `cocoindex.sources.AmazonS3(bucket_name, sqs_queue_url, ...)` | `ingestion/pipeline.py:742` |
| `cocoindex.sources.LocalFile(path, binary, included_patterns)` | `ingestion/pipeline.py:751` |
| `data_scope.add_collector()` / `.collect()` / `.export()` | `ingestion/pipeline.py:760-777` |
| `@cocoindex.op.function()` | `ingestion/pipeline.py:543`, `cocoindex_ops.py:19,27,34` |
| `cocoindex.op.TargetSpec` + `@cocoindex.op.target_connector` | `ingestion/target_connector.py:22,150` |
| `cocoindex.init()`, `cocoindex.setup_all_flows()` | `ingestion/pipeline.py:808,823` |
| `cocoindex.update_all_flows_async(FlowLiveUpdaterOptions(live_mode=...))` | `ingestion/pipeline.py:836-837` |
| `COCOINDEX_DATABASE_URL` → Postgres | `ingestion/pipeline.py:792`, `config/settings.py:60` |
| Tracking table `ragingestion__cocoindex_tracking` | `scripts/doctor.py:30` |

**All of the above are removed in v1.**

## Core model change

v0 was a **declarative dataflow DSL**: build a flow graph out of `DataScope`/`DataSlice`,
push rows into a collector, `export()` them to a target.

v1 is **plain async Python**: an `App` binds a `@coco.fn` main function; you mount
processing components over source items and *declare target states* inside them.
CocoIndex diffs declared state against actual state and handles create/update/delete.

Principle stays the same — `TargetState = Transform(SourceState)` — the expression of it
is entirely different.

### v0 → v1 mapping (from the skill's own table, annotated for Spektr)

| v0 (removed) | v1 equivalent | Spektr impact |
|-|-|-|
| `@cocoindex.flow_def`, `FlowBuilder`, `Flow`, `open_flow` | `coco.App(coco.AppConfig(...), app_main, **params)` + `@coco.fn` main | Rewrite `rag_ingestion_flow` |
| `DataScope`, `DataSlice`, `add_collector()`, `collect()`, `export()` | declare target states inside mounted components (`declare_row`, `declare_point`, `declare_file`) | Collector/export block deleted |
| `cocoindex.sources.LocalFile` | `localfs.walk_dir(path, recursive=True, path_matcher=PatternFilePathMatcher(...), live=True)` | Direct swap; `live=True` replaces `FlowLiveUpdaterOptions` for the local path |
| `cocoindex.sources.AmazonS3` | `amazon_s3.list_objects(client, bucket, prefix=..., path_matcher=...)` with an aiobotocore client via `ContextKey` | **SQS support unverified — see Blockers** |
| `cocoindex.functions.*` | `cocoindex.ops.*` (`RecursiveSplitter`, `detect_code_language`, `SentenceTransformerEmbedder`) | Spektr doesn't use these (own embedders/chunker) — no impact |
| `cocoindex.targets.*` / `storages.*` | connector targets (`qdrant.mount_collection_target`, `neo4j.mount_table_target`, …) | Opportunity to drop the custom connector |
| `transform_flow`, `cocoindex.op.function()` | plain `@coco.fn` functions | Mechanical: 4 call sites |
| `cocoindex.init()`, `settings`, `COCOINDEX_DATABASE_URL` | `coco.App(...)`; state in a local db path (`builder.settings.db_path`) | **Breaks doctor/backup — see Blockers** |
| CLI `cocoindex setup` | no setup step; `cocoindex update main.py` (`-L` for live) | `setup_all_flows()` call disappears |

## New in v1 that Spektr could use

**Memoization as a first-class primitive.** `@coco.fn(memo=True)` skips re-execution when
inputs *and function code* are unchanged; `version=N` forces re-execution. v0 change
detection was source-hash-only at the row level. This is directly relevant to
`ingest_file`, which does embedding + LLM entity extraction per file.

**Native Qdrant connector** (`from cocoindex.connectors import qdrant`) — target only,
which is all Spektr needs:

```python
collection = await qdrant.mount_collection_target(
    QDRANT_DB, collection_name="documents_dense",
    schema=await qdrant.CollectionSchema.create(vectors={...}),
)
collection.declare_point(point=qdrant.PointStruct(id=..., vector={...}, payload={...}))
```

Named vectors are supported natively (`vectors={"dense": QdrantVectorDef(...), ...}`),
which lines up with the `documents_dense` named dense + sparse migration in `d735689`.
Distance metrics: `cosine` / `dot` / `euclid`.

**Native Neo4j connector** — target only, node tables via `mount_table_target`,
edges via `mount_relation_target` + `declare_relation`, plus `declare_vector_index`.
This is a *structural* graph writer (declared nodes/edges diffed and reconciled), which
is a different model from Spektr's Graphiti episodic writer. Relevant to the GLiNER2
engine path; **not** a replacement for Graphiti.

**Live mode is a source property, not a flow option.** `LiveMapView` sources
(`localfs.walk_dir(..., live=True)`) snapshot then watch; `LiveMapFeed` sources
(`kafka.topic_as_map()`) stream only. `mount_each()` auto-detects feeds. Mode is chosen
at run time (`app.update_blocking(live=True)` or `cocoindex update main.py -L`) against
the same pipeline code.

**Other connectors now available:** SQLite, LanceDB, SurrealDB, Apache Doris, Kafka
(source *and* target), Google Drive. Kafka-as-source is a plausible future alternative to
the SQS daemon.

**`ContextKey` / `@coco.lifespan`** for shared resources (clients, pools, models),
with `Annotated[NDArray, EMBEDDER]` inferring vector dimensions from the embedder's
context key. Would replace Spektr's ad-hoc singletons (`_get_qdrant_client`,
`get_driver`, graph-engine singleton).

**Stable ID helpers** — `generate_id(dep)` (deterministic) and `IdGenerator.next_id(dep)`
(always distinct), both from `cocoindex.resources.id`.

**CLI:** `cocoindex init | update [-L] [--full-reprocess] | drop | ls | show [--tree]`.

## Blockers / open questions

### 1. S3 + SQS event-driven ingestion — RESOLVED: confirmed dropped in v1

> **Update 2026-08-01:** researched and confirmed. v1 has no SQS support and no
> replacement for S3 change-capture. See
> [`2026-08-01-cocoindex-v1-sqs-and-voice-research.md`](./2026-08-01-cocoindex-v1-sqs-and-voice-research.md)
> for evidence and the options. Recommended fix: keep a thin SQS consumer but use it
> purely as a **trigger** — on event, debounce and call `await app.update()`. No files
> are copied anywhere, no broker is added. (An earlier draft suggested mirroring S3 to a
> local volume; rejected — it pays for the same bytes twice.)
>
> **Scope correction — this is smaller than "blocker" implies.** Incremental
> reconciliation is fully intact in v1: changed files still reprocess in isolation and
> deleted files still have their Qdrant points and graph nodes removed automatically.
> What v1 lost is only the *push trigger* for S3, i.e. discovery latency drops from
> seconds to the poll interval. A catch-up scan is a `list_objects` metadata call and
> re-embeds nothing. `local` and `sharepoint` sources are unaffected — `localfs` has a
> real live watcher in v1. Original notes below.

The `ingest-live` prod service is an SQS-driven daemon: `AmazonS3(sqs_queue_url=...)`
makes CocoIndex poll the queue and process only the objects named in S3 events.

The v1 skill's `amazon_s3` connector documents only `list_objects` / `get_object` /
`read`, is marked "Source only", and **"sqs" does not appear anywhere in the skill or its
references.** If v1 dropped SQS integration, live S3 ingestion would degrade to full
bucket re-scans, or would need a Kafka bridge or a hand-rolled SQS consumer.

**Action: verify against upstream v1 source/docs before scoping anything else.** This
alone can make the migration a non-starter for the S3 deployment.

### 2. Internal state leaves PostgreSQL

v1 stores its internal state at `builder.settings.db_path` (a local file, e.g.
`./cocoindex.db`; `COCOINDEX_DB` env var is a fallback). **Confirmed 2026-08-01: the store
is LMDB** — single file, default 4 GiB virtual address space, holding the target-state
ledger, memoization cache, and component-path tree. Postgres leaves the stack entirely.
Consequences:

- `scripts/doctor.py` shells out to `psql` against the `cocoindex` database and reads
  `ragingestion__cocoindex_tracking` to diff tracking rows vs. Qdrant. Table name, schema,
  and access method all change. `task doctor` / `task doctor-fix` need rewriting.
- `scripts/backup.py` does `pg_dump -Fc cocoindex`. Would become a file copy.
- `config/settings.py:60` `database_url` and the `COCOINDEX_DATABASE_URL` export become
  dead unless Postgres is retained for something else.
- The Postgres service in `docker-compose.yml` / `docker-compose.prod.yml` may be
  removable — but the db file then needs a persistent volume, and concurrent access
  from multiple containers needs checking.

### 3. Poison-pill retry semantics — RESOLVED: contract survives 1:1

Verified 2026-08-01 against `programming_guide/function.mdx`,
`programming_guide/processing_component.mdx`, `advanced_topics/exception_handlers.mdx` @ main.

**Retry-on-failure is the v1 default.** From `function.mdx`, verbatim:

> If a memoized function raises, no cache entry is written for that call. The next
> invocation with the same inputs sees a cache miss and re-executes the body — exceptions
> never poison the cache, so you don't need to wrap calls defensively.

The CLAUDE.md contract maps directly:

| Behaviour | v0 mechanism | v1 mechanism |
|-|-|-|
| Retry next run | re-raise → tracking row not written | re-raise → no cache entry written |
| Give up after `PIPELINE_MAX_RETRIES` | swallow → row written, file marked done | swallow + return normally → cache entry written, file not retried |
| Failure counting | `state/ingestion_failures.db` | unchanged — Spektr's own sqlite, independent of CocoIndex |

**Failure isolation is now free.** `processing_component.mdx`: *"A failure in one child does
not affect the parent or siblings — by default the exception is logged and other components
continue. One bad file shouldn't take down the entire pipeline."* Half the poison-pill's
purpose (keep the batch moving) becomes framework behaviour; the counter's only remaining
job is to stop retrying a file forever.

**Two-phase processing removes partial writes.** Processing declares target states in memory
and is side-effect-free; submit runs only after processing succeeds. Today `ingest_file`
writes Qdrant + Neo4j inline, so a mid-file failure can leave partial data — v1 makes that
structurally impossible, *but only if writes are declared rather than kept as direct side
effects*.

**Better home for the counter:** `builder.set_exception_handler()` (global, in lifespan) or
`coco.exception_handler()` (scoped async context manager). Handlers receive
`ExceptionContext.stable_path`, a stable per-component key, so the failure counter can be
driven from there rather than by wrapping the function body.

**Operational catch — affects the SQS-trigger daemon.** `mount()` / `mount_each()` failures
**do not propagate**; they are logged at `ERROR` and nothing else, so **`await app.update()`
returns successfully even when files failed.** The trigger loop must check
`handle.stats().total.num_errored` before deleting SQS messages, otherwise failures are
*silent* — strictly worse than v0, where a raise was visible. `UpdateHandle.watch()` streams
progress snapshots for the same data.

### 4. Custom target connector — CORRECTION: v1 *does* have this

An earlier draft of this doc claimed v1 removed the custom-target-connector extension point.
**That was wrong.** v1 documents it at `advanced_topics/custom_target_connector.mdx` with a
public `TargetHandler` protocol — the same API the built-in Qdrant connector is implemented
against (`coco.TargetHandler`, `coco.TargetActionSink`,
`coco.register_root_target_states_provider`, `coco.ChildTargetDef`).

`RagTarget` / `@cocoindex.op.target_connector` exists solely so that deleting a source file
deletes its Qdrant points and Graphiti episodes; upserts are no-ops because `ingest_file`
writes directly.

- **Qdrant cleanup:** solved natively — declared points reconcile per key, so deletions come
  free. See the sibling doc for verified reconciliation scope and the `managed_by` decision.
- **Graphiti episode cleanup:** an earlier draft said "no equivalent" — also wrong, given the
  above. A custom `TargetHandler` can do what `RagTarget` does today, with real tracking
  records instead of a no-op upsert path. Remaining options unchanged otherwise: keep a
  side-channel hook, accept orphans, or move graph writes to the native Neo4j connector
  (structural, so GLiNER2-path only).

The docs' own guidance is worth heeding before building one: *"For simple use cases where you
just need to write data to an external system without sophisticated change tracking, consider
using a regular function with memoization instead."*

### 5. Python version

Spektr runs Python 3.13 (venv shows 3.14 site-packages). v1 dependency requirements
weren't checked. Worth confirming before any spike.

## Suggested next step

Don't scope the migration yet. Resolve Blocker #1 (SQS) first against upstream v1 —
it's cheap and it gates everything. If SQS is supported, the natural follow-up is a
throwaway spike: local-source-only v1 pipeline writing to Qdrant via the native
connector, to measure how much of `ingest_file` survives and to pin down the retry
semantics in #3.

## References

- Project skill: `.claude/skills/cocoindex/SKILL.md` (+ `references/api_reference.md`,
  `connectors.md`, `patterns.md`, `setup_project.md`, `setup_database.md`)
- Upstream examples: https://github.com/cocoindex-io/cocoindex/tree/main/examples
  (closest to Spektr: `text_embedding_qdrant`, `meeting_notes_graph_neo4j`, `pdf_to_markdown`)
- Full docs text: https://cocoindex.io/docs/llms-full.txt
