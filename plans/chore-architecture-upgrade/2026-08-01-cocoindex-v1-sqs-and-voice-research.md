# CocoIndex v1: SQS Gap (resolved) + Voice/Transcript Opportunity

Date: 2026-08-01
Status: research complete — decisions still open.
**Re-evaluated 2026-08-01 against the finished `chore/architecture-upgrade` branch — see
[Part 4](#part-4--re-evaluation-against-the-retrieval-upgrade-branch). Parts 1–3 were written
against `develop` and are amended there, not rewritten in place.**
Follows: [`2026-08-01-cocoindex-v1-vs-v0.md`](./2026-08-01-cocoindex-v1-vs-v0.md) (resolves its Blocker #1)
Method: upstream docs, GitHub repo/API inspection, issue & code search. Latest release: **cocoindex 1.0.18**, `requires_python >=3.11` (Spektr's 3.13 is fine).

---

## Part 1 — SQS: confirmed dropped in v1

**Verdict: v1 has no SQS support, and no v1 replacement for S3 change-capture. This is a
real regression against v0, not a documentation gap.**

### First, scope it correctly — what is *not* lost

Two separate things get conflated here:

| Concern | Mechanism | v1 status |
|-|-|-|
| **Incremental reconciliation** — only changed files reprocessed; deleted files have their Qdrant points and graph nodes removed | Target-state ledger + memoization | **Fully intact.** Better than v0: memoization keys on function *code* as well as inputs |
| **Change *discovery*** — how CocoIndex learns something changed | Catch-up scan (list + compare) *or* push event (SQS / file watcher / Kafka) | Push trigger for S3 is **gone**; scan still works |

So "S3 changes → Qdrant and Neo4j update automatically" **still holds in v1**. What changes
is *when it notices*: on the next scan rather than within seconds of the S3 event. And a
scan is a `list_objects` metadata call — it does not re-embed or re-extract anything. Cost
scales with object count per cycle, not change count.

The regression is **latency and scan cost on the direct-S3 path only**, not correctness and
not automation. Per Spektr's `DOCUMENT_SOURCE`:

| Source | v1 trigger |
|-|-|
| `local` | `localfs.walk_dir(live=True)` — real filesystem watcher, true push |
| `sharepoint` | Already syncs to a local mirror → same watcher, true push |
| `s3` (direct) | Poll/scan only — **the sole regression** |

The fix is therefore about restoring the *trigger*, not about relocating files — see
Option E below.

Five independent confirmations:

1. **No connector module.** `python/cocoindex/connectors/` contains exactly 20 modules —
   `amazon_s3, azure_blob, bigquery, doris, falkordb, google_drive, iggy, kafka, lancedb,
   localfs, neo4j, oci_object_storage, postgres, qdrant, snowflake, sqlite, surrealdb,
   turbopuffer, valkey, zvec`. There is no `sqs`.

2. **The v1 `amazon_s3` API has no event hook.** Public surface is only:
   ```
   list_objects(client, bucket_name, *, prefix="", path_matcher=None, max_file_size=None)
   get_object(client, bucket_name_or_uri, key=None)
   read(client, uri, size=-1)
   ```
   No `sqs_queue_url`, no `live=` parameter (unlike `localfs.walk_dir(..., live=True)`).

3. **The official v1 example says so outright.** `examples/amazon_s3_embedding/README.md`:
   > "the `amazon_s3` source does not support live mode, so this is a one-shot catch-up
   > run (scan the bucket, sync, exit)"

4. **The v0 SQS example was deleted and redirected.** A repo-wide code search for `sqs`
   returns one meaningful hit — `docs/src/data/redirects.ts`:
   ```
   '/examples/s3_sqs_pipeline': '/docs/examples/amazon-s3-embedding/'
   ```
   The v0 S3+SQS pipeline example now redirects to the v1 example that explicitly has no
   live mode.

5. **No v1 issue tracks restoring it.** The only related open issue is
   [#601](https://github.com/cocoindex-io/cocoindex/issues/601) — "Support event
   notification (e.g. for S3) by exposing a webhook", opened **June 2025** (v0 era), no
   comments, no activity since. Its body reads "Currently S3 supports event notifications
   by SQS", i.e. it was written *about v0*. The v0-era siblings —
   [#477](https://github.com/cocoindex-io/cocoindex/issues/477) (change event push for S3),
   [#600](https://github.com/cocoindex-io/cocoindex/issues/600) (Kafka queue for event
   notification), [#599](https://github.com/cocoindex-io/cocoindex/issues/599) (Redis
   queue) — are all closed.

Also confirmed from the [V1 launch post](https://cocoindex.io/blogs/cocoindex-v1/): v1
stores internal state in **LMDB**, a single local file (default 4 GiB virtual address
space), holding "the target-state ledger, the memoization cache, and the component-path
tree". This upgrades Blocker #2 of the previous doc from *likely* to *certain*: Postgres
leaves the stack, and `doctor.py` / `backup.py` / `docker-compose*.yml` all change.

### Four ways forward for Path A

| Option | Shape | Cost |
|-|-|-|
**Hard constraint (2026-08-01): no solution may duplicate stored bytes.** Documents live in
S3 / SharePoint / local disk and must not be copied to a second location that costs storage
again. This rules out the mirror approach outright.

| Option | Shape | Duplicates storage? | Cost |
|-|-|-|-|
| **A. Stay on v0** | Pin `<1.0` indefinitely | No | No work now; v0 is frozen and its docs are already relegated to `/docs-v0/`. Accrues debt. |
| **B. v1 + timer** | Periodic `app.update()`; catch-up scan reprocesses only changed objects | No | Simplest thing that works. Latency = interval. |
| **C. SQS → Kafka/Iggy bridge** | Consumer republishes S3 *event keys* to a topic; `kafka.topic_as_map()` live source; component fetches the body from S3 on demand | No | Event-driven latency, but adds a broker to the stack. |
| **D. SQS → local mirror** | Download changed objects to a watched volume | **Yes — rejected** | Pays for the same bytes twice. |
| **E. SQS as trigger** ⭐ **chosen** | Thin consumer long-polls SQS; on event(s), debounce then call `await app.update()`; plus a daily sweep and a startup run | No | Event-driven latency, no broker, no duplication. Smaller than today's daemon. |

**DECIDED (2026-08-01): Option E, hybrid triggering.** SQS is used as a *trigger*, never as
a transport — nothing is downloaded except objects that actually changed, and those are read
straight from S3 by the pipeline as it already does. A periodic full sweep backs it up so a
missed event is self-healing rather than permanent.

Three triggers, all calling the same `await app.update()`:

| Trigger | Purpose |
|-|-|
| SQS event (debounced) | Normal path — seconds of latency |
| Interval timer (default **24h**) | Safety net for missed or expired events |
| Daemon startup | Recovers changes made while the daemon was down — SQS retention is 4 days default, 14 max, so anything older is unrecoverable by event replay |

```python
last_full = 0.0
while True:
    msgs = sqs.receive_message(WaitTimeSeconds=20, ...)   # long poll, free while idle
    due = (time.monotonic() - last_full) > FULL_SCAN_INTERVAL
    if not msgs and not due:
        continue
    if msgs:
        await asyncio.sleep(DEBOUNCE)     # coalesce a burst into one run
        drain_more_messages()
    await app.update()                    # LIST + reprocess only what changed
    last_full = time.monotonic()
    if msgs:
        sqs.delete_message_batch(...)     # only after a successful update
```

Notes:
- One loop ⇒ updates are naturally serialised; no overlapping scans to guard against.
- Delete SQS messages **only after** `update()` succeeds, so a crash replays rather than drops.
- New settings: `S3_SQS_DEBOUNCE_SECONDS`, `S3_FULL_SCAN_INTERVAL_HOURS` (default 24).
- **`app.update()` does not raise when files fail** — `mount_each` failures are logged at
  `ERROR` and never propagate. Check `handle.stats().total.num_errored` before deleting SQS
  messages, or failures are silent. See the sibling doc, Blocker 3, for the verified
  poison-pill semantics.

`coco.App.update()` is a documented public API (`await app.update()` /
`app.update_blocking()`), so this replaces
`cocoindex.update_all_flows_async(FlowLiveUpdaterOptions(live_mode=True))` at
`ingestion/pipeline.py:836` with a loop. `_use_s3_source()` branching and the S3 source
config stay as they are.

**Scan cost is negligible.** A triggered run does one `list_objects` sweep — metadata only,
no object bodies. S3 LIST is ~$0.005 per 1,000 requests at 1,000 keys per request, so a
10k-object bucket costs ~$0.00005 per scan. Scanning every minute around the clock is cents
per month. Option B (drop SQS, just use a 1–5 minute timer) is defensible on the same
arithmetic and is strictly simpler.

**Trap — do not optimise the LIST away without verifying.** The tempting move is to skip
the scan and mount a component only for the object key named in the SQS event. CocoIndex
reconciles against the set of target states declared *during that run*; if a run declares
one object, everything else may be treated as removed and deleted from Qdrant and Neo4j.
Declaring the full set each run is what makes deletion handling work. Whether v1 offers an
escape hatch (partial/scoped reconciliation) is **unverified** — check before relying on it.

Note also [#2111](https://github.com/cocoindex-io/cocoindex/issues/2111) (closed): a
crash-on-rerun deserialization bug in `FileLike.__coco_memo_state__` specific to the
AmazonS3 source. The v1 S3 path has less production mileage than localfs.

---

## Part 2 — Voice & transcripts: what's actually there

The homepage's "Voice · Transcripts" is backed by real shipped capability, not marketing.

### Built-in speech-to-text

`cocoindex.ops.litellm.LiteLLMTranscriber` — **in the library**, not example code
(`python/cocoindex/ops/litellm.py`). Landed via
[PR #1889](https://github.com/cocoindex-io/cocoindex/pull/1889), merged 2026-04-27.

```python
from cocoindex.ops.litellm import LiteLLMTranscriber

transcriber = LiteLLMTranscriber("whisper-1", language="en")   # kwargs pass through to litellm.atranscription
text: str = await transcriber.transcribe(file)                 # file: FileLike (e.g. localfs.File)
```

Requires `pip install cocoindex[litellm]`. Any LiteLLM-supported STT backend: OpenAI
Whisper, ElevenLabs (`elevenlabs/scribe_v1`), Groq, self-hosted endpoints. Combined with
`@coco.fn(memo=True)`, a file is transcribed once and never again unless it or the code
changes.

Still open: [#1828](https://github.com/cocoindex-io/cocoindex/issues/1828) asks for a
broader configurable local+remote STT provider abstraction (opened April 2026). LiteLLM
covers the remote case today; fully-local Whisper means bringing your own.

### Relevant v1 examples

| Example | What it does |
|-|-|
| `audio_to_text` | Walk a dir of `.mp3/.wav/.m4a/.flac/.ogg/.webm/.aac/.aiff` → `LiteLLMTranscriber` → one transcript row per file in Postgres |
| **`entire_session_search`** | **Closest match to Spektr's live path.** Session checkpoint folders → `process_file` routes by filename → `full.jsonl` parsed into per-turn transcript chunks, `prompt.txt` embedded whole, `context.md` split with overlap, `metadata.json` → a second structured table. Fans out via `coco.map(process_chunk, ...)` into pgvector |
| `conversation_to_knowledge` | YouTube → yt-dlp + AssemblyAI **diarized** transcription → two-step LLM extraction (speakers, statements, mentioned entities) → embedding-based entity resolution → SurrealDB graph |
| `meeting_notes_graph_neo4j` | Google Drive notes → LLM extraction of organizers/attendees/tasks → Neo4j, kept in sync as notes change |
| `kafka_to_lancedb` / `csv_to_kafka` | Live Kafka source/target; offsets committed only after the row is durably written, so a crash replays cleanly |
| `slides_to_speech` | Vision LLM speaker notes → Pocket TTS locally on CPU → LanceDB |

---

## Part 3 — Path B and the v1 upgrade

### 3a. What a CocoIndex upgrade actually changes for Path B

**Directly: nothing.** `ingestion/live_ingest.py` imports no CocoIndex at all — verified.
Per `docs/ingestion/overview.md:80`, Path B embeds synchronously, upserts to Qdrant, returns,
and runs Graphiti as a background task. CocoIndex is Path A only
(`docs/ingestion/overview.md:183-186`).

Three indirect couplings, in order of risk:

**1. Shared Qdrant collection — the serious one.** Both paths write `DENSE_COLLECTION`.
Today Path A's cleanup is *scoped*: `RagTarget` deletes only points whose `source_file`
matches the removed file key, and Path B's points carry
`source_file = "session:<id>"` (`live_ingest.py:142`), so they are structurally invisible
to it.

If Path A moves to `qdrant.mount_collection_target()` + `declare_point`, CocoIndex owns the
collection as *declared state* and reconciles it. Points it never declared — every live
session point — may be deleted as orphans on each Path A run, silently wiping live data.

**VERIFIED 2026-08-01** against `python/cocoindex/connectors/qdrant/_target.py` @ main.
The concern above is **disproved at the point level**, but a different destructive path is
real at the collection level.

**Point reconciliation is per-key — safe.** `_PointHandler.reconcile` (L286) receives one
`key` at a time and emits a delete only when the desired state is non-existence *and*
CocoIndex already tracked that key (`prev_possible_records` / `prev_may_be_missing`);
otherwise it returns `None`. `_apply_actions` deletes via
`PointIdsList(points=[explicit ids])` (L277) — never a filter, never a scroll. **It never
enumerates the collection; there is no orphan sweep.** Points CocoIndex did not declare are
invisible to it. Path B's session points would not be deleted by Path A's reconciliation.

**Collection replacement IS destructive — the actual hazard.**
`_CollectionHandler._apply_actions` (L385):

```python
if action.main_action in ("replace", "delete"):
    await asyncio.to_thread(client.delete_collection, collection_name=key.collection_name)
```

L492 states it outright: `# Collection replacement destroys all points.` The tracking record
carries `vectors=desired_state.schema.vectors` (L479), so **any change to the declared
vector schema yields `replace` → drop and recreate the whole collection**, taking every
Path B point with it. Removing the target does the same via `delete`. Given `d735689` just
reshaped `documents_dense` to named dense + sparse, schema evolution here is a live concern.

**Mitigation — supported, one keyword.** All three entry points (`collection_target`,
`declare_collection_target`, `mount_collection_target`) accept
`managed_by: target.ManagedBy = target.ManagedBy.SYSTEM`. `ManagedBy` is `SYSTEM | USER`,
documented as: *"`SYSTEM` (default): CocoIndex creates and drops the index automatically.
`USER`: assumes the index already exists and never drops it."*

**Decision: pass `managed_by=ManagedBy.USER`.** `ensure_collections()` stays the
provisioning authority, CocoIndex never drops `documents_dense`, and per-key point
reconciliation still works — so `RagTarget` can still be retired. This also removes coupling
#2 below: Path B stops depending on a Path A run having created the collection.

*Scope of verification:* the Python connector, where action decisions are made. The ledger
semantics behind `prev_possible_records` / `statediff.resolve_system_transition` are
engine-side and unread; the action set is unambiguous regardless — point deletes are
explicit IDs only.

**2. Provisioning ordering.** `ensure_collections()` is called only from `run_pipeline()`
(`pipeline.py:811`); Path B assumes the collection already exists. Under v1, schema creation
moves inside `mount_collection_target(schema=...)`, making provisioning a side effect of a
Path A run. New failure modes: Path B starting on a fresh deploy before Path A has ever run,
and the two disagreeing on named-vector layout.

**3. Shared venv.** One dependency tree for both processes. cocoindex 1.x resolves
differently from 0.3.x; a conflict with `qdrant-client` / `graphiti` / `fastapi` reaches
Path B even though it imports no CocoIndex. Low risk, non-zero.

### 3b. Should `live_ingest.py` be rewritten on v1? — Probably not

**Partly. The hard constraint: v1 has no HTTP/webhook source.** None of the 20 connectors
accepts a push. Issue #601 asks for exactly this and has been dormant for over a year.
So the FastAPI endpoint cannot be replaced by CocoIndex — it can only be demoted to a
thin producer.

### Realistic target shape

```
client ──HTTP──> FastAPI (keeps INGEST_API_KEY → ephemeral session-token auth)
                     │  writes transcript chunk / audio blob
                     ▼
              Kafka/Iggy topic   OR   watched session dir on disk
                     │
                     ▼
          CocoIndex v1 live app (`cocoindex update main.py -L`)
                     │  @coco.fn(memo=True): transcribe? → chunk → embed → extract
                     ▼
          Qdrant (native connector, named vectors)  +  graph target
```

FastAPI keeps the two-layer auth (`INGEST_API_KEY` gates `/session/start`, which returns
the ephemeral per-session token for `/ingest/transcript` and `/session/end`) — that logic
has no CocoIndex analogue and shouldn't move.

### Correction: latency is the requirement, and CocoIndex works against it

An earlier draft called this rewrite "genuinely attractive". Re-reading
`docs/ingestion/overview.md`, that was badly weighted: **the stated aim of Path B is a
low-latency path to make text searchable within seconds.** CocoIndex is a reconciliation
engine — scan, mount, declare, diff. Path B's synchronous embed → upsert → return is already
the minimum-latency shape.

Routing it through CocoIndex would add a broker hop (there is no HTTP source) plus component
scheduling — strictly more latency — in exchange for incremental reprocessing that a
never-replayed live stream gains nothing from. Memoization has no value when every chunk is
seen exactly once.

**The one v1 feature genuinely worth taking is `LiteLLMTranscriber` for audio input, and it
needs no restructuring at all** — it is a plain class Path B can call directly, entirely
independent of whether the pipeline runs on CocoIndex.

Recommendation: leave Path B's architecture alone. Adopt STT if audio ingestion is wanted.

### What a rewrite would buy (for the record)

- **Audio ingestion Spektr does not have today.** Path B currently accepts text only;
  callers must transcribe upstream. `LiteLLMTranscriber` moves that inside the pipeline,
  memoized.
- **Incremental everything.** Re-processing a session re-embeds only changed turns.
- **Deletes handled by reconciliation.** Declared target states mean removing a session
  cleans up its Qdrant points — which is the entire reason `ingestion/target_connector.py`
  exists.
- **One engine for both paths.** Path A and Path B would share components, one state store,
  one deployment story.

### DECIDED 2026-08-01: Graphiti stays. Temporal episodic memory is wanted.

Consequences, both paths:

- **Path B** — unchanged in every respect. Already staying off CocoIndex for latency
  reasons; Graphiti stays with it.
- **Path A** — the native `neo4j` connector is **not** adopted. It is structural
  (nodes/edges reconciled as declared state), not temporal, so it only ever fitted the
  GLiNER2 path, and `GRAPH_ENGINE=graphiti` is the default.
- Graphiti writes therefore remain **side effects inside `ingest_file`**, not declared
  target states. Two consequences follow, neither a regression — both are true today:
  - v1's two-phase "no partial writes" guarantee does **not** extend to graph writes. A
    mid-file failure can still leave episodes in Neo4j.
  - Episode cleanup on source-file deletion still needs explicit handling: either a custom
    `TargetHandler` (v1 supports this — see sibling doc, Blocker 4) or a side-channel hook
    equivalent to today's `RagTarget`.
- Net: the Qdrant half of `RagTarget` is replaced by the native connector; the Graphiti half
  still needs code. That is the one piece of `target_connector.py` that survives migration.

### What it would have cost (retained for context)

**Graphiti's episodic model does not fit CocoIndex's target-state model.** Path B's value
is temporal episodic memory: Graphiti *appends* episodes with validity intervals.
CocoIndex reconciles *declared* state — it diffs what you say should exist against what
does. Episodes appended by Graphiti aren't CocoIndex-managed state, so either:

- graph writes stay outside CocoIndex (as today, via a side-effect in a `@coco.fn`), losing
  reconciliation for the graph half; or
- graph writes move to the native `neo4j` connector (`mount_table_target` /
  `mount_relation_target` / `declare_relation`), which is **structural, not temporal** —
  that's abandoning Graphiti for Path B, not porting it.

This is a product decision about whether temporal episodic memory is load-bearing for
Spektr, not an engineering detail. It should be settled before any Path B spike.

Secondary: adding Kafka or Iggy adds an operational component. The watched-directory
variant avoids that and reuses the same `localfs` live source as Option D above — worth
prototyping first precisely because it converges the two tracks.

---

## Recommendation

Split the work; do not couple the tracks.

**Track 1 — Path A (bulk).** Unblocked. Take Option E: keep SQS purely as a trigger for
`await app.update()`, duplicating no storage and adding no broker. Then the v1 migration is
scoped by the previous doc's remaining items (LMDB state → rewrite `doctor.py`/`backup.py`,
drop Postgres, re-derive the poison-pill semantics, replace `RagTarget` with the native
Qdrant connector).

**Track 2 — Path B (live).** **Leave the architecture as it is.** Path B's requirement is
low latency; CocoIndex's reconciliation model adds latency and offers incrementality a
single-pass live stream cannot use. It imports no CocoIndex today and should keep it that
way. Adopt `LiteLLMTranscriber` directly if audio ingestion is wanted — no restructuring
needed. The Graphiti-vs-target-state tension (3b) therefore stays theoretical unless Path B
is ever reconsidered on other grounds.

**Cross-cutting — resolved, no longer blocking.** The shared-collection question in 3a.1 is
verified: point reconciliation is per-key and never sweeps orphans, so `RagTarget` can be
retired safely. The one hazard is collection *replacement* on schema change, which drops all
points; pass `managed_by=ManagedBy.USER` and keep `ensure_collections()` authoritative.

**Do not** pin new work to v0. It is frozen and its docs are already segregated under
`/docs-v0/`.

---

## Part 4 — Re-evaluation against the retrieval-upgrade branch

Parts 1–3 were researched while `chore/architecture-upgrade` was still at `develop`. The
branch has since landed 44 commits that reshape exactly the surface a v1 migration touches.
This part records what that invalidates, what it confirms, and what it adds.

**Verification method changed too.** Parts 1–3 leaned on the vendored project skill
(`.claude/skills/cocoindex/`). That skill is **incomplete**: its `references/connectors.md`
documents only dense named vectors and never mentions sparse vectors or `managed_by`. Part 4
is verified against a local upstream clone at **`/home/rodo/Coding/cocoindex`** @ `5aa593f4`,
reading `python/cocoindex/connectors/qdrant/_target.py` and
`python/cocoindex/connectorkits/statediff.py` directly. Prefer the clone over the skill for
anything load-bearing.

### 4a. The new blocker that isn't one: sparse vectors are natively supported

`d735689` made `documents_dense` depend on a **sparse named vector with the IDF modifier**
(`ingestion/qdrant_setup.py:40-42`). Parts 1–3 never asked whether the v1 native Qdrant
connector can express that — the skill's dense-only examples made it look like it could not.
It can, verified at source:

```python
class QdrantSparseVectorDef(NamedTuple):        # _target.py:72
    modifier: Literal["idf"] | None = None
```

- `CollectionSchema.create(vectors=...)` accepts
  `dict[str, QdrantVectorDef | QdrantSparseVectorDef]` — dense and sparse share one namespace
  in the same dict (`_target.py:182-215`), and the connector splits them into Qdrant's
  `vectors_config` / `sparse_vectors_config` on create (`_target.py:429-455`).
- Sparse defs must be named; passing one bare raises with an explicit message
  (`_target.py:199-202`).
- `PointStruct.vector` accepts a `qdrant_models.SparseVector` alongside dense lists — the same
  shape `_build_chunk_point` already builds (`ingestion/pipeline.py:221-225`).

**Verdict: the branch's schema is fully expressible in v1's native connector.** This was the
one change with the potential to make the migration a non-starter, and it comes back green.

### 4b. New work item: embedders must implement `__coco_vector_schema__`

`QdrantVectorDef.schema` does **not** take a dimension int. It takes a
`VectorSchemaProvider` — a protocol requiring
`async def __coco_vector_schema__(self) -> VectorSchema`, where `VectorSchema` carries
`dtype: np.dtype` and `size: int` (`resources/schema.py:17-29`).

Spektr's embedders are custom classes (`ingestion/embedders/{jina,voyage,openrouter}.py`) that
today expose plain `.dim` / `.model_name` attributes. Each needs that one async method added,
or a small adapter, before `mount_collection_target` can be called. Not hard — but it is real
work that appears nowhere in Parts 1–3. The sparse side needs nothing:
`QdrantSparseVectorDef` carries no schema at all.

### 4c. `managed_by=USER` is promoted from prudent to mandatory — and is now source-verified

Part 3a called this "supported, one keyword" on the strength of docstrings. It is now
verified through the full chain:

| Step | Location | Behaviour with `managed_by="user"` |
|-|-|-|
| Tracking record built | `_target.py:474-482` | `MutualTrackingRecord(..., managed_by=...)` |
| Transition resolved | `statediff.py:128-129` | **returns `None`** for a user-managed desired state |
| Action computed | `statediff.py:169-170` | `diff(None)` → **`None`**, no action |
| Drop executed | `_target.py:385-392` | only on `"replace"` / `"delete"` — never reached |

So `managed_by=ManagedBy.USER` provably prevents CocoIndex from ever calling
`delete_collection`, while per-key point reconciliation (explicit point IDs only, no orphan
sweep) keeps working.

**Why it is now mandatory rather than advisable.** The branch demonstrated that schema
evolution on this collection is a *live event*, not a hypothetical: `d735689` reshaped
`documents_dense` and required a full re-ingest from source. Under the `SYSTEM` default, the
next such change emits `replace` → drop and recreate — and `99a4368` put **Path B's live
session points in that same collection with the same named-vector layout**. A single future
schema tweak would silently wipe live session data along with the corpus. Part 3a's hazard
became materially more likely and more costly during this branch.

Corollary: `ensure_collections()` (`ingestion/qdrant_setup.py`) must stay the provisioning
authority. This also cancels coupling #2 in Part 3a — Path B stops depending on a Path A run
having created the collection.

### 4d. The migration now has a correctness gate it did not have before

The strongest new fact. When Parts 1–3 were written, "did the migration break retrieval?" was
a judgment call. The branch added a measurement:

- `tests/eval/retrieval_set.yaml` — 7 labeled queries
- `tests/eval/test_retrieval_metrics.py` — recall@10, nDCG@10, MRR, **no LLM in the loop**
- an ablation matrix over the channel/stage combinations
- recorded baselines on the 68-point corpus: dense-only `0.714 / 0.504 / 0.449` → all stages
  `0.929 / 0.743 / 0.719`

A v1 migration is a rewrite of the write path. Re-running these metrics against a v1-ingested
corpus turns "the rewrite preserved retrieval behaviour" from an assertion into a number.
**This materially de-risks Track 1 and is the best argument for doing the migration now rather
than deferring it** — the gate exists today and will not get cheaper to build later.

Caveat, recorded honestly: 68 points and 7 queries is a thin corpus. The metrics are sensitive
at that size and will catch gross regressions, not subtle ones.

### 4e. Amendments to specific Part 1–3 claims

| Claim | Status after the branch |
|-|-|
| Part 1, Option E (SQS as trigger) | **Unchanged.** Nothing on the branch touches SQS, the daemon, or `_use_s3_source()`. Still the recommendation. |
| Part 2 (`LiteLLMTranscriber`) | **Unchanged.** Still present upstream at `python/cocoindex/ops/litellm.py`. Still an independent adopt needing no restructuring. |
| Part 3b (leave Path B alone) | **Unchanged and reinforced.** The branch touched Path B's vector shape only; its latency argument is untouched. |
| Part 3a.1 (collection replacement hazard) | **Escalated** — see 4c. |
| "Only the Graphiti half of `RagTarget` survives" | **Incomplete.** `ingestion/target_connector.py` has *two* graph cleanup paths — `_remove_graphiti_episodes` (L66) and `_remove_gliner_entities` (L89). Both survive the Qdrant half going native. |
| `scripts/doctor.py` "needs rewriting" | **Overstated.** It grew Qdrant-side checks that are source-agnostic and survive v1 untouched: missing-sparse detection (L99-128) and embedder-model drift. Only the Postgres tracking-table half (L30-75) is replaced by LMDB. The rewrite is partial. |
| "Do not pin new work to v0" | **Consistent with `434d6fa`.** That commit pinned `cocoindex>=0.3.39,<1.0` as a *resolver guard* against an accidental jump to the 1.x rewrite mid-branch — not as a commitment to stay on v0. Removing the ceiling is step one of the migration. |

### 4f. One bonus the migration now picks up

`_build_chunk_point` calls `encode_documents([text])[0]` — one unbatched miniCOIL encode per
chunk, already on the branch's follow-up list. Under v1 this sits inside a `@coco.fn(memo=True)`
boundary, so the migration is the natural place to batch it and get memoization across runs for
free. Minor, but it closes a recorded follow-up rather than deferring it again.

### 4g. Net recommendation

**Unchanged in shape, stronger in confidence.** Track 1 (Path A → v1 with SQS-as-trigger)
remains the work; Track 2 (Path B) remains untouched. The branch did not introduce a blocker —
the one candidate (sparse + IDF) is natively supported. It added one small task (4b), escalated
one mitigation from advisable to mandatory (4c), shrank one estimate (`doctor.py`), and — most
importantly — built the measurement that makes the rewrite verifiable (4d).

Remaining unknowns, unchanged from Parts 1–3: LMDB state handling in `backup.py`, the
`num_errored` check in the trigger loop, and whether the S3 source's thinner production mileage
(issue #2111) bites.

---

## Sources

- **Local upstream clone: `/home/rodo/Coding/cocoindex` @ `5aa593f4`** — authoritative for
  Part 4. Key files: `python/cocoindex/connectors/qdrant/_target.py`,
  `python/cocoindex/connectorkits/statediff.py`, `python/cocoindex/resources/schema.py`,
  `python/cocoindex/ops/litellm.py`. Prefer this over `.claude/skills/cocoindex/`, whose
  `references/connectors.md` omits sparse vectors and `managed_by` entirely.
- [CocoIndex homepage](https://cocoindex.io/) — "Voice · Transcripts", "Message Queues", "Images · Video"
- [CocoIndex V1 is Live!](https://cocoindex.io/blogs/cocoindex-v1/) — LMDB state, DSL removal, connector porting
- [v1 Amazon S3 connector docs](https://cocoindex.io/docs/connectors/amazon_s3/) — no SQS
- [v1 connectors index](https://cocoindex.io/docs/connectors/) — full 20-connector list
- [v1 examples index](https://cocoindex.io/docs/examples/)
- [`examples/amazon_s3_embedding`](https://github.com/cocoindex-io/cocoindex/tree/main/examples/amazon_s3_embedding) — "does not support live mode"
- [`examples/entire_session_search`](https://github.com/cocoindex-io/cocoindex/tree/main/examples/entire_session_search)
- [`examples/audio_to_text`](https://github.com/cocoindex-io/cocoindex/tree/main/examples/audio_to_text)
- [`examples/conversation_to_knowledge`](https://github.com/cocoindex-io/cocoindex/tree/main/examples/conversation_to_knowledge)
- [`examples/kafka_to_lancedb`](https://github.com/cocoindex-io/cocoindex/tree/main/examples/kafka_to_lancedb)
- [litellm ops docs](https://cocoindex.io/docs/ops/litellm) — `LiteLLMTranscriber`
- Issues: [#601](https://github.com/cocoindex-io/cocoindex/issues/601) (open, v0-era webhook request), [#1828](https://github.com/cocoindex-io/cocoindex/issues/1828) (open, STT providers), [#2111](https://github.com/cocoindex-io/cocoindex/issues/2111) (closed, S3 memo-state crash), [#477](https://github.com/cocoindex-io/cocoindex/issues/477) / [#599](https://github.com/cocoindex-io/cocoindex/issues/599) / [#600](https://github.com/cocoindex-io/cocoindex/issues/600) (closed, v0 event-notification requests)
- PR [#1889](https://github.com/cocoindex-io/cocoindex/pull/1889) — STT support, merged 2026-04-27
- [PyPI cocoindex](https://pypi.org/project/cocoindex/) — 1.0.18, `requires_python >=3.11`
