# Graphiti Cost Fixes — Design

Two defects found while investigating a $7.04 ingest of three PDFs on
2026-08-30. Both are bugs, not tradeoffs: neither trades quality for money,
and neither is a knob the operator was supposed to have tuned.

## Measured baseline

Recorded so the post-fix run has something to be compared against. Corpus:
`documents/` — `2605.27295v1.pdf` (21p), `arxiv.pdf` (5p), `sample.pdf` (2p).

| Quantity | Value |
|-|-|
| Source text | 93,315 chars ≈ 23,328 tokens |
| Wall clock | 2h 49m (14:47:12 → 17:36:06) |
| Haiku 4.5 | 5.31M tokens → **$6.97** (~4.90M in / ~415K out) |
| Gemini embedding 2 | 345K tokens → $0.07 |
| **Token amplification** | **228×** |
| Successful `chat/completions` | 2,920 |
| Embedding requests | 8,656 |
| HTTP 402 rejections | 468 |
| `add_episode` completions | 66 |
| Qdrant points | 146 (`google/gemini-embedding-2`, 768d) |
| Neo4j | 818 Entity, 157 Episodic, 1,006 RELATES_TO, 1,224 MENTIONS |
| Duplicate entity names | **0** |
| Duplicate facts | 4 of 1,006 |
| Duplicate Episodic names | **64** |
| Edges with `expired_at` | **1** of 1,006 |

The last row is the context for everything below: Graphiti charges per edge
largely to maintain bi-temporal invalidation, and on a static document corpus
that machinery fired exactly once.

## Defect 1 — `ModelSize.small` is ignored

Graphiti tags its high-volume maintenance calls as cheap-model-safe:

```
edge_operations.py:507, 559, 621   per-edge resolution        (1,006 edges)
node_operations.py:542, 645        per-node dedupe/attributes (818 entities)
```

That is 1,824 of 2,920 calls — **62%**.

`BaseOpenAIClient` honours the tag via `_get_model_for_size`
(`openai_base_client.py:111`). `OpenAIGenericClient` — the only client that
speaks to an arbitrary OpenAI-compatible endpoint, and therefore the only one
usable with OpenRouter — does **not**. It accepts `model_size`, records it as a
tracing attribute (`'model.size': model_size.value`, line 157), and then sends:

```python
model=self.model or DEFAULT_MODEL          # openai_generic_client.py:124
```

`ingestion/graphiti_client.py:153-164` also never sets `LLMConfig.small_model`.
So every "small" call ran on `anthropic/claude-haiku-4.5` at full rate.

### Decision: delegate to a second client instance

Three ways to fix it:

1. **Copy `_generate_response` with one line changed.** Rejected: duplicates
   ~40 lines of upstream code that will drift on the next `graphiti-core` bump.
2. **Mutate `self.model` around a `super()` call.** Rejected outright — it is a
   data race. `SEMAPHORE_LIMIT` is 10, so ten coroutines share the instance and
   one would read the model another had just swapped.
3. **Hold a second `OpenAIGenericClient` configured with the small model and
   delegate to it.** Chosen. No duplication, no mutation, no shared state.

The subclass overrides only `_generate_response` and dispatches on
`model_size`. The outer `generate_response` keeps its tracing and retry
handling, so the delegate's `_generate_response` is called directly.

### Consequences

- **We depend on a private method.** `_generate_response` is not public API. If
  a future `graphiti-core` changes its signature, our override silently stops
  being called and we quietly return to paying full price. Mitigated by a test
  that pins the upstream signature (Task 1) — it fails loudly on drift.
- **The setting must default to empty.** With `GRAPH_SMALL_MODEL=""` there is no
  delegate and behaviour is byte-identical to today. The fix is inert until
  configured, so it cannot regress the eval gate on its own.
- **The small model must support `response_format: json_schema`.** Every call
  goes out with a JSON-schema response format. A model that ignores or rejects
  it will fail every dedupe and edge-resolution call, which is *worse* than the
  status quo. This is why the setting is opt-in and why Task 4 requires a real
  ingest before the value is committed to `.env.prod`.
- **This is not free money.** I previously said it costs "zero change to graph
  quality"; that was too strong. Graphiti's judgement that these calls are
  small-model-safe is calibrated against `gpt-4.1-nano`, not against an
  arbitrary cheap model. Weaker dedupe means more duplicate entities and worse
  invalidation. The baseline above records 0 duplicate entity names and 4
  duplicate facts precisely so the post-fix run can be checked against it.

## Defect 2 — the bulk fallback restarts instead of resuming

`ingestion/graph_writer.py:162-194`:

```python
try:
    await client.add_episode_bulk(episodes)
    return
except Exception:
    logger.warning("Bulk ingestion failed ... falling back to sequential")

for episode in episodes:          # the full original list, from index 0
    await client.add_episode(...)
```

`add_episode_bulk` commits to Neo4j incrementally. When it raised on a 402,
part of the document was already written; nothing records which part, so the
fallback reprocesses everything.

Evidence: `2605.27295v1.pdf` has **135 Episodic nodes under 71 distinct names**
— 64 episodes ingested twice, each through the full ~44-call pipeline. Call
counts either side of the fallback line: 1,516 before, 1,646 after.

The trigger is `except Exception` — any one of them. A single 429, 500 or
timeout on the last episode of a long document restarts the whole file, and the
later the failure lands the more it costs. The redo is expensive rather than
fast precisely *because* dedupe works: every re-extracted entity must be
LLM-compared against the now-populated graph to conclude "already have this".

### Decision: query committed episode names, filter the fallback list

Before the fallback loop, ask Neo4j which episodes for this `source_key`
already exist and skip them.

### Consequences

- **Filter at fallback time, not up front.** Episode names
  (`{source_key}:p{page}:c{chunk}`) are content-independent. Filtering before
  the bulk call would also skip episodes left by an *earlier run*, which is
  wrong when the document has since been edited — same names, new content,
  silently not re-ingested. Filtering only after a failure in *this* call keeps
  the meaning precise: "skip what I just wrote."
- **A failing resume query must not fail the ingest.** It degrades to the
  current behaviour (reprocess everything) and logs a warning. Graph writes are
  side effects per the pipeline contract; this optimisation must not promote a
  recoverable failure into a hard one.
- **Couples `graph_writer` to Cypher and the `Episodic` label.** Accepted:
  `graph_target.py` already reaches into the graph directly for cleanup.
- **Existing pollution is not cleaned up.** The 64 duplicate Episodic nodes
  already in the dev graph stay. They are provenance clutter, not corruption —
  entity and edge dedupe caught essentially everything. Cleanup is out of scope.

## Explicitly out of scope

- **Silent episode loss.** The fallback loop still catches per-episode failures
  and continues, so a run reports `0 errored files` while dropping episodes
  (7 on the baseline run). Real, adjacent, and a separate change.
- **`GRAPH_EPISODE_TARGET_SIZE`.** Investigated and dropped: per-episode
  overhead is ~4 of 44 calls, so regrouping saves single-digit percent while
  risking worse extraction from longer prompts. It is a quality knob, not a
  cost one.
- **Architecture change** (GLiNER for bulk, `add_triplet` hybrid, LazyGraphRAG).
  Both fixes here are prerequisites for measuring that choice honestly, not
  alternatives to it.
