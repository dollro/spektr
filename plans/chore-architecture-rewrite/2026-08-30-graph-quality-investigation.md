# Why doesn't the GLiNER graph look like a knowledge graph?

**Status:** open investigation. Written 2026-08-30 as a handoff — a future
session (possibly a different model) should be able to pick this up cold.

**The question:** `GRAPH_ENGINE=gliner` extracts good entities and almost no
relationships. 920 entities, **19 relationships**, 897 entities with no edges at
all. `graph_search` consequently returns a keyword list, not facts. Why?

**Decision already made:** we are keeping GLiNER. It is 46x faster and
essentially free, and the vector side carries retrieval. This investigation is
about whether the graph can be made useful at that speed — not about reverting.

---

## 1. The anomaly, with evidence

Same corpus (`documents/`: `2605.27295v1.pdf` 21p, `arxiv.pdf` 5p,
`sample.pdf` 2p — 93,315 chars, 140 text chunks), three ingests on 2026-08-30.

| | Graphiti (baseline) | Graphiti + fixes | **GLiNER** |
|-|-|-|-|
| Wall clock | 2h 49m | 22m 06s | **3m 39s** |
| LLM calls | 2,920 | 709 | **2** (schema induction) |
| Est. spend | $6.97 | ~$0.74 | **~$0.01** |
| Entities | 818 | 912 | 920 |
| Relationships | 1,006 `RELATES_TO` | 1,038 `RELATES_TO` | **19** |
| Chunk provenance | 1,224 `MENTIONS` | 1,330 `MENTIONS` | **0 edges** |
| Isolated entities | — | — | **897 / 920** |
| Qdrant points | 146 | 146 | 146 |

Entity quality is genuinely fine — "Charging Station", "Occupancy
Probabilities", "Frontier-Based Exploration", "Log-Odds Representation" are the
right concepts for this corpus. The problem is exclusively relations.

What `graph_search` returns for "how do robots coordinate exploration":

```
Graphiti: "Multiple robots form teams together during exploration."
          "Team leader robots select frontier cells as exploration targets"

GLiNER:   "Robots (robot)"
          "Decentralized Exploration Method (concept)"
          "Frontier-Based Exploration (concept)"
```

`task smoke-graph` renders the relation as `?` because there isn't one. Note
that `graph_facts` is still populated, so **nothing reports as `degraded`** —
the channel looks healthy while returning entity names.

## 2. The strongest clue

The 19 surviving relationships, by type:

```
EVALUATED_ON 8, OPTIMIZED_FOR 5, MAINTAINS 1, INTEGRATES_MAP_WITH 1,
USES 1, IDENTIFIES_AS_TARGET 1, COMPUTES_ROUTE_TO 1, SCHEDULED_FOR 1
```

The declared base vocabulary (`config/constants.py:39`, `RELATIONSHIP_TYPES`) is:

```
created_by, owned_by, uses, part_of, measured_by, requires,
applies_to, succeeds, conflicts_with, valued_at, scheduled_for, mentions
```

**17 of 19 surviving relations are schema-induced domain-specific types**
(`EVALUATED_ON`, `OPTIMIZED_FOR`, `INTEGRATES_MAP_WITH`, `COMPUTES_ROUTE_TO`…
— none are in the base list). Only `USES` and `SCHEDULED_FOR` come from the
base vocabulary. And `mentions` — which per project history exists precisely to
give cheap "honest textual co-occurrence" edges, replacing the removed
`related_to` super-edge anti-pattern — produced **zero** edges.

That asymmetry is the thing to explain. Schema-induced types survive; base
types almost never do.

There is a mechanism that would produce exactly this pattern.
`GLiNEREngine._violates_constraint` (`ingestion/graph_engine.py:224-237`):

```python
constraint = RELATION_CONSTRAINTS.get(rel_key)
if constraint is None:
    return False  # unknown rel — allow (schema-induced types)
allowed_sources, allowed_targets = constraint
return not (head_types & allowed_sources and tail_types & allowed_targets)
```

Schema-induced types are absent from `RELATION_CONSTRAINTS` (12 keys, all base
types), so they bypass validation entirely. Base types get domain/range checked
against `name_types` — the entity types GLiNER assigned **in that same chunk**.
If GLiNER's type assignment disagrees with the constraint, the triple is
dropped at `ingestion/graph_engine.py:313`.

**This is a hypothesis, not a finding.** It has not been confirmed, because:

## 3. The blocker — we cannot yet tell "dropped" from "never extracted"

The drop is logged at DEBUG (`graph_engine.py:314`):

```python
logger.debug("Dropped %s -[%s]-> %s (constraint violation)", ...)
```

The ingest ran at INFO, so **no drop counts exist**. We literally cannot
distinguish:

- **(A)** GLiNER2 extracted plenty of base-type relations and the constraint
  filter rejected them, from
- **(B)** GLiNER2 never predicted them in the first place.

These have completely different fixes. Resolving this is step 1 for whoever
picks this up, and it is cheap — one ingest at DEBUG.

## 4. Hypotheses, ranked, each with its test

**H1 — the constraint filter is eating base relations.**
Test: re-ingest with `LOG_LEVEL=DEBUG`, then
`grep -c "constraint violation"` the log and group the dropped triples by
`rel_key`. If the count is large and dominated by base types, H1 is confirmed
and the fix is in the constraint table or in how `name_types` is built.
Suspect specifically: `name_types` is built per-chunk from
`result["entities"]`, so an entity whose type GLiNER assigned differently in
this chunk than the constraint expects fails the check even when the triple is
semantically fine.

**H2 — GLiNER2 does not predict abstract relation labels well.**
`mentions`, `applies_to`, `part_of` are abstract; `EVALUATED_ON` is concrete
and domain-specific. If the schema-induced types are the only ones GLiNER
predicts, the base vocabulary may simply be the wrong prompt surface for this
model. Test: run `gliner2` directly on a handful of chunks with (a) the base
schema and (b) the induced schema, and compare raw
`result["relation_extraction"]` counts before any filtering. This isolates the
model from our pipeline entirely.

**H3 — `mentions` is being asked of the wrong layer.**
`mentions` is co-occurrence — it does not need a model at all. Graphiti's 1,330
`MENTIONS` edges are entity→episode provenance links, not model predictions.
GLiNER's ingest writes provenance as a **property** (`e.source = source_key`)
and creates no chunk nodes, so there is nothing to link to. If graph provenance
matters, this is a pipeline design gap, not a model failure — and it is fixable
deterministically without any inference.

**H4 — episode/chunk granularity.** Chunks are merged per page before
extraction (`_merge_chunks_by_page`, `graph_engine.py:~210`). Longer text may
dilute relation extraction. Test: compare raw relation counts per-chunk vs
per-page.

Try them in order H1 → H3 → H2 → H4. H1 and H3 are cheap and would together
explain most of the gap.

## 5. What is already ruled out

- **Not an entity-extraction failure.** 920 entities, sensible names and types.
- **Not a crash or silent error.** `0 errored files`, 0 ERROR lines, `task
  doctor` reports all 3 files in sync, 146 Qdrant points.
- **Not the constraint code being inverted.** `_violates_constraint` reads
  correctly: `False` means "keep". The comment is confusingly worded but the
  logic is right.
- **Not a schema-induction failure.** Induction ran (2 LLM calls, one per
  non-trivial document) and its types are the ones that *did* survive.
- **Not a Neo4j write failure.** The 19 relations that passed the filter were
  written fine via `apoc.merge.relationship`.

## 6. Reproduce it

```bash
# stack up
docker compose up -d

# clean slate (destroys the dev graph + vectors — they rebuild in ~4 min)
curl -s -X DELETE http://localhost:6333/collections/documents_dense
docker exec spektr-neo4j-1 cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n) DETACH DELETE n"
rm -rf state/cocoindex.db state/ingestion_failures.db

# the run that matters — DEBUG is the whole point
LOG_LEVEL=DEBUG task ingest 2>&1 | tee /tmp/gliner-debug.log

# the decisive number
grep -c "constraint violation" /tmp/gliner-debug.log
grep -o "\-\[[a-z_]*\]->" /tmp/gliner-debug.log | sort | uniq -c | sort -rn

# the shape of what landed
docker exec spektr-neo4j-1 cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain \
  "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY c DESC"
docker exec spektr-neo4j-1 cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain \
  "MATCH (n:Entity) WHERE NOT (n)--() RETURN count(n) AS isolated"
```

Relevant config at time of writing: `GRAPH_ENGINE=gliner`,
`SCHEMA_INDUCTION_ENABLED=true`, `EMBEDDING_MODEL=gemini-2` (768d),
`GRAPH_SMALL_MODEL=google/gemini-2.5-flash-lite`, `GRAPH_SEMAPHORE_LIMIT=10`.

## 7. Key code

| What | Where |
|-|-|
| GLiNER engine, extraction + filtering | `ingestion/graph_engine.py:243-330` |
| Constraint validation | `ingestion/graph_engine.py:224-237` |
| The DEBUG drop log (raise this to INFO to diagnose) | `ingestion/graph_engine.py:314` |
| Per-page chunk merging before extraction | `ingestion/graph_engine.py:~205-218` |
| Entity/relationship vocabularies + constraints | `config/constants.py:22,39,58` |
| Schema induction | `ingestion/schema_inducer.py` |
| Graph engine docs | `docs/ingestion/knowledge-graph.md:112` |

---

## 8. Prior art — read this before "fixing" the relation extraction

Researched 2026-08-30. This matters because it reframes the question: a sparse,
cheaply-built graph is not automatically a broken one. Two of these findings
argue the current design may be closer to the state of the art than to a bug.

**LazyGraphRAG (Microsoft Research)** is the most directly relevant. Its
indexing cost is **identical to plain vector RAG — 0.1% of GraphRAG's** — with
comparable answer quality and >700x lower query cost than GraphRAG global
search. It achieves that by doing **no LLM extraction and no summarisation at
index time at all**: it uses NLP noun-phrase extraction to find concepts, builds
a **co-occurrence** graph, refines it with graph statistics, and defers all LLM
reasoning to query time.

Read that against our situation. We are asking *extraction time* to produce
semantically typed edges — expensively under Graphiti, and apparently not at all
under GLiNER. LazyGraphRAG's claim is that you should not do that work at index
time in the first place. If it is right, the fix for our sparse graph is not
"make GLiNER predict more relations" but "stop needing predicted relations" —
build the cheap co-occurrence layer deterministically (see H3, which needs no
model whatsoever) and let query-time reasoning do the rest.

**GraphRAG-Bench (ICLR'26, "When to use Graphs in RAG")** is the sobering one:
*"GraphRAG frequently underperforms vanilla RAG on many real-world tasks."*
Graphs pay off for hierarchical knowledge retrieval and deep multi-hop
reasoning, and their value scales with query complexity. For fact lookup, plain
hybrid retrieval often wins outright.

**So establish whether the graph channel matters before investing in it.** Our
ablation numbers were only ever measured on the Graphiti corpus. Re-run
`task eval-retrieval` on the GLiNER graph: if recall/nDCG/MRR are unchanged
against a 19-edge graph, the graph channel was contributing nothing to
retrieval even when it had 1,038 edges — and this whole investigation is
optional. That is a one-command experiment and it should come first.

**Other approaches worth knowing:** HippoRAG 2 (personalised PageRank over the
graph, ~10-30x cheaper multi-hop), PathRAG (flow-based pruning, 44% context
reduction at equal accuracy), LightRAG (dual-level retrieval, lightweight
index).

### The GLiNER-extracts / Graphiti-stores hybrid

If typed, temporal edges do turn out to be worth having, there is a path that
keeps GLiNER's speed. **Verified by reading the installed graphiti-core 0.28.1
source**, not from documentation:

`Graphiti.add_triplet()` (`graphiti.py:1450`) skips node extraction, edge
extraction, node attribute extraction and summarisation entirely. What remains:

- **Node dedupe** fires only when `EntityNode.get_by_uuid` misses. Give nodes
  **deterministic UUIDs** and the LLM call never happens. Spektr already does
  exactly this for Qdrant point ids (uuid5), so the pattern is established.
- **Edge resolution** (`resolve_extracted_edge`) **short-circuits to zero LLM
  calls** when `related_edges` and `existing_edges` are both empty and no custom
  edge attributes are declared (`edge_operations.py:492`). On a populated graph
  the `existing_edges` hybrid search usually returns something, so budget ~1
  small-model call per edge for contradiction checking.

So: GLiNER extracts on CPU → `add_triplet` writes → Graphiti supplies temporal
invalidation and provenance. This is precisely what
[graphiti issue #1193](https://github.com/getzep/graphiti/issues/1193) requests
("bring your own extraction, let Graphiti do temporal logic") and it is already
possible in the pinned version. Cost would be ~1 small call per edge instead of
the ~44 calls per episode the full pipeline charges.

Weigh it against the measured fact that **1 edge of 1,038 ever carried
`expired_at`** on this corpus. The temporal layer is what you would be buying,
and on static PDFs it did essentially nothing.

**Sources:**
[LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) ·
[GraphRAG-Bench paper](https://arxiv.org/abs/2506.05690) ·
[GraphRAG-Bench repo](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) ·
[RAG vs GraphRAG systematic evaluation](https://arxiv.org/html/2502.11371v3) ·
[graphiti issue #1193](https://github.com/getzep/graphiti/issues/1193) ·
[Zep: custom entity and edge types](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types)

## 9. Other findings from the same session (context, not the investigation)

These are settled and shipped; recorded so a future reader knows why the code
looks the way it does.

**Graphiti's cost model.** It bills roughly **one LLM call per entity plus one
per edge** — cost tracks extracted structure, not page count. The baseline run
showed **228x token amplification**: 23,328 tokens of source text produced 5.31M
LLM tokens. Written up in `docs/ingestion/knowledge-graph.md`.

**Two bugs found and fixed** (see `2026-08-30-graphiti-cost-design.md` and
`-plan.md`):

1. `OpenAIGenericClient` ignores graphiti's `ModelSize.small` tag — it records
   `model.size` as a tracing attribute then sends `self.model` regardless
   (`openai_generic_client.py:124,157`). 62% of calls were paying full price.
   Fixed by `SizeAwareGenericClient` (`ingestion/graphiti_llm.py`), gated on
   `GRAPH_SMALL_MODEL`, empty by default.
2. A failed `add_episode_bulk` restarted the whole document instead of
   resuming, reprocessing 64 of 71 episodes (~half the run's spend) and
   duplicating Episodic nodes. Fixed in `ingest_bulk`; the lookup is bounded by
   `created_at` so it matches only the failed call's own writes. **This fired
   for real on the next run** — `"Resuming arxiv.pdf: 20 of 20 episodes already
   committed"` after a Neo4j `IncompleteCommit`.

**Graphiti's temporal machinery is unused on static documents.** Exactly
**1 edge of 1,038** carried `expired_at`. That is the core argument for GLiNER
on bulk: you pay Graphiti's per-edge invalidation cost, in money and wall
clock, to invalidate one edge. Keep Graphiti for Path B live sessions, where
facts genuinely change.

**Changing the dense model invalidates the eval fixture.**
`tests/eval/fixtures/retrieval_corpus.json.gz` was a 512-d snapshot; after
switching to `gemini-2` (768d) Qdrant rejected it with
`expected dim: 768, got 512`. Regenerate with
`uv run python -m scripts.make_eval_fixture`.

**The retrieval eval thresholds are non-gating.** `tests/eval/thresholds.yaml`
has `recall_at_10` / `ndcg_at_10` / `mrr` commented out pending a larger query
set, so `task eval-retrieval` reporting PASSED enforces nothing. Current
measured values on the Graphiti corpus, for reference:

| config | recall@10 | ndcg@10 | mrr |
|-|-|-|-|
| dense-only | 0.429 | 0.272 | 0.238 |
| dense+sparse | 0.429 | 0.341 | 0.314 |
| dense+rerank | 0.429 | 0.376 | 0.357 |
| all | 0.429 | 0.429 | 0.429 |

Recall is flat while nDCG/MRR climb — the fusion and rerank stages are
re-ranking, not recalling more. **These have not been re-measured on the GLiNER
graph**, which is itself a worthwhile follow-up: if the numbers are unchanged,
the graph channel was contributing nothing to retrieval anyway, and that
materially changes how much the sparse graph matters.

**Still open, unrelated to graph quality:**

- Ingestion silently drops Graphiti episodes that exhaust their retries and
  still reports `0 errored files` (7 dropped on the baseline run). `task doctor`
  cannot see it — it diffs the ledger against Qdrant, and the graph is not in
  that comparison.
- The Docling→PyMuPDF fallback is invisible in the corpus: nothing records
  which processor produced a point, so a silent quality cliff looks like a green
  ingest.
- `ensure_collections` skips an existing collection rather than checking its
  vector size against `settings.dense_dimensions`, so a model/corpus dimension
  mismatch only surfaces as per-point write rejections.
- `*_MAX_CONCURRENT` bounds concurrency per event loop, not per process.
- `GRAPH_SEMAPHORE_LIMIT` was never raised above 10. On the Graphiti path each
  LLM call took ~12s and wall clock was `calls x latency / concurrency`, so
  raising it was the untested cheap lever if anyone returns to Graphiti.
