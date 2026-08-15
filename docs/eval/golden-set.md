# Retrieval Quality Eval (RAGAS)

`task eval` runs an end-to-end retrieval-quality check over a curated Q&A set. Fails CI (and PR merges, once wired) if any metric drops below its threshold.

This page also covers `task eval-retrieval` and `task eval-ablation` — retrieval-only metrics with no LLM in the loop, scored against a separate labelled set (`tests/eval/retrieval_set.yaml`). See [Retrieval-Only Metrics](#retrieval-only-metrics-task-eval-retrieval) below.

## Running it

```bash
uv sync --extra eval           # one-time: ragas + datasets + langchain + sentence-transformers
task ingest                    # populate Qdrant
task eval                      # score against golden_set.yaml
```

Output lands in `./eval-reports/<timestamp>.json` with per-item answer previews and the averaged metric scores.

## Metrics and thresholds

Defined in `tests/eval/thresholds.yaml`:

| Metric | Default | Meaning |
|-|-|-|
| `faithfulness` | 0.80 | Every claim in the answer is supported by retrieved context (no hallucination). |
| `context_precision` | 0.50 | Retrieved chunks are relevant to the question (signal-to-noise). |
| `answer_relevancy` | 0.75 | The answer actually addresses the question asked. |

Raise thresholds as the system matures. **Never silently lower them** — if a threshold needs to drop, note the reason in the commit message.

## Schema of `fixtures/golden_set.yaml`

```yaml
items:
  - id: unique-slug                      # never rename
    question: What is …?
    expected_context_substrings:         # must appear in retrieved text
      - substring1
      - substring2
    expected_answer_hint: free-form cue  # for human reviewers only
    required_source: arxiv.pdf           # skip Q if this doc isn't ingested
    tags: [topic, methodology, ...]
```

### Negative examples

Q&A items with `expected_context_substrings: []` test that the system declines when no relevant data exists. The answer should be "I don't have enough information to answer this" rather than fabrication. The eval enforces this with an explicit refusal-phrase check on negative rows (see `REFUSAL_PHRASES` in `tests/eval/test_retrieval_quality.py`); negative items can be marked `skip_in_ragas: true` so they don't drag down the RAGAS metric averages.

## Adding a new fixture

1. Append to `items` in `fixtures/golden_set.yaml`.
2. Pick a stable `id` slug (lowercase, hyphen-separated, never reused).
3. Set `required_source` to the filename that must be ingested for the Q to run.
4. `task eval` — confirm scores stay above thresholds.

## Why the eval path bypasses the agent

The eval directly calls `hybrid_search` and then one LLM completion. It does NOT go through the Pydantic AI agent + MCP server, because:

- Reproducibility: agent tool-choice is stochastic; we want the same retrieval path every run.
- Speed: one LLM call per Q instead of agent-loop with tool calls.
- CI cost: no need to spin up the MCP server inside the test runner.

The production agent's answer quality is still bounded by retrieval quality, so this eval is a lower bound on what users experience.

## Retrieval-Only Metrics (`task eval-retrieval`)

`task eval-retrieval` scores `multi_search` directly against a labelled set — no LLM completion, no RAGAS judge model. This makes it far less noisy than the metrics above: recall/nDCG/MRR are computed from exact chunk-identity matches, not from an LLM's judgment of an answer.

```bash
task up
task ingest
task eval-retrieval
```

Source: `tests/eval/test_retrieval_metrics.py::test_retrieval_quality_meets_thresholds`, backed by `tests/eval/retrieval_set.yaml`.

!!! warning "task test-integration can wipe your corpus"
    `tests/conftest.py`'s `qdrant_client` fixture deletes `documents_dense` and `documents_multivec` before *and* after every test that requests it (`test_qdrant_setup.py`, `test_tools.py`, `test_e2e.py`, `test_integration_live_*.py`), against the real collection names — not a sandboxed copy. Running `task test-integration` therefore destroys whatever was actually ingested, including the corpus this eval and the ablation run depend on. If you run `task test-integration` and then `task eval-retrieval`/`task eval-ablation` in the same environment, re-ingest first (`task ingest`) or the retrieval set will score against an empty collection.

### The three metrics

| Metric | Meaning |
|-|-|
| `recall@10` | Fraction of the labelled-relevant chunks that appear anywhere in the top 10 results. |
| `nDCG@10` | Binary-gain normalized discounted cumulative gain over the top 10 — rewards relevant chunks ranked *higher*, not just present. |
| `MRR` | Reciprocal rank of the first relevant chunk (`1/rank`). Punishes burying the right answer under near-misses. |

All three are computed per query and averaged. A chunk match is keyed on `source_file` + `page_number` + `chunk_index` — `page_number` is included because chunk indices reset per page in this corpus's chunking scheme, so `source_file#chunk_index` alone would collide across pages.

### `retrieval_set.yaml` format

```yaml
queries:
  - query: "What is the maximum number of retries allowed for LLM API calls before falling back to the baseline destination-selection method?"
    category: exact_id
    relevant:
      - source_file: "arxiv.pdf"
        page_number: 3
        chunk_index: 5
```

Each entry needs a `query`, a `category`, and at least one `relevant` label. `test_retrieval_set_is_wellformed` and `test_all_three_categories_are_represented` in `test_retrieval_metrics.py` enforce this shape as unit tests — a malformed or category-incomplete set fails fast, before the integration run even starts.

### The three query categories

| Category | Tests | Why it exists |
|-|-|-|
| `exact_id` | Part numbers, error codes, section references, proper nouns — literal string matches. | This is what the sparse (miniCOIL) channel exists for. A regression here that recall doesn't catch means sparse retrieval broke silently. |
| `multi_hop` | Questions needing two or more chunks to answer, ideally spanning non-adjacent parts of a document. | This is what query decomposition exists for. |
| `semantic` | Paraphrased questions with little or no lexical overlap with the source text. | This is what the dense channel exists for. |

A set missing any one category cannot detect a regression in the stage that category exercises — `test_all_three_categories_are_represented` gates on this directly.

## Per-Stage Attribution (`task eval-ablation`)

`task eval-ablation` runs the same labelled set through four retrieval configurations and prints a comparison table. It is **not a gate** — it has no threshold and never fails the build. Run it when deciding whether a stage (sparse channel, reranker) still earns its cost, or after a change that might affect one stage without affecting the others.

```bash
task eval-ablation
```

Source: `tests/eval/test_retrieval_metrics.py::test_ablation_matrix`. It patches `retrieval.pipeline.settings.sparse_enabled` / `.rerank_enabled` per config and calls `multi_search` for every query, so decomposition and the retry gate (hybrid_search-only stages) are never in the loop — this isolates the fusion and reranking stages specifically.

### Measured baseline (2026-08, 68-point corpus: `arxiv.pdf` 66 chunks + `sample.pdf` 2 chunks)

| config | recall@10 | nDCG@10 | MRR |
|-|-|-|-|
| dense-only | 0.714 | 0.504 | 0.449 |
| dense + sparse | 0.786 | 0.603 | 0.564 |
| dense + rerank | 0.714 | 0.662 | 0.643 |
| all | 0.929 | 0.743 | 0.719 |

### How to read this table

- **Recall is a candidate-set property; nDCG/MRR are an ordering property.** The reranker doesn't add new candidates to the pool — it only reorders the ones already fetched by dense (or dense+sparse). That's why `dense + rerank`'s recall is identical to `dense-only`'s (0.714) while nDCG and MRR both jump: reranking moves relevant chunks that were already retrieved up toward rank 1, it doesn't retrieve chunks that weren't found at all.
- **Sparse moves recall because it changes the candidate set.** `dense + sparse` recall (0.786) is higher than `dense-only` (0.714) because miniCOIL surfaces exact-match chunks the dense embedding ranked outside the top 10. That's the effect the `exact_id` category is designed to catch.
- **The two effects compose.** `all` (sparse + rerank together) beats every single-stage config on every metric — sparse widens the candidate pool, rerank orders it well. Neither stage alone reaches `all`'s numbers.
- If a future ablation run shows a stage no longer moving its expected metric (e.g. sparse stops improving recall, or rerank stops improving nDCG/MRR), treat that as a regression signal worth investigating even though the run isn't gated.

## Expanding the retrieval set

Be direct about what this eval currently is: **`tests/eval/retrieval_set.yaml` is a smoke-level set, not a benchmark.** The corpus is two documents — `arxiv.pdf` (a real multi-robot exploration paper, 66 chunks) and `sample.pdf` (a 2-page placeholder whose only text is literally "Page 1" / "Page 2", contributing no real queries). Every labelled query in the set is drawn from `arxiv.pdf` alone.

This is why `tests/eval/thresholds.yaml`'s `recall_at_10` / `ndcg_at_10` / `mrr` floors are **commented out and non-gating** — a floor calibrated against a single-document corpus would either be trivially easy to pass or meaningless the moment a second real document is ingested. `test_retrieval_quality_meets_thresholds` still runs and prints scores every time, but nothing fails if they move.

It's also worth naming the `multi_hop` gap directly: the set contains exactly **one** honest `multi_hop` entry. A single-paper corpus doesn't have much room for questions that genuinely need two non-adjacent chunks — most candidate two-fact questions turned out to be answerable from a single chunk once the full text was read, which would have made the label dishonest. One honest example beats several dishonest ones, but it also means the `multi_hop` category is currently under-tested relative to `exact_id` and `semantic`.

**When the corpus grows, do this:**

1. Write 40+ labelled queries, balanced roughly evenly across `exact_id` / `multi_hop` / `semantic` (see the category table above for what each is for). A multi-document corpus makes genuine `multi_hop` questions — spanning two documents, not just two chunks of one — possible for the first time.
2. Run `task eval-retrieval` against the expanded set and record the resulting recall@10 / nDCG@10 / MRR as the new baseline.
3. Uncomment the floors in `tests/eval/thresholds.yaml`, set each to roughly **baseline minus 0.05** (leaves headroom for run-to-run noise without being toothless).
4. Note the measuring commit (SHA) in the commit message that uncomments the floors, so a future reader can reproduce or re-baseline.

This section is the tracking record for that work — if you're the one growing the corpus, this is where to start.
