# Retrieval Quality Eval (RAGAS)

`task eval` runs an end-to-end retrieval-quality check over a curated Q&A set. Fails CI (and PR merges, once wired) if any metric drops below its threshold.

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
| `context_precision` | 0.70 | Retrieved chunks are relevant to the question (signal-to-noise). |
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

Q&A items with `expected_context_substrings: []` test that the system declines when no relevant data exists. The answer should be "I don't have enough information to answer this" rather than fabrication. RAGAS's `faithfulness` metric naturally rewards this behaviour.

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
