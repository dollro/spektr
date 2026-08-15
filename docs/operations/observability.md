# Observability

Spektr emits OpenTelemetry traces for every significant operation (ingest, embed, Qdrant write, MCP tool call, LLM request). Traces are stitched by trace_id across processes and also injected into the JSON log stream so logs and traces correlate.

## Default: local-only

Out of the box, tracing is **on** and **stays in the process** — no token, no network egress, no paid tier required. Set:

```bash
OBSERVABILITY_LOCAL_ONLY=true      # default
LOGFIRE_TOKEN=                     # empty
SERVICE_NAME=spektr                # optional; defaults to "spektr"
```

You'll see `trace_id` / `span_id` appear in every JSON log once something starts a span. Useful for:
- correlating a slow `task ask` call across ingest, Qdrant, Neo4j, and LLM.
- reproducing a failing user query by filtering logs on one trace_id.

## Ship to Logfire Cloud

If you want the hosted UI (nice flame graphs, search, alerts), sign up at <https://logfire.pydantic.dev> and:

```bash
LOGFIRE_TOKEN=your-token-here
OBSERVABILITY_LOCAL_ONLY=false
```

Restart your processes. Traces land in the dashboard within seconds.

## What gets instrumented

| Library | What you see |
|-|-|
| pydantic-ai | agent runs, tool calls, LLM requests (prompts + completions + token counts) |
| httpx | every outbound HTTP call (Jina/Voyage embeddings, Anthropic/OpenRouter, Qdrant REST) |
| FastAPI | live-ingest endpoints + agent `/query` endpoint (request + response + status) |

Instrumentation is wired in these entrypoints:
- `ingestion/runner.py::run_pipeline`
- `ingestion/live_ingest.py` (module-level)
- `server/mcp_server.py`
- `agent/api.py::lifespan`
- `scripts/ask.py::main`
- `services/sharepoint_sync/main.py::main`

FastAPI apps additionally call `instrument_fastapi(app)` after `setup_observability()`. The two FastAPI entrypoints are:
- `ingestion/live_ingest.py` (module-level)
- `agent/api.py` (module-level, after app construction)

## Correlating logs with traces

The JSON log formatter (`config/logging.py`) adds two extra keys when an OTEL span is active:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "logger": "ingestion.pipeline",
  "message": "Finished file: arxiv.pdf in 90000ms",
  "trace_id": "e1b7...",
  "span_id": "a9f0...",
  "duration_ms": 90000,
  "file_name": "arxiv.pdf"
}
```

Grep by `trace_id` to reconstruct a single operation end-to-end.

## Relevance-gate telemetry

`retrieval/gate.py::log_gate_decision` emits **one INFO record per gated pipeline run** — whether or not the retry fired. Quiet runs are logged deliberately: a count of retries without a denominator cannot answer "how often".

```json
{
  "message": "Relevance gate evaluated",
  "gate_fired": true,
  "gate_reranked": true,
  "top_score": -0.12,
  "top_score_after": 0.31,
  "gate_widened_to": 30,
  "retry_helped": true
}
```

| Field | Meaning |
|-|-|
| `gate_fired` | The retry ran. Present on every record, so it doubles as the marker for gate telemetry |
| `gate_reranked` | A rerank score was available to test. **When false the run is inert** — see below |
| `top_score` | Top-1 score before any retry. `null` when the pass returned nothing |
| `top_score_after` | Top-1 score after the widened pass. Only when fired |
| `gate_widened_to` | Widened candidate pool size. Only when fired |
| `retry_helped` | The widened pass beat the first one. Only when fired |

**Inert runs must be excluded from the rate.** When the reranker is disabled or degraded, `FusedResult.score` carries the RRF fusion score instead — always positive, therefore never below the default `RERANK_SCORE_FLOOR` of `0.0`. Such runs cannot fire by construction, and counting them as "gate did not fire" understates the true rate. `gate_reranked` distinguishes them.

Aggregate with `task retry-stats`:

```bash
docker compose logs mcp | task retry-stats
task retry-stats -- --from logs/mcp.log
```

**Reading the result.** `retry_helped` is the diagnostic that matters. A high fire rate where the retry *does* improve top-1 means the right chunk existed but ranked outside the initial pool — a first-stage **recall** problem, addressable with better retrieval (late interaction, a stronger embedding model, a larger `RERANK_CANDIDATES`). A high fire rate where it *does not* means the content is absent or the query is unanswerable — a corpus coverage problem no retrieval change will fix.

Volume is one line per `hybrid_search` call; `multi_search` (`fast_pipeline`) has no gate and emits nothing.

## Local trace viewer (optional)

If you want a local UI without Logfire Cloud, add a Jaeger container:

```yaml
# docker-compose.obs.yml (new file)
services:
  jaeger:
    image: jaegertracing/all-in-one:1.60
    ports:
      - "16686:16686"   # UI
      - "4318:4318"     # OTLP HTTP
```

Run with both compose files:

```bash
docker compose -f docker-compose.yml -f docker-compose.obs.yml up -d
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Open <http://localhost:16686>. (Jaeger profile is not wired into `task up` yet — add it when you need it.)
