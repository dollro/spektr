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
- `ingestion/pipeline.py::run_pipeline`
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
