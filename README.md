# Spektr

**RAG-as-MCP-Server for AI agents.** Spektr ingests your documents and live text streams into a dual knowledge store (vector + graph), then exposes session-aware search tools to any LLM agent over the [Model Context Protocol](https://modelcontextprotocol.io/).

If you've ever wanted "give the agent access to my company knowledge" without building the retrieval stack from scratch, Spektr is the retrieval layer you point your agent at.

```
   Documents (PDF, images)            Live text streams
   from local | S3 | SharePoint       via authenticated HTTP
              │                                   │
              ▼                                   ▼
       ┌──────────────────────────────────────────────┐
       │  Spektr ingestion (Path A bulk · Path B live)│
       │  embeddings · entities · relationships       │
       └──────────────────────────────────────────────┘
              │                                   │
              ▼                                   ▼
            Qdrant (vector)              Neo4j + Graphiti (graph)
              │                                   │
              └────────────────┬──────────────────┘
                               ▼
                       FastMCP server
                    (7 tools, 4 session-aware)
                               ▼
                          Your LLM agent
```

---

## Table of contents

- [What Spektr is — and is not](#what-spektr-is--and-is-not)
- [Features (in scope)](#features-in-scope)
- [Out of scope](#out-of-scope)
- [Architecture overview](#architecture-overview)
- [Quick start (local development)](#quick-start-local-development)
- [Production deployment](#production-deployment)
- [Configuration](#configuration)
- [Search tools exposed to agents](#search-tools-exposed-to-agents)
- [Operations](#operations)
- [Stack](#stack)
- [License](#license)

---

## What Spektr is — and is not

**Spektr is a self-hostable retrieval backend** for AI agents. It does three things:

1. **Ingests** documents (PDF, images, text) from local storage, S3, or SharePoint, plus real-time text streams over HTTP.
2. **Indexes** that content as dense vectors (Qdrant) **and** as a temporal knowledge graph (Neo4j + Graphiti).
3. **Serves** the index to any LLM agent through an MCP server with seven search/listing tools — including session-scoped search that lets agents combine live session data with the bulk knowledge base.

**Spektr is not** a chatbot UI, an end-user product, or a multi-tenant SaaS. It is a single-tenant infrastructure component you run alongside your agent.

---

## Features (in scope)

### Ingestion

- **Path A — Bulk documents.** Incremental ingestion via [CocoIndex](https://cocoindex.io). PDF (Docling-aided OCR + PyMuPDF rendering), images, plain text. Three document sources, selected by `DOCUMENT_SOURCE`:
  - `local` — local filesystem watch
  - `s3` — S3 + SQS event-driven daemon
  - `sharepoint` — SharePoint sync to a local volume
- **Path B — Live streaming text.** FastAPI endpoint for real-time text input. Each session gets isolated `session_id`-scoped points.
- **Embeddings**: `EMBEDDING_MODEL` (`jina-v4` | `voyage-4` | `gemini-2`) × `EMBEDDING_ROUTE` (`native` | `openrouter`). Default: `gemini-2` via `openrouter`.
- **Optional ColBERT multi-vector** (`MULTIVEC_ENABLED=true`, Jina only) for layout-aware visual search.
- **Pluggable graph engine** (`GRAPH_ENGINE`):
  - `graphiti` — LLM-driven temporal episodic memory (default; used unconditionally for Path B)
  - `gliner` — local CPU entity/relation extraction with dynamic schema induction (Path A only)
- **Failure tracking with poison-pill semantics.** Per-file retry counts in SQLite; after N failures Spektr lets the rest of the batch proceed. See [docs/operations/atomicity.md](docs/operations/atomicity.md).

### Serving (MCP)

- **FastMCP** server with three transports: `streamable-http` (default), `sse` (legacy), `stdio`.
- **Seven tools** exposed to agents: `vector_search`, `visual_search`, `graph_search`, `multi_search`, `hybrid_search`, `list_documents`, `list_document_chunks`.
- **Hybrid retrieval** — dense (Qdrant named vector) + sparse (miniCOIL, local CPU, no API cost) fused with Reciprocal Rank Fusion (RRF, `k=60`), then reranked with `jina-reranker-v3.5` (listwise). `multi_search` is the deterministic version of this pipeline (no LLM calls, the default general-purpose tool); `hybrid_search` wraps the same core with query decomposition and a relevance-gated retry.
- **Session-aware filtering** — combine bulk KB + a live session in one query.
- **Bearer auth** via `MCP_API_KEY` (when set).
- **Two-layer auth** for live ingestion (`INGEST_API_KEY` → ephemeral per-session token).

### Operations

- **Observability** — Logfire/OpenTelemetry instrumentation across every entrypoint; JSON logs with `trace_id`/`span_id` correlation. See [docs/operations/observability.md](docs/operations/observability.md).
- **Backup & restore** — `task backup` snapshots Qdrant + Neo4j + Postgres + manifest. See [docs/operations/backup-restore.md](docs/operations/backup-restore.md).
- **RAGAS evaluation gate** — golden-set retrieval-quality CI gate with thresholds in `tests/eval/thresholds.yaml`. See [docs/eval/golden-set.md](docs/eval/golden-set.md).
- **Drift detection** — `task doctor` cross-checks CocoIndex tracking against Qdrant; flags missing/orphan rows.

### Agent (optional)

A reference [Pydantic AI](https://ai.pydantic.dev/) agent is included for end-to-end testing (`task ask`) and as an example HTTP front-end. See [docs/agent/overview.md](docs/agent/overview.md).

---

## Out of scope

So you know what you're getting:

|Not included|Why / what to use instead|
|-|-|
|Chatbot UI / web frontend|Spektr is the backend. Plug your own UI or use Claude Desktop / Cursor / etc. as the MCP client.|
|Multi-tenant SaaS|Single-instance design. Run one Spektr per tenant if you need isolation.|
|File ingestion via Path B|Path B is text-only. Files go through Path A.|
|Local dense embeddings|All dense embedding providers are remote APIs. (GLiNER2 entity extraction and miniCOIL sparse retrieval run on local CPU.)|
|Model fine-tuning / training|Spektr indexes; it does not train.|
|Real-time file streaming|Path A is event-driven (S3/SQS, filesystem watch), not millisecond-latency streaming.|
|Built-in crawler / browser|Bring your own document acquisition; Spektr ingests what you give it.|
|ColBERT for non-Jina providers|Multi-vector is currently Jina-only (`MULTIVEC_ENABLED=true`).|
|Image-only PDF pages with OpenRouter|Mixed text+image PDFs work; pure-image pages with OpenRouter raise per-page errors today.|

---

## Architecture overview

```
   Path A (Bulk)                                   Path B (Live)
─────────────────────────                       ────────────────────
local | S3 | SharePoint                         HTTP POST (auth)
        │                                              │
   CocoIndex pipeline                              FastAPI :8001
        │                                              │
   ┌────┴────┬─────────────┐                    ┌──────┴──────┐
   │         │             │                    │             │
embeddings  graph engine  Qdrant            embeddings    Graphiti
(jina /     (graphiti     dense             (provider)    (always)
 voyage /   default,
 openrouter) gliner opt-in)
   │         │             │                    │             │
   ▼         ▼             ▼                    ▼             ▼
 Qdrant    Neo4j        documents_dense      Qdrant         Neo4j
                       (session_id +         (is_live=true) (group_id=
                        is_live tagged                       session_id)
                        on Path B)

                              │
                              ▼
                        FastMCP server :8080
              ┌───────────────┼───────────────┐
              │               │               │
       vector_search    graph_search    multi_search
       visual_search    list_documents  hybrid_search
                         list_document_chunks
                              │
                              ▼
                          LLM agent
```

**Two ingestion paths, one query surface.** Bulk and live data live side-by-side in Qdrant/Neo4j; live data is tagged with `session_id` and `is_live` so agents can scope queries per session.

**Pluggable graph engine.** Path A picks Graphiti (LLM, temporal) or GLiNER2 (CPU, schema-driven). Path B always uses Graphiti.

**Authentication.** MCP server uses Bearer (`MCP_API_KEY`). Live ingest uses two-layer auth: `INGEST_API_KEY` opens a session and returns an ephemeral token used for chunk uploads. Both layers opt-in.

Deeper dives:
- [docs/architecture/overview.md](docs/architecture/overview.md) — full architecture
- [docs/architecture/data-flow.md](docs/architecture/data-flow.md) — ingest + query data flow
- [docs/ingestion/overview.md](docs/ingestion/overview.md) — Path A vs Path B
- [docs/ingestion/knowledge-graph.md](docs/ingestion/knowledge-graph.md) — Graphiti vs GLiNER2
- [docs/server/overview.md](docs/server/overview.md) — MCP server internals

---

## Quick start (local development)

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/), Docker, [go-task](https://taskfile.dev).

```bash
git clone <repo-url> && cd spektr
cp .env.example .env              # set JINA_API_KEY, LLM_API_KEY, NEO4J_PASSWORD

task setup                        # install uv + dependencies
task up                           # start Qdrant, Neo4j, Postgres
task ingest                       # bulk-ingest documents/ (default DOCUMENT_SOURCE=local)
task serve                        # MCP server on http://localhost:8080/mcp
```

Connect any MCP client:

```json
{
  "mcpServers": {
    "spektr": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

Smoke-test it without a client:

```bash
task smoke                        # direct vector_search test
task ask -- "what is in my docs?" # end-to-end: agent → MCP → LLM
```

Full walkthrough: [docs/deployment/local-development.md](docs/deployment/local-development.md).

---

## Production deployment

Spektr ships a production Docker Compose stack (`docker-compose.prod.yml`) designed for a single Linux VM with an external **Traefik** reverse proxy.

```bash
cp .env.example .env.prod          # set strong passwords, MCP_API_KEY, MCP_PUBLIC_DOMAIN

task prod:build
task prod:up
task prod:ingest                   # one-shot bulk ingest
```

Connect with bearer auth:

```json
{
  "mcpServers": {
    "spektr": {
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer <MCP_API_KEY>" }
    }
  }
}
```

Detailed runbooks:
- [docs/deployment/production.md](docs/deployment/production.md) — VM + Traefik deployment (with a Caddy alternative)
- [docs/deployment/aws-setup.md](docs/deployment/aws-setup.md) — S3 + SQS event-driven ingest
- [docs/ingestion/sharepoint-setup.md](docs/ingestion/sharepoint-setup.md) — SharePoint source

---

## Configuration

`config/settings.py` (Pydantic Settings) is the single source of truth. `.env.example` is the working template.

### Required (varies with provider/source)

|Variable|When required|
|-|-|
|`JINA_API_KEY` / `VOYAGE_API_KEY` / `OPENROUTER_API_KEY`|Whichever `EMBEDDING_ROUTE` you pick|
|`NEO4J_PASSWORD`|Always|
|`LLM_API_KEY`|For Graphiti, schema induction, the agent|
|`S3_BUCKET_NAME`, `S3_SQS_QUEUE_URL`|`DOCUMENT_SOURCE=s3`|
|`SHAREPOINT_*`|`DOCUMENT_SOURCE=sharepoint`|
|`MCP_API_KEY`|Production (Bearer auth on MCP)|
|`INGEST_API_KEY`|Production (auth on Path B live ingest)|
|`MCP_PUBLIC_DOMAIN`|Production (Traefik routing)|

### Most-used options

|Variable|Default|Description|
|-|-|-|
|`EMBEDDING_MODEL`|`gemini-2`|`jina-v4` \| `voyage-4` \| `gemini-2`|
|`EMBEDDING_ROUTE`|`openrouter`|`native` \| `openrouter`|
|`EMBEDDING_DIMENSIONS`|`0`|`0` = model default; MRL models accept less|
|`MCP_TRANSPORT`|`http`|`http` (streamable-http) \| `sse` \| `stdio`|
|`GRAPH_ENGINE`|`graphiti`|`graphiti` \| `gliner` (Path A only)|
|`GRAPH_ENABLED`|`true`|Disable Neo4j writes entirely|
|`DOCUMENT_SOURCE`|`local`|`local` \| `s3` \| `sharepoint`|
|`LOCAL_DOCUMENTS_PATH`|`documents`|Where the local source reads from|
|`MULTIVEC_ENABLED`|`false`|ColBERT multi-vector for visual search (Jina only)|
|`RERANK_ENABLED`|`true`|Cross-encoder reranking on results|
|`PIPELINE_MAX_RETRIES`|`3`|Per-file retry budget before poison-pill kicks in|

Full env reference: [docs/configuration/environment.md](docs/configuration/environment.md).

---

## Search tools exposed to agents

|Tool|Backend|`session_id`|Description|
|-|-|-|-|
|`vector_search`|Qdrant dense|Yes|Semantic similarity search|
|`visual_search`|Qdrant ColBERT|No|Layout-aware search for charts, tables, diagrams (requires `MULTIVEC_ENABLED`)|
|`graph_search`|Neo4j|Yes|Entity and relationship lookup|
|`multi_search`|Qdrant dense + sparse|Yes|Fused dense + sparse retrieval (RRF), reranked. No LLM calls — the default general-purpose tool|
|`hybrid_search`|Qdrant dense + sparse|Yes|Same fused core as `multi_search`, plus query decomposition and a relevance-gated retry|
|`list_documents`|Qdrant|—|Enumerate ingested documents (chunks, pages, content types)|
|`list_document_chunks`|Qdrant|—|Per-chunk listing for a single document|

Schemas, parameters, and limits: [docs/server/search-tools.md](docs/server/search-tools.md).
Auth: [docs/server/authentication.md](docs/server/authentication.md).
Client setup (Claude Desktop, Cursor, etc.): [docs/server/client-setup.md](docs/server/client-setup.md).

---

## Operations

```bash
task smoke           # vector_search smoke test
task smoke-graph     # graph_search smoke test
task doctor          # diff CocoIndex tracking vs Qdrant, flag drift
task doctor-fix      # repair drift (deletes orphan tracking rows)
task backup          # snapshot Qdrant + Neo4j + Postgres → ./backups/<ts>/
task restore -- --from backups/<ts> --target all --yes-i-know-this-wipes-things
task eval            # RAGAS retrieval-quality evaluation
task test            # unit tests
task test-integration
task lint
task typecheck
task check           # lint + typecheck + test
task docs-serve      # MkDocs dev server
```

Runbooks: [docs/operations/](docs/operations/).

---

## Stack

|Component|Technology|
|-|-|
|Language|Python 3.13|
|Package manager|uv|
|Bulk ingestion|CocoIndex|
|Live ingestion|FastAPI|
|Document processing|Docling + PyMuPDF|
|Embeddings|Jina v4 / Voyage AI / OpenRouter|
|Vector store|Qdrant v1.17|
|Knowledge graph|Neo4j 5.26 + Graphiti (default) / GLiNER2 (opt-in)|
|State DB|PostgreSQL 17|
|MCP server|FastMCP (streamable-http default + SSE legacy + stdio)|
|Reference agent|Pydantic AI|
|Observability|Logfire / OpenTelemetry|
|Document sources|Local FS / AWS S3+SQS / SharePoint|

---

## License

Spektr is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, research, educational, and other noncommercial use.

**For commercial use, a separate license is required.** Get in touch at roland@dolltons.com.
