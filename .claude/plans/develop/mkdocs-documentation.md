# Plan: MkDocs Full Documentation

## Context

Spektr has no structured documentation beyond `docs/AWS_SETUP.md` and the architecture blueprint. The CLAUDE.md references `mkdocs serve` but there's no `mkdocs.yml`, no Makefile, and `mkdocs` isn't in dependencies. This plan builds the full documentation site.

## Deliverables

### 1. Infrastructure Setup

**Files to create/modify:**
- `mkdocs.yml` — MkDocs config with Material theme, nav structure, plugins
- `pyproject.toml` — add `mkdocs`, `mkdocs-material` to dev dependencies
- `Makefile` — add `docs-serve`, `docs-build` targets

```yaml
# mkdocs.yml (target structure)
site_name: Spektr
theme:
  name: material
  palette:
    scheme: slate
nav:
  - Home: index.md
  - Architecture:
    - Overview: architecture/overview.md
    - Data Flow: architecture/data-flow.md
  - Ingestion Pipeline:
    - Overview: ingestion/overview.md
    - File Processing: ingestion/file-processing.md
    - Embeddings (Jina v4): ingestion/embeddings.md
    - Knowledge Graph (Graphiti): ingestion/knowledge-graph.md
    - CocoIndex Pipeline: ingestion/cocoindex.md
  - MCP Server:
    - Overview: server/overview.md
    - Search Tools: server/search-tools.md
    - Authentication: server/authentication.md
  - Agent:
    - Pydantic AI Agent: agent/overview.md
    - HTTP API: agent/http-api.md
  - Configuration:
    - Environment Variables: configuration/environment.md
    - Infrastructure (Docker): configuration/infrastructure.md
  - Deployment:
    - AWS Setup: deployment/aws-setup.md
    - Local Development: deployment/local-development.md
  - API Reference:
    - Response Models: api/models.md
```

### 2. Documentation Pages

#### `docs/index.md` — Home
- What Spektr is (1 paragraph)
- Key features bullet list
- Architecture diagram (Mermaid)
- Quick start (link to deployment/local-development)
- Link to each major section

#### `docs/architecture/overview.md`
- System diagram (Mermaid — S3→SQS→Pipeline→Stores→MCP→Agent)
- Component roles table
- Technology choices rationale (why Jina v4, why Graphiti, why Qdrant)
- Source: extract from `docs/resources/rag-mcp-architecture-blueprint.md`

#### `docs/architecture/data-flow.md`
- Step-by-step data flow from S3 upload to agent query
- Sequence diagram (Mermaid)
- Covers: ingest path, query path, delete/invalidation path

#### `docs/ingestion/overview.md`
- Pipeline stages: classify → chunk → embed → store → extract entities
- Which files handle what (with links)
- Supported file types table

#### `docs/ingestion/file-processing.md`
- `file_processor.py` — MIME classification, PDF-to-images, semantic chunking
- `page_number` tracking through chunks
- Code examples: `file_to_pages()`, `semantic_chunk()`

#### `docs/ingestion/embeddings.md`
- Jina v4 API: dense (2048d) + ColBERT multi-vector (128d)
- `JinaV4Embedder` class API
- Retry logic (429/5xx only, exponential backoff)
- Concurrency limiting (`jina_max_concurrent`)
- CocoIndex ops wrapper (`jina_cocoindex_ops.py`)

#### `docs/ingestion/knowledge-graph.md`
- Graphiti integration: what it does, how episodes work
- `GraphitiWriter.ingest_chunk()` — episode-based ingestion
- Temporal awareness: `created_at`, `expired_at` on edges
- Entity/relationship types are dynamic (LLM-discovered, not hardcoded)
- Legacy writer: `_LegacyGraphWriter` and migration notes

#### `docs/ingestion/cocoindex.md`
- Pipeline definition (`rag_ingestion_flow`)
- S3 vs local source switching
- State tracking in PostgreSQL
- `ingest_file` custom op walkthrough

#### `docs/server/overview.md`
- FastMCP server setup
- Tool registration pattern
- Transport options (SSE, stdio)

#### `docs/server/search-tools.md`
- Each tool with:
  - Purpose, parameters, return schema
  - Example usage (from agent perspective)
- `vector_search` — dense Qdrant query with filters
- `visual_search` — ColBERT multi-vector, optional VLM generation
- `graph_search` — Graphiti `client.search()`, temporal metadata
- `hybrid_search` — parallel execution, partial failure handling, optional reranking

#### `docs/server/authentication.md`
- `BearerAuthMiddleware` — how it works
- `MCP_API_KEY` configuration
- Behavior when key is empty (auth disabled)
- Client-side header format

#### `docs/agent/overview.md`
- `create_rag_agent()` — what it returns
- System prompt and tool selection guidance
- Using `agent.override()` for testing

#### `docs/agent/http-api.md`
- FastAPI endpoints: `GET /health`, `POST /query`
- Request/response models
- Streaming SSE support
- Lifespan MCP connection management

#### `docs/configuration/environment.md`
- Full table of all env vars from `.env.example`
- Grouped by: Jina, Qdrant, Neo4j, PostgreSQL, AWS, LLM, MCP, Auth, Resilience, Feature Flags, Observability
- Which are required vs optional

#### `docs/configuration/infrastructure.md`
- `docker-compose.yml` services: Neo4j, Qdrant, PostgreSQL
- Container versions and ports
- `wait-for-services.sh` usage
- Health check endpoints

#### `docs/deployment/aws-setup.md`
- Move existing `docs/aws-setup.md` content here
- IAM, SQS policy, S3 event notifications, LocalStack

#### `docs/deployment/local-development.md`
- Prerequisites (Python 3.13, uv, Docker)
- Step-by-step setup
- Running pipeline, server, agent
- Running tests

#### `docs/api/models.md`
- `SearchResult`, `VisualSearchResult`, `GraphFact`, `HybridSearchResponse`
- Field descriptions and example JSON

### 3. Content Sources

| Doc Page | Primary Source |
|-|-|
| Architecture overview | `docs/resources/rag-mcp-architecture-blueprint.md` |
| AWS setup | `docs/aws-setup.md` (move) |
| Environment vars | `.env.example` |
| Tool docs | Docstrings in `server/tools/*.py` |
| Agent docs | `agent/agent.py`, `agent/api.py` |
| Ingestion docs | `ingestion/*.py` docstrings + pipeline logic |
| Auth docs | `server/mcp_server.py` BearerAuthMiddleware |

### 4. File Structure

```
docs/
├── index.md
├── architecture/
│   ├── overview.md
│   └── data-flow.md
├── ingestion/
│   ├── overview.md
│   ├── file-processing.md
│   ├── embeddings.md
│   ├── knowledge-graph.md
│   └── cocoindex.md
├── server/
│   ├── overview.md
│   ├── search-tools.md
│   └── authentication.md
├── agent/
│   ├── overview.md
│   └── http-api.md
├── configuration/
│   ├── environment.md
│   └── infrastructure.md
├── deployment/
│   ├── aws-setup.md
│   └── local-development.md
├── api/
│   └── models.md
└── resources/
    └── rag-mcp-architecture-blueprint.md  (existing, keep as reference)
```

**Total: 17 new markdown files + mkdocs.yml + Makefile + dep updates**

## Execution Order

1. **Setup** — `mkdocs.yml`, deps, Makefile (must be first so we can preview)
2. **Index + Architecture** — home page and architecture overview (sets context for everything else)
3. **Ingestion** — 5 pages (self-contained module)
4. **Server + Auth** — 3 pages (depends on ingestion for context)
5. **Agent** — 2 pages
6. **Configuration + Deployment** — 4 pages (reference material, can be last)
7. **API Reference** — 1 page

Steps 2-6 are parallelizable across agents (each section is self-contained).

## Verification

- `uv run mkdocs serve` renders without errors
- All nav links resolve
- No broken internal cross-references
- Mermaid diagrams render correctly
- Existing `aws-setup.md` content preserved (moved to new location)
- All doc filenames are lowercase with hyphens (no uppercase, no underscores)
