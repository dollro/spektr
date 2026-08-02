# Local Development

Step-by-step guide to running Spektr on your local machine.

## Prerequisites

| Tool | Version | Install |
|-|-|-|
| Python | 3.13+ | [python.org](https://www.python.org/downloads/) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | 24+ | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | v2+ | Included with Docker Desktop |

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> spektr
cd spektr
```

### 2. Install Python dependencies

```bash
uv sync --extra gliner
```

This installs all base, dev, and test dependencies from `pyproject.toml`, plus
`gliner2`, which `GRAPH_ENGINE=gliner` needs. `task setup` runs the same command.

Keep the `--extra gliner`. `uv sync` is exact by default, so a bare sync
*uninstalls* `gliner2` — and because the import is lazy, nothing complains until
the graph engine is first built, at which point ingestion and `graph_search`
fail with `No module named 'gliner2'`.

### 3. Start infrastructure services

```bash
docker compose up -d
```

This starts Qdrant and Neo4j. CocoIndex's pipeline state needs no service — it lives in a local LMDB directory (`COCOINDEX_DB_PATH`, default `state/cocoindex.db`). See [Infrastructure Services](../configuration/infrastructure.md) for details on each service.

Wait for both services to be ready:

```bash
curl -sf http://localhost:6333/healthz && curl -sf http://localhost:7474
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values. At minimum you need:

- `JINA_API_KEY` (or `VOYAGE_API_KEY` / `OPENROUTER_API_KEY` matching `EMBEDDING_PROVIDER`)
- `NEO4J_PASSWORD` — Neo4j password (must match `docker-compose.yml`)
- `LLM_API_KEY` — LLM provider API key
- `MCP_API_KEY` — Bearer token for MCP server auth (leave empty to disable auth in dev)

`DOCUMENT_SOURCE` defaults to `local` and the pipeline reads files from `LOCAL_DOCUMENTS_PATH` (default `documents/` in the repo). Drop PDFs / images / text files into that directory and they will be picked up. LocalStack/S3 is **optional** — only needed if you set `DOCUMENT_SOURCE=s3` to test the S3 + SQS path locally (see [AWS Setup](aws-setup.md#localstack-local-development)).

See [Environment Variables](../configuration/environment.md) for the full reference.

### 5. Run the ingestion pipeline

```bash
uv run python -m ingestion.pipeline
```

With the default `DOCUMENT_SOURCE=local`, this reads from `LOCAL_DOCUMENTS_PATH`, generates embeddings, stores vectors in Qdrant, and builds the knowledge graph in Neo4j. With `DOCUMENT_SOURCE=s3` (or `sharepoint`), it pulls from the configured remote source instead.

### 6. Start the MCP server

```bash
uv run python -m server.mcp_server
```

The server starts on `http://localhost:8080/mcp` by default (streamable-http transport). Configurable via `MCP_HOST`, `MCP_PORT`, `MCP_PATH`, and `MCP_TRANSPORT`. Set `MCP_HOST=127.0.0.1` to keep it local-only.

### 7. Start the agent API (optional)

```bash
uv run python -m agent.api
```

This exposes a FastAPI HTTP endpoint for interacting with the Pydantic AI agent.

## Testing

Run all tests:

```bash
uv run pytest
```

Run unit tests only (no Docker services needed):

```bash
uv run pytest -m "not integration"
```

Run integration tests (requires running Docker services):

```bash
uv run pytest -m integration
```

Integration tests never touch your dev data:

- **Qdrant** — the suite is redirected to `test_documents_dense` /
  `test_documents_multivec`, which it drops and recreates per test.
- **Neo4j** — the suite starts its own **ephemeral `neo4j:5.26-community`
  container** (Testcontainers) and points `settings.neo4j_uri` at it for the
  session. Community Edition has a single database, so the per-test
  `MATCH (n) DETACH DELETE n` would otherwise wipe your knowledge graph. The
  container is created and destroyed by the test run — nothing to start
  beforehand, nothing left behind. It needs Docker and the `neo4j:5.26-community`
  image; unit runs (`uv run pytest`) collect no integration tests and start no
  container.

Run a specific test file:

```bash
uv run pytest tests/test_tools.py -v
```

## Troubleshooting

### Services won't start

Check Docker is running and ports are not in use:

```bash
docker compose ps
lsof -i :6333 -i :7474 -i :7687 -i :5432
```

### Neo4j authentication errors

Ensure `NEO4J_PASSWORD` in `.env` matches the Docker Compose configuration. If you changed the password after first run, remove the volume and restart:

```bash
docker compose down -v
docker compose up -d
```

### Qdrant connection refused

Verify Qdrant is healthy:

```bash
curl http://localhost:6333/healthz
```

### Pipeline hangs on S3 source

Only relevant when `DOCUMENT_SOURCE=s3`. Ensure `S3_BUCKET_NAME` is configured (the pipeline fails fast at startup otherwise). `S3_SQS_QUEUE_URL` is optional — without it, `task ingest-live` degrades to a sweep every `S3_FULL_SCAN_INTERVAL_HOURS` (default 24), which looks a lot like a hang. For S3 testing without real AWS, use [LocalStack](aws-setup.md#localstack-local-development). For purely local work, leave `DOCUMENT_SOURCE` unset (defaults to `local`).

### Import errors

Ensure dependencies are installed:

```bash
uv sync --extra gliner
```

`No module named 'gliner2'` specifically means a bare `uv sync` pruned the
optional extra — see [step 2](#2-install-python-dependencies).

### Port conflicts

Change default ports in `.env`:

```bash
MCP_PORT=9000
```

For infrastructure ports, modify `docker-compose.yml` host port mappings.
