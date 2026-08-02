# Infrastructure Services

Spektr uses two infrastructure services managed via Docker Compose. All service images are pinned to specific versions for reproducibility.

!!! note "Two compose files"
    `docker-compose.yml` is the **development** stack — it runs only Qdrant and Neo4j with host ports published, so the app processes (`task serve`, `task ingest-live`) run natively on the host and connect to `localhost`. For the production stack (app + data + reverse proxy, all containerized), see [Production Deployment](../deployment/production.md) and `docker-compose.prod.yml`.

## Docker Compose Overview

Start all services:

```bash
docker compose up -d
```

Stop all services (data is preserved in named volumes):

```bash
docker compose down
```

Stop and remove all data:

```bash
docker compose down -v
```

## Services

### Qdrant (Vector Store)

| Property | Value |
|-|-|
| Image | `qdrant/qdrant:v1.17.0` |
| HTTP API | `localhost:6333` |
| gRPC API | `localhost:6334` |
| Volume | `qdrant_data:/qdrant/storage` |
| Health check | `http://localhost:6333/healthz` |

Qdrant stores two collections:

- **`documents_dense`** — dense vectors. Dimensionality follows the active embedding provider (`JINA_DENSE_DIMENSIONS` default `2048`, `VOYAGE_DENSE_DIMENSIONS` default `1024`, `OPENROUTER_DENSE_DIMENSIONS` default `3072`)
- **`documents_multivec`** — 128-dimensional ColBERT multi-vectors (Jina v4 only, opt-in via `MULTIVEC_ENABLED=true`)

Collections are provisioned automatically by the ingestion pipeline on first run.

### Neo4j (Knowledge Graph)

| Property | Value |
|-|-|
| Image | `neo4j:5.26-community` |
| Browser UI | `http://localhost:7474` |
| Bolt protocol | `bolt://localhost:7687` |
| Volume | `neo4j_data:/data` |
| Health check | `http://localhost:7474` |
| Plugins | APOC |

The Neo4j password is set via the `NEO4J_PASSWORD` environment variable (defaults to `password` in Docker Compose). The APOC plugin is installed automatically for advanced graph operations used by Graphiti.

## Pipeline State (no service)

CocoIndex keeps its target-state ledger, memoization cache and component tree in a local **LMDB directory** — there is no database service for it.

| Property | Value |
|-|-|
| Setting | `COCOINDEX_DB_PATH` |
| Default | `state/cocoindex.db` (a *directory*, not a file) |
| Map size | `COCOINDEX_LMDB_MAP_SIZE`, read by CocoIndex itself; default 4 GiB |

The path lives under `state/` so the `ingest_state` volume covers it in the production stack. LMDB has no safe hot-copy — see [Backup and Restore](../operations/backup-restore.md#cocoindex-state-lmdb).

## Persistent Volumes

All data is stored in Docker named volumes:

| Volume | Service | Contents |
|-|-|-|
| `qdrant_data` | Qdrant | Vector index and collection data |
| `neo4j_data` | Neo4j | Graph database files |

Volumes persist across `docker compose down`. Use `docker compose down -v` to remove them.

The production stack adds an `ingest_state` volume mounted on every service that runs the pipeline; it holds `state/ingestion_failures.db` and the CocoIndex LMDB directory.

## Health Check Script

The `scripts/wait-for-services.sh` script polls the services until they report healthy (or times out after 30 seconds):

```bash
./scripts/wait-for-services.sh
```

Output:

```
Waiting for services to become healthy...
Waiting... (0s / 30s) qdrant=false neo4j=false postgres=false
Waiting... (2s / 30s) qdrant=true neo4j=false postgres=false
All services are healthy.
```

!!! warning "The script still probes PostgreSQL"
    `scripts/wait-for-services.sh` has not been updated for the LMDB-backed pipeline state and still requires a `pg_isready` check to pass, so it will time out against the current compose stack. Use `docker compose ps` or the manual checks below until the script is fixed.

The script checks:

| Service | Check method |
|-|-|
| Qdrant | `curl -sf http://localhost:6333/healthz` |
| Neo4j | `curl -sf http://localhost:7474` |
| PostgreSQL | `pg_isready -h localhost -p 5432 -U cocoindex` (obsolete — see warning above) |

## Manual Health Checks

```bash
# Qdrant
curl http://localhost:6333/healthz

# Neo4j
curl http://localhost:7474
```

For environment variable configuration of these services, see [Environment Variables](environment.md).
