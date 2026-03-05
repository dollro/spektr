# Infrastructure Services

Spektr uses three infrastructure services managed via Docker Compose. All service images are pinned to specific versions for reproducibility.

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
| Image | `qdrant/qdrant:v1.13.2` |
| HTTP API | `localhost:6333` |
| gRPC API | `localhost:6334` |
| Volume | `qdrant_data:/qdrant/storage` |
| Health check | `http://localhost:6333/healthz` |

Qdrant stores two collections:

- **`documents_dense`** — 2048-dimensional dense vectors (Jina v4 single-vector)
- **`documents_multivec`** — 128-dimensional ColBERT multi-vectors (Jina v4)

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

### PostgreSQL (Pipeline State)

| Property | Value |
|-|-|
| Image | `postgres:17.2` |
| Port | `localhost:5432` |
| Volume | `postgres_data:/var/lib/postgresql/data` |
| Health check | `pg_isready -h localhost -p 5432 -U cocoindex` |
| Default database | `cocoindex` |
| Default user | `cocoindex` |
| Default password | `cocoindex` |

PostgreSQL stores CocoIndex pipeline state (checkpoint tracking, deduplication).

## Persistent Volumes

All data is stored in Docker named volumes:

| Volume | Service | Contents |
|-|-|-|
| `qdrant_data` | Qdrant | Vector index and collection data |
| `neo4j_data` | Neo4j | Graph database files |
| `postgres_data` | PostgreSQL | Relational database files |

Volumes persist across `docker compose down`. Use `docker compose down -v` to remove them.

## Health Check Script

The `scripts/wait-for-services.sh` script polls all three services until they report healthy (or times out after 30 seconds):

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

The script checks:

| Service | Check method |
|-|-|
| Qdrant | `curl -sf http://localhost:6333/healthz` |
| Neo4j | `curl -sf http://localhost:7474` |
| PostgreSQL | `pg_isready -h localhost -p 5432 -U cocoindex` |

## Manual Health Checks

```bash
# Qdrant
curl http://localhost:6333/healthz

# Neo4j
curl http://localhost:7474

# PostgreSQL
pg_isready -h localhost -p 5432 -U cocoindex
```

For environment variable configuration of these services, see [Environment Variables](environment.md).
