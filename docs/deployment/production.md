# Production Deployment (Docker Compose)

This page describes the fully containerized deployment for a single Linux VM. The dev flow (`task up` + `task serve` on the host) is unchanged and recommended for iteration — this page is for turning a VM into a running Spektr instance.

## Architecture

Everything runs inside one Docker Compose project on a private `spektr-net`. The `mcp` service additionally joins an external `proxy` network so an existing Traefik instance can route traffic to it. Data services are not reachable from outside `spektr-net`.

```
Internet
   │
   ▼ :80 / :443
┌─────────────────┐
│  Traefik        │  external instance, auto-TLS (Let's Encrypt)
│  (proxy net)    │
└────┬────────────┘
     │ proxy network (external)
     ▼
┌─────────────────┐
│  mcp            │  python -m server.mcp_server, :8080
└────┬────────────┘
     │ spektr-net
     ├──► agent-api       (python -m agent.api,                :8001)
     ├──► ingest-live     (python -m ingestion.pipeline --live)
     ├──► sharepoint-sync (python -m services.sharepoint_sync, profile: sharepoint)
     │
     ├──► qdrant    (:6333, internal only; 127.0.0.1 publish for backup scripts)
     └──► neo4j     (:7687, internal only)
```

CocoIndex's pipeline state needs no service: it lives in an LMDB directory under `state/`, on the `ingest_state` volume shared by every service that runs the pipeline.

The app services (`mcp`, `agent-api`, `ingest-live`, `sharepoint-sync`, one-shot `ingest`) all share a single image built from the repo `Dockerfile`.

## Files

| File | Purpose |
|-|-|
| `Dockerfile` | Multi-stage Python 3.13 + uv image. Non-root user, tini as PID 1 |
| `.dockerignore` | Keeps the build context small (excludes `documents/`, `backups/`, `state/`, …) |
| `docker-compose.prod.yml` | Full production stack with Traefik labels on `mcp` |
| `Caddyfile` | Optional sample config for a self-managed Caddy reverse proxy — only used if you choose the "Alternative: Caddy" path below |
| `.env.example` | Template; copy to `.env.prod` on the VM and override hostnames + secrets |

## Prerequisites on the VM

| Tool | Why |
|-|-|
| Docker Engine 24+ | Container runtime |
| Docker Compose plugin v2+ | `docker compose ...` |
| go-task | Task shortcuts (`task prod:up` …). Optional — you can call `docker compose` directly. |
| External Traefik | An existing Traefik instance with a Docker-attached external network named `proxy`, configured with a `tls_resolver` certresolver (e.g. Let's Encrypt). Spektr's `mcp` service joins this network. |
| Public DNS | A/AAAA record for `${MCP_PUBLIC_DOMAIN}` (and any other hostnames you front) pointing at the Traefik VM |

If you don't have an external Traefik, see [Alternative: Caddy](#alternative-caddy) or [Alternative: any other reverse proxy](#alternative-any-other-reverse-proxy).

## Deploy (Traefik — default)

### 1. Clone the repo on the VM

```bash
git clone <repo-url> /opt/spektr
cd /opt/spektr
```

### 2. Configure environment

```bash
cp .env.example .env.prod
```

Edit `.env.prod`. Required values in production:

- `NEO4J_PASSWORD` — strong, randomly generated
- `JINA_API_KEY` (or `VOYAGE_API_KEY` / `OPENROUTER_API_KEY` depending on `EMBEDDING_PROVIDER`)
- `LLM_API_KEY`
- `MCP_API_KEY` — Bearer token clients must present
- `MCP_PUBLIC_DOMAIN` — public hostname Traefik should route to `mcp` (e.g. `mcp.example.com`)
- `INGEST_API_KEY` — gates `/session/start` on the live-ingest endpoint
- AWS block (`AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`) when `DOCUMENT_SOURCE=s3`; add `S3_SQS_QUEUE_URL` unless you're happy with interval-only sweeps
- SharePoint block (`SHAREPOINT_*`) when `DOCUMENT_SOURCE=sharepoint`

Service hostnames use container names: `qdrant`, `neo4j`. Do not change these — other services resolve each other by name on `spektr-net`.

### 3. Build the image and start the stack

```bash
task prod:build
task prod:up
```

Or without go-task:

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d
```

Traefik auto-discovers the `mcp` container via the Docker labels in `docker-compose.prod.yml` and provisions a Let's Encrypt certificate the first time the route is hit. The labels also disable response buffering so streamable-http / SSE responses flow through immediately.

### 4. Verify

```bash
task prod:ps                          # docker compose ps
task prod:logs -- mcp                 # follow mcp logs
curl https://${MCP_PUBLIC_DOMAIN}/mcp # Traefik routes to mcp:8080
```

## Day-2 operations

### Bulk ingestion (one-shot batch)

The `ingest` service is behind the `oneshot` profile and does not start with `prod:up`. Run it on demand:

```bash
task prod:ingest
# or:
docker compose -f docker-compose.prod.yml --profile oneshot run --rm ingest
```

Mount host documents into the container via the volume already declared in compose (`./documents:/app/documents:ro`).

### Logs

```bash
task prod:logs               # all services
task prod:logs -- mcp        # one service
task prod:logs -- "--tail=200 ingest-live"
```

### Updating the stack

```bash
cd /opt/spektr
git pull
task prod:build
task prod:up                 # recreates containers with the new image
```

If the pull crosses the CocoIndex v0→v1 boundary (the release that removed PostgreSQL), migrate `.env.prod` first — see below — and expect to rebuild the corpus: the `documents_dense` vector config changed and cannot be migrated in place. [Re-indexing](../operations/reindex.md) covers the collection rebuild; [First Ingest](../operations/first-ingest.md) covers a clean slate.

### Migrating an existing `.env.prod`

An env file written for the old stack carries variables nothing reads any more and is missing ones the new pipeline expects. Neither is reported at boot — the pipeline just runs on defaults. `scripts/migrate_env.py` reconciles it:

```bash
python3 scripts/migrate_env.py .env.prod          # report only, changes nothing
python3 scripts/migrate_env.py .env.prod --write  # rewrite in place, keeps .env.prod.bak
task prod:migrate-env -- --write                  # same, via go-task
```

Unlike the rest of `scripts/`, this one is stdlib-only and runs under plain `python3` — a deploy VM has Docker but usually no project virtualenv. Your comments, ordering and untouched lines are preserved; the output is written mode 600. Values are never printed, only variable names, so the report is safe to paste into a ticket.

What it does:

| Action | Detail |
|-|-|
| Drops | `DATABASE_URL`, `POSTGRES_USER/PASSWORD/DB/PORT/HOST` — dead since the LMDB ledger replaced PostgreSQL |
| Adds | `COCOINDEX_DB_PATH`, `PIPELINE_MAX_CONCURRENT_FILES`, `S3_PREFIX`, `S3_SQS_DEBOUNCE_SECONDS`, `S3_FULL_SCAN_INTERVAL_HOURS`, at their documented defaults |
| Retunes | `JINA_DENSE_DIMENSIONS` 512 → 2048 and `LLM_MODEL` → `claude-sonnet-5`, but *only* where the file holds exactly the stale value, so a deliberate override survives |
| Flags | `QDRANT_DENSE_COLLECTION` / `QDRANT_MULTIVEC_COLLECTION` — test-suite overrides that point production at throwaway collections |
| Validates | required variables, the embedding key matching `EMBEDDING_PROVIDER`, and `S3_BUCKET_NAME` when `DOCUMENT_SOURCE=s3` |

It exits 1 when the migrated file would still be invalid, so a deploy script can gate on it. Re-running against an already-migrated file is a no-op, which makes it safe in automation.

The migration does not invent secrets. Anything the new schema needs but the old file never had — `MCP_API_KEY` on a stack that ran unauthenticated, for instance — is reported as a problem for you to fill in.

### SharePoint sync

`sharepoint-sync` is behind the `sharepoint` profile, so `task prod:up` skips it. It refuses to start unless `DOCUMENT_SOURCE=sharepoint` (exit 2), which combined with `restart: unless-stopped` would crash-loop on a local or S3 deployment. When you do want it:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod --profile sharepoint up -d
```

### Health check

```bash
task prod:doctor             # diffs the CocoIndex ledger against Qdrant, inside the stack
```

Unlike `task doctor`, this runs in a container, so the VM needs no `uv` or Python environment. It flags drift, mixed `embedder_model`/`embedder_dim`, and text chunks missing their `sparse` vector.

### Stopping

```bash
task prod:down               # keeps volumes (qdrant_data, neo4j_data, ingest_state)
```

### Wiping everything

For a clean slate — a schema change that cannot be migrated in place, or standing the instance back up from nothing:

```bash
task prod:nuke -- --yes-i-know-this-wipes-things
```

!!! danger "This deletes all data"
    Every vector, the whole graph, and all pipeline state. Run `task prod:backup` first unless you genuinely intend to lose it. The task refuses to run without the flag.

It brings the stack down with `-v --remove-orphans` across all profiles, then removes `spektr_postgres_data` explicitly — a leftover from the pre-CocoIndex-v1 stack that this compose file no longer declares and therefore cannot clean up on its own.

Afterwards, follow [First Ingest](../operations/first-ingest.md) from step 2: the collections and Neo4j constraints are recreated by the next ingestion run, since `ingestion/runner.py::_provision()` is the only code that provisions them. Nothing is searchable until that run succeeds.

### Backups

Two dedicated tasks target the prod compose file and use the host-local Qdrant publish:

```bash
task prod:backup                                       # snapshot into ./backups/<ts>/
task prod:backup -- --prune-older-than 14              # drop backups older than N days first
task prod:restore -- --from backups/20260419-153000 \
                    --target all \
                    --yes-i-know-this-wipes-things
```

Under the hood:

- `scripts/backup.py --compose-file docker-compose.prod.yml` threads `-f docker-compose.prod.yml` into every `docker compose ...` shell-out (neo4j stop/run/start).
- The CocoIndex state is archived by tarring `COCOINDEX_DB_PATH`. LMDB has no safe hot-copy, so stop `ingest-live` (or schedule the backup between ingests) before running it.
- `QDRANT_URL=http://127.0.0.1:6333` is exported by the task so the host-side script can reach the Qdrant HTTP snapshot API via the port the prod compose publishes on `127.0.0.1` only.
- Neo4j Community 5 has no online backup — the script stops the `neo4j` service for ~10-30s, runs `neo4j-admin database dump`, then starts it again. Plan backups outside peak traffic.
- Output lands in `./backups/<timestamp>/` on the VM with a `manifest.json`.

Schedule nightly via host cron:

```cron
# /etc/cron.d/spektr-backup
30 3 * * *  deploy  cd /opt/spektr && /usr/local/bin/task prod:backup -- --prune-older-than 14
```

See [Backup & Restore](../operations/backup-restore.md) for the full procedure (verification, retention, restore smoke test).

### Secrets

`.env.prod` lives on the VM and is gitignored. Keep it `chmod 600`, owner root (or the user running Docker). For multi-operator setups, graduate to Docker secrets / a vault — not included by default because a single-VM deploy does not need it.

## Alternative: any other reverse proxy

If you don't have an external Traefik, publish the app services on localhost and front them with your own reverse proxy (nginx, cloud LB, etc.):

```yaml
mcp:
  ports:
    - "127.0.0.1:8080:8080"

agent-api:
  ports:
    - "127.0.0.1:8001:8001"
```

Remove the `proxy` external network entry and the `traefik.*` labels from the `mcp` service. Then point your reverse proxy at `127.0.0.1:8080`.

## Alternative: Caddy

The repo ships a sample `Caddyfile` for users who prefer Caddy with built-in Let's Encrypt over Traefik. **Caddy is not part of `docker-compose.prod.yml`** — you have to add the service yourself. A minimal addition:

```yaml
caddy:
  image: caddy:2-alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./Caddyfile:/etc/caddy/Caddyfile:ro
    - caddy_data:/data
    - caddy_config:/config
  networks:
    - spektr-net
  restart: unless-stopped

# also add to top-level volumes:
# caddy_data:
# caddy_config:
```

Then drop the `proxy` external network and the `traefik.*` labels from the `mcp` service so it stays on `spektr-net` only, and edit `Caddyfile` to set the email and your two domains:

```caddyfile
{
    email you@yourdomain.tld
}

mcp.yourdomain.tld {
    reverse_proxy mcp:8080
}

agent.yourdomain.tld {
    reverse_proxy agent-api:8001
}
```

Caddy takes ~10-60 seconds to obtain a certificate the first time. Open ports 80 and 443 on the VM (security group / UFW) so ACME HTTP-01 succeeds.

## Sizing

Rough baseline for a small instance:

| Service | Memory | Notes |
|-|-|-|
| qdrant | 1-2 GB | grows with corpus size |
| neo4j | 1-2 GB | plus APOC plugin |
| mcp | 512 MB | mostly idle |
| agent-api | 512 MB | |
| ingest-live | 1-2 GB | spikes during extraction |

A 4 vCPU / 8 GB VM handles a modest deployment. Bulk ingestion is the CPU-heavy step — run it off-peak or on a larger instance.

## Image size

The app image weighs in at ~2-3 GB because of `docling`, `onnxruntime`, and `pymupdf`. All app services share the same image, so you only pay for it once on disk. If that becomes a constraint, we can split into a slim `spektr-mcp` image (no ingestion deps) and a full `spektr-ingest` image — not worth the complexity today.

## Troubleshooting

### Traefik isn't routing to mcp

- Confirm the external `proxy` network exists and Traefik is attached to it: `docker network inspect proxy`.
- Check Traefik picked up the labels: visit Traefik's dashboard or `docker logs <traefik-container>`.
- Verify `MCP_PUBLIC_DOMAIN` resolves to the Traefik VM's public IP.
- Inspect: `task prod:logs -- mcp` for app-side errors.

### MCP returns 401

`MCP_API_KEY` is set, clients must send `Authorization: Bearer <key>`. See [MCP Authentication](../server/authentication.md).

### A service keeps restarting

```bash
docker compose -f docker-compose.prod.yml logs <service>
```

Usually a missing required env var or an unreachable data service. Check `depends_on` ordering and that the data services passed their own health checks.

### Changes to `.env.prod` aren't picked up

Compose reads env vars at container start. Recreate:

```bash
task prod:up   # equivalent to `up -d`; compose recreates containers whose config changed
```
