# Production Deployment (Docker Compose)

This page describes the fully containerized deployment for a single Linux VM. The dev flow (`task up` + `task serve` on the host) is unchanged and recommended for iteration — this page is for turning a VM into a running Spektr instance.

## Architecture

Everything runs inside one Docker Compose project on a private `spektr-net`. Only Caddy publishes ports to the internet; data services are not reachable from outside the network.

```
Internet
   │
   ▼ :80 / :443
┌─────────┐
│  caddy  │  auto-TLS (Let's Encrypt)
└────┬────┘
     │ spektr-net
     ├──► mcp          (python -m server.mcp_server, :8080)
     ├──► agent-api    (python -m agent.api,         :8001)
     └──► ingest-live  (python -m ingestion.pipeline --live)
              │
              ├──► qdrant    (:6333, internal only)
              ├──► neo4j     (:7687, internal only)
              └──► postgres  (:5432, internal only)
```

The three app services (`mcp`, `agent-api`, `ingest-live`) all share a single image built from the repo `Dockerfile`.

## Files

| File | Purpose |
|-|-|
| `Dockerfile` | Multi-stage Python 3.13 + uv image. Non-root user, tini as PID 1 |
| `.dockerignore` | Keeps the build context small (excludes `documents/`, `backups/`, `state/`, …) |
| `docker-compose.prod.yml` | Full production stack |
| `Caddyfile` | Reverse proxy config with auto-TLS — **edit before deploy** |
| `.env.example` | Template; copy to `.env.prod` on the VM and override hostnames + secrets |

## Prerequisites on the VM

| Tool | Why |
|-|-|
| Docker Engine 24+ | Container runtime |
| Docker Compose plugin v2+ | `docker compose ...` |
| go-task | Task shortcuts (`task prod:up` …). Optional — you can call `docker compose` directly. |
| Public DNS | A/AAAA records for `mcp.yourdomain.tld` + `agent.yourdomain.tld` pointing at the VM |
| Open ports | 80 and 443 inbound (ACME HTTP-01 + TLS traffic) |

If you're fronting Spektr with an existing reverse proxy (nginx, Traefik on the host, a cloud load balancer), skip Caddy — see [Without Caddy](#without-caddy) below.

## Deploy

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

- `NEO4J_PASSWORD`, `POSTGRES_PASSWORD` — strong, randomly generated
- `JINA_API_KEY`
- `LLM_API_KEY`
- `MCP_API_KEY` — Bearer token clients must present
- `INGEST_API_KEY` — gates `/session/start` on the live-ingest endpoint
- AWS block (`AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_SQS_QUEUE_URL`) if you're using live ingestion

Service hostnames use container names: `qdrant`, `neo4j`, `postgres`. Do not change these — other services resolve each other by name on `spektr-net`.

### 3. Configure Caddy

Edit `Caddyfile`:

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

### 4. Build the image and start the stack

```bash
task prod:build
task prod:up
```

Or without go-task:

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d
```

### 5. Verify

```bash
task prod:ps                 # docker compose ps
task prod:logs -- mcp        # follow mcp logs
curl https://mcp.yourdomain.tld/health   # or whatever health endpoint you expose
```

Caddy takes ~10-60 seconds to obtain a certificate the first time.

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

### Stopping

```bash
task prod:down               # keeps volumes (qdrant_data, neo4j_data, postgres_data)
```

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

- `scripts/backup.py --compose-file docker-compose.prod.yml` threads `-f docker-compose.prod.yml` into every `docker compose ...` shell-out (neo4j stop/run/start, postgres exec).
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

## Traefik Integration (default)

The prod compose is designed for an external Traefik instance. The `mcp` service joins Traefik's `proxy` network and declares routing labels. Caddy is not included.

Set `MCP_PUBLIC_DOMAIN` in `.env.prod` to the domain Traefik should route:

```bash
MCP_PUBLIC_DOMAIN=mcp.example.com
```

Traefik auto-discovers the container via Docker labels and provisions a Let's Encrypt certificate. The `MCP_API_KEY` bearer token handles application-level auth.

## Without Traefik

If you don't have an external Traefik, publish the app services on localhost and front them with your own reverse proxy (nginx, Caddy, cloud LB):

```yaml
mcp:
  ports:
    - "127.0.0.1:8080:8080"

agent-api:
  ports:
    - "127.0.0.1:8001:8001"
```

Remove the `proxy` external network and Traefik labels from the `mcp` service.

## Sizing

Rough baseline for a small instance:

| Service | Memory | Notes |
|-|-|-|
| qdrant | 1-2 GB | grows with corpus size |
| neo4j | 1-2 GB | plus APOC plugin |
| postgres | 256 MB | CocoIndex tracking only |
| mcp | 512 MB | mostly idle |
| agent-api | 512 MB | |
| ingest-live | 1-2 GB | spikes during extraction |
| caddy | 64 MB | |

A 4 vCPU / 8 GB VM handles a modest deployment. Bulk ingestion is the CPU-heavy step — run it off-peak or on a larger instance.

## Image size

The app image weighs in at ~2-3 GB because of `docling`, `onnxruntime`, and `pymupdf`. All three app services share the same image, so you only pay for it once on disk. If that becomes a constraint, we can split into a slim `spektr-mcp` image (no ingestion deps) and a full `spektr-ingest` image — not worth the complexity today.

## Troubleshooting

### Caddy can't get a certificate

- Check that ports 80 and 443 are open on the VM (security group / UFW).
- Verify `mcp.yourdomain.tld` and `agent.yourdomain.tld` resolve to the VM's public IP.
- Inspect: `task prod:logs -- caddy`.

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
