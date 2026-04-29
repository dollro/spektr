# Plan: Expose MCP via Traefik

## Goal

Wire `htgf.spektr.everken.com` → Traefik → Spektr MCP container, secured with bearer token.

## Decisions

- Subdomain: `htgf.spektr.everken.com`
- No CrowdSec bouncer on this route
- Only MCP exposed (no agent-api, no live-ingest)
- Bulk ingest via cron (`task prod:ingest`), not HTTP — no Traefik route needed
- Drop Caddy from prod compose (Traefik replaces it)

## Changes

### 1. `docker-compose.prod.yml`

- Remove `caddy` service + `caddy_data`/`caddy_config` volumes
- Add external `proxy` network (Traefik's network)
- `mcp` service: join both `spektr-net` + `proxy`, add Traefik labels
- Data services stay on `spektr-net` only (not exposed)

### 2. `.env.prod`

- Generate `MCP_API_KEY` via `secrets.token_urlsafe(48)`
- Set `MCP_TRANSPORT=http`
- Set `MCP_HOST=0.0.0.0`, `MCP_PORT=8080`, `MCP_PATH=/mcp`

### 3. DNS

- A/AAAA record: `htgf.spektr.everken.com` → server public IP

### 4. Cron for bulk ingest

- Schedule `task prod:ingest` at desired interval

## Auth layers

| Layer | Responsibility |
|-------|---------------|
| Traefik TLS | HTTPS termination, Let's Encrypt cert |
| BearerAuthMiddleware | App-level, rejects `tools/call` without valid token |
