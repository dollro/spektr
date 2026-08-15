# Upgrading a Running Deployment

How to move an existing production instance across the **CocoIndex v0 → v1** boundary — the release that removed PostgreSQL and changed the `documents_dense` vector configuration.

Use this when you are upgrading a deployment that already holds data. If you are standing up a new environment, you want [First Ingest](first-ingest.md) instead; if you are only changing embedding provider or dimensionality on an already-v1 stack, you want [Re-indexing](reindex.md).

!!! danger "There is no in-place upgrade"
    Three of the changes below are not migrations — they are rebuilds. Plan a maintenance window. Search is unavailable from the moment you drop the collection until the first re-ingest completes, which scales with corpus size and your embedding provider's rate limits.

## Am I on the old version?

```bash
docker compose -f docker-compose.prod.yml ps | grep postgres
grep -c DATABASE_URL .env.prod
```

Either one matching means you are on the pre-v1 stack.

## What changed, and why it forces a rebuild

| Change | Consequence |
|-|-|
| PostgreSQL removed | The ledger, memo cache and component tree moved to an LMDB directory (`COCOINDEX_DB_PATH`, default `state/cocoindex.db`) on the existing `ingest_state` volume. The old tracking table cannot be carried forward. |
| `documents_dense` vector config | Moved from a single *unnamed* vector to named `dense` + sparse `sparse` (miniCOIL, IDF). Named and unnamed vectors cannot coexist in one collection, and Qdrant cannot alter it in place. |
| Payload schema | Points now carry `embedder_model` and `embedder_dim`. `task prod:doctor` reports pre-versioning points as "unversioned". |
| MCP tool surface | `multi_search` added, `hybrid_search` reworked onto fused retrieval, and `tools/list` now requires authentication. **Existing clients break.** See [Contract & Client Migration](../server/contract.md). |

The first two are why "just pull and restart" cannot work: the new code has nowhere to read its old state from, and the collection it expects has a shape the old one cannot take.

## Decide the blast radius

Both paths rebuild `documents_dense`. They differ on the knowledge graph.

**Wipe everything (recommended).** Simplest, and the only option that leaves no ambiguity about what survived.

**Keep Neo4j.** Tempting, since the graph and the vector store are independent. But with the default `GRAPH_ENGINE=graphiti`, re-ingesting writes episodes again, and Graphiti's episodic nodes have no uniqueness constraint in `ingestion/neo4j_setup.py` — only `:Document`, `:Entity` and `:Chunk` do. Re-ingesting your whole corpus on top of a surviving graph will therefore accumulate duplicate episodes. Keep the graph only if you are re-ingesting with `GRAPH_ENABLED=false`, or if you have verified the duplication behaviour for your engine.

## Procedure

Commands assume `/opt/spektr` on the VM. ☠ marks irreversible steps.

### 1. Back up, unless you have decided otherwise

```bash
task prod:backup
```

Captures Qdrant, Neo4j and the CocoIndex state to `./backups/<ts>/`. Stop `ingest-live` first — LMDB has no safe hot-copy. See [Backup & Restore](backup-restore.md).

Note that a backup taken here is a backup of the *old* schema. It is a way back to the previous version, not something the new version can consume.

### 2. Wipe — before you pull ☠

!!! warning "Ordering is load-bearing"
    Run the teardown while the checkout is still on the **old** revision. Its compose file is the last one that declares the `postgres` service and `postgres_data` volume. Pull first and `down -v` can no longer see them: Compose only removes volumes the current file declares, and you are left with an orphaned volume no tooling will ever clean up.

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod \
  --profile oneshot down -v --remove-orphans        # ☠
docker volume ls | grep spektr                      # expect no output
```

After the upgrade, `task prod:nuke -- --yes-i-know-this-wipes-things` does the same job and additionally removes `spektr_postgres_data` by name, precisely because the current compose file can no longer reach it.

### 3. Pull the new code

```bash
git fetch --all --prune
git checkout develop && git pull --ff-only
```

### 4. Migrate the environment file

```bash
python3 scripts/migrate_env.py .env.prod            # report first
python3 scripts/migrate_env.py .env.prod --write    # apply, keeps .env.prod.bak
```

Drops the dead `POSTGRES_*` / `DATABASE_URL` block, adds the new LMDB and S3-trigger variables, and validates what is required. It exits 1 if the result is still invalid. Full detail in [Production Deployment](../deployment/production.md#migrating-an-existing-envprod).

Two things it cannot do for you: it will not invent secrets the old stack never had (`MCP_API_KEY` on a previously unauthenticated deployment, for instance), and it will not decide your `DOCUMENT_SOURCE`. Both surface as reported problems.

### 5. Build and start the data services

```bash
task prod:build
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d qdrant neo4j
docker compose -f docker-compose.prod.yml --env-file .env.prod \
  exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN apoc.version();"
```

Retry the last command until APOC answers — typically 60–120s. There are no healthchecks and no `depends_on: condition:` in the compose files, and `ingestion/neo4j_setup.py::create_neo4j_schema` fails hard when APOC has not loaded yet. Combined with `restart: unless-stopped`, starting the ingest too early produces a crash-loop rather than a clear error.

### 6. Re-ingest

```bash
task prod:ingest ; echo "exit=$?"
```

This is the step that recreates the collections and the Neo4j constraints: `ingestion/runner.py::_provision()` is the only code that provisions them, and neither `mcp` nor `agent-api` ever does. Nothing is searchable until it succeeds.

Exit 0 means zero per-file errors. Exit 1 means some files failed and the run continued — `app.update()` never raises on a per-file failure, so the exit code is the signal, not the absence of a traceback. See [Ingestion Failure Semantics](atomicity.md).

Expect a long first run: every document is re-embedded, and miniCOIL downloads on first use.

### 7. Verify before exposing the stack

```bash
curl -s localhost:6333/collections/documents_dense | jq '.result.config.params'
```

Expect `vectors.dense` (your provider's dimensionality, `Cosine`) **and** `sparse_vectors.sparse` with `"modifier": "idf"`. A collection with only a dense vector means the upgrade silently half-applied.

```bash
task prod:doctor
```

Expect no drift, no mixed `embedder_model`/`embedder_dim`, and no text chunks missing their `sparse` vector. Then bring up the application services and check the public path:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d mcp agent-api
task prod:kb:list -- --limit 20
task prod:ask -- 'name three documents in the knowledge base'
```

[First Ingest](first-ingest.md#3-verify) has the full layer-by-layer verification if any of these disappoint.

### 8. Start the live path

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d ingest-live
```

With `DOCUMENT_SOURCE=s3` this is the SQS-triggered catch-up daemon, not an HTTP endpoint. Confirm `sharepoint-sync` is *absent* from `ps` — it now sits behind the `sharepoint` profile.

### 9. Migrate the clients

`tools/list` requires a bearer token from this release on, so every MCP client configuration needs `MCP_API_KEY` before it will enumerate tools again. Announce this: from the client's perspective the server simply stops working, and the failure looks like an outage rather than a config change. [Contract & Client Migration](../server/contract.md) has the details.

## Rolling back

Code rollback is straightforward; data rollback is not.

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod down --remove-orphans
git checkout <old-revision>
cp .env.prod.bak .env.prod
task prod:build && task prod:up
```

That returns you to the old stack with empty volumes. You then restore the backup from step 1 with `task prod:restore -- --from backups/<ts> --target all --yes-i-know-this-wipes-things`, which re-provisions the Postgres-era state the old code expects.

If you skipped the backup there is no way back to the previous corpus — fix forward instead.

## Related

- [First Ingest](first-ingest.md) — clean-slate bring-up and per-layer verification
- [Re-indexing](reindex.md) — rebuilding a corpus after a config change, narrower blast radius
- [Backup & Restore](backup-restore.md) — capturing state before a destructive operation
- [Production Deployment](../deployment/production.md) — the deployment itself, and the env migration script
- [Contract & Client Migration](../server/contract.md) — what changed for MCP clients
