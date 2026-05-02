# Backup and Restore

Spektr's state lives in three Docker volumes: Qdrant (vectors), Neo4j (knowledge graph), Postgres (CocoIndex pipeline state). All three are captured by `task backup`; `task restore` reverses the process.

!!! note "Dev vs prod"
    `task backup` / `task restore` operate on the **dev** compose stack (`docker-compose.yml`). For the production stack (`docker-compose.prod.yml`) use `task prod:backup` / `task prod:restore` instead — same output format, just with `--compose-file docker-compose.prod.yml` threaded through and `QDRANT_URL` pointed at the host-local Qdrant publish. See [Production Deployment](../deployment/production.md#backups) for details.

## Backup

```bash
task backup                         # everything (dev stack)
task backup -- qdrant               # just Qdrant
task backup -- postgres             # just Postgres
task backup -- all --prune-older-than 30
task prod:backup                    # same, against docker-compose.prod.yml
```

Artefacts land in `./backups/<YYYYMMDD-HHMMSS>/`:

```
backups/20260419-153000/
├── manifest.json
├── qdrant/
│   ├── documents_dense_<snap>.snapshot
│   └── documents_multivec_<snap>.snapshot     (if enabled)
├── neo4j/
│   └── neo4j.dump                             (or a backup dir structure)
└── postgres/
    └── cocoindex.dump
```

The `manifest.json` records service versions, collection names, and per-artefact sizes. `task restore` reads it first and refuses if it's missing.

## Restore

Destructive: Qdrant collections are deleted first, Neo4j is loaded with `--overwrite-destination=true`, Postgres gets `pg_restore --clean --if-exists`.

```bash
task restore -- --from backups/20260419-153000 \
    --target all \
    --yes-i-know-this-wipes-things
```

Without the `--yes-i-know-this-wipes-things` flag the script refuses to run. `--target {qdrant,neo4j,postgres,all}` lets you restore piecewise.

### Neo4j restore requires service downtime

Neo4j Community Edition doesn't support online restore. `task restore` stops the neo4j container, runs `neo4j-admin database load`, and restarts. Expect a ~10-30s gap during which any MCP graph query will fail.

### Postgres restore rewinds CocoIndex state

After a postgres-only restore, the `ragingestion__cocoindex_tracking` table reflects whatever was tracked at backup time. If the surrounding Qdrant/Neo4j state is newer, run `task doctor` and then `task doctor-fix` to reconcile.

## Recommended cadence

For a single-node production deployment:

- **Nightly**: `task backup -- all --prune-older-than 30` via cron.
- **Before any risky change**: model swap, schema migration, Neo4j version bump.
- **Weekly**: ship a copy to off-host storage (e.g. `rclone sync backups/ s3:spektr-backups/`).

## Round-trip drill

Periodically verify the pipeline actually works end-to-end:

```bash
task backup
docker compose down -v              # wipe volumes
docker compose up -d
task restore -- --from backups/<ts> --target all --yes-i-know-this-wipes-things
task doctor                         # should exit 0
task smoke                          # should return same top-k as before
```

If `task smoke` scores change significantly after restore, something is off — investigate before trusting backups for real recovery.

## Recovery time expectations

With a small corpus (a handful of PDFs):
- Qdrant snapshot + restore: ~seconds per collection.
- Neo4j dump + load: ~10-30s total downtime.
- Postgres dump + restore: near-instant.

At 100 GB of Qdrant data, expect 10-30 minutes for the Qdrant snapshot + restore round trip; plan maintenance windows accordingly.
