# Backup and Restore

Spektr's state lives in three places: Qdrant (vectors), Neo4j (knowledge graph), and CocoIndex's LMDB state directory (`COCOINDEX_DB_PATH`, default `state/cocoindex.db` — the target-state ledger and memoization cache). All three are captured by `task backup`; `task restore` reverses the process.

!!! note "Dev vs prod"
    `task backup` / `task restore` operate on the **dev** compose stack (`docker-compose.yml`). For the production stack (`docker-compose.prod.yml`) use `task prod:backup` / `task prod:restore` instead — same output format, just with `--compose-file docker-compose.prod.yml` threaded through and `QDRANT_URL` pointed at the host-local Qdrant publish. See [Production Deployment](../deployment/production.md#backups) for details.

## Backup

```bash
task backup                         # everything (dev stack)
task backup -- qdrant               # just Qdrant
task backup -- cocoindex            # just the CocoIndex LMDB state
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
└── cocoindex/
    └── cocoindex-state.tar.gz
```

The `manifest.json` records service versions, collection names, and per-artefact sizes. `task restore` reads it first and refuses if it's missing.

## Restore

Destructive: Qdrant collections are deleted first, Neo4j is loaded with `--overwrite-destination=true`, and the CocoIndex state directory is removed and replaced by the archived one.

```bash
task restore -- --from backups/20260419-153000 \
    --target all \
    --yes-i-know-this-wipes-things
```

Without the `--yes-i-know-this-wipes-things` flag the script refuses to run. `--target {qdrant,neo4j,cocoindex,all}` lets you restore piecewise.

### Neo4j restore requires service downtime

Neo4j Community Edition doesn't support online restore. `task restore` stops the neo4j container, runs `neo4j-admin database load`, and restarts. Expect a ~10-30s gap during which any MCP graph query will fail.

### CocoIndex state (LMDB)

LMDB offers no safe hot-copy: copying `data.mdb` while a writer is mid-transaction can capture a torn page. **Stop the ingest process before backing up**, or take the backup between scheduled ingests. The same applies to restore — swapping the directory underneath a live writer corrupts it, because LMDB keeps a lock file and open readers.

Losing this state is recoverable but expensive: it costs a full reprocess, not data loss, because Qdrant point ids are deterministic (uuid5 over `source_file::pX::cY`), so re-ingesting rewrites the same points rather than duplicating them.

After a cocoindex-only restore, the ledger reflects whatever was tracked at backup time. If the surrounding Qdrant/Neo4j state is newer, run `task doctor` and then `task doctor-fix` to reconcile.

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
- CocoIndex state tar + untar: near-instant.

At 100 GB of Qdrant data, expect 10-30 minutes for the Qdrant snapshot + restore round trip; plan maintenance windows accordingly.
