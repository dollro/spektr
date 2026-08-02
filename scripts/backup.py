"""Snapshot Qdrant + Neo4j + CocoIndex state to ./backups/<timestamp>/.

Subcommands:
    qdrant     — snapshot every Qdrant collection via the native snapshot API
    neo4j      — neo4j-admin database backup + docker cp out
    cocoindex  — tar the LMDB state directory (ledger + memoization cache)
    all        — run every subcommand sequentially

The top-level manifest.json records service versions, collection names,
and per-artifact sizes so `restore.py` can validate before wiping state.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.settings import settings


def _now_stamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _make_dest(dest_root: Path | None = None, timestamp: str | None = None) -> Path:
    root = dest_root or Path("backups")
    ts = timestamp or _now_stamp()
    dest = root / ts
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _compose_cmd(compose_file: str | None, *args: str) -> list[str]:
    """Build a `docker compose [-f FILE] ...` command vector."""
    cmd = ["docker", "compose"]
    if compose_file:
        cmd += ["-f", compose_file]
    cmd.extend(args)
    return cmd


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def _qdrant_collections(client: httpx.Client) -> list[str]:
    """Return collection names actually present in Qdrant."""
    r = client.get(f"{settings.qdrant_url}/collections", timeout=30)
    r.raise_for_status()
    return [c["name"] for c in r.json()["result"]["collections"]]


def backup_qdrant(dest: Path) -> dict[str, Any]:
    """Snapshot each Qdrant collection and download the tarball.

    Qdrant's snapshot API:
        POST /collections/<name>/snapshots        -> {name: 'xxx.snapshot'}
        GET  /collections/<name>/snapshots/<name> -> tarball stream
    """
    out = dest / "qdrant"
    out.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []

    # Prefer the actually-present collections; fall back to config'd ones.
    with httpx.Client() as client:
        present = set(_qdrant_collections(client))
        wanted = [
            c for c in (DENSE_COLLECTION, MULTIVEC_COLLECTION) if c in present
        ]
        if not wanted:
            print("  (no configured collections exist; skipping Qdrant)")
            return {"collections": [], "status": "nothing-to-do"}

        for name in wanted:
            print(f"  snapshotting Qdrant:{name} …")
            r = client.post(
                f"{settings.qdrant_url}/collections/{name}/snapshots",
                timeout=300,
            )
            r.raise_for_status()
            snap_name = r.json()["result"]["name"]
            snap_path = out / f"{name}_{snap_name}"

            with (
                client.stream(
                    "GET",
                    f"{settings.qdrant_url}/collections/{name}/snapshots/{snap_name}",
                    timeout=600,
                ) as stream,
                snap_path.open("wb") as fh,
            ):
                stream.raise_for_status()
                for chunk in stream.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
            size = snap_path.stat().st_size
            print(f"    → {snap_path.name} ({size/1e6:.1f} MB)")
            results.append(
                {"collection": name, "file": snap_path.name, "bytes": size},
            )
    return {"collections": results, "status": "ok"}


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


def backup_neo4j(dest: Path, compose_file: str | None = None) -> dict[str, Any]:
    """Stop → dump via ephemeral neo4j container mounted onto the host → start.

    Neo4j Community 5 has no online `database backup`; `database dump` needs
    the target DB stopped. We accept ~10-30s of downtime, then restart.
    The ephemeral `docker compose run` container shares the neo4j data
    volume (because compose re-uses it) and we mount the host destination
    directly so the dump lands on the host without a separate docker cp.
    """
    out = dest / "neo4j"
    out.mkdir(exist_ok=True)
    out_abs = out.resolve()

    print("  stopping neo4j (required for offline dump) …")
    subprocess.run(_compose_cmd(compose_file, "stop", "neo4j"), check=True)
    try:
        print("  running neo4j-admin database dump …")
        r = subprocess.run(
            _compose_cmd(
                compose_file,
                "run", "--rm", "-T", "--no-deps",
                "-v", f"{out_abs}:/export",
                "--entrypoint", "neo4j-admin",
                "neo4j",
                "database", "dump", "neo4j",
                "--to-path=/export",
                "--overwrite-destination=true",
            ),
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            msg = (
                f"neo4j-admin database dump failed (rc={r.returncode})\n"
                f"stdout: {r.stdout}\nstderr: {r.stderr}"
            )
            raise RuntimeError(msg)
    finally:
        print("  starting neo4j …")
        subprocess.run(_compose_cmd(compose_file, "start", "neo4j"), check=True)

    files = [f for f in out.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    if not files:
        raise RuntimeError(f"dump wrote no files to {out}")
    for f in files:
        print(f"    → {f.relative_to(out)} ({f.stat().st_size/1e6:.1f} MB)")
    return {"files": [str(f.relative_to(out)) for f in files], "bytes": total}


# ---------------------------------------------------------------------------
# CocoIndex state (LMDB)
# ---------------------------------------------------------------------------


def backup_cocoindex(dest: Path, state_path: str | None = None) -> dict[str, Any]:
    """Archive CocoIndex's LMDB state directory.

    LMDB offers no safe hot-copy: copying ``data.mdb`` while a writer is
    mid-transaction can capture a torn page. Stop the ingest process (or run
    between scheduled ingests) before taking this backup.

    Losing this state is recoverable but expensive — it costs a full reprocess,
    not data loss, because Qdrant point ids are deterministic (uuid5).
    """
    out = dest / "cocoindex"
    out.mkdir(exist_ok=True)
    src = Path(state_path or settings.cocoindex_db_path)

    if not src.exists():
        print(f"  no CocoIndex state at {src} — skipping")
        return {"skipped": True, "reason": f"{src} does not exist"}

    print(f"  archiving CocoIndex state from {src} …")
    archive = shutil.make_archive(str(out / "cocoindex-state"), "gztar", root_dir=str(src))
    target = Path(archive)
    size = target.stat().st_size
    print(f"    → {target.name} ({size/1e6:.1f} MB)")
    return {"file": target.name, "source": str(src), "bytes": size}


# ---------------------------------------------------------------------------
# Manifest + retention
# ---------------------------------------------------------------------------


def _write_manifest(dest: Path, entries: dict[str, Any]) -> Path:
    manifest = dest / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "timestamp": dest.name,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "qdrant_url": settings.qdrant_url,
                "neo4j_uri": settings.neo4j_uri,
                "entries": entries,
            },
            indent=2,
        )
    )
    return manifest


def _prune_older_than(root: Path, days: int) -> int:
    """Remove dated subdirs older than `days`. Returns count removed."""
    if not root.exists():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for sub in root.iterdir():
        if sub.is_dir() and sub.stat().st_mtime < cutoff:
            shutil.rmtree(sub)
            removed += 1
            print(f"  pruned {sub.name}")
    return removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["qdrant", "neo4j", "cocoindex", "all"],
        nargs="?",
        default="all",
        help="What to back up (default: all).",
    )
    parser.add_argument(
        "--prune-older-than",
        type=int,
        metavar="DAYS",
        help="Before running, remove dated backups older than DAYS days.",
    )
    parser.add_argument(
        "--compose-file",
        metavar="FILE",
        default=None,
        help=(
            "Path to a docker-compose file (passed as `-f FILE` to every "
            "`docker compose` call). Defaults to compose's own discovery "
            "(docker-compose.yml in cwd)."
        ),
    )
    args = parser.parse_args(argv)

    root = Path("backups")
    if args.prune_older_than is not None:
        removed = _prune_older_than(root, args.prune_older_than)
        print(f"Pruned {removed} old backup(s).")

    dest = _make_dest(root)
    print(f"Backup target: {dest}")

    entries: dict[str, Any] = {}
    try:
        if args.target in ("qdrant", "all"):
            entries["qdrant"] = backup_qdrant(dest)
        if args.target in ("neo4j", "all"):
            entries["neo4j"] = backup_neo4j(dest, args.compose_file)
        if args.target in ("cocoindex", "all"):
            entries["cocoindex"] = backup_cocoindex(dest)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest = _write_manifest(dest, entries)
    print(f"\nManifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
