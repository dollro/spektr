"""Restore Qdrant + Neo4j + CocoIndex state from a ./backups/<timestamp>/.

Dangerous: Qdrant restore deletes the target collection, Neo4j load
replaces the database, and the CocoIndex restore replaces the LMDB state
directory outright. Refuses to run without --yes-i-know-this-wipes-things.

Usage:
    python -m scripts.restore --from backups/20260419-153000 \
        --target all --yes-i-know-this-wipes-things
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from config.settings import settings


def _read_manifest(src: Path) -> dict:  # type: ignore[type-arg]
    m = src / "manifest.json"
    if not m.exists():
        raise FileNotFoundError(f"No manifest at {m}")
    data: dict = json.loads(m.read_text())  # type: ignore[type-arg]
    return data


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


def restore_qdrant(src: Path, manifest: dict) -> None:  # type: ignore[type-arg]
    """Upload each snapshot tarball, recover into the named collection."""
    qdrant_entry = manifest.get("entries", {}).get("qdrant", {})
    snap_dir = src / "qdrant"
    if not snap_dir.exists():
        print("  no qdrant dir in backup; skipping")
        return

    with httpx.Client() as client:
        for item in qdrant_entry.get("collections", []):
            name = item["collection"]
            file_name = item["file"]
            snap_path = snap_dir / file_name
            if not snap_path.exists():
                print(f"  missing {snap_path}, skipping")
                continue

            # Delete existing collection so recovery doesn't conflict
            print(f"  restoring Qdrant:{name} from {file_name} …")
            client.delete(
                f"{settings.qdrant_url}/collections/{name}", timeout=30
            )

            # Upload and recover in one shot
            with snap_path.open("rb") as fh:
                r = client.post(
                    f"{settings.qdrant_url}/collections/{name}/snapshots/upload",
                    files={"snapshot": (file_name, fh, "application/octet-stream")},
                    timeout=600,
                )
            if r.status_code >= 400:
                print(f"    ERROR: {r.status_code} {r.text[:300]}")
                raise RuntimeError(f"Qdrant snapshot upload failed for {name}")
            print(f"    → {name} restored")


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


def restore_neo4j(
    src: Path,
    manifest: dict,  # type: ignore[type-arg]
    compose_file: str | None = None,
) -> None:
    """Load the dump (or backup dir) back into the container."""
    neo4j_dir = src / "neo4j"
    if not neo4j_dir.exists():
        print("  no neo4j dir in backup; skipping")
        return

    container_path = "/tmp/spektr-restore"
    subprocess.run(
        _compose_cmd(
            compose_file, "exec", "-T", "neo4j",
            "rm", "-rf", container_path,
        ),
        check=False,
    )
    subprocess.run(
        _compose_cmd(
            compose_file, "exec", "-T", "neo4j",
            "mkdir", "-p", container_path,
        ),
        check=True,
    )

    # Copy dump/backup artefacts INTO the container
    subprocess.run(
        _compose_cmd(
            compose_file, "cp",
            str(neo4j_dir) + "/.", f"neo4j:{container_path}",
        ),
        check=True,
    )

    print("  stopping neo4j service (Community load requires DB offline) …")
    subprocess.run(
        _compose_cmd(compose_file, "stop", "neo4j"), check=True,
    )
    try:
        # Prefer `database load` (from dump); fall back to `database restore`
        # (from backup dir structure) if load fails.
        print("  loading dump …")
        r = subprocess.run(
            _compose_cmd(
                compose_file, "run", "--rm", "-T", "--no-deps", "neo4j",
                "neo4j-admin", "database", "load", "neo4j",
                f"--from-path={container_path}", "--overwrite-destination=true",
            ),
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            print("    load failed, trying restore …")
            r = subprocess.run(
                _compose_cmd(
                    compose_file, "run", "--rm", "-T", "--no-deps", "neo4j",
                    "neo4j-admin", "database", "restore",
                    f"--from-path={container_path}", "neo4j",
                    "--overwrite-destination=true",
                ),
                capture_output=True, text=True, check=False,
            )
            if r.returncode != 0:
                msg = (
                    f"neo4j restore failed.\nstdout: {r.stdout}\n"
                    f"stderr: {r.stderr}"
                )
                raise RuntimeError(msg)
        print("    → neo4j restored")
    finally:
        subprocess.run(
            _compose_cmd(compose_file, "start", "neo4j"), check=True,
        )


# ---------------------------------------------------------------------------
# CocoIndex state (LMDB)
# ---------------------------------------------------------------------------


def restore_cocoindex(
    src: Path,
    manifest: dict,  # type: ignore[type-arg]
    state_path: str | None = None,
) -> None:
    """Replace the LMDB state directory with the archived one.

    The ingest process must not be running: LMDB keeps a lock file and open
    readers, and swapping the directory underneath a live writer corrupts it.
    """
    entry = manifest.get("entries", {}).get("cocoindex", {})
    if entry.get("skipped"):
        print("  backup recorded no CocoIndex state; skipping")
        return
    file_name = entry.get("file", "cocoindex-state.tar.gz")
    archive = src / "cocoindex" / file_name
    if not archive.exists():
        print(f"  no archive at {archive}; skipping")
        return

    dest = Path(state_path or settings.cocoindex_db_path)
    print(f"  restoring CocoIndex state to {dest} …")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(dest))
    print("    → cocoindex state restored")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="src", required=True,
                        help="Path to a backups/<timestamp>/ directory.")
    parser.add_argument("--target",
                        choices=["qdrant", "neo4j", "cocoindex", "all"],
                        default="all")
    parser.add_argument("--yes-i-know-this-wipes-things",
                        action="store_true",
                        help="Required to actually run. Restore is destructive.")
    parser.add_argument(
        "--compose-file",
        metavar="FILE",
        default=None,
        help=(
            "Path to a docker-compose file (passed as `-f FILE` to every "
            "`docker compose` call). Defaults to compose's own discovery."
        ),
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"No backup at {src}", file=sys.stderr)
        return 2

    manifest = _read_manifest(src)
    print(f"Manifest: created {manifest.get('created_at')} for {manifest.get('qdrant_url')}")

    if not args.yes_i_know_this_wipes_things:
        print(
            "Refusing to run without --yes-i-know-this-wipes-things. "
            "Restore wipes current Qdrant collections, the Neo4j DB, "
            "and the CocoIndex LMDB state directory.",
            file=sys.stderr,
        )
        return 3

    try:
        if args.target in ("qdrant", "all"):
            restore_qdrant(src, manifest)
        if args.target in ("cocoindex", "all"):
            restore_cocoindex(src, manifest)
        if args.target in ("neo4j", "all"):
            restore_neo4j(src, manifest, args.compose_file)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nRestore complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
