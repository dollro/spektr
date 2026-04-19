"""Diff CocoIndex's tracked files against what's retrievable in Qdrant.

CocoIndex (PostgreSQL) = what the pipeline thinks it has processed.
list_documents (Qdrant) = what the retriever can actually find.

Mismatches indicate: failed ingestion, wiped Qdrant collection,
manual data, or deleted source files.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

from server.tools.list_documents import list_documents

TRACKING_TABLE = "ragingestion__cocoindex_tracking"


def _tracked_files() -> set[str]:
    """Read distinct source_key values from CocoIndex's tracking table via docker psql."""
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "postgres",
            "psql", "-U", "cocoindex", "-d", "cocoindex", "-At",
            "-c", f"SELECT source_key FROM {TRACKING_TABLE};",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"psql failed: {result.stderr.strip()}"
        raise RuntimeError(msg)
    return {
        line.strip().strip('"')
        for line in result.stdout.splitlines()
        if line.strip()
    }


async def main() -> int:
    tracked = _tracked_files()
    indexed_entries = await list_documents(limit=1000)
    indexed = {e["source_file"] for e in indexed_entries if "source_file" in e}

    both = sorted(tracked & indexed)
    only_cocoindex = sorted(tracked - indexed)
    only_qdrant = sorted(indexed - tracked)

    print(f"Tracked by CocoIndex : {len(tracked)}")
    print(f"Present in Qdrant    : {len(indexed)}")
    print(f"In sync              : {len(both)}")
    print()

    if both:
        print("✓ Healthy:")
        for s in both:
            entry = next(e for e in indexed_entries if e["source_file"] == s)
            print(f"  - {s}  ({entry['chunk_count']} chunks, {entry['page_count']} pages)")

    if only_cocoindex:
        print("\n⚠ Tracked but missing from Qdrant (ingestion failed or collection wiped):")
        for s in only_cocoindex:
            print(f"  - {s}")
        print("  Fix: drop cocoindex tracking rows and re-run `task ingest`.")

    if only_qdrant:
        print("\n⚠ In Qdrant but not tracked by CocoIndex (orphan or test data):")
        for s in only_qdrant:
            print(f"  - {s}")

    if not only_cocoindex and not only_qdrant:
        print("✓ All in sync.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
