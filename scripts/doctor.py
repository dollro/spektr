"""Diff CocoIndex's tracked files against what's retrievable in Qdrant.

CocoIndex (PostgreSQL) = what the pipeline thinks it has processed.
list_documents (Qdrant) = what the retriever can actually find.

Mismatches indicate: failed ingestion, wiped Qdrant collection,
manual data, or deleted source files.

Also detects mixed embedder models across the dense collection,
which silently poisons retrieval quality (query-model mismatch).

Usage:
    python -m scripts.doctor                    # report only
    python -m scripts.doctor --fix              # repair drift interactively
    python -m scripts.doctor --fix --yes        # repair non-interactively
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from collections import Counter

from config.constants import DENSE_COLLECTION, SPARSE_VECTOR_NAME
from config.settings import settings
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


def _delete_tracking_rows(filenames: list[str]) -> None:
    """Delete orphan tracking rows so the next `task ingest` reprocesses them."""
    if not filenames:
        return
    # psql expects JSON-style quoted strings matching the jsonb source_key column
    values = ",".join(f"'\"{name}\"'" for name in filenames)
    sql = f"DELETE FROM {TRACKING_TABLE} WHERE source_key::text IN ({values});"
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "postgres",
            "psql", "-U", "cocoindex", "-d", "cocoindex",
            "-c", sql,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"DELETE failed: {result.stderr.strip()}"
        raise RuntimeError(msg)
    print(f"Deleted {len(filenames)} tracking row(s).")


def _check_embedder_consistency() -> list[str]:
    """Scan a sample of Qdrant points for mixed embedder_model / embedder_dim
    and text chunks missing a sparse vector.

    Returns a list of warning lines (empty = all consistent).
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    points, _ = client.scroll(
        collection_name=DENSE_COLLECTION,
        limit=500,
        with_payload=["embedder_model", "embedder_dim", "content_type"],
        with_vectors=True,
    )
    if not points:
        return []

    models_seen: Counter[str] = Counter()
    dims_seen: Counter[int] = Counter()
    unversioned = 0
    missing_sparse: list[str] = []
    for p in points:
        payload = p.payload or {}
        model = payload.get("embedder_model")
        dim = payload.get("embedder_dim")
        if model is None or dim is None:
            unversioned += 1
        else:
            models_seen[str(model)] += 1
            dims_seen[int(dim)] += 1

        if payload.get("content_type") == "text_chunk":
            vectors = p.vector or {}
            if SPARSE_VECTOR_NAME not in vectors:
                missing_sparse.append(p.id)

    warnings: list[str] = []
    if len(models_seen) > 1:
        breakdown = ", ".join(f"{m}={c}" for m, c in models_seen.most_common())
        warnings.append(f"⚠ Mixed embedder_model in collection: {breakdown}")
    if len(dims_seen) > 1:
        breakdown = ", ".join(f"{d}={c}" for d, c in dims_seen.most_common())
        warnings.append(f"⚠ Mixed embedder_dim in collection: {breakdown}")
    if unversioned:
        warnings.append(
            f"ℹ {unversioned}/{len(points)} sampled points have no embedder_* "
            "tags (pre-versioning ingest — reingest to tag)."
        )
    if missing_sparse:
        warnings.append(f"  ✗ {len(missing_sparse)} text chunks missing sparse vectors")
        warnings.append("    Re-ingest required: task ingest")
    return warnings


async def main(fix: bool = False, yes: bool = False) -> int:
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
        if fix:
            if not yes:
                resp = input(f"Delete {len(only_cocoindex)} tracking row(s)? [y/N] ")
                if resp.strip().lower() != "y":
                    print("Aborted.")
                    return 1
            _delete_tracking_rows(only_cocoindex)
            print("Run `task ingest` to reprocess.")
        else:
            print("  Fix: rerun with --fix to delete tracking rows, then `task ingest`.")

    if only_qdrant:
        print("\n⚠ In Qdrant but not tracked by CocoIndex (orphan or test data):")
        for s in only_qdrant:
            print(f"  - {s}")

    # Embedder consistency check
    try:
        warnings = _check_embedder_consistency()
    except Exception as exc:  # noqa: BLE001
        print(f"\nEmbedder check skipped: {exc}")
        warnings = []
    if warnings:
        print()
        for w in warnings:
            print(w)

    drift = bool(only_cocoindex or only_qdrant)
    has_model_drift = any("Mixed embedder" in w for w in warnings)
    has_missing_sparse = any("missing sparse vectors" in w for w in warnings)

    if not drift and not has_model_drift and not has_missing_sparse:
        print("\n✓ All in sync.")
        return 0
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete orphan CocoIndex tracking rows so next ingest reprocesses.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (use with --fix).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(main(fix=args.fix, yes=args.yes)))
