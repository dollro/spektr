"""Diff CocoIndex's tracked files against what's retrievable in Qdrant.

CocoIndex (LMDB ledger) = what the pipeline thinks it has processed.
list_documents (Qdrant) = what the retriever can actually find.

Since the CocoIndex v1 migration, points are *declared* on the native Qdrant
target rather than upserted as a side effect, so CocoIndex owns them and the
"tracked but missing" class of drift should no longer occur — it is still
reported, because a non-empty result now means files errored during processing.
The drift CocoIndex cannot self-heal is the other direction: points in Qdrant
that no CocoIndex run declared (v0 leftovers, manual data, test fixtures). That
is what ``--fix`` removes. Live-session points (``is_live=True``) are excluded
throughout — they belong to Path B and are legitimately untracked.

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
import sys
from collections import Counter

from config.constants import DENSE_COLLECTION, SPARSE_VECTOR_NAME
from config.settings import settings
from server.tools.list_documents import list_documents

# Components mounted per source file live at "/@process_file/<source_key>".
# The prefix follows ingestion.app.process_file.__name__.
_COMPONENT_PREFIX = "/@process_file/"


async def _tracked_files() -> set[str] | None:
    """Read the source keys CocoIndex has components for, via its LMDB ledger.

    Returns ``None`` (rather than raising) when the ledger cannot be read — a
    fresh checkout with no state directory is a normal condition, and the
    Qdrant-side checks below are still worth running.
    """
    try:
        from cocoindex import inspect

        from ingestion.app import build_app

        app = build_app()
        tracked: set[str] = set()
        async for info in inspect.iter_stable_paths(app):
            path = info.path.to_string()
            if not path.startswith(_COMPONENT_PREFIX):
                continue
            parts = info.path.parts()
            if len(parts) == 2 and isinstance(parts[1], str):
                tracked.add(parts[1])
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read CocoIndex state ({exc.__class__.__name__}: {exc}).")
        print(f"  State directory: {settings.cocoindex_db_path}")
        return None
    return tracked


def _delete_orphan_points(source_files: list[str]) -> None:
    """Delete Qdrant points for source files CocoIndex does not track."""
    if not source_files:
        return
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=settings.qdrant_url)
    client.delete(
        collection_name=DENSE_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchAny(any=source_files),
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="is_live",
                        match=models.MatchValue(value=True),
                    ),
                ],
            ),
        ),
    )
    print(f"Deleted orphan points for {len(source_files)} source file(s).")


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
                missing_sparse.append(str(p.id))

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


def _dense_collection_exists() -> bool:
    """Whether the dense collection has been provisioned yet."""
    from qdrant_client import QdrantClient

    try:
        return bool(QdrantClient(url=settings.qdrant_url).collection_exists(DENSE_COLLECTION))
    except Exception:  # noqa: BLE001
        # Qdrant unreachable is a different problem; let the checks below report it.
        return True


async def main(fix: bool = False, yes: bool = False) -> int:
    if not _dense_collection_exists():
        # A fresh install is not "drift" — say so plainly instead of letting
        # list_documents and the embedder scan each raise a raw Qdrant 404.
        print(f"Collection '{DENSE_COLLECTION}' does not exist yet.")
        print(f"  Qdrant: {settings.qdrant_url}")
        print("  Nothing has been ingested. Run `task ingest` to create it.")
        return 0

    tracked = await _tracked_files()
    indexed_entries = await list_documents(limit=1000)
    indexed = {e["source_file"] for e in indexed_entries if "source_file" in e}

    only_cocoindex: list[str] = []
    only_qdrant: list[str] = []
    if tracked is None:
        print("Tracked by CocoIndex : unknown (state unreadable — diff skipped)")
        print(f"Present in Qdrant    : {len(indexed)}")
        both = sorted(indexed)
    else:
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
            entry = next((e for e in indexed_entries if e["source_file"] == s), None)
            if entry is None:
                print(f"  - {s}")
            else:
                print(
                    f"  - {s}  ({entry['chunk_count']} chunks, {entry['page_count']} pages)"
                )

    if only_cocoindex:
        print("\n⚠ Tracked but missing from Qdrant (processing errored or collection wiped):")
        for s in only_cocoindex:
            print(f"  - {s}")
        print(
            "  Fix: re-run ingestion for these files — "
            "`task ingest -- --full-reprocess` (--fix does not touch the ledger)."
        )

    if only_qdrant:
        print("\n⚠ In Qdrant but not tracked by CocoIndex (v0 leftovers or test data):")
        for s in only_qdrant:
            print(f"  - {s}")
        if fix and not tracked:
            # An empty-but-readable ledger makes *every* indexed document look
            # like an orphan. That is exactly the state after a legitimate
            # `rm -rf state/cocoindex.db` reindex, so auto-deleting here would
            # wipe the whole corpus. Refuse and let the operator re-ingest.
            print(
                "\n  REFUSING to --fix: CocoIndex tracks nothing at all, so every\n"
                "  indexed document looks orphaned. This is the expected state\n"
                "  after clearing the LMDB directory. Run `task ingest` to\n"
                "  repopulate the ledger, then re-run doctor."
            )
        elif fix:
            if not yes:
                resp = input(
                    f"Delete Qdrant points for {len(only_qdrant)} source file(s)? [y/N] "
                )
                if resp.strip().lower() != "y":
                    print("Aborted.")
                    return 1
            _delete_orphan_points(only_qdrant)
        else:
            print("  Fix: rerun with --fix to delete these orphan points.")

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
        help="Delete Qdrant points that no CocoIndex run declared.",
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
