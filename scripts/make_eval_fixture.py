"""Freeze the ingested corpus into a committed fixture for retrieval eval.

The retrieval metrics (recall@10, nDCG@10, MRR) score whatever is in Qdrant,
but `tests/conftest.py` repoints every pytest run at `test_documents_dense`,
which nothing writes to. This script snapshots the real collection once so the
gate has a fixed, reproducible corpus on any checkout — no embedding API calls
at test time, and identical numbers on every machine.

Regenerate only when the corpus or the chunking scheme changes, and regenerate
`tests/eval/retrieval_set.yaml` alongside it: that file labels relevance by
``source_file#page_number#chunk_index``, so both are coupled to chunking.

Usage (needs `task ingest` to have run against ./documents):

    uv run python -m scripts.make_eval_fixture
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from config.constants import DENSE_COLLECTION, DENSE_DIM
from config.settings import settings

OUT_PATH = Path("tests/eval/fixtures/retrieval_corpus.json.gz")


def _serialise_vectors(vector: Any) -> dict[str, Any]:
    """Flatten Qdrant's named-vector struct into plain JSON.

    Text chunks carry both `dense` and `sparse`; image and VLM-caption points
    carry `dense` only, so `sparse` is optional here by design.
    """
    if not isinstance(vector, dict):
        msg = f"expected named vectors, got {type(vector).__name__}"
        raise TypeError(msg)

    out: dict[str, Any] = {"dense": list(vector["dense"])}
    sparse = vector.get("sparse")
    if sparse is not None:
        out["sparse"] = {
            "indices": list(sparse.indices),
            "values": list(sparse.values),
        }
    return out


def export() -> None:
    client = QdrantClient(url=settings.qdrant_url)
    try:
        if not client.collection_exists(DENSE_COLLECTION):
            msg = (
                f"collection {DENSE_COLLECTION!r} does not exist — "
                "run `task ingest` before generating the fixture"
            )
            raise SystemExit(msg)

        points, _ = client.scroll(
            DENSE_COLLECTION,
            limit=10_000,
            with_vectors=True,
            with_payload=True,
        )
    finally:
        client.close()

    if not points:
        raise SystemExit(f"collection {DENSE_COLLECTION!r} is empty — nothing to freeze")

    records = [
        {
            "id": str(point.id),
            "payload": point.payload,
            "vector": _serialise_vectors(point.vector),
        }
        for point in points
    ]
    sources = sorted({str(r["payload"].get("source_file", "")) for r in records})  # type: ignore[union-attr]

    blob = {
        "dense_dim": DENSE_DIM,
        "source_collection": DENSE_COLLECTION,
        "sources": sources,
        "points": records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as fh:
        json.dump(blob, fh)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(
        f"wrote {OUT_PATH} — {len(records)} points, "
        f"{len(sources)} sources, {size_kb:.0f} KB"
    )
    for source in sources:
        count = sum(1 for r in records if r["payload"].get("source_file") == source)  # type: ignore[union-attr]
        print(f"  {source}: {count} points")


if __name__ == "__main__":
    export()
