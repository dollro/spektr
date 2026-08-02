"""Fixtures for the RAG evaluation harness.

Run with `task eval` (pytest -m eval). Requires the `eval` extra:
    uv sync --extra eval
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

EVAL_DIR = Path(__file__).parent
FIXTURES = EVAL_DIR / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Eval tests don't run under `task test` — only `task eval`.

    Checks the actual `eval` marker via `get_closest_marker`, not substring
    containment on `item.keywords` — that mapping also matches node IDs, so
    `"eval" in item.keywords` was true for any test under a path containing
    the substring "retrieval" (e.g. test_retrieval_metrics.py), silently
    skipping unrelated tests regardless of `-m`.
    """
    if config.getoption("-m") and "eval" in config.getoption("-m"):
        return
    skip_eval = pytest.mark.skip(reason="Use `task eval` (pytest -m eval) to run.")
    for item in items:
        if item.get_closest_marker("eval") is not None:
            item.add_marker(skip_eval)


@pytest.fixture(scope="session")
def golden_set() -> list[dict[str, Any]]:
    """Parsed Q&A fixtures."""
    with (FIXTURES / "golden_set.yaml").open() as fh:
        return yaml.safe_load(fh)["items"]


@pytest.fixture(scope="session")
def thresholds() -> dict[str, float]:
    """Minimum per-metric scores for a green run."""
    with (EVAL_DIR / "thresholds.yaml").open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def ingested_sources() -> set[str]:
    """Set of source_file values currently present in Qdrant.

    Items whose required_source is missing are skipped rather than failing.
    """
    import asyncio

    from server.tools.list_documents import list_documents

    entries = asyncio.run(list_documents(limit=1000))
    return {e["source_file"] for e in entries if "source_file" in e}


@pytest.fixture(scope="session")
def frozen_corpus() -> int:
    """Load the committed corpus snapshot into the throwaway test collection.

    `tests/conftest.py` repoints QDRANT_DENSE_COLLECTION at
    `test_documents_dense` for every pytest run, and nothing else writes to it,
    so retrieval metrics would otherwise score an empty collection and report a
    vacuous 0.0 for every metric. This restores a fixed corpus — the same 32
    points on every machine, with no embedding API calls, so the gate is
    reproducible on a fresh checkout.

    Regenerate with `uv run python -m scripts.make_eval_fixture`.
    """
    import gzip
    import json

    from qdrant_client import QdrantClient, models

    from config.constants import DENSE_COLLECTION
    from config.settings import settings
    from ingestion.qdrant_setup import ensure_collections

    snapshot = FIXTURES / "retrieval_corpus.json.gz"
    if not snapshot.exists():
        pytest.skip(f"{snapshot} missing — run `uv run python -m scripts.make_eval_fixture`")

    with gzip.open(snapshot, "rt", encoding="utf-8") as fh:
        blob = json.load(fh)

    # Belt and braces: this fixture deletes a collection, so refuse to run
    # unless the name is the throwaway one conftest redirects to.
    assert DENSE_COLLECTION.startswith("test_"), (
        f"Refusing to load fixtures into {DENSE_COLLECTION!r} — expected the "
        "test_-prefixed collection set by tests/conftest.py"
    )

    points = [
        models.PointStruct(
            id=record["id"],
            payload=record["payload"],
            vector=_rebuild_vectors(record["vector"], models),
        )
        for record in blob["points"]
    ]

    client = QdrantClient(url=settings.qdrant_url)
    try:
        client.delete_collection(DENSE_COLLECTION)
        ensure_collections(client)
        client.upsert(DENSE_COLLECTION, points=points, wait=True)
        loaded = client.count(DENSE_COLLECTION).count
    finally:
        client.close()

    assert loaded == len(points), f"loaded {loaded} of {len(points)} fixture points"
    return loaded


def _rebuild_vectors(vector: dict[str, Any], models: Any) -> dict[str, Any]:
    """Turn the flat JSON vector record back into Qdrant's named-vector form."""
    out: dict[str, Any] = {"dense": vector["dense"]}
    sparse = vector.get("sparse")
    if sparse is not None:
        out["sparse"] = models.SparseVector(
            indices=sparse["indices"],
            values=sparse["values"],
        )
    return out


@pytest.fixture(scope="session")
def eval_reports_dir() -> Iterator[Path]:
    """Gitignored dir for dated JSON run artifacts."""
    reports = Path("eval-reports")
    reports.mkdir(exist_ok=True)
    yield reports
