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
def eval_reports_dir() -> Iterator[Path]:
    """Gitignored dir for dated JSON run artifacts."""
    reports = Path("eval-reports")
    reports.mkdir(exist_ok=True)
    yield reports
