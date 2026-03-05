# Custom Target Connector for Automatic Deletion Handling

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken `handle_file_delete` with a CocoIndex custom target connector so file deletions from source (local or S3) automatically clean up Qdrant vectors and Graphiti episodes.

**Architecture:** Create a CocoIndex `TargetSpec` + `target_connector` that receives mutations from CocoIndex's incremental engine. When a mutation value is `None` (deletion), it removes Qdrant points by `source_file` filter and Graphiti episodes by `source_description`. When a mutation value is present (upsert), it delegates to the existing `ingest_file` logic. The flow definition is updated to export to this custom target instead of (or alongside) the Postgres log.

**Tech Stack:** CocoIndex custom targets (`cocoindex.op.TargetSpec`, `@cocoindex.op.target_connector`), Qdrant client, Graphiti `retrieve_episodes` + `remove_episode`

---

## Current State

- `ingest_file` is a `@cocoindex.op.function()` that writes directly to Qdrant + Neo4j as a side effect
- `handle_file_delete()` at `ingestion/pipeline.py:641` is broken — uses `edge.source_description` which doesn't exist on `EntityEdge`
- The flow exports only to Postgres (`ingestion_log`) — no custom target
- Deletions from source are not propagated to Qdrant or Graphiti

## Design Decisions

1. **Keep `ingest_file` as the CocoIndex custom op** — it does the heavy lifting (file processing, embedding, upsert). The custom target connector only needs to handle the *deletion* path and log upserts.
2. **The target connector's `mutate()` receives `(filename, result_or_None)`** — when `None`, call cleanup. When present, it's a no-op (ingest already happened in the custom op).
3. **Replace the Postgres export** with the custom target export. The target connector can optionally log to Postgres if needed, but the primary purpose is deletion handling.
4. **Fix Graphiti cleanup** to use `retrieve_episodes` + `remove_episode` (proper API).

## Files Overview

| Action | File | Purpose |
|-|-|-|
| Create | `ingestion/target_connector.py` | Custom target connector (TargetSpec + connector class) |
| Modify | `ingestion/pipeline.py` | Wire custom target into flow, remove old `handle_file_delete` |
| Create | `tests/test_target_connector.py` | Unit tests for the connector |
| Modify | `tests/test_pipeline_delete.py` | Update to test via connector instead of old function |

---

### Task 1: Create the Custom Target Connector

**Files:**
- Create: `ingestion/target_connector.py`
- Test: `tests/test_target_connector.py`

**Step 1: Write failing tests for the connector**

```python
# tests/test_target_connector.py
from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.target_connector import (
    RagTarget,
    RagTargetConnector,
    RagTargetValues,
)


class TestRagTargetConnectorMutateDelete:
    """When mutation value is None, connector deletes from Qdrant."""

    def test_delete_removes_dense_points(self) -> None:
        mock_qdrant = MagicMock()
        spec = RagTarget(qdrant_url="http://localhost:6333")

        RagTargetConnector.mutate((spec, {"report.pdf": None}))

        mock_qdrant_calls = mock_qdrant.delete.call_args_list
        # Verify delete was called (we'll refine after implementation)

    def test_delete_skipped_for_upsert(self) -> None:
        """Non-None values are no-ops (ingest already happened)."""
        spec = RagTarget(qdrant_url="http://localhost:6333")
        values = RagTargetValues(result="report.pdf")

        # Should not raise or call any external service
        RagTargetConnector.mutate((spec, {"report.pdf": values}))


class TestRagTargetConnectorSetup:
    def test_get_persistent_key(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        key = RagTargetConnector.get_persistent_key(spec, "rag_target")
        assert key == "http://localhost:6333"

    def test_apply_setup_change_noop(self) -> None:
        """Setup changes are no-ops (Qdrant collections managed elsewhere)."""
        RagTargetConnector.apply_setup_change("key", None, None)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_target_connector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.target_connector'`

**Step 3: Implement the target connector**

```python
# ingestion/target_connector.py
"""CocoIndex custom target connector for RAG pipeline.

Handles deletion of Qdrant points and Graphiti episodes when
source files are removed. Upserts are no-ops here since
ingest_file handles them as a CocoIndex custom op.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import cocoindex
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.logging import get_logger
from config.settings import settings
from ingestion._utils import run_async

logger = get_logger(__name__)


class RagTarget(cocoindex.op.TargetSpec):
    """Target spec for RAG pipeline cleanup."""

    qdrant_url: str


@dataclasses.dataclass
class RagTargetValues:
    """Value fields from the collector export."""

    result: str


def _delete_qdrant_points(qdrant: QdrantClient, source_key: str) -> None:
    """Delete all Qdrant points for a source file."""
    qdrant.delete(
        collection_name=DENSE_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_key),
                    ),
                ],
            ),
        ),
    )
    if settings.multivec_enabled:
        qdrant.delete(
            collection_name=MULTIVEC_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_key),
                        ),
                    ],
                ),
            ),
        )


async def _remove_graphiti_episodes(source_key: str) -> None:
    """Remove Graphiti episodes matching source_description."""
    from ingestion.graphiti_client import close_graphiti, get_graphiti

    client = await get_graphiti()
    episodes = await client.retrieve_episodes(
        reference_time=datetime.now(tz=UTC),
        last_n=10000,
    )
    removed = 0
    for ep in episodes:
        if ep.source_description == source_key:
            await client.remove_episode(ep.uuid)
            removed += 1
    logger.info(
        "Removed %d Graphiti episodes for %s",
        removed,
        source_key,
        extra={"file_name": source_key},
    )
    await close_graphiti()


def _handle_delete(source_key: str) -> None:
    """Delete Qdrant points and Graphiti episodes for a source file."""
    logger.info(
        "Handling file delete for %s",
        source_key,
        extra={"file_name": source_key},
    )

    try:
        qdrant = QdrantClient(url=settings.qdrant_url)
        _delete_qdrant_points(qdrant, source_key)
        logger.info(
            "Deleted Qdrant points for %s",
            source_key,
            extra={"file_name": source_key},
        )
    except Exception:
        logger.exception(
            "Failed to delete Qdrant points for %s",
            source_key,
            extra={"file_name": source_key},
        )

    if not settings.graph_enabled:
        return
    try:
        run_async(_remove_graphiti_episodes(source_key))
    except Exception:
        logger.exception(
            "Failed to remove Graphiti data for %s",
            source_key,
            extra={"file_name": source_key},
        )


@cocoindex.op.target_connector(spec_cls=RagTarget)
class RagTargetConnector:
    """CocoIndex target connector that cleans up on file deletion."""

    @staticmethod
    def get_persistent_key(spec: RagTarget, target_name: str) -> str:
        return spec.qdrant_url

    @staticmethod
    def describe(key: str) -> str:
        return f"RAG target (Qdrant: {key})"

    @staticmethod
    def apply_setup_change(
        key: str,
        previous: RagTarget | None,
        current: RagTarget | None,
    ) -> None:
        pass  # Collections managed by ensure_collections()

    @staticmethod
    def mutate(
        *all_mutations: tuple[RagTarget, dict[str, RagTargetValues | None]],
    ) -> None:
        for _spec, mutations in all_mutations:
            for filename, value in mutations.items():
                if value is None:
                    _handle_delete(filename)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_target_connector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ingestion/target_connector.py tests/test_target_connector.py
git commit -m "feat: add CocoIndex custom target connector for deletion handling"
```

---

### Task 2: Refine Tests with Proper Mocking

**Files:**
- Modify: `tests/test_target_connector.py`

**Step 1: Write comprehensive tests with mocking**

Replace the initial test file with properly mocked tests:

```python
# tests/test_target_connector.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.target_connector import (
    RagTarget,
    RagTargetConnector,
    RagTargetValues,
    _delete_qdrant_points,
    _handle_delete,
)


class TestDeleteQdrantPoints:
    def test_deletes_dense_collection(self) -> None:
        mock_qdrant = MagicMock()
        with patch("ingestion.target_connector.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            _delete_qdrant_points(mock_qdrant, "report.pdf")

        mock_qdrant.delete.assert_called_once()
        call_kwargs = mock_qdrant.delete.call_args
        assert call_kwargs.kwargs["collection_name"] == "documents_dense"

    def test_deletes_multivec_when_enabled(self) -> None:
        mock_qdrant = MagicMock()
        with patch("ingestion.target_connector.settings") as mock_settings:
            mock_settings.multivec_enabled = True
            _delete_qdrant_points(mock_qdrant, "report.pdf")

        assert mock_qdrant.delete.call_count == 2


class TestHandleDelete:
    def test_calls_qdrant_and_graphiti(self) -> None:
        with (
            patch("ingestion.target_connector.QdrantClient") as mock_cls,
            patch("ingestion.target_connector._delete_qdrant_points") as mock_del,
            patch("ingestion.target_connector.settings") as mock_settings,
            patch("ingestion.target_connector.run_async") as mock_run,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.graph_enabled = True
            _handle_delete("report.pdf")

        mock_del.assert_called_once()
        mock_run.assert_called_once()

    def test_skips_graphiti_when_disabled(self) -> None:
        with (
            patch("ingestion.target_connector.QdrantClient"),
            patch("ingestion.target_connector._delete_qdrant_points"),
            patch("ingestion.target_connector.settings") as mock_settings,
            patch("ingestion.target_connector.run_async") as mock_run,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.graph_enabled = False
            _handle_delete("report.pdf")

        mock_run.assert_not_called()


class TestRagTargetConnectorMutate:
    def test_delete_calls_handle_delete(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        with patch("ingestion.target_connector._handle_delete") as mock_del:
            RagTargetConnector.mutate((spec, {"report.pdf": None}))
        mock_del.assert_called_once_with("report.pdf")

    def test_upsert_is_noop(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        values = RagTargetValues(result="report.pdf")
        with patch("ingestion.target_connector._handle_delete") as mock_del:
            RagTargetConnector.mutate((spec, {"report.pdf": values}))
        mock_del.assert_not_called()

    def test_multiple_mutations(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        values = RagTargetValues(result="keep.pdf")
        with patch("ingestion.target_connector._handle_delete") as mock_del:
            RagTargetConnector.mutate(
                (spec, {"delete.pdf": None, "keep.pdf": values})
            )
        mock_del.assert_called_once_with("delete.pdf")


class TestRagTargetConnectorSetup:
    def test_get_persistent_key(self) -> None:
        spec = RagTarget(qdrant_url="http://localhost:6333")
        key = RagTargetConnector.get_persistent_key(spec, "rag_target")
        assert key == "http://localhost:6333"

    def test_apply_setup_change_noop(self) -> None:
        RagTargetConnector.apply_setup_change("key", None, None)

    def test_describe(self) -> None:
        desc = RagTargetConnector.describe("http://localhost:6333")
        assert "Qdrant" in desc
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_target_connector.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_target_connector.py
git commit -m "test: comprehensive tests for RagTargetConnector"
```

---

### Task 3: Wire Custom Target into the Flow Definition

**Files:**
- Modify: `ingestion/pipeline.py:737-785` (the `rag_ingestion_flow` function)

**Step 1: Update the flow to export to the custom target**

Replace the collector export from Postgres to the custom target in `rag_ingestion_flow`:

```python
# At top of pipeline.py, add import:
from ingestion.target_connector import RagTarget

# In rag_ingestion_flow, replace:
#     collector.export(
#         "ingestion_log",
#         cocoindex.targets.Postgres(),
#         primary_key_fields=["filename"],
#     )
# With:
    collector.export(
        "rag_target",
        RagTarget(qdrant_url=settings.qdrant_url),
        primary_key_fields=["filename"],
    )
```

**Step 2: Run the pipeline to verify it works**

Run: `GRAPH_ENABLED=false uv run python -m ingestion.pipeline`
Expected: Pipeline runs successfully, shows the custom target instead of Postgres export

**Step 3: Commit**

```bash
git add ingestion/pipeline.py
git commit -m "feat: wire RagTarget connector into CocoIndex flow for auto-deletion"
```

---

### Task 4: Remove Old `handle_file_delete` and Update References

**Files:**
- Modify: `ingestion/pipeline.py` — remove `handle_file_delete`, `handle_s3_delete`
- Modify: `tests/test_pipeline_delete.py` — update or remove

**Step 1: Remove `handle_file_delete` from pipeline.py**

Delete lines 641-734 from `ingestion/pipeline.py` (the `handle_file_delete` function and `handle_s3_delete` alias).

**Step 2: Add backward-compatible aliases in target_connector.py**

At the bottom of `ingestion/target_connector.py`:

```python
# Backward-compatible aliases
handle_file_delete = _handle_delete
handle_s3_delete = _handle_delete
```

**Step 3: Update test_pipeline_delete.py**

```python
# tests/test_pipeline_delete.py
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestHandleFileDelete:
    def test_deletes_dense_points_by_source_file(self) -> None:
        """Qdrant dense collection points are deleted by source_file filter."""
        mock_qdrant = MagicMock()
        with (
            patch("ingestion.target_connector.QdrantClient", return_value=mock_qdrant),
            patch("ingestion.target_connector.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.multivec_enabled = False
            mock_settings.graph_enabled = False

            from ingestion.target_connector import handle_file_delete

            handle_file_delete("docs/report.pdf")

        mock_qdrant.delete.assert_called_once()

    def test_deletes_multivec_when_enabled(self) -> None:
        """Qdrant multivec collection is also cleaned when enabled."""
        mock_qdrant = MagicMock()
        with (
            patch("ingestion.target_connector.QdrantClient", return_value=mock_qdrant),
            patch("ingestion.target_connector.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.multivec_enabled = True
            mock_settings.graph_enabled = False

            from ingestion.target_connector import handle_file_delete

            handle_file_delete("docs/report.pdf")

        assert mock_qdrant.delete.call_count == 2

    def test_backward_compat_alias(self) -> None:
        """handle_s3_delete is an alias for handle_file_delete."""
        from ingestion.target_connector import handle_file_delete, handle_s3_delete

        assert handle_s3_delete is handle_file_delete
```

**Step 4: Run all tests**

Run: `uv run pytest tests/test_target_connector.py tests/test_pipeline_delete.py -v`
Expected: All PASS

**Step 5: Check no other imports reference the old location**

Run: `uv run ruff check . && uv run pytest`
Expected: No import errors, all tests pass

**Step 6: Commit**

```bash
git add ingestion/pipeline.py ingestion/target_connector.py tests/test_pipeline_delete.py
git commit -m "refactor: move deletion logic to target connector, remove old handle_file_delete"
```

---

### Task 5: Integration Test — Delete a File and Verify Cleanup

**Files:**
- Modify: `tests/test_integration_ingestion.py` (or create `tests/test_integration_delete.py`)

**Step 1: Write integration test**

```python
# tests/test_integration_delete.py
"""Integration test: verify file deletion cleans up Qdrant.

Requires Docker services (Qdrant). Run with:
    uv run pytest tests/test_integration_delete.py -m integration
"""
from __future__ import annotations

import pytest

from ingestion.target_connector import RagTarget, RagTargetConnector


@pytest.mark.integration
class TestDeletionIntegration:
    def test_delete_nonexistent_is_idempotent(self) -> None:
        """Deleting a file that was never ingested should not error."""
        spec = RagTarget(qdrant_url="http://localhost:6333")
        # Should not raise
        RagTargetConnector.mutate((spec, {"nonexistent.pdf": None}))
```

**Step 2: Run integration test**

Run: `uv run pytest tests/test_integration_delete.py -m integration -v`
Expected: PASS (idempotent delete on non-existent data)

**Step 3: Commit**

```bash
git add tests/test_integration_delete.py
git commit -m "test: integration test for deletion idempotency"
```

---

### Task 6: Lint, Format, Type Check

**Step 1: Run linting and formatting**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: Clean

**Step 2: Run type check**

Run: `uv run mypy ingestion/target_connector.py`
Expected: Clean (or only pre-existing issues)

**Step 3: Run full test suite**

Run: `uv run pytest`
Expected: All pass

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: lint and format after target connector changes"
```

---

## Summary of Changes

1. **New file `ingestion/target_connector.py`** — CocoIndex custom target connector (`RagTarget` + `RagTargetConnector`) that handles deletions via `mutate()` receiving `None` values
2. **Fixed Graphiti cleanup** — uses `retrieve_episodes` + `remove_episode` instead of broken `search` + `edge.source_description`
3. **Updated flow definition** — exports to `RagTarget` instead of `Postgres`, so CocoIndex drives deletion automatically for both local and S3 sources
4. **Removed broken `handle_file_delete`** from `pipeline.py`, backward-compat aliases in new module
5. **Works identically for local and S3** — CocoIndex abstracts source-level change detection

## Risks / Open Questions

- **`retrieve_episodes` with `last_n=10000`**: If a project has >10k episodes, this won't find all of them. Consider paginating or using a direct Cypher query against Neo4j to find episodes by `source_description`. For now, 10k is a reasonable upper bound.
- **Postgres ingestion log**: We're replacing the Postgres export. If other tooling reads `ingestion_log`, we'd need to keep both exports (a second `collector.export` for Postgres). Check if anything depends on it.
- **CocoIndex target connector thread safety**: `mutate()` is called by CocoIndex's engine — verify it handles the sync `run_async` call for Graphiti without deadlocking.
