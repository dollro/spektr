"""Integration gate for the CocoIndex v1 write path.

These are the assertions the whole v0 -> v1 migration rests on. Everything else
about the migration is mechanical; if any of these break, points are being
silently added or destroyed.

1. **A re-run is a no-op.** Memoized components must keep their previously
   declared target states. If they did not, an unchanged file's points would be
   reconciled to non-existence on the very next run.
2. **Deleting a source file deletes exactly its points.** This replaces v0's
   filter-delete in ``RagTarget``.
3. **Path B's live-session points survive.** They share ``documents_dense`` but
   were never declared by CocoIndex, and per-point reconciliation issues
   explicit-id deletes with no orphan sweep — so they must be untouchable.
4. **The source key stays relative.** ``source_file`` must remain ``a.txt``,
   not an absolute path: every payload, ``list_documents`` and the eval
   fixtures are keyed on it.

Requires Docker (Qdrant). Run with:
    uv run pytest tests/test_integration_cocoindex_v1.py -m integration
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from config.settings import settings

_LIVE_POINT_ID = "00000000-0000-0000-0000-0000000000ff"


class _StubEmbedder:
    """Deterministic embedder — keeps the gate off any embedding provider."""

    model_name = "stub-embedder"
    tokens_used = 0.0

    def __init__(self, dim: int) -> None:
        self.dim = dim

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]:
        return [[0.01] * self.dim for _ in texts]

    async def embed_text_query(self, query: str, dimensions: int | None = None) -> list[float]:
        return [0.01] * self.dim

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        return [0.02] * self.dim

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        raise NotImplementedError

    def reset_token_counter(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _fake_sparse(texts: list[str]) -> list[models.SparseVector]:
    return [models.SparseVector(indices=[1, 2], values=[0.5, 0.5]) for _ in texts]


def _sources(client: QdrantClient) -> set[str]:
    points, _ = client.scroll(DENSE_COLLECTION, limit=1000, with_payload=["source_file"])
    return {(p.payload or {}).get("source_file", "") for p in points}


def _count(client: QdrantClient) -> int:
    return int(client.count(DENSE_COLLECTION, exact=True).count)


@pytest.fixture
async def v1_app(tmp_path: pathlib.Path, qdrant_client: QdrantClient) -> Any:
    """Build a real CocoIndex app over a temp doc dir with a stub embedder.

    CocoIndex's default environment is a process-wide singleton bound to the
    event loop that first started it, and pytest-asyncio gives each test a fresh
    loop. Without the ``coco.stop()`` teardown below, the second test in this
    module fails with "Event loop is closed".
    """
    import cocoindex as coco

    import ingestion.app as app_mod
    import ingestion.page_processor as page_processor

    docs = tmp_path / "docs"
    docs.mkdir()

    dim = settings.dense_dimensions
    originals = (
        settings.local_documents_path,
        settings.cocoindex_db_path,
        settings.document_source,
        settings.graph_enabled,
        settings.multivec_enabled,
    )
    settings.local_documents_path = str(docs)
    settings.cocoindex_db_path = str(tmp_path / "state" / "cocoindex.db")
    settings.document_source = "local"
    settings.graph_enabled = False
    settings.multivec_enabled = False

    with (
        patch.object(app_mod, "create_embedder", lambda *a, **k: _StubEmbedder(dim)),
        patch.object(page_processor, "encode_documents", side_effect=_fake_sparse),
    ):
        yield docs, app_mod.build_app()

    await coco.stop()
    (
        settings.local_documents_path,
        settings.cocoindex_db_path,
        settings.document_source,
        settings.graph_enabled,
        settings.multivec_enabled,
    ) = originals


@pytest.mark.integration
class TestCocoIndexV1WritePath:
    async def test_declare_reconcile_lifecycle(
        self, v1_app: Any, qdrant_client: QdrantClient
    ) -> None:
        docs, app = v1_app
        (docs / "a.txt").write_text("Alpha document about revenue growth in Q3.\n" * 5)
        (docs / "b.txt").write_text("Beta document about expenses and staffing.\n" * 5)

        # A Path B point CocoIndex never declared.
        qdrant_client.upsert(
            DENSE_COLLECTION,
            points=[
                models.PointStruct(
                    id=_LIVE_POINT_ID,
                    vector={
                        DENSE_VECTOR_NAME: [0.5] * settings.dense_dimensions,
                        SPARSE_VECTOR_NAME: models.SparseVector(indices=[3], values=[1.0]),
                    },
                    payload={"source_file": "session:live-1", "is_live": True},
                )
            ],
        )

        # --- run 1: initial ingest -------------------------------------------------
        handle = app.update()
        await handle.result()
        first = handle.stats().total
        assert first.num_errors == 0
        after_first = _count(qdrant_client)
        assert after_first > 1

        # Source keys stay relative to the document root, as under v0.
        assert {"a.txt", "b.txt"} <= _sources(qdrant_client)

        # --- run 2: nothing changed ------------------------------------------------
        handle = app.update()
        await handle.result()
        second = handle.stats().total
        assert second.num_errors == 0
        # The load-bearing assertion: memoized components keep their declared
        # target states, so no point is re-added and none is swept away.
        assert second.num_adds == 0
        assert second.num_unchanged >= 2
        assert _count(qdrant_client) == after_first

        # --- run 3: delete one source file -----------------------------------------
        (docs / "b.txt").unlink()
        handle = app.update()
        await handle.result()
        third = handle.stats().total
        assert third.num_errors == 0

        remaining = _sources(qdrant_client)
        assert "b.txt" not in remaining
        assert "a.txt" in remaining
        # Path B's point was never declared by CocoIndex, so it must survive.
        assert "session:live-1" in remaining
        assert _count(qdrant_client) < after_first

    async def test_failing_file_keeps_its_previous_points(
        self, v1_app: Any, qdrant_client: QdrantClient
    ) -> None:
        """A transient failure must not destroy already-indexed content.

        This is why ``page_processor`` re-raises embedding failures instead of
        logging and returning: a page that is not declared gets reconciled to
        non-existence. Raising must leave the previous state untouched.
        """
        import ingestion.page_processor as page_processor

        docs, app = v1_app
        target = docs / "d.txt"
        target.write_text("Content that indexes fine the first time.\n" * 5)

        await app.update().result()
        before = _count(qdrant_client)
        assert before > 0

        # Change the file so it must be reprocessed, then make embedding fail.
        target.write_text("Changed content that will fail to embed.\n" * 5)
        with patch.object(
            page_processor,
            "encode_documents",
            side_effect=RuntimeError("sparse encoder unavailable"),
        ):
            handle = app.update()
            await handle.result()
            stats = handle.stats().total

        assert stats.num_errors > 0, "the failure must be reported, not hidden"
        assert _count(qdrant_client) == before
        assert "d.txt" in _sources(qdrant_client)

    async def test_changed_file_is_reprocessed(
        self, v1_app: Any, qdrant_client: QdrantClient
    ) -> None:
        docs, app = v1_app
        target = docs / "c.txt"
        target.write_text("Original content about widgets.\n" * 5)

        await app.update().result()
        before = {
            p.id: (p.payload or {}).get("text_content")
            for p in qdrant_client.scroll(
                DENSE_COLLECTION, limit=100, with_payload=["text_content"]
            )[0]
        }

        target.write_text("Replacement content about sprockets entirely.\n" * 5)
        handle = app.update()
        await handle.result()
        stats = handle.stats().total

        assert stats.num_errors == 0
        assert stats.num_unchanged == 0  # the changed file must not be skipped
        after = {
            p.id: (p.payload or {}).get("text_content")
            for p in qdrant_client.scroll(
                DENSE_COLLECTION, limit=100, with_payload=["text_content"]
            )[0]
        }
        assert after != before
        assert any("sprockets" in (t or "") for t in after.values())
