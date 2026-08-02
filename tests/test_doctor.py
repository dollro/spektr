"""Tests for scripts/doctor drift detection + --fix mode."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import scripts.doctor as doctor


class _FakePath:
    def __init__(self, text: str, parts: list[object]) -> None:
        self._text = text
        self._parts = parts

    def to_string(self) -> str:
        return self._text

    def parts(self) -> list[object]:
        return self._parts


class _FakeInfo:
    def __init__(self, path: _FakePath) -> None:
        self.path = path


def _fake_stable_paths(*paths: _FakePath):  # type: ignore[no-untyped-def]
    async def _iter(_app):  # type: ignore[no-untyped-def]
        for p in paths:
            yield _FakeInfo(p)

    return _iter


class TestTrackedFiles:
    """CocoIndex v1 keeps its ledger in LMDB; doctor reads it via cocoindex.inspect."""

    async def test_extracts_source_keys_from_component_paths(self) -> None:
        paths = [
            _FakePath("/", []),
            _FakePath("/@process_file", ["sym"]),
            _FakePath('/@process_file/"a.pdf"', ["sym", "a.pdf"]),
            _FakePath('/@process_file/"sub/b.pdf"', ["sym", "sub/b.pdf"]),
            _FakePath("/@cocoindex/mount_target/x", ["mt", "x"]),
        ]
        with (
            patch("ingestion.app.build_app"),
            patch("cocoindex.inspect.iter_stable_paths", new=_fake_stable_paths(*paths)),
        ):
            assert await doctor._tracked_files() == {"a.pdf", "sub/b.pdf"}

    async def test_unreadable_state_returns_none_not_raise(self) -> None:
        """A fresh checkout with no LMDB directory is a normal condition."""
        with patch("ingestion.app.build_app", side_effect=RuntimeError("no db_path")):
            assert await doctor._tracked_files() is None


class TestDeleteOrphanPoints:
    def test_noop_on_empty(self) -> None:
        with patch("qdrant_client.QdrantClient") as mock_cls:
            doctor._delete_orphan_points([])
        mock_cls.assert_not_called()

    def test_deletes_by_source_file_excluding_live_sessions(self) -> None:
        with patch("qdrant_client.QdrantClient") as mock_cls:
            doctor._delete_orphan_points(["a.pdf", "b.pdf"])
        client = mock_cls.return_value
        client.delete.assert_called_once()
        selector = client.delete.call_args.kwargs["points_selector"]
        condition = selector.filter.must[0]
        assert condition.key == "source_file"
        assert condition.match.any == ["a.pdf", "b.pdf"]
        assert selector.filter.must_not[0].key == "is_live"


class TestFixSafety:
    """--fix deletes Qdrant data, so its guard rails are load-bearing."""

    async def test_refuses_to_wipe_corpus_when_ledger_is_empty(self) -> None:
        """After `rm -rf state/cocoindex.db` every document looks orphaned."""
        entries = [
            {"source_file": "a.pdf", "chunk_count": 1, "page_count": 1},
            {"source_file": "b.pdf", "chunk_count": 1, "page_count": 1},
        ]
        with (
            patch.object(doctor, "_dense_collection_exists", return_value=True),
            patch.object(doctor, "_tracked_files", return_value=set()),
            patch.object(doctor, "list_documents", return_value=entries),
            patch.object(doctor, "_check_embedder_consistency", return_value=[]),
            patch.object(doctor, "_delete_orphan_points") as mock_delete,
        ):
            rc = await doctor.main(fix=True, yes=True)

        mock_delete.assert_not_called()
        assert rc == 1

    async def test_deletes_genuine_orphans_when_ledger_is_populated(self) -> None:
        entries = [
            {"source_file": "a.pdf", "chunk_count": 1, "page_count": 1},
            {"source_file": "stale.pdf", "chunk_count": 1, "page_count": 1},
        ]
        with (
            patch.object(doctor, "_dense_collection_exists", return_value=True),
            patch.object(doctor, "_tracked_files", return_value={"a.pdf"}),
            patch.object(doctor, "list_documents", return_value=entries),
            patch.object(doctor, "_check_embedder_consistency", return_value=[]),
            patch.object(doctor, "_delete_orphan_points") as mock_delete,
        ):
            await doctor.main(fix=True, yes=True)

        mock_delete.assert_called_once_with(["stale.pdf"])

    async def test_unreadable_ledger_skips_the_diff_entirely(self) -> None:
        entries = [{"source_file": "a.pdf", "chunk_count": 1, "page_count": 1}]
        with (
            patch.object(doctor, "_dense_collection_exists", return_value=True),
            patch.object(doctor, "_tracked_files", return_value=None),
            patch.object(doctor, "list_documents", return_value=entries),
            patch.object(doctor, "_check_embedder_consistency", return_value=[]),
            patch.object(doctor, "_delete_orphan_points") as mock_delete,
        ):
            rc = await doctor.main(fix=True, yes=True)

        mock_delete.assert_not_called()
        assert rc == 0


class TestFreshInstall:
    async def test_missing_collection_reports_cleanly(self) -> None:
        """Before the first ingest there is no collection — that is not drift.

        Without this guard, list_documents and the embedder scan each raise a
        raw Qdrant 404 and the run still prints "All in sync".
        """
        with (
            patch.object(doctor, "_dense_collection_exists", return_value=False),
            patch.object(doctor, "list_documents") as mock_list,
            patch.object(doctor, "_delete_orphan_points") as mock_delete,
        ):
            rc = await doctor.main(fix=True, yes=True)

        assert rc == 0
        mock_list.assert_not_called()
        mock_delete.assert_not_called()


class TestEmbedderConsistency:
    def test_empty_collection_is_silent(self) -> None:
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = ([], None)
            assert doctor._check_embedder_consistency() == []

    def test_single_model_is_silent(self) -> None:
        points = [
            MagicMock(payload={"embedder_model": "jina-embeddings-v4", "embedder_dim": 512}),
            MagicMock(payload={"embedder_model": "jina-embeddings-v4", "embedder_dim": 512}),
        ]
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = (points, None)
            assert doctor._check_embedder_consistency() == []

    def test_mixed_models_warn(self) -> None:
        points = [
            MagicMock(payload={"embedder_model": "jina-embeddings-v4", "embedder_dim": 512}),
            MagicMock(payload={"embedder_model": "voyage-4-large", "embedder_dim": 1024}),
        ]
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = (points, None)
            warnings = doctor._check_embedder_consistency()
            assert any("Mixed embedder_model" in w for w in warnings)
            assert any("Mixed embedder_dim" in w for w in warnings)

    def test_unversioned_points_are_informational(self) -> None:
        points = [
            MagicMock(payload={"embedder_model": "jina-embeddings-v4", "embedder_dim": 512}),
            MagicMock(payload={}),
        ]
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = (points, None)
            warnings = doctor._check_embedder_consistency()
            assert any("no embedder_* tags" in w for w in warnings)
            assert not any("Mixed" in w for w in warnings)


class TestMissingSparseVectors:
    def test_scroll_requests_vectors(self) -> None:
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = ([], None)
            doctor._check_embedder_consistency()
            call_kwargs = mock_cls.return_value.scroll.call_args.kwargs
            assert call_kwargs["with_vectors"] is True

    def test_no_missing_sparse_is_silent(self) -> None:
        points = [
            MagicMock(
                payload={
                    "embedder_model": "jina-embeddings-v4",
                    "embedder_dim": 512,
                    "content_type": "text_chunk",
                },
                vector={"dense": [0.1], "sparse": {"indices": [1], "values": [0.5]}},
            ),
        ]
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = (points, None)
            warnings = doctor._check_embedder_consistency()
            assert not any("missing sparse" in w for w in warnings)

    def test_missing_sparse_warns(self) -> None:
        points = [
            MagicMock(
                payload={
                    "embedder_model": "jina-embeddings-v4",
                    "embedder_dim": 512,
                    "content_type": "text_chunk",
                },
                vector={"dense": [0.1]},
            ),
        ]
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = (points, None)
            warnings = doctor._check_embedder_consistency()
            assert any("1 text chunks missing sparse vectors" in w for w in warnings)

    def test_non_text_chunk_points_are_ignored(self) -> None:
        points = [
            MagicMock(
                payload={
                    "embedder_model": "jina-embeddings-v4",
                    "embedder_dim": 512,
                    "content_type": "image",
                },
                vector={"dense": [0.1]},
            ),
        ]
        with patch("qdrant_client.QdrantClient") as mock_cls:
            mock_cls.return_value.scroll.return_value = (points, None)
            warnings = doctor._check_embedder_consistency()
            assert not any("missing sparse" in w for w in warnings)
