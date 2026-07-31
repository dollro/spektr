"""Tests for scripts/doctor drift detection + --fix mode."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

import scripts.doctor as doctor


@pytest.fixture
def patched_subprocess() -> Iterator[MagicMock]:
    """Patch subprocess.run so tests don't hit a real docker container."""
    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock_run


class TestDeleteTrackingRows:
    def test_noop_on_empty(self, patched_subprocess: MagicMock) -> None:
        doctor._delete_tracking_rows([])
        patched_subprocess.assert_not_called()

    def test_issues_delete_for_listed_files(self, patched_subprocess: MagicMock) -> None:
        doctor._delete_tracking_rows(["a.pdf", "b.pdf"])
        assert patched_subprocess.call_count == 1
        args = patched_subprocess.call_args[0][0]
        sql = next(arg for arg in args if arg.startswith("DELETE"))
        assert '"a.pdf"' in sql
        assert '"b.pdf"' in sql
        assert "ragingestion__cocoindex_tracking" in sql


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
