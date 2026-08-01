"""Tests for dense and sparse retrieval channels."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from retrieval.channels import build_filter, dense_channel, sparse_channel


def _point(point_id: str, score: float, text: str) -> MagicMock:
    p = MagicMock()
    p.id = point_id
    p.score = score
    p.payload = {
        "text_content": text,
        "source_file": "doc.pdf",
        "page_number": 2,
        "chunk_index": 3,
        "metadata": {"mime_type": "application/pdf"},
    }
    return p


@pytest.mark.asyncio
async def test_dense_channel_targets_named_dense_vector() -> None:
    """The dense query must specify using='dense'."""
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_point("p1", 0.9, "hello")]
    embedder = MagicMock()
    embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

    with (
        patch("retrieval.channels._get_qdrant_client", return_value=qdrant),
        patch("retrieval.channels._get_embedder", return_value=embedder),
    ):
        out = await dense_channel("hello", limit=10, query_filter=None)

    assert qdrant.query_points.call_args.kwargs["using"] == "dense"
    assert len(out) == 1
    assert out[0].channel == "dense"
    assert out[0].id == "p1"
    assert out[0].chunk_index == 3


@pytest.mark.asyncio
async def test_sparse_channel_targets_named_sparse_vector() -> None:
    """The sparse query must specify using='sparse'."""
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_point("p2", 4.2, "world")]

    with (
        patch("retrieval.channels._get_qdrant_client", return_value=qdrant),
        patch(
            "retrieval.channels.encode_query",
            return_value=models.SparseVector(indices=[1], values=[0.5]),
        ),
    ):
        out = await sparse_channel("world", limit=10, query_filter=None)

    assert qdrant.query_points.call_args.kwargs["using"] == "sparse"
    assert out[0].channel == "sparse"


@pytest.mark.asyncio
async def test_dense_channel_empty_query_returns_empty() -> None:
    """A blank query short-circuits without touching Qdrant."""
    with patch("retrieval.channels._get_qdrant_client") as client:
        assert await dense_channel("  ", limit=10, query_filter=None) == []
    client.assert_not_called()


def test_build_filter_combines_conditions() -> None:
    """Content type, source file, and session all become must-conditions."""
    f = build_filter(content_type="text_chunk", source_file="a.pdf", session_id="s1")
    assert f is not None
    assert len(f.must) == 3


def test_build_filter_returns_none_when_unfiltered() -> None:
    """No filters means no Filter object."""
    assert build_filter(content_type=None, source_file=None, session_id=None) is None
