"""Tests for dense and sparse retrieval channels."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from retrieval.channels import (
    build_filter,
    build_kb_filter,
    build_live_filter,
    dense_channel,
    sparse_channel,
)


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


def _live_point(point_id: str, score: float, text: str, session_id: str) -> MagicMock:
    """A point shaped exactly like what ingestion/live_ingest.py:141-164 writes.

    is_live, session_id, and content_type all live at the payload TOP LEVEL,
    with metadata left empty — not nested, unlike bulk KB points.
    """
    p = MagicMock()
    p.id = point_id
    p.score = score
    p.payload = {
        "source_file": f"session:{session_id}",
        "content_type": "live",
        "is_live": True,
        "session_id": session_id,
        "timestamp": "2026-08-01T00:00:00",
        "text_content": text,
        "page_number": 0,
        "metadata": {},
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
async def test_dense_channel_synthesises_source_type_for_live_points() -> None:
    """A live payload (is_live=True, top-level session_id) gets source_type
    synthesised into its Candidate metadata at read time.

    Production live points never carry source_type in their stored payload
    (ingestion/live_ingest.py writes metadata={}) — this is what makes it
    possible for callers to distinguish live from KB hits at all.
    """
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_live_point("p3", 0.5, "hi", "s1")]
    embedder = MagicMock()
    embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

    with (
        patch("retrieval.channels._get_qdrant_client", return_value=qdrant),
        patch("retrieval.channels._get_embedder", return_value=embedder),
    ):
        out = await dense_channel("hi", limit=10, query_filter=None)

    assert out[0].metadata.get("source_type") == "live"


@pytest.mark.asyncio
async def test_dense_channel_leaves_bulk_kb_metadata_untouched() -> None:
    """A payload with no is_live field gets no source_type injected."""
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_point("p1", 0.9, "hello")]
    embedder = MagicMock()
    embedder.embed_text_query = AsyncMock(return_value=[0.1] * 512)

    with (
        patch("retrieval.channels._get_qdrant_client", return_value=qdrant),
        patch("retrieval.channels._get_embedder", return_value=embedder),
    ):
        out = await dense_channel("hello", limit=10, query_filter=None)

    assert "source_type" not in out[0].metadata


@pytest.mark.asyncio
async def test_dense_channel_empty_query_returns_empty() -> None:
    """A blank query short-circuits without touching Qdrant."""
    with patch("retrieval.channels._get_qdrant_client") as client:
        assert await dense_channel("  ", limit=10, query_filter=None) == []
    client.assert_not_called()


def test_build_filter_combines_conditions() -> None:
    """Content type and source file both become must-conditions."""
    f = build_filter(content_type="text_chunk", source_file="a.pdf")
    assert f is not None
    assert len(f.must) == 2


def test_build_filter_returns_none_when_unfiltered() -> None:
    """No filters means no Filter object."""
    assert build_filter(content_type=None, source_file=None) is None


def test_build_kb_filter_excludes_live_but_admits_missing_is_live() -> None:
    """KB filter excludes is_live=True but admits points with no field at all.

    Bulk points written by Path A never set is_live, so a `must_not:
    is_live==True` condition is required (not `is_live==False OR
    is_null(is_live)` — Qdrant's is_null matches an explicit JSON null, not
    a missing key, so that shape matches zero real KB points).
    """
    f = build_kb_filter(content_type=None, source_file=None)
    assert f.must_not is not None
    assert len(f.must_not) == 1
    condition = f.must_not[0]
    assert condition.key == "is_live"
    assert condition.match.value is True


def test_build_live_filter_scopes_to_session_id() -> None:
    """Live filter matches only the given session_id, at the payload top level."""
    f = build_live_filter("s1")
    assert f.must is not None
    assert len(f.must) == 1
    condition = f.must[0]
    assert condition.key == "session_id"
    assert condition.match.value == "s1"
