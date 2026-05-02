"""Unit tests for `list_document_chunks` MCP tool.

The tool enumerates all chunks for a single source file in (page, chunk_index)
order, with no similarity ranking. This is the right primitive when an agent
needs exhaustive coverage of a long document, where vector-similarity top-k
inevitably truncates.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from server.tools import list_document_chunks as mod


def _point(
    *,
    source_file: str = "doc.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
    text_content: str = "hello",
    content_type: str = "text_chunk",
    is_live: bool = False,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    p = MagicMock()
    p.payload = {
        "source_file": source_file,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text_content": text_content,
        "content_type": content_type,
        "is_live": is_live,
        "metadata": metadata or {"mime_type": "application/pdf"},
    }
    return p


def _fake_qdrant(points: list[MagicMock]) -> MagicMock:
    """Mock that returns all points in a single scroll page (no pagination)."""
    qdrant = MagicMock()
    qdrant.scroll.return_value = (points, None)
    return qdrant


@pytest.fixture(autouse=True)
def _reset_client_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_qdrant_client", None)


async def test_returns_chunks_for_source(monkeypatch: pytest.MonkeyPatch) -> None:
    points = [_point(page_number=2, chunk_index=0), _point(page_number=1, chunk_index=0)]
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: _fake_qdrant(points))

    out = await list_document_chunks_async("doc.pdf")

    assert len(out) == 2
    # Sorted by (page, chunk_index)
    assert out[0]["page_number"] == 1
    assert out[1]["page_number"] == 2


async def test_sort_by_page_then_chunk_index(monkeypatch: pytest.MonkeyPatch) -> None:
    points = [
        _point(page_number=1, chunk_index=2),
        _point(page_number=1, chunk_index=0),
        _point(page_number=2, chunk_index=0),
        _point(page_number=1, chunk_index=1),
    ]
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: _fake_qdrant(points))

    out = await list_document_chunks_async("doc.pdf")
    order = [(c["page_number"], c["chunk_index"]) for c in out]
    assert order == [(1, 0), (1, 1), (1, 2), (2, 0)]


async def test_page_range_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """page_from / page_to bounds are translated into a Qdrant range filter."""
    captured: dict[str, Any] = {}

    def _scroll(**kwargs: Any) -> tuple[list[MagicMock], None]:
        captured.update(kwargs)
        return [], None

    qdrant = MagicMock()
    qdrant.scroll.side_effect = _scroll
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: qdrant)

    await list_document_chunks_async("doc.pdf", page_from=5, page_to=10)

    # Inspect the filter sent to qdrant
    f = captured["scroll_filter"]
    must = f.must
    range_conds = [c for c in must if getattr(c, "range", None) is not None]
    assert len(range_conds) == 1
    rng = range_conds[0].range
    assert rng.gte == 5
    assert rng.lte == 10


async def test_limit_and_offset_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    points = [
        _point(page_number=p, chunk_index=0)
        for p in range(1, 11)  # 10 chunks
    ]
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: _fake_qdrant(points))

    page1 = await list_document_chunks_async("doc.pdf", limit=4, offset=0)
    page2 = await list_document_chunks_async("doc.pdf", limit=4, offset=4)
    page3 = await list_document_chunks_async("doc.pdf", limit=4, offset=8)

    assert [c["page_number"] for c in page1] == [1, 2, 3, 4]
    assert [c["page_number"] for c in page2] == [5, 6, 7, 8]
    assert [c["page_number"] for c in page3] == [9, 10]
    assert len(page3) == 2  # signals end of pagination


async def test_excludes_live_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live chunks must never bleed into the bulk-KB enumeration."""
    captured: dict[str, Any] = {}

    def _scroll(**kwargs: Any) -> tuple[list[MagicMock], None]:
        captured.update(kwargs)
        return [], None

    qdrant = MagicMock()
    qdrant.scroll.side_effect = _scroll
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: qdrant)

    await list_document_chunks_async("doc.pdf")

    f = captured["scroll_filter"]
    # Either must_not is_live=True, or must is_live=False — both are valid
    has_live_guard = any(
        getattr(c, "match", None) is not None and getattr(c.match, "value", None) is False
        for c in f.must
    ) or any(getattr(c, "key", None) == "is_live" for c in (f.must_not or []))
    assert has_live_guard, "filter must exclude live chunks"


async def test_content_type_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _scroll(**kwargs: Any) -> tuple[list[MagicMock], None]:
        captured.update(kwargs)
        return [], None

    qdrant = MagicMock()
    qdrant.scroll.side_effect = _scroll
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: qdrant)

    await list_document_chunks_async("doc.pdf", content_type="text_chunk")

    f = captured["scroll_filter"]
    types = [c for c in f.must if getattr(c, "key", None) == "content_type"]
    assert len(types) == 1
    assert types[0].match.value == "text_chunk"


async def test_empty_source_file_returns_error_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await list_document_chunks_async("")
    assert len(out) == 1
    assert "error" in out[0]


async def test_qdrant_failure_returns_error_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qdrant = MagicMock()
    qdrant.scroll.side_effect = RuntimeError("boom")
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: qdrant)

    out = await list_document_chunks_async("doc.pdf")
    assert len(out) == 1
    assert "error" in out[0]


async def test_limit_capped_at_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool must clamp limit to a sane upper bound to protect token budgets."""
    points = [_point(page_number=p, chunk_index=0) for p in range(1, 1001)]
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: _fake_qdrant(points))

    out = await list_document_chunks_async("doc.pdf", limit=10_000)
    assert len(out) == mod.MAX_LIMIT


async def test_returned_chunk_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_get_qdrant_client", lambda: _fake_qdrant([_point()]))
    out = await list_document_chunks_async("doc.pdf")
    chunk = out[0]
    assert set(chunk.keys()) >= {
        "source_file",
        "page_number",
        "chunk_index",
        "text",
        "content_type",
        "metadata",
    }
    assert chunk["text"] == "hello"


# Helper: the actual MCP-decorated tool is async; this trampolines through it.
async def list_document_chunks_async(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return await mod.list_document_chunks(*args, **kwargs)
