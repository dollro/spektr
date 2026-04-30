"""Enumerate all chunks for a single source document.

Use this when an agent needs **exhaustive coverage** of a long document and
vector-similarity top-k inevitably truncates. Unlike `vector_search`, this
tool returns chunks in deterministic (page, chunk_index) order with no
ranking — the agent paginates with `limit` + `offset`.

Typical agent workflow:
1. `list_documents()` to discover available `source_file` names.
2. `list_document_chunks(source_file=...)` to read the document end-to-end.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION
from config.settings import settings

logger = logging.getLogger(__name__)

MAX_LIMIT = 500
DEFAULT_LIMIT = 200

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def _build_filter(
    source_file: str,
    page_from: int | None,
    page_to: int | None,
    content_type: str | None,
) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(
            key="source_file",
            match=models.MatchValue(value=source_file),
        ),
    ]
    if content_type is not None:
        must.append(
            models.FieldCondition(
                key="content_type",
                match=models.MatchValue(value=content_type),
            )
        )
    if page_from is not None or page_to is not None:
        must.append(
            models.FieldCondition(
                key="page_number",
                range=models.Range(gte=page_from, lte=page_to),
            )
        )
    return models.Filter(
        must=must,
        must_not=[
            models.FieldCondition(
                key="is_live",
                match=models.MatchValue(value=True),
            ),
        ],
    )


async def list_document_chunks(
    source_file: str,
    page_from: int | None = None,
    page_to: int | None = None,
    content_type: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return all chunks for `source_file`, ordered by (page, chunk_index).

    No similarity ranking. Use this when the agent needs exhaustive coverage
    of a single document — for example, enumerating every cluster on a 90+
    page portfolio matrix where vector top-k truncates the long tail.

    Args:
        source_file: Document key (as returned by `list_documents`). Required.
        page_from: Optional inclusive lower page bound.
        page_to: Optional inclusive upper page bound.
        content_type: Optional payload filter (e.g. "text_chunk").
        limit: Max chunks per response, capped at 500. Default 200.
        offset: Number of chunks to skip (for pagination across calls).

    Returns:
        Chunks ordered by (page_number, chunk_index). When `len(result) ==
        limit`, the caller should call again with `offset += limit` to
        continue. An empty list (or fewer than `limit` chunks) means the
        document is fully enumerated.
    """
    if not source_file or not source_file.strip():
        return [{"error": "source_file is required and must be non-empty"}]
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)

    qdrant = _get_qdrant_client()
    scroll_filter = _build_filter(source_file, page_from, page_to, content_type)

    try:
        all_points: list[Any] = []
        next_offset: Any = None
        # Hard cap on total scanned to bound work even for very large documents
        scanned = 0
        max_scan = MAX_LIMIT + offset + 1  # enough to fulfil offset+limit ordering

        while True:
            points, next_offset = qdrant.scroll(
                collection_name=DENSE_COLLECTION,
                scroll_filter=scroll_filter,
                limit=256,
                offset=next_offset,
                with_payload=True,
            )
            if not points:
                break
            all_points.extend(points)
            scanned += len(points)
            if next_offset is None or scanned >= max_scan:
                break

        # Deterministic ordering — Qdrant scroll returns by id, not by payload
        all_points.sort(
            key=lambda p: (
                (p.payload or {}).get("page_number", 0),
                (p.payload or {}).get("chunk_index", 0),
            )
        )

        sliced = all_points[offset : offset + limit]
        return [_to_chunk(p) for p in sliced]
    except Exception as exc:
        logger.exception("list_document_chunks failed")
        return [{"error": f"list_document_chunks failed: {exc}"}]


def _to_chunk(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "source_file": payload.get("source_file", ""),
        "page_number": payload.get("page_number", 0),
        "chunk_index": payload.get("chunk_index", 0),
        "content_type": payload.get("content_type", ""),
        "text": payload.get("text_content", payload.get("text", "")),
        "metadata": payload.get("metadata", {}),
    }
