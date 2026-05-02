"""List documents ingested into the knowledge base.

Scrolls the Qdrant dense collection and returns the distinct
source files with per-document chunk + page counts. Excludes
live session data (is_live=True).
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION
from config.settings import settings

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


async def list_documents(limit: int = 100) -> list[dict]:  # type: ignore[type-arg]
    """List distinct documents in the knowledge base.

    Returns one entry per source file with chunk and page counts.
    Excludes live session data (use session-aware search for that).

    Args:
        limit: Max number of documents to return (default 100).
    """
    limit = max(1, min(limit, 1000))
    qdrant = _get_qdrant_client()

    kb_filter = models.Filter(
        must_not=[
            models.FieldCondition(
                key="is_live",
                match=models.MatchValue(value=True),
            ),
        ],
    )

    try:
        docs: dict[str, dict] = {}  # type: ignore[type-arg]
        next_offset = None
        scanned = 0
        max_scan = 10_000

        while scanned < max_scan:
            points, next_offset = qdrant.scroll(
                collection_name=DENSE_COLLECTION,
                scroll_filter=kb_filter,
                limit=256,
                offset=next_offset,
                with_payload=["source_file", "page_number", "content_type"],
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                src = payload.get("source_file")
                if not src:
                    continue
                entry = docs.setdefault(
                    src,
                    {
                        "source_file": src,
                        "chunk_count": 0,
                        "pages": set(),
                        "content_types": set(),
                    },
                )
                entry["chunk_count"] += 1
                if (page := payload.get("page_number")) is not None:
                    entry["pages"].add(page)
                if (ct := payload.get("content_type")) is not None:
                    entry["content_types"].add(ct)

            scanned += len(points)
            if next_offset is None:
                break

        results = [
            {
                "source_file": d["source_file"],
                "chunk_count": d["chunk_count"],
                "page_count": len(d["pages"]),
                "content_types": sorted(d["content_types"]),
            }
            for d in docs.values()
        ]
        results.sort(key=lambda r: r["source_file"])
        return results[:limit]
    except Exception as exc:
        logger.exception("list_documents failed")
        return [{"error": f"list_documents failed: {exc}"}]
