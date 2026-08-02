"""Live text ingestion via FastAPI.

Provides HTTP endpoints for real-time text input:
- POST /session/start — create a live session (requires INGEST_API_KEY)
- POST /ingest/chunk — ingest a single text chunk (requires session token)
- POST /session/end — end session (requires session token)
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from config.settings import settings
from ingestion.embedder import Embedder, create_embedder
from ingestion.graphiti_client import get_graphiti
from ingestion.sparse_embedder import encode_documents
from server.models import (
    IngestResponse,
    LiveChunk,
    SessionEndRequest,
    SessionStartRequest,
)

logger = logging.getLogger(__name__)

from config.observability import instrument_fastapi, setup_observability  # noqa: E402

setup_observability()
app = FastAPI(title="Spektr Live Ingest")
instrument_fastapi(app)

_qdrant_client: QdrantClient | None = None
_embedder: Embedder | None = None
_active_session: dict | None = None  # type: ignore[type-arg]


def _extract_bearer(request: Request) -> str:
    """Extract Bearer token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    return auth.removeprefix("Bearer ")


def _require_ingest_api_key(request: Request) -> None:
    """Validate INGEST_API_KEY on session start."""
    if not settings.ingest_api_key:
        return  # auth disabled
    token = _extract_bearer(request)
    if not secrets.compare_digest(token, settings.ingest_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def _require_session_token(request: Request) -> None:
    """Validate the per-session token on ingest/end calls."""
    if not settings.ingest_api_key:
        return  # auth disabled
    if _active_session is None:
        raise HTTPException(status_code=404, detail="No active session")
    token = _extract_bearer(request)
    if not secrets.compare_digest(token, _active_session["session_token"]):
        raise HTTPException(status_code=403, detail="Invalid session token")


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def _get_embedder() -> Embedder:
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


def _make_point_id(key: str) -> str:
    """Deterministic UUID from a string key."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@app.post("/session/start", dependencies=[Depends(_require_ingest_api_key)])
async def start_session(req: SessionStartRequest) -> dict:  # type: ignore[type-arg]
    """Create a new live session. Returns a session_token for subsequent calls."""
    global _active_session  # noqa: PLW0603

    if _active_session is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Active session already exists: {_active_session['session_id']}",
        )

    session_token = secrets.token_urlsafe(32)
    _active_session = {
        "session_id": req.session_id,
        "session_token": session_token,
        "metadata": req.metadata,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }

    logger.info("Session started: %s", req.session_id)
    return {
        "session_id": req.session_id,
        "session_token": session_token,
        "status": "active",
        "created_at": _active_session["created_at"],
    }


@app.post("/ingest/chunk", dependencies=[Depends(_require_session_token)])
async def ingest_chunk(chunk: LiveChunk) -> IngestResponse:
    """Ingest a single text chunk."""
    if _active_session is None:
        raise HTTPException(status_code=404, detail="No active session")
    if _active_session["session_id"] != chunk.session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Session mismatch: active={_active_session['session_id']}",
        )

    # 1. Embed and upsert to Qdrant (immediate)
    embedder = _get_embedder()
    vectors = await embedder.embed_text([chunk.text])
    # miniCOIL sparse encoding runs on local CPU (fastembed) and is blocking;
    # offload to a thread so it doesn't stall the event loop. Sparse is
    # included so live-session points remain reachable via sparse_channel,
    # matching Path A's schema on the same collection.
    sparse_vectors = await asyncio.to_thread(encode_documents, [chunk.text])
    point_key = f"{chunk.session_id}::{chunk.timestamp.isoformat()}"

    _get_qdrant_client().upsert(
        collection_name=DENSE_COLLECTION,
        points=[
            models.PointStruct(
                id=_make_point_id(point_key),
                vector={
                    DENSE_VECTOR_NAME: vectors[0],
                    SPARSE_VECTOR_NAME: sparse_vectors[0],
                },
                payload={
                    "source_file": f"session:{chunk.session_id}",
                    "content_type": "live",
                    "is_live": True,
                    "session_id": chunk.session_id,
                    "timestamp": chunk.timestamp.isoformat(),
                    "text_content": chunk.text,
                    "page_number": 0,
                    "embedder_model": embedder.model_name,
                    "embedder_dim": embedder.dim,
                    "metadata": {},
                },
            ),
        ],
    )

    # 2. Graphiti ingest (background)
    asyncio.create_task(_graphiti_ingest(chunk))

    return IngestResponse(
        status="accepted",
        vector_indexed=True,
        graph_status="processing",
    )


async def _graphiti_ingest(chunk: LiveChunk) -> None:
    """Background task: ingest chunk as Graphiti episode."""
    try:
        client = await get_graphiti()
        episode_name = f"{chunk.session_id}:t{chunk.timestamp.isoformat()}"
        await client.add_episode(
            name=episode_name,
            episode_body=chunk.text,
            source_description=f"Live session chunk for session {chunk.session_id}",
            reference_time=chunk.timestamp,
            group_id=chunk.session_id,
        )
        logger.info("Graphiti episode ingested: %s", episode_name)
    except Exception:
        logger.exception("Graphiti background ingest failed for %s", chunk.session_id)


@app.post("/session/end", dependencies=[Depends(_require_session_token)])
async def end_session(req: SessionEndRequest) -> dict:  # type: ignore[type-arg]
    """End a session: archive (keep data) or discard (delete data)."""
    global _active_session  # noqa: PLW0603

    if _active_session is None or _active_session["session_id"] != req.session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    qdrant = _get_qdrant_client()

    if req.archive:
        # Set is_live=false on all session points
        points, _ = qdrant.scroll(
            collection_name=DENSE_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=req.session_id),
                    ),
                ],
            ),
            limit=10000,
        )
        if points:
            qdrant.set_payload(
                collection_name=DENSE_COLLECTION,
                payload={"is_live": False},
                points=[p.id for p in points],
            )
        status = "archived"
    else:
        # Delete all points for this session
        qdrant.delete(
            collection_name=DENSE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=req.session_id),
                        ),
                    ],
                ),
            ),
        )
        # Delete Graphiti data
        try:
            client = await get_graphiti()
            await client.delete_group(req.session_id)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to delete Graphiti group %s", req.session_id)
        status = "discarded"

    _active_session = None
    logger.info("Session %s: %s", req.session_id, status)
    return {"session_id": req.session_id, "status": status}
