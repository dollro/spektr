"""Pydantic response models for MCP search tools."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SearchResult(BaseModel):
    """Dense vector search result."""

    score: float
    text: str
    source_file: str
    page_number: int
    content_type: str
    metadata: dict = {}  # type: ignore[type-arg]


class VisualSearchResult(BaseModel):
    """ColBERT multi-vector visual search result."""

    score: float
    source_file: str
    page_number: int
    content_type: str
    source_key: str
    metadata: dict = {}  # type: ignore[type-arg]


class GraphFact(BaseModel):
    """Knowledge graph fact from any graph engine."""

    fact: str
    source: str | None = None
    created_at: str | None = None
    expired_at: str | None = None
    entities: list[str] | None = None
    relation_type: str | None = None
    confidence: float | None = None


class HybridSearchResponse(BaseModel):
    """Combined vector + graph search response."""

    vector_results: list[SearchResult] = []
    graph_results: list[GraphFact] = []
    live_results: list[SearchResult] = []
    query: str
    session_id: str | None = None
    strategy: str = "parallel"
    errors: list[str] | None = None


class LiveChunk(BaseModel):
    """A single text chunk from a live session."""

    session_id: str
    text: str
    timestamp: datetime


class SessionStartRequest(BaseModel):
    """Request to start a new live session."""

    session_id: str
    metadata: dict = {}  # type: ignore[type-arg]


class SessionEndRequest(BaseModel):
    """Request to end a live session."""

    session_id: str
    archive: bool = False


class IngestResponse(BaseModel):
    """Response from live text ingestion."""

    status: str
    vector_indexed: bool
    graph_status: str
