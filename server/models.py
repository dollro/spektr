"""Pydantic response models for MCP search tools."""

from __future__ import annotations

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
    query: str
    strategy: str = "parallel"
