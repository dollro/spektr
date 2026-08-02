"""Shared dataclasses for the retrieval pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A single ranked hit from one retrieval channel."""

    id: str
    text: str
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    score: float
    channel: str
    metadata: dict = Field(default_factory=dict)  # type: ignore[type-arg]


class FusedResult(BaseModel):
    """A candidate after rank fusion, optionally rescored by the reranker."""

    id: str
    text: str
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    score: float  # rerank score when reranked, else equal to fusion_score
    fusion_score: float
    channels: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)  # type: ignore[type-arg]
