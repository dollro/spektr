"""Tests for query decomposition."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.decompose import _model_name, _parse_subqueries, decompose


def test_model_name_falls_back_to_llm_model() -> None:
    """An unset decompose_model routes to the primary LLM model."""
    with (
        patch("retrieval.decompose.settings.decompose_model", ""),
        patch("retrieval.decompose.settings.llm_model", "primary/model"),
    ):
        assert _model_name() == "primary/model"


def test_model_name_prefers_explicit_decompose_model() -> None:
    """An explicit decompose_model wins over llm_model."""
    with (
        patch("retrieval.decompose.settings.decompose_model", "cheap/model"),
        patch("retrieval.decompose.settings.llm_model", "primary/model"),
    ):
        assert _model_name() == "cheap/model"


def test_parse_extracts_numbered_lines() -> None:
    """The model returns one sub-query per line."""
    raw = "1. What is the warranty period?\n2. What voids the warranty?"
    assert _parse_subqueries(raw, max_n=4) == [
        "What is the warranty period?",
        "What voids the warranty?",
    ]


def test_parse_handles_bare_lines() -> None:
    """Unnumbered lines are accepted too."""
    assert _parse_subqueries("first thing\nsecond thing", max_n=4) == [
        "first thing",
        "second thing",
    ]


def test_parse_respects_max() -> None:
    """More sub-queries than allowed are truncated."""
    raw = "\n".join(f"{i}. q{i}" for i in range(1, 10))
    assert len(_parse_subqueries(raw, max_n=4)) == 4


def test_parse_ignores_blank_and_noise_lines() -> None:
    """Empty lines and a preamble are dropped."""
    raw = "Here are the sub-queries:\n\n1. real one\n\n"
    assert _parse_subqueries(raw, max_n=4) == ["Here are the sub-queries:", "real one"]


@pytest.mark.asyncio
async def test_decompose_disabled_returns_identity() -> None:
    """With the flag off, no LLM call happens."""
    with (
        patch("retrieval.decompose.settings.decompose_enabled", False),
        patch("retrieval.decompose._call_llm") as llm,
    ):
        assert await decompose("a and b") == ["a and b"]
    llm.assert_not_called()


@pytest.mark.asyncio
async def test_decompose_llm_failure_returns_identity() -> None:
    """An LLM error degrades to the original query."""
    with patch(
        "retrieval.decompose._call_llm", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        assert await decompose("a and b") == ["a and b"]


@pytest.mark.asyncio
async def test_decompose_single_subquery_returns_original() -> None:
    """If the model returns one part, the query was not multi-part."""
    with patch("retrieval.decompose._call_llm", AsyncMock(return_value="1. a and b")):
        assert await decompose("a and b") == ["a and b"]


@pytest.mark.asyncio
async def test_decompose_returns_subqueries() -> None:
    """A genuine multi-part question splits."""
    with patch(
        "retrieval.decompose._call_llm",
        AsyncMock(return_value="1. warranty period\n2. warranty exclusions"),
    ):
        assert await decompose("what is the warranty and what voids it") == [
            "warranty period",
            "warranty exclusions",
        ]
