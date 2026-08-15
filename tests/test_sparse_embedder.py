"""Tests for the miniCOIL sparse embedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config.constants import MINICOIL_AVG_LEN
from config.settings import settings
from ingestion import sparse_embedder
from ingestion.sparse_embedder import encode_documents, encode_query, reset_model


class _FakeEmbedding:
    """Mimics a fastembed SparseEmbedding."""

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    reset_model()


def test_encode_documents_returns_sparse_vectors() -> None:
    """Document texts map to Qdrant SparseVector objects."""
    fake = MagicMock()
    fake.embed.return_value = iter([_FakeEmbedding([1, 5], [0.4, 0.7])])

    with patch("ingestion.sparse_embedder._load_model", return_value=fake):
        out = encode_documents(["hello world"])

    assert len(out) == 1
    assert out[0].indices == [1, 5]
    assert out[0].values == [0.4, 0.7]


def test_encode_documents_empty_input_skips_model() -> None:
    """No texts means no model load and no output."""
    with patch("ingestion.sparse_embedder._load_model") as loader:
        assert encode_documents([]) == []
    loader.assert_not_called()


def test_encode_query_returns_single_vector() -> None:
    """Query encoding returns one SparseVector, not a list."""
    fake = MagicMock()
    fake.query_embed.return_value = iter([_FakeEmbedding([2], [0.9])])

    with patch("ingestion.sparse_embedder._load_model", return_value=fake):
        out = encode_query("hello")

    assert out.indices == [2]
    assert out.values == [0.9]


def test_model_is_loaded_once() -> None:
    """The model is cached across calls — loading is expensive."""
    fake = MagicMock()
    fake.embed.return_value = iter([_FakeEmbedding([1], [0.5])])

    with patch("ingestion.sparse_embedder._load_model", return_value=fake) as loader:
        encode_documents(["a"])
        fake.embed.return_value = iter([_FakeEmbedding([1], [0.5])])
        encode_documents(["b"])

    assert loader.call_count == 1


def test_load_model_passes_avg_len_as_constructor_kwarg() -> None:
    """avg_len must be a constructor kwarg to SparseTextEmbedding, not a per-call option.

    fastembed 0.8.0's ``embed()``/``query_embed()`` accept no ``options`` kwarg — an
    ``avg_len`` passed that way is silently swallowed into unused **kwargs and never
    reaches the BM25 length-normalisation term. The real API only respects ``avg_len``
    when set at construction time (``SparseTextEmbedding(..., avg_len=...)``); the
    document/query asymmetry is then handled internally by the library, not by us.
    """
    fake_cls = MagicMock()

    with patch("fastembed.SparseTextEmbedding", fake_cls):
        sparse_embedder._load_model()

    fake_cls.assert_called_once_with(
        model_name=settings.sparse_model, avg_len=MINICOIL_AVG_LEN
    )
