from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.cocoindex_ops import (
    op_embed_image,
    op_embed_image_multivec,
    op_embed_text,
)


@pytest.fixture(autouse=True)
def _reset_embedder() -> None:  # type: ignore[misc]
    """Reset the module-level embedder singleton between tests."""
    import ingestion.cocoindex_ops as mod

    mod._embedder = None
    yield  # type: ignore[misc]
    mod._embedder = None


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=[[0.1] * 2048])
    embedder.embed_image = AsyncMock(return_value=[0.1] * 2048)
    embedder.embed_multi_vector = AsyncMock(return_value=[[0.1] * 128, [0.2] * 128])
    return embedder


class TestEmbedText:
    def test_calls_embed_text(self, mock_embedder: MagicMock) -> None:
        with patch(
            "ingestion.cocoindex_ops._get_embedder",
            return_value=mock_embedder,
        ):
            result = op_embed_text("hello world")

        mock_embedder.embed_text.assert_called_once_with(["hello world"])
        assert len(result) == 2048

    def test_returns_single_vector(self, mock_embedder: MagicMock) -> None:
        with patch(
            "ingestion.cocoindex_ops._get_embedder",
            return_value=mock_embedder,
        ):
            result = op_embed_text("test")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)


class TestEmbedImage:
    def test_calls_embed_image(self, mock_embedder: MagicMock) -> None:
        with patch(
            "ingestion.cocoindex_ops._get_embedder",
            return_value=mock_embedder,
        ):
            result = op_embed_image(b"fake_image")

        mock_embedder.embed_image.assert_called_once_with(b"fake_image")
        assert len(result) == 2048


class TestEmbedImageMultivec:
    def test_calls_embed_multi_vector(self, mock_embedder: MagicMock) -> None:
        with patch(
            "ingestion.cocoindex_ops._get_embedder",
            return_value=mock_embedder,
        ):
            result = op_embed_image_multivec(b"fake_image")

        mock_embedder.embed_multi_vector.assert_called_once_with(b"fake_image")
        assert len(result) == 2
        assert all(len(v) == 128 for v in result)
