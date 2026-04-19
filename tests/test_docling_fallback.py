from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.file_processor import _extract_text_docling


class TestDoclingFallback:
    def test_returns_text_when_docling_available(self) -> None:
        """Docling extracts text from image bytes when available."""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Extracted OCR text"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch(
            "ingestion.file_processor._get_docling_converter",
            return_value=mock_converter,
        ):
            text = _extract_text_docling(b"fake-image-bytes")

        assert text == "Extracted OCR text"

    def test_returns_empty_when_docling_unavailable(self) -> None:
        """Returns empty string when docling is not installed."""
        with patch(
            "ingestion.file_processor._get_docling_converter",
            return_value=None,
        ):
            text = _extract_text_docling(b"fake-image-bytes")

        assert text == ""
