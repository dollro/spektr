from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.file_processor import (
    FileProcessingResult,
    docling_chunk,
    file_to_pages,
    semantic_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestFileToPages:
    def test_png_returns_single_image_page(self) -> None:
        content = (FIXTURES / "sample.png").read_bytes()
        result = file_to_pages("photo.png", content)
        assert isinstance(result, FileProcessingResult)
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "image"
        assert pages[0].image_bytes == content
        assert pages[0].text == ""
        assert pages[0].page_number == 1

    def test_jpg_detected_as_image(self) -> None:
        result = file_to_pages("photo.jpg", b"\xff\xd8\xff\xe0fake")
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "image"

    def test_txt_returns_text_page(self) -> None:
        content = (FIXTURES / "sample.txt").read_bytes()
        result = file_to_pages("doc.txt", content)
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "text"
        assert pages[0].text == content.decode("utf-8")
        assert pages[0].image_bytes == b""

    def test_markdown_detected_as_text(self) -> None:
        result = file_to_pages("readme.md", b"# Hello")
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "text"
        assert pages[0].text == "# Hello"

    def test_pdf_extracts_text_and_image(self) -> None:
        """PDF pages with text layer have both text and image_bytes."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        result = file_to_pages("doc.pdf", content)
        pages = result.pages

        assert len(pages) == 2
        for i, page in enumerate(pages):
            assert page.content_type == "pdf"
            assert page.page_number == i + 1
            assert len(page.image_bytes) > 0
            assert page.image_bytes[:4] == b"\x89PNG"
            assert len(page.text) > 0

    def test_pdf_image_at_150dpi(self) -> None:
        """PDF pages rendered at 150 DPI are smaller than 300 DPI."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        result = file_to_pages("doc.pdf", content)
        pages = result.pages

        for page in pages:
            assert len(page.image_bytes) < 1_500_000

    def test_unknown_type_returns_empty(self) -> None:
        result = file_to_pages("data.xyz123", b"mystery")
        assert result.pages == []

    def test_pdf_returns_docling_document(self) -> None:
        """PDF processing populates docling_document when Docling available."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc

        with patch("ingestion.file_processor._get_docling_converter") as mock_get:
            mock_converter = MagicMock()
            mock_converter.convert.return_value = mock_result
            mock_get.return_value = mock_converter

            result = file_to_pages("doc.pdf", content)

        assert result.docling_document is mock_doc

    def test_pdf_docling_document_none_when_unavailable(self) -> None:
        """PDF processing returns None docling_document when Docling absent."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        with patch(
            "ingestion.file_processor._get_docling_converter",
            return_value=None,
        ):
            result = file_to_pages("doc.pdf", content)

        assert result.docling_document is None

    def test_text_file_has_no_docling_document(self) -> None:
        """Text files don't produce a docling_document."""
        result = file_to_pages("readme.md", b"# Hello")
        assert result.docling_document is None


class TestSemanticChunk:
    def test_short_text_single_chunk(self) -> None:
        chunks = semantic_chunk("Short text.", max_chunk_size=512)

        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].chunk_index == 0
        assert chunks[0].page_number == 1

    def test_paragraph_split(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = semantic_chunk(text, max_chunk_size=30)

        assert len(chunks) == 3
        assert chunks[0].text == "First paragraph."
        assert chunks[1].text == "Second paragraph."
        assert chunks[2].text == "Third paragraph."

    def test_merges_small_paragraphs(self) -> None:
        text = "A.\n\nB.\n\nC."
        chunks = semantic_chunk(text, max_chunk_size=512)

        assert len(chunks) == 1
        assert "A." in chunks[0].text
        assert "B." in chunks[0].text
        assert "C." in chunks[0].text

    def test_long_text_produces_multiple_chunks(self) -> None:
        # 1500 chars total, max_chunk_size=512 -> at least 3 chunks
        text = ("word " * 100).strip()  # 499 chars
        text = text + "\n\n" + text + "\n\n" + text
        chunks = semantic_chunk(text, max_chunk_size=512)

        assert len(chunks) >= 3

    def test_chunk_indices_sequential(self) -> None:
        text = "A.\n\nB.\n\nC."
        chunks = semantic_chunk(text, max_chunk_size=5)

        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(indices)

    def test_empty_text_returns_empty(self) -> None:
        chunks = semantic_chunk("", max_chunk_size=512)
        assert chunks == []

    def test_only_whitespace_returns_empty(self) -> None:
        chunks = semantic_chunk("   \n\n   ", max_chunk_size=512)
        assert chunks == []


class TestDoclingChunk:
    def test_returns_text_chunks_with_page_numbers(self) -> None:
        """docling_chunk produces TextChunks with contextualized text."""
        mock_doc = MagicMock()
        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "First heading content"
        mock_chunk_1.meta = MagicMock()
        mock_chunk_1.meta.headings = ["Introduction"]
        mock_chunk_1.meta.doc_items = [MagicMock()]
        mock_chunk_1.meta.doc_items[0].prov = [MagicMock(page_no=1)]

        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "Table content here"
        mock_chunk_2.meta = MagicMock()
        mock_chunk_2.meta.headings = ["Results", "Table 1"]
        mock_chunk_2.meta.doc_items = [MagicMock()]
        mock_chunk_2.meta.doc_items[0].prov = [MagicMock(page_no=2)]

        with patch("ingestion.file_processor._get_hybrid_chunker") as mock_chunker_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = [mock_chunk_1, mock_chunk_2]
            mock_chunker.contextualize.side_effect = [
                "Introduction\nFirst heading content",
                "Results > Table 1\nTable content here",
            ]
            mock_chunker_factory.return_value = mock_chunker

            chunks = docling_chunk(mock_doc)

        assert len(chunks) == 2
        assert chunks[0].text == "First heading content"
        assert chunks[0].contextualized_text == ("Introduction\nFirst heading content")
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[1].text == "Table content here"
        assert chunks[1].contextualized_text == ("Results > Table 1\nTable content here")
        assert chunks[1].page_number == 2
        assert chunks[1].chunk_index == 1

    def test_returns_empty_for_none_document(self) -> None:
        """docling_chunk returns empty list for None input."""
        assert docling_chunk(None) == []

    def test_falls_back_on_chunker_error(self) -> None:
        """docling_chunk returns empty list if HybridChunker fails."""
        mock_doc = MagicMock()
        with patch("ingestion.file_processor._get_hybrid_chunker") as mock_factory:
            mock_factory.side_effect = Exception("chunker init failed")
            result = docling_chunk(mock_doc)

        assert result == []

    def test_returns_empty_when_chunker_unavailable(self) -> None:
        """docling_chunk returns empty when HybridChunker not installed."""
        mock_doc = MagicMock()
        with patch(
            "ingestion.file_processor._get_hybrid_chunker",
            return_value=None,
        ):
            result = docling_chunk(mock_doc)

        assert result == []

    def test_contextualize_failure_uses_raw_text(self) -> None:
        """If contextualize() fails, fall back to raw chunk.text."""
        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Raw text content"
        mock_chunk.meta = MagicMock()
        mock_chunk.meta.doc_items = [MagicMock()]
        mock_chunk.meta.doc_items[0].prov = [MagicMock(page_no=1)]

        with patch("ingestion.file_processor._get_hybrid_chunker") as mock_chunker_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = [mock_chunk]
            mock_chunker.contextualize.side_effect = Exception("contextualize failed")
            mock_chunker_factory.return_value = mock_chunker

            chunks = docling_chunk(mock_doc)

        assert len(chunks) == 1
        assert chunks[0].text == "Raw text content"
        assert chunks[0].contextualized_text is None
