from __future__ import annotations

from pathlib import Path

from ingestion.file_processor import file_to_pages, semantic_chunk

FIXTURES = Path(__file__).parent / "fixtures"


class TestFileToPages:
    def test_png_returns_single_image_page(self) -> None:
        content = (FIXTURES / "sample.png").read_bytes()
        pages = file_to_pages("photo.png", content)

        assert len(pages) == 1
        assert pages[0].content_type == "image"
        assert pages[0].image_bytes == content
        assert pages[0].text == ""
        assert pages[0].page_number == 1

    def test_jpg_detected_as_image(self) -> None:
        pages = file_to_pages("photo.jpg", b"\xff\xd8\xff\xe0fake")

        assert len(pages) == 1
        assert pages[0].content_type == "image"

    def test_txt_returns_text_page(self) -> None:
        content = (FIXTURES / "sample.txt").read_bytes()
        pages = file_to_pages("doc.txt", content)

        assert len(pages) == 1
        assert pages[0].content_type == "text"
        assert pages[0].text == content.decode("utf-8")
        assert pages[0].image_bytes == b""

    def test_markdown_detected_as_text(self) -> None:
        pages = file_to_pages("readme.md", b"# Hello")

        assert len(pages) == 1
        assert pages[0].content_type == "text"
        assert pages[0].text == "# Hello"

    def test_pdf_returns_multiple_pages(self) -> None:
        content = (FIXTURES / "sample.pdf").read_bytes()
        pages = file_to_pages("doc.pdf", content)

        assert len(pages) == 2
        for i, page in enumerate(pages):
            assert page.content_type == "pdf"
            assert page.page_number == i + 1
            assert len(page.image_bytes) > 0
            assert page.text == ""
            # Verify PNG signature
            assert page.image_bytes[:4] == b"\x89PNG"

    def test_unknown_type_returns_empty(self) -> None:
        pages = file_to_pages("data.xyz123", b"mystery")
        assert pages == []


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
