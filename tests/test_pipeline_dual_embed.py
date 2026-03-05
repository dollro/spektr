"""Tests for dual-embedding behavior on mixed PDF pages.

Mixed pages (text + visual content) should get BOTH text and image embeddings.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.file_processor import Page
from ingestion.pipeline import _build_page_tasks


class TestDualEmbedMixedPages:
    """Verify that PDF pages with both text and image_bytes get dual embeddings."""

    def test_mixed_pdf_page_gets_both_text_and_image_tasks(self) -> None:
        """A PDF page with text AND image_bytes gets both task types."""
        page = Page(
            image_bytes=b"fake-png",
            text="Revenue grew 15% in Q3 2024.",
            page_number=1,
            content_type="pdf",
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="report.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graphiti_writer=None,
            )

        assert len(tasks.text) >= 1, "Mixed page should have text tasks"
        assert len(tasks.image) >= 1, "Mixed page should have image tasks"

    def test_pdf_page_no_text_gets_only_image_task(self) -> None:
        """A PDF page with no text (scanned, OCR failed) gets only image task."""
        page = Page(
            image_bytes=b"fake-png",
            text="",
            page_number=1,
            content_type="pdf",
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="scan.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graphiti_writer=None,
            )

        assert len(tasks.text) == 0, "No-text PDF page should have no text tasks"
        assert len(tasks.image) >= 1, "No-text PDF page should still have image task"

    def test_text_only_page_gets_only_text_task(self) -> None:
        """A plain text page (not PDF) gets only text task, no image task."""
        page = Page(
            image_bytes=b"",
            text="Plain text document content.",
            page_number=1,
            content_type="text",
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="readme.md",
                mime="text/markdown",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graphiti_writer=None,
            )

        assert len(tasks.text) >= 1, "Text page should have text tasks"
        assert len(tasks.image) == 0, "Text page should have no image tasks"

    def test_image_page_gets_only_image_task(self) -> None:
        """A standalone image gets only image task, no text task."""
        page = Page(
            image_bytes=b"fake-png",
            text="",
            page_number=1,
            content_type="image",
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="photo.jpg",
                mime="image/jpeg",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graphiti_writer=None,
            )

        assert len(tasks.text) == 0, "Image page should have no text tasks"
        assert len(tasks.image) >= 1, "Image page should have image tasks"
