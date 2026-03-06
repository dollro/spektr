"""Tests for dual-embedding behavior on mixed PDF pages.

Mixed pages (text + visual content) should get BOTH text and image embeddings.
Image embedding is gated by image_embed_strategy and has_visual_content.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.file_processor import Page
from ingestion.pipeline import _build_page_tasks


class TestDualEmbedMixedPages:
    """Verify that PDF pages with both text and image_bytes get dual embeddings."""

    def test_mixed_pdf_page_gets_both_text_and_image_tasks(self) -> None:
        """A PDF page with text AND has_visual_content gets both task types."""
        page = Page(
            image_bytes=b"fake-png",
            text="Revenue grew 15% in Q3 2024.",
            page_number=1,
            content_type="pdf",
            has_visual_content=True,
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.image_embed_strategy = "all"
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="report.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graph_engine=None,
            )

        assert len(tasks.text) >= 1, "Mixed page should have text tasks"
        assert len(tasks.image) >= 1, "Mixed page should have image tasks"
        for coro in tasks.text + tasks.image:
            coro.close()

    def test_pdf_page_no_text_gets_only_image_task(self) -> None:
        """A PDF page with no text (scanned, OCR failed) gets only image task."""
        page = Page(
            image_bytes=b"fake-png",
            text="",
            page_number=1,
            content_type="pdf",
            has_visual_content=True,
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.image_embed_strategy = "smart"
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="scan.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graph_engine=None,
            )

        assert len(tasks.text) == 0, "No-text PDF page should have no text tasks"
        assert len(tasks.image) >= 1, "Visual PDF page should still have image task"
        for coro in tasks.text + tasks.image:
            coro.close()

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
                graph_engine=None,
            )

        assert len(tasks.text) >= 1, "Text page should have text tasks"
        assert len(tasks.image) == 0, "Text page should have no image tasks"
        for coro in tasks.text + tasks.image:
            coro.close()

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
                graph_engine=None,
            )

        assert len(tasks.text) == 0, "Image page should have no text tasks"
        assert len(tasks.image) >= 1, "Image page should have image tasks"
        for coro in tasks.text + tasks.image:
            coro.close()


class TestImageEmbedStrategy:
    """Verify image_embed_strategy gating behavior."""

    def test_smart_strategy_skips_image_for_text_only_page(self) -> None:
        """Smart strategy skips image embedding when has_visual_content=False."""
        page = Page(
            image_bytes=b"fake-png",
            text="Plain text paragraph with no figures.",
            page_number=1,
            content_type="pdf",
            has_visual_content=False,
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.image_embed_strategy = "smart"
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="report.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graph_engine=None,
            )

        assert len(tasks.text) >= 1, "Should still have text tasks"
        assert len(tasks.image) == 0, "Smart strategy should skip image for text-only"
        for coro in tasks.text + tasks.image:
            coro.close()

    def test_smart_strategy_embeds_image_for_visual_page(self) -> None:
        """Smart strategy embeds image when has_visual_content=True."""
        page = Page(
            image_bytes=b"fake-png",
            text="Chart showing revenue growth.",
            page_number=1,
            content_type="pdf",
            has_visual_content=True,
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.image_embed_strategy = "smart"
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="report.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graph_engine=None,
            )

        assert len(tasks.text) >= 1, "Should have text tasks"
        assert len(tasks.image) >= 1, "Smart strategy should embed visual pages"
        for coro in tasks.text + tasks.image:
            coro.close()

    def test_all_strategy_always_embeds_image(self) -> None:
        """All strategy embeds image even when has_visual_content=False."""
        page = Page(
            image_bytes=b"fake-png",
            text="Plain text paragraph.",
            page_number=1,
            content_type="pdf",
            has_visual_content=False,
        )

        with patch("ingestion.pipeline.settings") as mock_settings:
            mock_settings.image_embed_strategy = "all"
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False

            tasks = _build_page_tasks(
                page,
                source_file="report.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graph_engine=None,
            )

        assert len(tasks.text) >= 1, "Should have text tasks"
        assert len(tasks.image) >= 1, "All strategy should always embed image"
        for coro in tasks.text + tasks.image:
            coro.close()
