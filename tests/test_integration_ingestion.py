"""Integration tests for the ingestion pipeline.

Tests the processing functions that the CocoIndex pipeline calls,
using real Qdrant and Neo4j services (Docker).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.constants import DENSE_COLLECTION, DENSE_DIM, MULTIVEC_COLLECTION
from config.settings import settings
from ingestion.file_processor import file_to_pages, semantic_chunk
from ingestion.pipeline import (
    _process_text_page,
    _process_visual_page,
    ingest_file,
)


@pytest.mark.integration
class TestTextPageIngestion:
    """Test text file → dense Qdrant points + Neo4j entities."""

    async def test_text_file_produces_dense_points(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_txt_bytes: bytes,
        sample_txt_name: str,
    ) -> None:
        result = file_to_pages(sample_txt_name, sample_txt_bytes)
        pages = result.pages
        assert len(pages) == 1
        assert pages[0].content_type == "text"

        await _process_text_page(
            source_file=sample_txt_name,
            text=pages[0].text,
            page_number=1,
            mime="text/plain",
            now="2025-01-01T00:00:00",
            qdrant=qdrant_client,
            embedder=mock_embedder,
            graphiti_writer=None,
        )

        # Verify points in dense collection
        result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        points = result[0]
        assert len(points) >= 1

        point = points[0]
        assert point.payload["source_file"] == sample_txt_name
        assert point.payload["content_type"] == "text_chunk"
        assert point.payload["page_number"] == 1
        assert point.payload["text_content"] != ""
        assert point.payload["metadata"]["mime_type"] == "text/plain"

    async def test_text_file_entities_written_to_neo4j(
        self,
        neo4j_driver,  # type: ignore[no-untyped-def]
        sample_txt_bytes: bytes,
        sample_txt_name: str,
        mock_llm_client: MagicMock,
    ) -> None:
        from ingestion.graph_writer import GraphWriter

        chunks = semantic_chunk(sample_txt_bytes.decode("utf-8"))
        assert len(chunks) >= 1

        graph_writer = GraphWriter()

        await graph_writer.upsert_document(
            source_key=sample_txt_name,
            filename=sample_txt_name,
            mime_type="text/plain",
            page_count=1,
            source_bucket="",
        )

        with patch(
            "ingestion.pipeline.GraphitiWriter",
        ):
            with patch(
                "ingestion.entity_extractor.get_llm_client",
                return_value=mock_llm_client,
            ):
                llm_client = mock_llm_client
                from ingestion.entity_extractor import extract_entities

                for chunk in chunks:
                    chunk_id = f"{sample_txt_name}::p{chunk.page_number}::c{chunk.chunk_index}"
                    await graph_writer.upsert_chunk(
                        chunk_id=chunk_id,
                        text_preview=chunk.text[:200],
                        page_number=chunk.page_number,
                        source_key=sample_txt_name,
                    )
                    result = await extract_entities(
                        chunk.text, llm_client,
                    )
                    await graph_writer.write_extraction_result(
                        source_key=sample_txt_name,
                        chunk_id=chunk_id,
                        extraction_result=result,
                    )

        await graph_writer.close()

        # Verify Neo4j has Document + Chunk nodes
        async with neo4j_driver.session() as session:
            doc_result = await session.run(
                "MATCH (d:Document {source_key: $key}) RETURN d",
                key=sample_txt_name,
            )
            docs = await doc_result.data()
            assert len(docs) == 1

            chunk_result = await session.run(
                "MATCH (d:Document {source_key: $key})-[:HAS_CHUNK]->(c:Chunk) RETURN c",
                key=sample_txt_name,
            )
            chunks_found = await chunk_result.data()
            assert len(chunks_found) >= 1

            # Entity should exist from mock extraction
            entity_result = await session.run(
                "MATCH (e:Entity {name: 'Test Entity'}) RETURN e",
            )
            entities = await entity_result.data()
            assert len(entities) >= 1


@pytest.mark.integration
class TestImagePageIngestion:
    """Test image file → dense + multivec Qdrant points."""

    async def test_image_file_produces_dense_points(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_png_bytes: bytes,
        sample_png_name: str,
    ) -> None:
        result = file_to_pages(sample_png_name, sample_png_bytes)
        pages = result.pages
        assert len(pages) == 1
        assert pages[0].content_type == "image"

        await _process_visual_page(
            source_file=sample_png_name,
            image_bytes=pages[0].image_bytes,
            page_number=1,
            content_type="image",
            mime="image/png",
            now="2025-01-01T00:00:00",
            qdrant=qdrant_client,
            embedder=mock_embedder,
        )

        dense_result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        dense_points = dense_result[0]
        assert len(dense_points) == 1
        assert dense_points[0].payload["content_type"] == "image"

    async def test_image_file_produces_multivec_when_enabled(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_png_bytes: bytes,
        sample_png_name: str,
    ) -> None:
        result = file_to_pages(sample_png_name, sample_png_bytes)
        pages = result.pages

        with patch.object(settings, "multivec_enabled", True):
            await _process_visual_page(
                source_file=sample_png_name,
                image_bytes=pages[0].image_bytes,
                page_number=1,
                content_type="image",
                mime="image/png",
                now="2025-01-01T00:00:00",
                qdrant=qdrant_client,
                embedder=mock_embedder,
            )

        multivec_result = qdrant_client.scroll(
            collection_name=MULTIVEC_COLLECTION,
            limit=100,
        )
        multivec_points = multivec_result[0]
        assert len(multivec_points) == 1
        assert multivec_points[0].payload["content_type"] == "image"


@pytest.mark.integration
class TestPdfIngestion:
    """Test PDF file → dense + multivec points for each page."""

    async def test_pdf_produces_dense_points_per_page(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_pdf_bytes: bytes,
        sample_pdf_name: str,
    ) -> None:
        result = file_to_pages(sample_pdf_name, sample_pdf_bytes)
        pages = result.pages
        assert len(pages) >= 1
        assert all(p.content_type == "pdf" for p in pages)

        for page in pages:
            await _process_visual_page(
                source_file=sample_pdf_name,
                image_bytes=page.image_bytes,
                page_number=page.page_number,
                content_type="pdf_page",
                mime="application/pdf",
                now="2025-01-01T00:00:00",
                qdrant=qdrant_client,
                embedder=mock_embedder,
            )

        dense_result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        assert len(dense_result[0]) == len(pages)


@pytest.mark.integration
class TestIdempotency:
    """Test that running twice on same file produces no duplicates."""

    async def test_no_duplicate_dense_points(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_txt_bytes: bytes,
        sample_txt_name: str,
    ) -> None:
        result = file_to_pages(sample_txt_name, sample_txt_bytes)
        pages = result.pages

        # Run twice
        for _ in range(2):
            await _process_text_page(
                source_file=sample_txt_name,
                text=pages[0].text,
                page_number=1,
                mime="text/plain",
                now="2025-01-01T00:00:00",
                qdrant=qdrant_client,
                embedder=mock_embedder,
                graphiti_writer=None,
            )

        # Deterministic IDs mean upsert overwrites, no duplicates
        result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        chunks = semantic_chunk(pages[0].text)
        assert len(result[0]) == len(chunks)

    async def test_no_duplicate_multivec_points(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_png_bytes: bytes,
        sample_png_name: str,
    ) -> None:
        with patch.object(settings, "multivec_enabled", True):
            for _ in range(2):
                await _process_visual_page(
                    source_file=sample_png_name,
                    image_bytes=sample_png_bytes,
                    page_number=1,
                    content_type="image",
                    mime="image/png",
                    now="2025-01-01T00:00:00",
                    qdrant=qdrant_client,
                    embedder=mock_embedder,
                )

        multivec_result = qdrant_client.scroll(
            collection_name=MULTIVEC_COLLECTION,
            limit=100,
        )
        assert len(multivec_result[0]) == 1


@pytest.mark.integration
class TestCorruptFiles:
    """Test that corrupt/empty files are skipped gracefully."""

    async def test_empty_file_skipped(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
    ) -> None:
        result = file_to_pages("empty.txt", b"")
        pages = result.pages
        assert len(pages) == 1
        assert pages[0].text == ""

        # Empty text produces no chunks, so no points
        await _process_text_page(
            source_file="empty.txt",
            text="",
            page_number=1,
            mime="text/plain",
            now="2025-01-01T00:00:00",
            qdrant=qdrant_client,
            embedder=mock_embedder,
            graphiti_writer=None,
        )

        result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        assert len(result[0]) == 0

    def test_unknown_file_type_returns_no_pages(self) -> None:
        result = file_to_pages("data.xyz", b"some binary data")
        assert len(result.pages) == 0

    def test_ingest_file_handles_corrupt_pdf(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
    ) -> None:
        # Corrupt PDF: valid header but invalid content
        corrupt_pdf = b"%PDF-1.4 corrupt data"

        with (
            patch(
                "ingestion.pipeline.create_embedder",
                return_value=mock_embedder,
            ),
            patch("ingestion.pipeline.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.document_source = "local"
            mock_settings.multivec_enabled = False
            mock_settings.image_embed_strategy = "smart"
            mock_settings.vlm_generation_enabled = False
            # ingest_file should not crash
            result = ingest_file(corrupt_pdf, "corrupt.pdf")
            assert result == "corrupt.pdf"


@pytest.mark.integration
class TestIngestFileEndToEnd:
    """Test the ingest_file function with real services."""

    def test_ingest_text_file(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_txt_bytes: bytes,
        sample_txt_name: str,
    ) -> None:
        mock_graphiti = MagicMock()
        mock_graphiti.ingest_chunk = AsyncMock()
        mock_graphiti.close = AsyncMock()

        with (
            patch(
                "ingestion.pipeline.create_embedder",
                return_value=mock_embedder,
            ),
            patch(
                "ingestion.pipeline.GraphitiWriter",
                return_value=mock_graphiti,
            ),
            patch("ingestion.pipeline.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.document_source = "local"
            mock_settings.multivec_enabled = False
            mock_settings.image_embed_strategy = "smart"
            mock_settings.vlm_generation_enabled = False

            result = ingest_file(sample_txt_bytes, sample_txt_name)

        assert result == sample_txt_name

        # Verify dense points exist
        dense_result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        assert len(dense_result[0]) >= 1

    def test_ingest_image_file(
        self,
        qdrant_client,  # type: ignore[no-untyped-def]
        mock_embedder,  # type: ignore[no-untyped-def]
        sample_png_bytes: bytes,
        sample_png_name: str,
    ) -> None:
        with (
            patch(
                "ingestion.pipeline.create_embedder",
                return_value=mock_embedder,
            ),
            patch("ingestion.pipeline.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.document_source = "local"
            mock_settings.multivec_enabled = False
            mock_settings.image_embed_strategy = "smart"
            mock_settings.vlm_generation_enabled = False

            result = ingest_file(sample_png_bytes, sample_png_name)

        assert result == sample_png_name

        dense_result = qdrant_client.scroll(
            collection_name=DENSE_COLLECTION,
            limit=100,
        )
        assert len(dense_result[0]) == 1
