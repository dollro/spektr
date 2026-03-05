from __future__ import annotations

import pytest

from ingestion.file_processor import TextChunk


class TestGroupChunksForGraph:
    def test_groups_small_chunks_to_target_size(self) -> None:
        """Adjacent chunks on same page are merged up to target size."""
        from ingestion.graph_writer import group_chunks_for_graph

        chunks = [
            TextChunk(text="A" * 400, chunk_index=0, page_number=1),
            TextChunk(text="B" * 400, chunk_index=1, page_number=1),
            TextChunk(text="C" * 400, chunk_index=2, page_number=1),
            TextChunk(text="D" * 400, chunk_index=3, page_number=1),
        ]

        grouped = group_chunks_for_graph(chunks, target_size=1000)
        # 400+400=800 < 1000 → group; 800+400=1200 > 1000 → new group
        assert len(grouped) == 2
        assert "A" * 400 in grouped[0].text
        assert "B" * 400 in grouped[0].text
        assert "C" * 400 in grouped[1].text
        assert "D" * 400 in grouped[1].text

    def test_preserves_page_boundaries(self) -> None:
        """Chunks from different pages are never grouped together."""
        from ingestion.graph_writer import group_chunks_for_graph

        chunks = [
            TextChunk(text="Page1 chunk", chunk_index=0, page_number=1),
            TextChunk(text="Page2 chunk", chunk_index=0, page_number=2),
        ]

        grouped = group_chunks_for_graph(chunks, target_size=5000)
        assert len(grouped) == 2

    def test_prefers_contextualized_text(self) -> None:
        """Grouped text uses contextualized_text when available."""
        from ingestion.graph_writer import group_chunks_for_graph

        chunks = [
            TextChunk(
                text="raw A",
                chunk_index=0,
                page_number=1,
                contextualized_text="## Heading\nraw A",
            ),
            TextChunk(text="raw B", chunk_index=1, page_number=1),
        ]

        grouped = group_chunks_for_graph(chunks, target_size=5000)
        assert len(grouped) == 1
        assert "## Heading\nraw A" in grouped[0].text
        assert "raw B" in grouped[0].text

    def test_single_large_chunk_passes_through(self) -> None:
        """A chunk already larger than target_size is kept as-is."""
        from ingestion.graph_writer import group_chunks_for_graph

        chunks = [
            TextChunk(text="X" * 2000, chunk_index=0, page_number=1),
        ]

        grouped = group_chunks_for_graph(chunks, target_size=1500)
        assert len(grouped) == 1
        assert grouped[0].text == "X" * 2000

    def test_empty_input_returns_empty(self) -> None:
        """Empty chunk list returns empty list."""
        from ingestion.graph_writer import group_chunks_for_graph

        assert group_chunks_for_graph([], target_size=1500) == []

    def test_grouped_chunk_metadata(self) -> None:
        """Grouped chunk uses first chunk's page_number and chunk_index."""
        from ingestion.graph_writer import group_chunks_for_graph

        chunks = [
            TextChunk(text="A" * 400, chunk_index=3, page_number=2),
            TextChunk(text="B" * 400, chunk_index=4, page_number=2),
        ]

        grouped = group_chunks_for_graph(chunks, target_size=1500)
        assert len(grouped) == 1
        assert grouped[0].page_number == 2
        assert grouped[0].chunk_index == 3
