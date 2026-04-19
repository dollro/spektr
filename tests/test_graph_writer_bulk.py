from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk


class TestGraphitiWriterBulk:
    @pytest.mark.asyncio
    async def test_ingest_bulk_builds_raw_episodes(self) -> None:
        """ingest_bulk creates RawEpisode per chunk and calls add_episode_bulk."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="chunk zero", chunk_index=0, page_number=1),
            TextChunk(
                text="chunk one raw",
                chunk_index=1,
                page_number=1,
                contextualized_text="## Heading\nchunk one raw",
            ),
            TextChunk(text="chunk two", chunk_index=2, page_number=2),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(return_value=MagicMock())

        with (
            patch(
                "ingestion.graph_writer.get_graphiti",
                return_value=mock_client,
            ),
            patch(
                "ingestion.graph_writer.settings",
            ) as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            writer = GraphitiWriter()
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        mock_client.add_episode_bulk.assert_called_once()
        episodes = mock_client.add_episode_bulk.call_args.args[0]
        assert len(episodes) == 2
        # Page 1 chunks grouped: chunk 0 (raw) + chunk 1 (contextualized)
        assert "chunk zero" in episodes[0].content
        assert "## Heading\nchunk one raw" in episodes[0].content
        # Page 2 chunk standalone
        assert episodes[1].content == "chunk two"
        # Names use anchor chunk
        assert episodes[0].name == "test.pdf:p1:c0"
        assert episodes[1].name == "test.pdf:p2:c2"
        assert episodes[0].source_description == "test.pdf"

    @pytest.mark.asyncio
    async def test_ingest_bulk_empty_chunks_is_noop(self) -> None:
        """ingest_bulk with empty chunk list does not call Graphiti."""
        from ingestion.graph_writer import GraphitiWriter

        mock_client = AsyncMock()

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            await writer.ingest_bulk(chunks=[], source_key="test.pdf")

        mock_client.add_episode_bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_bulk_falls_back_on_error(self) -> None:
        """When add_episode_bulk fails, falls back to sequential add_episode."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="chunk A", chunk_index=0, page_number=1),
            TextChunk(text="chunk B", chunk_index=1, page_number=1),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(
            side_effect=Exception("NodeResolutions ValidationError")
        )
        mock_client.add_episode = AsyncMock()

        with (
            patch(
                "ingestion.graph_writer.get_graphiti",
                return_value=mock_client,
            ),
            patch(
                "ingestion.graph_writer.settings",
            ) as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            writer = GraphitiWriter()
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        # Bulk failed; chunks grouped into 1 episode, so 1 sequential call
        assert mock_client.add_episode.call_count == 1

    @pytest.mark.asyncio
    async def test_ingest_bulk_fallback_logs_individual_failures(
        self,
    ) -> None:
        """Sequential fallback continues on individual chunk failures."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="ok chunk", chunk_index=0, page_number=1),
            TextChunk(text="bad chunk", chunk_index=1, page_number=1),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(side_effect=Exception("bulk failed"))
        # Single grouped episode fails
        mock_client.add_episode = AsyncMock(side_effect=Exception("single failed"))

        with (
            patch(
                "ingestion.graph_writer.get_graphiti",
                return_value=mock_client,
            ),
            patch(
                "ingestion.graph_writer.settings",
            ) as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            writer = GraphitiWriter()
            # Should not raise
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        # Chunks grouped into 1 episode, so 1 sequential call
        assert mock_client.add_episode.call_count == 1

    @pytest.mark.asyncio
    async def test_ingest_bulk_groups_chunks(self) -> None:
        """ingest_bulk groups small chunks into fewer episodes."""
        from ingestion.graph_writer import GraphitiWriter

        # 10 small chunks (100 chars each) → should group into fewer episodes
        chunks = [
            TextChunk(text=f"chunk {i} " * 10, chunk_index=i, page_number=1) for i in range(10)
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(return_value=MagicMock())

        with (
            patch(
                "ingestion.graph_writer.get_graphiti",
                return_value=mock_client,
            ),
            patch(
                "ingestion.graph_writer.settings",
            ) as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            writer = GraphitiWriter()
            await writer.ingest_bulk(chunks=chunks, source_key="test.pdf")

        mock_client.add_episode_bulk.assert_called_once()
        episodes = mock_client.add_episode_bulk.call_args.args[0]
        # 10 chunks of ~80 chars → should be grouped (fewer than 10 episodes)
        assert len(episodes) < 10
        assert len(episodes) >= 1
