# Graphiti Bulk Ingestion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace sequential `add_episode()` calls with `add_episode_bulk()` for dramatically faster knowledge graph ingestion, with proper error handling and SEMAPHORE_LIMIT tuning.

**Architecture:** The current pipeline calls `GraphitiWriter.ingest_chunk()` once per text chunk in a sequential loop (~74 calls for a 25-page PDF). We replace this with a single `add_episode_bulk()` call that processes all chunks in parallel internally. The `GraphitiWriter` gets a new `ingest_bulk()` method. The pipeline's `_ingest_to_graphiti()` collects all chunks across all pages first, then does one bulk call. A `SEMAPHORE_LIMIT` env var is exposed for LLM concurrency tuning. Fallback to sequential `add_episode()` is provided when bulk fails.

**Tech Stack:** graphiti-core (RawEpisode, EpisodeType, add_episode_bulk), asyncio, Pydantic Settings

---

## Context for Implementer

### Key API signatures (verified against installed graphiti-core)

```python
# graphiti_core.utils.bulk_utils.RawEpisode
RawEpisode(
    name: str,           # episode identifier (e.g., "doc.pdf:p1:c0")
    content: str,        # the chunk text
    source: EpisodeType, # EpisodeType.text for document chunks
    source_description: str,  # source file key
    reference_time: datetime,
    uuid: str | None = None,
)

# graphiti_core.Graphiti.add_episode_bulk
await client.add_episode_bulk(
    bulk_episodes: list[RawEpisode],
    group_id: str | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    ...
) -> AddBulkEpisodeResults
```

### Known bugs in `add_episode_bulk`
- **#879**: `NodeResolutions ValidationError` — LLM returns malformed JSON for entity deduplication
- **#882**: `IndexError` during node resolution — same root cause (bad LLM output)
- These are LLM-dependent; use capable models (Claude Sonnet, GPT-4o) to minimize risk

### Important files
- `ingestion/graph_writer.py` — `GraphitiWriter` class (lines 19-56)
- `ingestion/graphiti_client.py` — singleton Graphiti client lifecycle
- `ingestion/pipeline.py` — `_ingest_to_graphiti()` (lines 469-495), `_process_text_page()` (lines 175-260)
- `config/settings.py` — Pydantic Settings (already has `graph_enabled`)
- `.env.example` — env var documentation
- `tests/test_pipeline_vlm_graphiti.py` — existing Graphiti pipeline tests

---

## Task 1: Add `SEMAPHORE_LIMIT` to settings and `.env.example`

**Files:**
- Modify: `config/settings.py:80` (near `graph_enabled`)
- Modify: `.env.example:154` (near `GRAPH_ENABLED`)

**Step 1: Add setting**

In `config/settings.py`, add after `graph_enabled`:

```python
graph_enabled: bool = True
graph_semaphore_limit: int = 10  # SEMAPHORE_LIMIT for Graphiti LLM concurrency
```

**Step 2: Wire SEMAPHORE_LIMIT env var in graphiti_client.py**

In `ingestion/graphiti_client.py`, in `get_graphiti()`, before creating the Graphiti client, set the env var:

```python
import os
from config.settings import settings

# Set before Graphiti reads it internally
os.environ.setdefault("SEMAPHORE_LIMIT", str(settings.graph_semaphore_limit))
```

Note: `graphiti_client.py` already imports `os` and `settings`. Add the `SEMAPHORE_LIMIT` line inside `get_graphiti()` before `_client = Graphiti(...)`.

**Step 3: Update `.env.example`**

Add after the `GRAPH_ENABLED` line:

```
GRAPH_SEMAPHORE_LIMIT=10               # Concurrent LLM calls in Graphiti (default: 10)
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_settings.py -v -x
```
Expected: PASS (new field has default, no breaking change)

**Step 5: Commit**

```bash
git add config/settings.py ingestion/graphiti_client.py .env.example
git commit -m "feat: add GRAPH_SEMAPHORE_LIMIT setting for Graphiti LLM concurrency"
```

---

## Task 2: Add `ingest_bulk()` method to `GraphitiWriter`

**Files:**
- Modify: `ingestion/graph_writer.py:19-56`
- Create: `tests/test_graph_writer_bulk.py`

**Step 1: Write the failing test**

Create `tests/test_graph_writer_bulk.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
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

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        mock_client.add_episode_bulk.assert_called_once()
        episodes = mock_client.add_episode_bulk.call_args.args[0]
        assert len(episodes) == 3
        # Chunk 1 should use contextualized_text
        assert episodes[1].content == "## Heading\nchunk one raw"
        # Chunk 0 should use raw text
        assert episodes[0].content == "chunk zero"
        # Names should encode source/page/chunk
        assert episodes[0].name == "test.pdf:p1:c0"
        assert episodes[2].name == "test.pdf:p2:c2"
        # source_description should be source_key
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

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        # Bulk failed, so sequential should have been called per chunk
        assert mock_client.add_episode.call_count == 2

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
        mock_client.add_episode_bulk = AsyncMock(
            side_effect=Exception("bulk failed")
        )
        # Second sequential call fails
        mock_client.add_episode = AsyncMock(
            side_effect=[None, Exception("single failed")]
        )

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ):
            writer = GraphitiWriter()
            # Should not raise
            await writer.ingest_bulk(
                chunks=chunks,
                source_key="test.pdf",
            )

        assert mock_client.add_episode.call_count == 2
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_graph_writer_bulk.py -v -x
```
Expected: FAIL with `AttributeError: 'GraphitiWriter' object has no attribute 'ingest_bulk'`

**Step 3: Implement `ingest_bulk()` in `GraphitiWriter`**

In `ingestion/graph_writer.py`, add imports at top:

```python
from datetime import UTC
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from ingestion.file_processor import TextChunk
```

Add method to `GraphitiWriter` class (after `ingest_chunk`, before `close`):

```python
    async def ingest_bulk(
        self,
        chunks: list[TextChunk],
        source_key: str,
        reference_time: datetime | None = None,
    ) -> None:
        """Ingest all chunks via add_episode_bulk with sequential fallback.

        Falls back to individual add_episode calls if bulk ingestion
        fails (known issues with LLM structured output parsing).
        """
        if not chunks:
            return

        ref_time = reference_time or datetime.now(tz=UTC)
        client = await get_graphiti()

        episodes = [
            RawEpisode(
                name=f"{source_key}:p{chunk.page_number}:c{chunk.chunk_index}",
                content=chunk.contextualized_text or chunk.text,
                source=EpisodeType.text,
                source_description=source_key,
                reference_time=ref_time,
            )
            for chunk in chunks
        ]

        try:
            await client.add_episode_bulk(episodes)
            logger.info(
                "Bulk ingested %d episodes for %s",
                len(episodes),
                source_key,
            )
            return
        except Exception:
            logger.warning(
                "Bulk ingestion failed for %s, falling back to sequential",
                source_key,
                exc_info=True,
            )

        # Sequential fallback
        for episode in episodes:
            try:
                await client.add_episode(
                    name=episode.name,
                    episode_body=episode.content,
                    source_description=episode.source_description,
                    reference_time=episode.reference_time,
                )
            except Exception:
                logger.exception(
                    "Sequential fallback failed for episode %s",
                    episode.name,
                )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_graph_writer_bulk.py -v -x
```
Expected: All 4 tests PASS

**Step 5: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v -x --ignore=tests/test_graph_writer.py -k "not integration"
```
Expected: PASS

**Step 6: Commit**

```bash
git add ingestion/graph_writer.py tests/test_graph_writer_bulk.py
git commit -m "feat: add ingest_bulk() to GraphitiWriter with sequential fallback"
```

---

## Task 3: Switch pipeline to use bulk ingestion

**Files:**
- Modify: `ingestion/pipeline.py:175-260` (`_process_text_page`)
- Modify: `ingestion/pipeline.py:469-495` (`_ingest_to_graphiti`)
- Modify: `ingestion/pipeline.py:556-606` (`ingest_file` inner function)
- Create: `tests/test_pipeline_bulk_graphiti.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline_bulk_graphiti.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from ingestion.file_processor import TextChunk


class TestPipelineBulkGraphiti:
    @pytest.mark.asyncio
    async def test_ingest_to_graphiti_calls_bulk(self) -> None:
        """_ingest_to_graphiti uses ingest_bulk instead of per-chunk calls."""
        from ingestion.pipeline import _ingest_to_graphiti

        chunks = [
            TextChunk(text="chunk 0", chunk_index=0, page_number=1),
            TextChunk(text="chunk 1", chunk_index=1, page_number=1),
            TextChunk(text="chunk 2", chunk_index=2, page_number=2),
        ]

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", chunks, mock_writer)

        # Should call ingest_bulk once, not ingest_chunk 3 times
        mock_writer.ingest_bulk.assert_called_once()
        call_kwargs = mock_writer.ingest_bulk.call_args.kwargs
        assert call_kwargs["source_key"] == "doc.pdf"
        assert call_kwargs["chunks"] == chunks
        mock_writer.ingest_chunk.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_to_graphiti_empty_chunks(self) -> None:
        """_ingest_to_graphiti with empty chunks is a no-op."""
        from ingestion.pipeline import _ingest_to_graphiti

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", [], mock_writer)

        mock_writer.ingest_bulk.assert_called_once()
        call_kwargs = mock_writer.ingest_bulk.call_args.kwargs
        assert call_kwargs["chunks"] == []
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_bulk_graphiti.py -v -x
```
Expected: FAIL — `_ingest_to_graphiti` still calls `ingest_chunk` in a loop

**Step 3: Rewrite `_ingest_to_graphiti` to use bulk**

Replace the entire `_ingest_to_graphiti` function in `ingestion/pipeline.py` (lines 469-495):

```python
async def _ingest_to_graphiti(
    source_file: str,
    chunks: list[TextChunk],
    graphiti_writer: GraphitiWriter,
) -> None:
    """Ingest chunks as Graphiti episodes using bulk API.

    Uses add_episode_bulk for speed. GraphitiWriter.ingest_bulk
    handles fallback to sequential on failure.
    """
    ref_time = datetime.now(tz=UTC)
    await graphiti_writer.ingest_bulk(
        chunks=chunks,
        source_key=source_file,
        reference_time=ref_time,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_pipeline_bulk_graphiti.py -v -x
```
Expected: PASS

**Step 5: Run existing Graphiti pipeline tests**

```bash
uv run pytest tests/test_pipeline_vlm_graphiti.py -v -x
```
Expected: PASS (VLM path still uses `ingest_chunk`, not affected)

**Step 6: Run full non-integration test suite**

```bash
uv run pytest tests/ -v -x -k "not integration"
```
Expected: PASS

**Step 7: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline_bulk_graphiti.py
git commit -m "feat: switch pipeline to Graphiti bulk ingestion for faster KG building"
```

---

## Task 4: Collect all document chunks before Graphiti ingestion

**Why:** Currently `_ingest_to_graphiti` is called per-page in `_process_text_page`. For bulk to be effective, we need to collect all chunks across all pages and call bulk once per document.

**Files:**
- Modify: `ingestion/pipeline.py:175-260` (`_process_text_page`) — remove Graphiti call
- Modify: `ingestion/pipeline.py:556-606` (`_process_all_pages` inner function) — collect chunks, call bulk after all pages
- Modify: `tests/test_pipeline_bulk_graphiti.py` — add integration-level test

**Step 1: Write the failing test**

Add to `tests/test_pipeline_bulk_graphiti.py`:

```python
    @pytest.mark.asyncio
    async def test_process_text_page_does_not_call_graphiti(self) -> None:
        """_process_text_page no longer calls Graphiti directly."""
        from ingestion.pipeline import _process_text_page

        mock_writer = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=[[0.1] * 2048])
        mock_qdrant = MagicMock()

        collected: list[TextChunk] = []

        await _process_text_page(
            source_file="test.pdf",
            text="Some text content here.",
            page_number=1,
            mime="application/pdf",
            now="2026-03-05T00:00:00",
            qdrant=mock_qdrant,
            embedder=mock_embedder,
            graphiti_writer=mock_writer,
            chunk_collector=collected,
        )

        # Should NOT call graphiti directly
        mock_writer.ingest_bulk.assert_not_called()
        mock_writer.ingest_chunk.assert_not_called()
        # Chunks should be collected instead
        assert len(collected) > 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_bulk_graphiti.py::TestPipelineBulkGraphiti::test_process_text_page_does_not_call_graphiti -v -x
```
Expected: FAIL — `_process_text_page` doesn't accept `chunk_collector` parameter

**Step 3: Modify `_process_text_page` to collect chunks instead of ingesting**

In `ingestion/pipeline.py`, modify `_process_text_page` signature to add `chunk_collector` parameter:

```python
async def _process_text_page(
    source_file: str,
    text: str,
    page_number: int,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graphiti_writer: GraphitiWriter | None,
    docling_chunks: list[TextChunk] | None = None,
    chunk_collector: list[TextChunk] | None = None,
) -> None:
```

Replace the Graphiti ingestion block at the end of `_process_text_page` (the `if graphiti_writer is not None:` block, around line 259-260):

```python
    # Collect chunks for bulk Graphiti ingestion (called after all pages)
    if chunk_collector is not None:
        chunk_collector.extend(chunks)
```

**Step 4: Update `_build_page_tasks` to pass `chunk_collector` through**

Modify `_build_page_tasks` signature to accept and forward `chunk_collector`:

```python
def _build_page_tasks(
    page,
    source_file: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graphiti_writer: GraphitiWriter | None,
    docling_chunks: list[TextChunk] | None = None,
    chunk_collector: list[TextChunk] | None = None,
) -> _PageTasks:
```

Pass `chunk_collector=chunk_collector` to both `_process_text_page` calls inside (lines 108 and 123).

**Step 5: Update `_process_all_pages` in `ingest_file` to collect and bulk-ingest**

In `ingest_file`, modify the `_process_all_pages` inner function:

```python
            async def _process_all_pages() -> None:
                embedder = create_embedder()
                sem = asyncio.Semaphore(2)
                all_chunks: list[TextChunk] = []

                async def _bounded(coro):
                    async with sem:
                        return await coro

                try:
                    text_tasks: list = []
                    image_tasks: list = []
                    for page in pages:
                        pt = _build_page_tasks(
                            page,
                            filename,
                            mime,
                            now,
                            qdrant,
                            embedder,
                            graphiti_writer,
                            docling_chunks=dl_chunks,
                            chunk_collector=all_chunks if graphiti_writer else None,
                        )
                        text_tasks.extend(pt.text)
                        image_tasks.extend(pt.image)

                    # Text: concurrent (lightweight, small TPM footprint)
                    if text_tasks:
                        await asyncio.gather(*[_bounded(t) for t in text_tasks])

                    # Bulk Graphiti ingestion after all text pages are processed
                    if graphiti_writer and all_chunks:
                        await _ingest_to_graphiti(
                            filename, all_chunks, graphiti_writer,
                        )

                    # Images: sequential (heavy, TPM-sensitive)
                    for task in image_tasks:
                        await task
                finally:
                    if hasattr(embedder, "tokens_used"):
                        logger.info(
                            "Token usage for %s: %.0f estimated tokens",
                            filename,
                            embedder.tokens_used,
                            extra={
                                "file_name": filename,
                                "estimated_tokens": embedder.tokens_used,
                            },
                        )
                    await embedder.close()
```

**Step 6: Run all tests**

```bash
uv run pytest tests/test_pipeline_bulk_graphiti.py tests/test_pipeline_vlm_graphiti.py tests/test_pipeline_chunking.py -v -x
```
Expected: PASS

**Step 7: Lint and type check**

```bash
uv run ruff check ingestion/pipeline.py ingestion/graph_writer.py config/settings.py && uv run ruff format --check ingestion/pipeline.py ingestion/graph_writer.py
```
Expected: PASS (fix any issues)

**Step 8: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline_bulk_graphiti.py
git commit -m "refactor: collect chunks across pages for single bulk Graphiti call"
```

---

## Task 5: Logging and observability

**Files:**
- Modify: `ingestion/graph_writer.py` (add timing logs to `ingest_bulk`)

**Step 1: Add timing to `ingest_bulk`**

Wrap the bulk call with timing in `GraphitiWriter.ingest_bulk()`:

```python
import time

# Inside ingest_bulk, around the try block:
        t0 = time.monotonic()
        try:
            await client.add_episode_bulk(episodes)
            duration_s = round(time.monotonic() - t0, 1)
            logger.info(
                "Bulk ingested %d episodes for %s in %ss",
                len(episodes),
                source_key,
                duration_s,
            )
            return
        except Exception:
            duration_s = round(time.monotonic() - t0, 1)
            logger.warning(
                "Bulk ingestion failed for %s after %ss, "
                "falling back to sequential",
                source_key,
                duration_s,
                exc_info=True,
            )
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_graph_writer_bulk.py -v -x
```
Expected: PASS

**Step 3: Commit**

```bash
git add ingestion/graph_writer.py
git commit -m "feat: add timing logs to bulk Graphiti ingestion"
```

---

## Task 6: Full lint, format, and verification

**Step 1: Lint entire project**

```bash
uv run ruff check . --fix && uv run ruff format .
```

**Step 2: Type check**

```bash
uv run mypy ingestion/graph_writer.py ingestion/pipeline.py config/settings.py
```

**Step 3: Full test suite**

```bash
uv run pytest tests/ -v -k "not integration"
```
Expected: All PASS

**Step 4: Final commit if any fixes**

```bash
git add -u && git commit -m "chore: lint and format after bulk ingestion changes"
```

---

## Summary of changes

| File | Change |
|-|-|
| `config/settings.py` | Add `graph_semaphore_limit: int = 10` |
| `ingestion/graphiti_client.py` | Set `SEMAPHORE_LIMIT` env var from settings |
| `ingestion/graph_writer.py` | Add `ingest_bulk()` with fallback + timing |
| `ingestion/pipeline.py` | Collect chunks across pages, single bulk call |
| `.env.example` | Document `GRAPH_SEMAPHORE_LIMIT` |
| `tests/test_graph_writer_bulk.py` | 4 unit tests for bulk writer |
| `tests/test_pipeline_bulk_graphiti.py` | 3 unit tests for pipeline integration |

## Manual verification (after all tasks)

Run the pipeline with a real document:

```bash
# Fast mode (vector only)
GRAPH_ENABLED=false uv run python -m ingestion.pipeline

# Bulk graph mode (should be much faster than before)
GRAPH_ENABLED=true GRAPH_SEMAPHORE_LIMIT=10 uv run python -m ingestion.pipeline
```

Compare timing against the previous ~10 minute run. Expected: significant speedup from parallelized LLM calls within `add_episode_bulk`.
