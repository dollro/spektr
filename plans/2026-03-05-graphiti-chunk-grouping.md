# Graphiti Episode Grouping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Group small text chunks into larger episodes (~1500 chars) before Graphiti ingestion to reduce LLM calls by ~5x while keeping vector search chunks unchanged.

**Architecture:** Vector embeddings continue using fine-grained chunks (Docling ~200-500 chars or semantic ~512 chars) for retrieval precision. Before sending to `GraphitiWriter.ingest_bulk()`, we group adjacent chunks from the same page into larger episodes (~1500 chars target). This reduces 74 episodes → ~15 episodes for a 25-page PDF, cutting ~296 LLM calls to ~60. The grouping happens in `_ingest_to_graphiti()` — a pure data transformation with no side effects on embedding or Qdrant paths. A `GRAPH_EPISODE_TARGET_SIZE` setting controls the target character count.

**Tech Stack:** Python dataclasses, existing `TextChunk`, Pydantic Settings

---

## Context for Implementer

### Why chunk grouping works

Graphiti's entity extraction LLM calls work *better* with more context per episode. Neo4j's own KG builder groups smaller chunks before extraction. We're not losing quality — we're improving it while also cutting cost and time.

### Key constraint

The grouping must preserve `contextualized_text` (heading-prefixed text from Docling). When chunks are grouped, concatenate `contextualized_text` (falling back to `text`) with `\n\n` separators. The grouped episode name uses the first chunk's page/index as anchor.

### Important files

- `ingestion/graph_writer.py:56-117` — `GraphitiWriter.ingest_bulk()` (receives chunks, builds RawEpisodes)
- `ingestion/pipeline.py:469-484` — `_ingest_to_graphiti()` (calls ingest_bulk)
- `ingestion/file_processor.py:26-31` — `TextChunk` dataclass
- `config/settings.py:83-84` — graph settings
- `.env.example:156` — near `GRAPH_SEMAPHORE_LIMIT`
- `tests/test_graph_writer_bulk.py` — existing bulk writer tests
- `tests/test_pipeline_bulk_graphiti.py` — existing pipeline bulk tests

---

## Task 1: Add `GRAPH_EPISODE_TARGET_SIZE` setting

**Files:**
- Modify: `config/settings.py:84` (after `graph_semaphore_limit`)
- Modify: `.env.example:157` (after `GRAPH_SEMAPHORE_LIMIT`)

**Step 1: Add setting**

In `config/settings.py`, add after `graph_semaphore_limit`:

```python
graph_episode_target_size: int = 1500  # Target chars per Graphiti episode
```

**Step 2: Update `.env.example`**

Add after the `GRAPH_SEMAPHORE_LIMIT` line:

```
GRAPH_EPISODE_TARGET_SIZE=1500          # Target chars per Graphiti episode (groups small chunks)
```

**Step 3: Run tests**

```bash
uv run pytest tests/test_settings.py -v -x
```
Expected: PASS (new field has default, no breaking change)

**Step 4: Commit**

```bash
git add config/settings.py .env.example
git commit -m "feat: add GRAPH_EPISODE_TARGET_SIZE setting for chunk grouping"
```

---

## Task 2: Add `group_chunks_for_graph()` function

**Files:**
- Modify: `ingestion/graph_writer.py:18` (add function after imports, before class)
- Create: `tests/test_graph_writer_grouping.py`

**Step 1: Write the failing tests**

Create `tests/test_graph_writer_grouping.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_graph_writer_grouping.py -v -x
```
Expected: FAIL with `ImportError: cannot import name 'group_chunks_for_graph'`

**Step 3: Implement `group_chunks_for_graph()`**

In `ingestion/graph_writer.py`, add after the imports (line 21, before the class):

```python
def group_chunks_for_graph(
    chunks: list[TextChunk],
    target_size: int = 1500,
) -> list[TextChunk]:
    """Group adjacent same-page chunks into larger episodes for Graphiti.

    Keeps vector search chunks untouched — this only affects what
    gets sent to the knowledge graph. Larger episodes give the LLM
    more context for entity extraction and reduce total LLM calls.
    """
    if not chunks:
        return []

    grouped: list[TextChunk] = []
    current_texts: list[str] = []
    current_len = 0
    anchor = chunks[0]

    for chunk in chunks:
        text = chunk.contextualized_text or chunk.text

        # Page boundary → flush
        if chunk.page_number != anchor.page_number:
            grouped.append(
                TextChunk(
                    text="\n\n".join(current_texts),
                    chunk_index=anchor.chunk_index,
                    page_number=anchor.page_number,
                )
            )
            current_texts = [text]
            current_len = len(text)
            anchor = chunk
        # Would exceed target → flush and start new group
        elif current_texts and current_len + len(text) + 2 > target_size:
            grouped.append(
                TextChunk(
                    text="\n\n".join(current_texts),
                    chunk_index=anchor.chunk_index,
                    page_number=anchor.page_number,
                )
            )
            current_texts = [text]
            current_len = len(text)
            anchor = chunk
        else:
            current_texts.append(text)
            current_len += len(text) + 2  # +2 for "\n\n" separator

    # Flush remaining
    if current_texts:
        grouped.append(
            TextChunk(
                text="\n\n".join(current_texts),
                chunk_index=anchor.chunk_index,
                page_number=anchor.page_number,
            )
        )

    return grouped
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_graph_writer_grouping.py -v -x
```
Expected: All 6 tests PASS

**Step 5: Run existing tests for regressions**

```bash
uv run pytest tests/test_graph_writer_bulk.py -v -x
```
Expected: PASS

**Step 6: Commit**

```bash
git add ingestion/graph_writer.py tests/test_graph_writer_grouping.py
git commit -m "feat: add group_chunks_for_graph() for KG episode grouping"
```

---

## Task 3: Wire grouping into `ingest_bulk()`

**Files:**
- Modify: `ingestion/graph_writer.py:56-82` (`ingest_bulk` method)
- Modify: `tests/test_graph_writer_bulk.py` (update test expectations)

**Step 1: Write the failing test**

Add to `tests/test_graph_writer_bulk.py` (new test in existing class):

```python
    @pytest.mark.asyncio
    async def test_ingest_bulk_groups_chunks(self) -> None:
        """ingest_bulk groups small chunks into fewer episodes."""
        from ingestion.graph_writer import GraphitiWriter

        # 10 small chunks (100 chars each) → should group into fewer episodes
        chunks = [
            TextChunk(text=f"chunk {i} " * 10, chunk_index=i, page_number=1)
            for i in range(10)
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(return_value=MagicMock())

        with patch(
            "ingestion.graph_writer.get_graphiti",
            return_value=mock_client,
        ), patch(
            "ingestion.graph_writer.settings",
        ) as mock_settings:
            mock_settings.graph_episode_target_size = 1500
            writer = GraphitiWriter()
            await writer.ingest_bulk(chunks=chunks, source_key="test.pdf")

        mock_client.add_episode_bulk.assert_called_once()
        episodes = mock_client.add_episode_bulk.call_args.args[0]
        # 10 chunks of ~80 chars → should be grouped (fewer than 10 episodes)
        assert len(episodes) < 10
        assert len(episodes) >= 1
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_graph_writer_bulk.py::TestGraphitiWriterBulk::test_ingest_bulk_groups_chunks -v -x
```
Expected: FAIL — `ingest_bulk` doesn't call `group_chunks_for_graph` yet

**Step 3: Wire grouping into `ingest_bulk`**

In `ingestion/graph_writer.py`, modify `ingest_bulk()`. Add import of settings at top of file (after existing imports):

```python
from config.settings import settings
```

Note: `settings` is already imported lower in the file (line 142) for the legacy writer. Move or add this import near the top-level imports (line 19 area). Check that the existing `from config.settings import settings  # noqa: E402` on line 142 doesn't conflict — it won't, since it's the same import.

Then in `ingest_bulk()`, after the early return for empty chunks and before building episodes, add the grouping call:

Replace lines 67-82 of `ingest_bulk` (from `if not chunks` through the episode list comprehension):

```python
        if not chunks:
            return

        ref_time = reference_time or datetime.now(tz=UTC)
        client = await get_graphiti()

        grouped = group_chunks_for_graph(
            chunks, target_size=settings.graph_episode_target_size,
        )

        episodes = [
            RawEpisode(
                name=f"{source_key}:p{chunk.page_number}:c{chunk.chunk_index}",
                content=chunk.text,
                source=EpisodeType.text,
                source_description=source_key,
                reference_time=ref_time,
            )
            for chunk in grouped
        ]
```

Note: The episode `content` changes from `chunk.contextualized_text or chunk.text` to just `chunk.text` because `group_chunks_for_graph()` already resolves contextualized_text during grouping.

**Step 4: Run all bulk writer tests**

```bash
uv run pytest tests/test_graph_writer_bulk.py tests/test_graph_writer_grouping.py -v -x
```
Expected: PASS

Note: The existing `test_ingest_bulk_builds_raw_episodes` test uses 3 chunks across 2 pages with short text — they'll be grouped into 2 episodes (one per page) instead of 3. Update that test's assertion:

In `tests/test_graph_writer_bulk.py`, in `test_ingest_bulk_builds_raw_episodes`, the assertion `assert len(episodes) == 3` will likely need to change to `assert len(episodes) == 2` because chunk 0 and chunk 1 (both page 1, short text) will be grouped. Also update the content assertions:
- `episodes[0].content` will contain both "chunk zero" and the contextualized "## Heading\nchunk one raw"
- `episodes[1].content` will be "chunk two"
- `episodes[0].name` will be "test.pdf:p1:c0" (anchor of first group)
- `episodes[1].name` will be "test.pdf:p2:c2"

Adjust the test accordingly:

```python
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
```

**Step 5: Run full non-integration test suite**

```bash
uv run pytest tests/ -v -x -k "not integration"
```
Expected: PASS

**Step 6: Commit**

```bash
git add ingestion/graph_writer.py tests/test_graph_writer_bulk.py
git commit -m "feat: wire chunk grouping into ingest_bulk for fewer LLM calls"
```

---

## Task 4: Add logging for grouping stats

**Files:**
- Modify: `ingestion/graph_writer.py` (add log line in `ingest_bulk`)

**Step 1: Add log line**

In `ingest_bulk()`, after the `group_chunks_for_graph` call and before building episodes, add:

```python
        if len(grouped) < len(chunks):
            logger.info(
                "Grouped %d chunks into %d episodes for %s (target: %d chars)",
                len(chunks),
                len(grouped),
                source_key,
                settings.graph_episode_target_size,
            )
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_graph_writer_bulk.py tests/test_graph_writer_grouping.py -v -x
```
Expected: PASS

**Step 3: Commit**

```bash
git add ingestion/graph_writer.py
git commit -m "feat: log chunk grouping stats before Graphiti ingestion"
```

---

## Task 5: Lint, format, and full verification

**Step 1: Lint and format**

```bash
uv run ruff check ingestion/graph_writer.py config/settings.py --fix && uv run ruff format ingestion/graph_writer.py config/settings.py tests/test_graph_writer_grouping.py tests/test_graph_writer_bulk.py
```

**Step 2: Full test suite**

```bash
uv run pytest tests/ -v -x -k "not integration"
```
Expected: All PASS

**Step 3: Commit if any fixes**

```bash
git add -u && git commit -m "chore: lint and format after chunk grouping changes"
```

---

## Summary of changes

| File | Change |
|-|-|
| `config/settings.py` | Add `graph_episode_target_size: int = 1500` |
| `.env.example` | Document `GRAPH_EPISODE_TARGET_SIZE` |
| `ingestion/graph_writer.py` | Add `group_chunks_for_graph()`, wire into `ingest_bulk()` |
| `tests/test_graph_writer_grouping.py` | 6 unit tests for grouping logic |
| `tests/test_graph_writer_bulk.py` | Update existing + add grouping integration test |

## Expected performance impact

| Metric | Before | After |
|-|-|-|
| Episodes for test.pdf | 74 | ~15 |
| Est. LLM calls | ~296 | ~60 |
| Est. ingestion time | ~28 min | ~5-6 min |

## Manual verification

```bash
# Clear previous graph data
# In Neo4j browser: MATCH (n) DETACH DELETE n

# Clear CocoIndex state
docker compose exec postgres psql -U cocoindex -d cocoindex -c "DROP SCHEMA IF EXISTS cocoindex CASCADE;"

# Run with grouping
GRAPH_ENABLED=true GRAPH_SEMAPHORE_LIMIT=10 LOG_FORMAT=text LOG_LEVEL=INFO uv run python -m ingestion.pipeline
```

Watch for log line: `Grouped 74 chunks into ~15 episodes for test.pdf (target: 1500 chars)`
