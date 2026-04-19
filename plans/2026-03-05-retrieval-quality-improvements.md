# Retrieval Quality & Data Integrity Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix image token estimation, upgrade chunking to Docling HybridChunker, wire local delete propagation, add token consumption logging, add Voyage+multivec config validation, add VLM→Graphiti for visual pages, and dual-embed mixed pages.

**Architecture:** Seven independent improvements applied in dependency order. Tasks 1/4/5 are isolated config/logging changes. Task 2 replaces the naive `semantic_chunk` with Docling's structure-aware HybridChunker. Task 3 generalizes S3 delete handling to work for local sources. Task 6 adds a VLM caption path from visual pages into Graphiti. Task 7 adds dual dense vectors (text + image) for mixed PDF pages.

**Tech Stack:** Python 3.13, Docling (HybridChunker), Jina v4, Qdrant, Graphiti/Neo4j, httpx, pytest-asyncio

---

## Task 1: Fix Image Token Estimation

Jina v4 tiles images into 28×28 patches at ~10 tokens per tile. The current `len(base64_data) / 4` heuristic dramatically underestimates image tokens, causing unexpected 429s.

**Files:**
- Modify: `ingestion/embedders/jina.py:172-185` (`_estimate_tokens`)
- Test: `tests/test_embedder.py`

**Step 1: Write the failing test**

Add to `tests/test_embedder.py`:

```python
class TestEstimateTokens:
    def test_text_token_estimation(self) -> None:
        """Text tokens estimated as len/4."""
        payload = {"input": [{"text": "hello world test string"}]}
        result = JinaV4Embedder._estimate_tokens(payload)
        assert result == len("hello world test string") / 4.0

    def test_image_token_estimation_uses_tile_calculation(self) -> None:
        """Image tokens estimated via 28x28 tile grid, not base64 length."""
        # A 400x300 image: ceil(400/28) * ceil(300/28) * 10
        # = 15 * 11 * 10 = 1650 tokens
        # Create a real 400x300 PNG to get realistic base64
        from PIL import Image
        import io
        import base64

        img = Image.new("RGB", (400, 300), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        payload = {"input": [{"image": data_uri}]}
        result = JinaV4Embedder._estimate_tokens(payload)
        # Should be ~1650, NOT len(b64)/4 which would be much larger
        assert 1000 <= result <= 2500
        # Verify it's NOT using the old base64/4 heuristic
        old_estimate = len(b64) / 4.0
        assert result < old_estimate * 0.5  # tile estimate is much smaller

    def test_image_token_estimation_small_image(self) -> None:
        """Small images still get a reasonable token count."""
        from PIL import Image
        import io
        import base64

        img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        payload = {"input": [{"image": data_uri}]}
        result = JinaV4Embedder._estimate_tokens(payload)
        # ceil(100/28) * ceil(100/28) * 10 = 4 * 4 * 10 = 160
        assert 100 <= result <= 300

    def test_mixed_payload_sums_correctly(self) -> None:
        """Payload with both text and image sums both estimates."""
        import base64

        payload = {
            "input": [
                {"text": "x" * 400},  # 400/4 = 100 tokens
                {"image": "data:image/png;base64," + "A" * 1000},
            ]
        }
        result = JinaV4Embedder._estimate_tokens(payload)
        assert result >= 100  # at minimum the text portion
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py::TestEstimateTokens -v`
Expected: FAIL — current implementation returns `len(b64)/4` for images

**Step 3: Write minimal implementation**

Replace `_estimate_tokens` in `ingestion/embedders/jina.py`:

```python
@staticmethod
def _estimate_tokens(payload: dict) -> float:  # type: ignore[type-arg]
    """Estimate token count from a Jina API payload.

    Text: len(text) / 4 (standard heuristic).
    Images: Jina v4 tiles into 28×28 patches at ~10 tokens/tile.
    Decoded image dimensions are read to compute the tile grid.
    Falls back to a conservative flat estimate if decoding fails.
    """
    total = 0.0
    for item in payload.get("input", []):
        if "text" in item:
            total += len(item["text"]) / 4.0
        elif "image" in item:
            total += _estimate_image_tokens(item["image"])
    return max(total, 1.0)
```

Add a module-level helper above the class (after imports):

```python
import math

def _estimate_image_tokens(data_uri: str) -> float:
    """Estimate Jina v4 token cost for an image.

    Jina v4 uses a Qwen2.5-VL vision encoder that tiles images
    into 28×28 patches with ~10 tokens per tile.
    """
    _TILE_SIZE = 28
    _TOKENS_PER_TILE = 10
    _FALLBACK_TOKENS = 2000  # conservative fallback for 400px images

    comma = data_uri.find(",")
    if comma < 0:
        return _FALLBACK_TOKENS

    b64_data = data_uri[comma + 1 :]
    try:
        import io
        from PIL import Image

        img_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        tiles_w = math.ceil(w / _TILE_SIZE)
        tiles_h = math.ceil(h / _TILE_SIZE)
        return float(tiles_w * tiles_h * _TOKENS_PER_TILE)
    except Exception:
        return _FALLBACK_TOKENS
```

Add `import math` to the top-level imports in jina.py.

**Step 4: Run test to verify it passes**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py::TestEstimateTokens -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py -v`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add ingestion/embedders/jina.py tests/test_embedder.py
git commit -m "fix: use tile-based token estimation for Jina v4 images

len(base64)/4 dramatically underestimates image tokens for Jina v4's
Qwen2.5-VL vision encoder (28x28 tile grid at ~10 tokens/tile).
A 400px image is ~1960 tokens, not ~50k from base64 length."
```

---

## Task 2: Replace Naive Chunking with Docling HybridChunker

The current `semantic_chunk` splits on `\n\n` with a 512-char limit, losing document structure. Docling's `HybridChunker` respects headings, tables, and lists from the Docling document model.

**Important context:** Docling is already a dependency but the HybridChunker only works on Docling `Document` objects (from `DocumentConverter.convert()`). We already run Docling in `_classify_pdf_pages_docling` — we need to reuse that Document for chunking. For non-PDF text files, we fall back to the existing paragraph chunker since there's no Docling document model.

**Files:**
- Modify: `ingestion/file_processor.py` (add `docling_chunk`, modify `_classify_pdf_pages_docling` to return the Document)
- Modify: `ingestion/pipeline.py` (use Docling chunks when available)
- Test: `tests/test_file_processor.py`

**Step 1: Write the failing tests**

Add to `tests/test_file_processor.py`:

```python
from unittest.mock import MagicMock, patch


class TestDoclingChunk:
    def test_returns_text_chunks_with_page_numbers(self) -> None:
        """docling_chunk produces TextChunks from a Docling Document."""
        from ingestion.file_processor import docling_chunk

        # Create a mock Docling Document with chunk output
        mock_doc = MagicMock()
        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "First heading content"
        mock_chunk_1.meta = MagicMock()
        mock_chunk_1.meta.doc_items = [MagicMock()]
        mock_chunk_1.meta.doc_items[0].prov = [MagicMock(page_no=1)]

        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "Table content here"
        mock_chunk_2.meta = MagicMock()
        mock_chunk_2.meta.doc_items = [MagicMock()]
        mock_chunk_2.meta.doc_items[0].prov = [MagicMock(page_no=2)]

        with patch(
            "ingestion.file_processor._get_hybrid_chunker"
        ) as mock_chunker_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = [mock_chunk_1, mock_chunk_2]
            mock_chunker_factory.return_value = mock_chunker

            chunks = docling_chunk(mock_doc)

        assert len(chunks) == 2
        assert chunks[0].text == "First heading content"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[1].text == "Table content here"
        assert chunks[1].page_number == 2
        assert chunks[1].chunk_index == 1

    def test_returns_empty_for_none_document(self) -> None:
        """docling_chunk returns empty list for None input."""
        from ingestion.file_processor import docling_chunk

        assert docling_chunk(None) == []

    def test_falls_back_on_chunker_error(self) -> None:
        """docling_chunk returns empty list if HybridChunker fails."""
        from ingestion.file_processor import docling_chunk

        mock_doc = MagicMock()
        with patch(
            "ingestion.file_processor._get_hybrid_chunker"
        ) as mock_factory:
            mock_factory.side_effect = Exception("chunker init failed")
            result = docling_chunk(mock_doc)

        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_file_processor.py::TestDoclingChunk -v`
Expected: FAIL — `docling_chunk` doesn't exist yet

**Step 3: Implement docling_chunk in file_processor.py**

Add after `semantic_chunk` in `ingestion/file_processor.py`:

```python
_hybrid_chunker = None
_hybrid_chunker_checked = False


def _get_hybrid_chunker() -> object | None:
    """Lazily initialize Docling HybridChunker."""
    global _hybrid_chunker, _hybrid_chunker_checked  # noqa: PLW0603
    if _hybrid_chunker_checked:
        return _hybrid_chunker
    _hybrid_chunker_checked = True
    try:
        from docling.chunking import HybridChunker

        _hybrid_chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=256,
        )
        logger.info("Docling HybridChunker available")
    except (ImportError, Exception):
        logger.info("Docling HybridChunker not available, using paragraph chunker")
    return _hybrid_chunker


def docling_chunk(doc: object | None) -> list[TextChunk]:
    """Chunk a Docling Document using HybridChunker.

    Returns TextChunks with page numbers extracted from chunk provenance.
    Returns empty list if doc is None or chunking fails.
    """
    if doc is None:
        return []

    try:
        chunker = _get_hybrid_chunker()
        if chunker is None:
            return []

        raw_chunks = chunker.chunk(doc)
        result: list[TextChunk] = []
        for idx, chunk in enumerate(raw_chunks):
            page_no = 1
            if (
                hasattr(chunk, "meta")
                and chunk.meta
                and hasattr(chunk.meta, "doc_items")
                and chunk.meta.doc_items
            ):
                first_item = chunk.meta.doc_items[0]
                if hasattr(first_item, "prov") and first_item.prov:
                    page_no = first_item.prov[0].page_no

            result.append(
                TextChunk(text=chunk.text, chunk_index=idx, page_number=page_no)
            )
        return result
    except Exception:
        logger.exception("Docling chunking failed, returning empty")
        return []
```

**Step 4: Modify `_classify_pdf_pages_docling` to also return the Document object**

Change the return type and implementation in `ingestion/file_processor.py`:

```python
def _classify_pdf_pages_docling(
    content: bytes,
) -> tuple[dict[int, bool], dict[int, str], object | None]:
    """Run Docling on entire PDF once. Return per-page visual flags + text + Document.

    Returns (visual_map, text_map, docling_document):
        visual_map: {page_num: True} for pages with figures/tables/formulas
        text_map: {page_num: markdown_text} for scanned pages needing OCR
        docling_document: the Docling Document for downstream chunking (or None)
    """
    converter = _get_docling_converter()
    if converter is None:
        return {}, {}, None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        result = converter.convert(tmp_path)
        doc = result.document
        visual_map: dict[int, bool] = {}
        text_map: dict[int, str] = {}

        for item, _level in doc.iterate_items():
            page_no = _get_item_page(item)
            if page_no is None:
                continue
            if type(item).__name__ in _VISUAL_TYPES:
                visual_map[page_no] = True
            if hasattr(item, "text") and item.text:
                text_map.setdefault(page_no, "")
                text_map[page_no] += item.text + "\n\n"

        return visual_map, text_map, doc
    except Exception:
        logger.exception("Docling PDF classification failed")
        return {}, {}, None
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
```

Update `_pdf_to_pages` to unpack the third return value:

```python
def _pdf_to_pages(content: bytes) -> list[Page]:
    # Change this line:
    visual_map, docling_text_map, _docling_doc = _classify_pdf_pages_docling(content)
    # ... rest unchanged
```

Wait — we also need the pipeline to access the docling document for chunking. But `file_to_pages` doesn't return it. We need a way to pass it through. The cleanest approach: add a module-level cache that `_pdf_to_pages` sets and the pipeline can read.

Actually, simpler: add a `docling_document` field to the return from `file_to_pages` by creating a small result dataclass.

Add to `ingestion/file_processor.py`:

```python
@dataclass
class FileProcessingResult:
    pages: list[Page]
    docling_document: object | None = None
```

Change `file_to_pages` to return `FileProcessingResult`:

```python
def file_to_pages(filename: str, content: bytes) -> FileProcessingResult:
    """MIME-classify file and convert to Pages + optional Docling Document."""
    mime_type, _ = mimetypes.guess_type(filename)
    ext = _get_extension(filename)

    if mime_type == "application/pdf" or ext == ".pdf":
        return _pdf_to_pages(content)
    if ext in _IMAGE_EXTENSIONS or (mime_type and mime_type.startswith("image/")):
        return FileProcessingResult(
            pages=[Page(image_bytes=content, text="", page_number=1, content_type="image")]
        )
    if ext in _TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        return FileProcessingResult(
            pages=[
                Page(
                    image_bytes=b"",
                    text=content.decode("utf-8"),
                    page_number=1,
                    content_type="text",
                )
            ]
        )

    logger.warning("Unknown file type for %s (mime=%s), skipping.", filename, mime_type)
    return FileProcessingResult(pages=[])
```

And `_pdf_to_pages` returns `FileProcessingResult`:

```python
def _pdf_to_pages(content: bytes) -> FileProcessingResult:
    visual_map, docling_text_map, docling_doc = _classify_pdf_pages_docling(content)
    # ... existing page building code ...
    return FileProcessingResult(pages=pages, docling_document=docling_doc)
```

**Step 5: Update pipeline.py to use Docling chunks when available**

In `ingestion/pipeline.py`, update the import and `ingest_file`:

```python
from ingestion.file_processor import (
    FileProcessingResult,
    TextChunk,
    docling_chunk,
    file_to_pages,
    resize_image_for_embedding,
    semantic_chunk,
)
```

In `ingest_file`, change:

```python
    try:
        result = file_to_pages(filename, content)
        pages = result.pages
    except Exception:
        # ... existing error handling
```

Pass `result.docling_document` to `_process_text_page` so it can use Docling chunks for PDF pages. Add a `docling_document` parameter to `_build_page_tasks` and `_process_text_page`:

In `_process_text_page`, change the chunking logic:

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
) -> None:
    """Process a text page: chunk, embed, store dense, ingest to Graphiti."""
    # Use pre-computed Docling chunks for this page if available
    if docling_chunks:
        chunks = [c for c in docling_chunks if c.page_number == page_number]
    else:
        chunks = semantic_chunk(text, page_number=page_number)

    if not chunks:
        return
    # ... rest unchanged
```

In `ingest_file`, compute Docling chunks once for the whole document:

```python
    # After file_to_pages, before _process_all_pages:
    dl_chunks = docling_chunk(result.docling_document) if result.docling_document else None
```

Pass `dl_chunks` through `_build_page_tasks` → `_process_text_page`.

**Step 6: Update existing tests for new return type**

In `tests/test_file_processor.py`, update all `file_to_pages` tests to use `.pages`:

```python
class TestFileToPages:
    def test_png_returns_single_image_page(self) -> None:
        content = (FIXTURES / "sample.png").read_bytes()
        result = file_to_pages("photo.png", content)
        pages = result.pages

        assert len(pages) == 1
        # ... rest same but using pages variable
```

Do this for every test in `TestFileToPages`.

**Step 7: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_file_processor.py tests/test_embedder.py -v`
Expected: All pass

**Step 8: Commit**

```bash
git add ingestion/file_processor.py ingestion/pipeline.py tests/test_file_processor.py
git commit -m "feat: replace naive paragraph chunking with Docling HybridChunker

Docling HybridChunker respects document structure (headings, tables, lists)
instead of splitting on paragraph boundaries. Falls back to semantic_chunk
for non-PDF files or when Docling is unavailable."
```

---

## Task 3: Wire Local Delete Propagation

`handle_s3_delete` already handles Qdrant + Graphiti cleanup but is only called for S3 sources. Local file deletions tracked by CocoIndex don't propagate.

**Files:**
- Modify: `ingestion/pipeline.py` (rename `handle_s3_delete` → `handle_file_delete`, wire into local source flow)
- Test: `tests/test_pipeline_delete.py` (new)

**Step 1: Write the failing test**

Create `tests/test_pipeline_delete.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.pipeline import handle_file_delete


class TestHandleFileDelete:
    def test_deletes_dense_points_by_source_file(self) -> None:
        """Qdrant dense collection points are deleted by source_file filter."""
        mock_qdrant = MagicMock()
        with patch("ingestion.pipeline._get_qdrant_client", return_value=mock_qdrant):
            with patch("ingestion.pipeline.settings") as mock_settings:
                mock_settings.multivec_enabled = False
                handle_file_delete("docs/report.pdf")

        mock_qdrant.delete.assert_called_once()
        call_args = mock_qdrant.delete.call_args
        assert call_args.kwargs["collection_name"] == "documents_dense"

    def test_deletes_multivec_when_enabled(self) -> None:
        """Qdrant multivec collection is also cleaned when enabled."""
        mock_qdrant = MagicMock()
        with patch("ingestion.pipeline._get_qdrant_client", return_value=mock_qdrant):
            with patch("ingestion.pipeline.settings") as mock_settings:
                mock_settings.multivec_enabled = True
                handle_file_delete("docs/report.pdf")

        assert mock_qdrant.delete.call_count == 2

    def test_invalidates_graphiti_edges(self) -> None:
        """Graphiti edges matching source are expired."""
        mock_qdrant = MagicMock()
        mock_edge = MagicMock()
        mock_edge.source_description = "docs/report.pdf"
        mock_edge.expired_at = None

        mock_graphiti = AsyncMock()
        mock_graphiti.search.return_value = [mock_edge]

        with (
            patch("ingestion.pipeline._get_qdrant_client", return_value=mock_qdrant),
            patch("ingestion.pipeline.settings") as mock_settings,
            patch("ingestion.pipeline.run_async") as mock_run_async,
        ):
            mock_settings.multivec_enabled = False
            handle_file_delete("docs/report.pdf")

        # Verify run_async was called for graph invalidation
        assert mock_run_async.called
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_delete.py -v`
Expected: FAIL — `handle_file_delete` doesn't exist (only `handle_s3_delete`)

**Step 3: Rename and generalize**

In `ingestion/pipeline.py`:
- Rename `handle_s3_delete` → `handle_file_delete`
- Keep `handle_s3_delete = handle_file_delete` as a backward-compat alias
- Update all log messages from "S3 delete" to "file delete"

```python
def handle_file_delete(source_key: str) -> None:
    """Handle file deletion: remove Qdrant points and invalidate graph.

    Works for both S3 and local file sources. Deletes all Qdrant points
    matching the source_file and invalidates related Graphiti episodes.
    """
    # ... existing body with "S3" replaced by "file" in log messages


# Backward-compatible alias
handle_s3_delete = handle_file_delete
```

**Step 4: Run test to verify it passes**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_delete.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest -v`
Expected: All pass (alias preserves backward compat)

**Step 6: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline_delete.py
git commit -m "feat: generalize delete propagation for local and S3 sources

Rename handle_s3_delete -> handle_file_delete (with alias).
Ensures Qdrant points and Graphiti edges are cleaned up regardless
of document source type."
```

**Note:** Wiring CocoIndex's delete events to call `handle_file_delete` requires CocoIndex callback support. If CocoIndex doesn't expose delete callbacks, the pragmatic fallback is a periodic reconciliation job or manual cleanup CLI. Document this limitation in the commit message or a follow-up issue.

---

## Task 4: Add Token Consumption Logging

Track total tokens consumed (text + image) per `ingest_file` call for cost forecasting.

**Files:**
- Modify: `ingestion/embedders/jina.py` (add token counter)
- Modify: `ingestion/pipeline.py` (log token count after file processing)
- Test: `tests/test_embedder.py`

**Step 1: Write the failing test**

Add to `tests/test_embedder.py`:

```python
class TestTokenCounter:
    async def test_tracks_estimated_tokens(self, embedder: JinaV4Embedder) -> None:
        """Token counter accumulates across calls."""
        mock_resp = _mock_response([{"embedding": [0.1] * DENSE_DIM}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            embedder.reset_token_counter()
            await embedder.embed_text(["hello world"])
            tokens_after_text = embedder.tokens_used

            await embedder.embed_text(["another query"])
            tokens_after_two = embedder.tokens_used

        assert tokens_after_text > 0
        assert tokens_after_two > tokens_after_text

    async def test_reset_clears_counter(self, embedder: JinaV4Embedder) -> None:
        """reset_token_counter zeroes the accumulator."""
        mock_resp = _mock_response([{"embedding": [0.1] * DENSE_DIM}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            await embedder.embed_text(["test"])
            embedder.reset_token_counter()

        assert embedder.tokens_used == 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py::TestTokenCounter -v`
Expected: FAIL — `tokens_used` and `reset_token_counter` don't exist

**Step 3: Implement token counter**

In `ingestion/embedders/jina.py`, add to `__init__`:

```python
self._tokens_used: float = 0.0
```

Add properties and method:

```python
@property
def tokens_used(self) -> float:
    """Total estimated tokens consumed since last reset."""
    return self._tokens_used

def reset_token_counter(self) -> None:
    """Reset the token consumption counter to zero."""
    self._tokens_used = 0.0
```

In `_request`, after `_estimate_tokens`:

```python
self._tokens_used += estimated_tokens
```

**Step 4: Add to Embedder protocol**

In `ingestion/embedder.py`, add to the `Embedder` Protocol:

```python
@property
def tokens_used(self) -> float: ...
def reset_token_counter(self) -> None: ...
```

Also add matching stubs to `VoyageEmbedder` in `ingestion/embedders/voyage.py`.

**Step 5: Log tokens in pipeline**

In `ingestion/pipeline.py`, in `_process_all_pages`, after the try/finally block but before `await embedder.close()`:

```python
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
```

**Step 6: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py -v`
Expected: All pass

**Step 7: Commit**

```bash
git add ingestion/embedder.py ingestion/embedders/jina.py ingestion/embedders/voyage.py ingestion/pipeline.py tests/test_embedder.py
git commit -m "feat: add per-file token consumption logging

Tracks estimated tokens (text + image) per ingest_file call.
Logged as structured extra for cost forecasting on Jina free tier."
```

---

## Task 5: Add Voyage + Multivec Config Validation

Fail fast at startup if `embedding_provider=voyage` and `multivec_enabled=True`.

**Files:**
- Modify: `config/settings.py` (add model validator)
- Test: `tests/test_settings.py` (new)

**Step 1: Write the failing test**

Create `tests/test_settings.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestSettingsValidation:
    def test_voyage_with_multivec_raises(self) -> None:
        """Voyage + multivec_enabled=True is an invalid combination."""
        from config.settings import Settings

        with pytest.raises(ValidationError, match="[Vv]oyage.*ColBERT|multivec"):
            Settings(
                jina_api_key="test",
                neo4j_password="test",
                embedding_provider="voyage",
                voyage_api_key="test",
                multivec_enabled=True,
                _env_file=None,
            )

    def test_jina_with_multivec_is_valid(self) -> None:
        """Jina + multivec_enabled=True is fine."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="test",
            neo4j_password="test",
            embedding_provider="jina",
            multivec_enabled=True,
            _env_file=None,
        )
        assert s.multivec_enabled is True

    def test_voyage_without_multivec_is_valid(self) -> None:
        """Voyage without multivec is fine."""
        from config.settings import Settings

        s = Settings(
            jina_api_key="test",
            neo4j_password="test",
            embedding_provider="voyage",
            voyage_api_key="test",
            multivec_enabled=False,
            _env_file=None,
        )
        assert s.embedding_provider == "voyage"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_settings.py -v`
Expected: FAIL — no validator exists, voyage+multivec doesn't raise

**Step 3: Add validator to Settings**

In `config/settings.py`, add import and validator:

```python
from pydantic import model_validator
```

Inside the `Settings` class, after all field declarations:

```python
    @model_validator(mode="after")
    def _validate_provider_features(self) -> Settings:
        if self.embedding_provider == "voyage" and self.multivec_enabled:
            msg = (
                "Voyage does not support ColBERT multi-vector embeddings. "
                "Set multivec_enabled=False or use embedding_provider=jina."
            )
            raise ValueError(msg)
        return self
```

**Step 4: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_settings.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest -v`
Expected: All pass (existing config uses jina+multivec_enabled=False)

**Step 6: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: validate Voyage + multivec_enabled is unsupported

Fail fast at startup with a clear error message instead of silently
breaking visual_search at runtime."
```

---

## Task 6: VLM Caption → Graphiti for Visual Pages

Visual pages currently contribute nothing to the knowledge graph. Use the existing VLM infrastructure to generate text descriptions of visual pages, then feed those to Graphiti for entity extraction.

**Files:**
- Modify: `ingestion/pipeline.py` (add VLM captioning step in `_process_visual_page`)
- Test: `tests/test_pipeline_vlm_graphiti.py` (new)

**Step 1: Write the failing test**

Create `tests/test_pipeline_vlm_graphiti.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVlmGraphitiIngestion:
    async def test_visual_page_caption_sent_to_graphiti(self) -> None:
        """When VLM is enabled, visual page captions are ingested to Graphiti."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_graphiti_writer = AsyncMock()
        mock_vlm_caption = AsyncMock(return_value="Chart showing Q3 revenue of $5M")

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graphiti_writer=mock_graphiti_writer,
            )

        mock_vlm_caption.assert_called_once_with(b"fake-png")
        mock_graphiti_writer.ingest_chunk.assert_called_once()
        call_kwargs = mock_graphiti_writer.ingest_chunk.call_args.kwargs
        assert "Q3 revenue" in call_kwargs["chunk_text"]
        assert call_kwargs["source_key"] == "report.pdf"
        assert call_kwargs["page_number"] == 3

    async def test_skipped_when_caption_is_empty(self) -> None:
        """No Graphiti ingestion when VLM returns empty caption."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_graphiti_writer = AsyncMock()
        mock_vlm_caption = AsyncMock(return_value="")

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graphiti_writer=mock_graphiti_writer,
            )

        mock_graphiti_writer.ingest_chunk.assert_not_called()

    async def test_skipped_when_vlm_fails(self) -> None:
        """Graceful fallback when VLM captioning fails."""
        from ingestion.pipeline import _caption_and_ingest_visual

        mock_graphiti_writer = AsyncMock()
        mock_vlm_caption = AsyncMock(side_effect=Exception("VLM timeout"))

        with patch(
            "ingestion.pipeline._caption_visual_page",
            mock_vlm_caption,
        ):
            # Should not raise
            await _caption_and_ingest_visual(
                source_file="report.pdf",
                image_bytes=b"fake-png",
                page_number=3,
                graphiti_writer=mock_graphiti_writer,
            )

        mock_graphiti_writer.ingest_chunk.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_vlm_graphiti.py -v`
Expected: FAIL — `_caption_and_ingest_visual` and `_caption_visual_page` don't exist

**Step 3: Implement**

Add to `ingestion/pipeline.py`:

```python
async def _caption_visual_page(image_bytes: bytes) -> str:
    """Generate a text description of a visual page using VLM.

    Uses the same VLM infrastructure as visual_search's vlm_generator,
    but with a structured extraction prompt instead of a query-answer prompt.
    """
    import base64 as _b64

    provider = settings.llm_api_type.lower()
    _b64_str = _b64.b64encode(image_bytes).decode()

    prompt = (
        "Describe the content of this document page in detail. "
        "Extract all entities (people, organizations, products, dates, numbers), "
        "relationships, and key facts. Be factual and concise."
    )

    if provider == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        resp = await client.messages.create(
            model=settings.llm_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _b64_str,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return resp.content[0].text
    else:
        import openai

        client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_b64_str}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""


async def _caption_and_ingest_visual(
    source_file: str,
    image_bytes: bytes,
    page_number: int,
    graphiti_writer: GraphitiWriter,
) -> None:
    """Caption a visual page and ingest the text to Graphiti."""
    try:
        caption = await _caption_visual_page(image_bytes)
        if not caption or not caption.strip():
            return

        await graphiti_writer.ingest_chunk(
            chunk_text=caption,
            source_key=source_file,
            page_number=page_number,
            chunk_index=0,
        )
        logger.info(
            "VLM caption ingested for %s page %d (%d chars)",
            source_file,
            page_number,
            len(caption),
        )
    except Exception:
        logger.exception(
            "VLM caption/ingestion failed for %s page %d",
            source_file,
            page_number,
        )
```

Wire it into `_process_visual_page` — add at the end, after the multivec block:

```python
    # VLM caption → Graphiti (when enabled)
    if settings.vlm_generation_enabled and graphiti_writer is not None:
        await _caption_and_ingest_visual(
            source_file, image_bytes, page_number, graphiti_writer,
        )
```

This requires passing `graphiti_writer` to `_process_visual_page`. Update its signature and all call sites in `_build_page_tasks`.

**Step 4: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_vlm_graphiti.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest -v`
Expected: All pass

**Step 6: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline_vlm_graphiti.py
git commit -m "feat: VLM caption visual pages into Graphiti knowledge graph

When vlm_generation_enabled=True, visual pages are captioned by the VLM
and the resulting text is ingested as Graphiti episodes for entity extraction.
This makes chart/diagram entities visible to graph_search."
```

---

## Task 7: Dual-Embed Mixed PDF Pages

Mixed pages (text + visual content) currently get image-embedded but the text doesn't get a separate dense vector. Add dual embedding with a modality suffix on the UUID5 key.

**Files:**
- Modify: `ingestion/pipeline.py` (adjust `_build_page_tasks` for mixed pages)
- Test: `tests/test_pipeline_dual_embed.py` (new)

**Step 1: Write the failing test**

Create `tests/test_pipeline_dual_embed.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import Page
from ingestion.pipeline import _build_page_tasks


class TestDualEmbedMixedPages:
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
            mock_settings.image_embed_strategy = "smart"
            mock_settings.multivec_enabled = False
            mock_settings.vlm_generation_enabled = False
            mock_settings.image_embed_max_px = 400

            tasks = _build_page_tasks(
                page,
                source_file="report.pdf",
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=MagicMock(),
                embedder=MagicMock(),
                graphiti_writer=None,
            )

        # Should have BOTH text tasks (chunking+embedding) AND image tasks
        assert len(tasks.text) >= 1, "Mixed page should have text tasks"
        assert len(tasks.image) >= 1, "Mixed page should have image tasks"

    def test_text_only_pdf_page_no_image_task(self) -> None:
        """A PDF page with text but NO visual content only gets text tasks."""
        page = Page(
            image_bytes=b"fake-png",
            text="Plain text paragraph.",
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
                graphiti_writer=None,
            )

        assert len(tasks.text) >= 1
        assert len(tasks.image) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_dual_embed.py -v`
Expected: FAIL on `test_mixed_pdf_page_gets_both_text_and_image_tasks` — currently mixed pages only get image tasks, not both

**Step 3: Modify `_build_page_tasks` for dual embedding**

In `ingestion/pipeline.py`, change the `elif page.content_type == "pdf"` block:

```python
    elif page.content_type == "pdf":
        # Text embedding (always, if text exists)
        if page.text.strip():
            tasks.text.append(
                _process_text_page(
                    source_file,
                    page.text,
                    page.page_number,
                    mime,
                    now,
                    qdrant,
                    embedder,
                    graphiti_writer,
                )
            )

        # Image embedding (based on strategy + visual content flag)
        should_embed_image = (
            settings.image_embed_strategy == "all"
            or (settings.image_embed_strategy == "smart" and page.has_visual_content)
        )

        if should_embed_image:
            tasks.image.append(
                _process_visual_page(
                    source_file,
                    page.image_bytes,
                    page.page_number,
                    "pdf_page",
                    mime,
                    now,
                    qdrant,
                    embedder,
                )
            )
        elif page.image_bytes:
            tasks.text.append(
                _store_page_thumbnail(
                    source_file,
                    page.image_bytes,
                    page.page_number,
                    qdrant,
                )
            )
```

The key change: remove the `if page.text.strip()` / `elif` conditional that made text and image mutually exclusive. Now both paths run independently.

**Important:** For mixed pages, the text chunks and image will have different point IDs in Qdrant because text uses `_make_chunk_id` (includes chunk_index) while images use `_make_point_id(page_key)`. No UUID5 collision.

**Step 4: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_dual_embed.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest -v`
Expected: All pass

**Step 6: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline_dual_embed.py
git commit -m "feat: dual-embed mixed PDF pages (text + image)

Mixed pages with both text and visual content now get both text chunk
embeddings AND image embeddings in documents_dense. Previously the
text embedding was skipped for visual pages."
```

---

## Summary

| Task | Files Changed | Risk | Est. Complexity |
|-|-|-|-|
| 1. Image token estimation | jina.py, test_embedder.py | Low | Small |
| 2. Docling HybridChunker | file_processor.py, pipeline.py, test_file_processor.py | Medium | Large |
| 3. Delete propagation | pipeline.py, test_pipeline_delete.py | Low | Small |
| 4. Token logging | jina.py, voyage.py, embedder.py, pipeline.py, test_embedder.py | Low | Small |
| 5. Voyage+multivec validation | settings.py, test_settings.py | Low | Small |
| 6. VLM→Graphiti | pipeline.py, test_pipeline_vlm_graphiti.py | Medium | Medium |
| 7. Dual-embed mixed pages | pipeline.py, test_pipeline_dual_embed.py | Low | Small |

**Dependencies:** Tasks 1-5 are independent. Task 6 depends on no other task but should come after Task 7 is understood. Task 7 is independent.

**Recommended execution:** Tasks 1, 3, 4, 5 can be parallelized (they touch different files/functions). Task 2 is the largest and should get focused attention. Tasks 6 and 7 can also be parallelized.
