# Hybrid PDF Processing + Jina Rate Limiting — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make PDF ingestion fast by adding text extraction (PyMuPDF + Docling fallback), parallel page processing, and configurable Jina rate limiting.

**Architecture:** PDFs get dual processing — PyMuPDF extracts native text (fast path to text embeddings + Graphiti), while pages are also rendered to images at 150 DPI for visual embeddings. When no text layer exists, Docling provides OCR fallback. All page processing runs concurrently, bounded by a token-bucket rate limiter that respects Jina's RPM/TPM limits per plan tier.

**Tech Stack:** PyMuPDF (text extraction + rendering), Docling (OCR fallback), asyncio (parallelism), token-bucket rate limiter

---

### Task 1: Add Rate Limit Settings

**Files:**
- Modify: `config/settings.py:11-15` (Jina section)
- Modify: `.env.example:14-19` (Jina section)

**Step 1: Add settings fields**

In `config/settings.py`, add after `jina_dense_dimensions`:

```python
jina_rpm: int = 100  # requests per minute (free=100, paid=500, premium=5000)
jina_tpm: int = 100_000  # tokens per minute (free=100K, paid=2M, premium=50M)
```

**Step 2: Update .env.example**

Add after `JINA_MAX_CONCURRENT` line:

```
JINA_RPM=100                            # Requests/min (free=100, paid=500, premium=5000)
JINA_TPM=100000                         # Tokens/min (free=100K, paid=2M, premium=50M)
```

**Step 3: Commit**

```bash
git add config/settings.py .env.example
git commit -m "feat(config): add JINA_RPM and JINA_TPM rate limit settings"
```

---

### Task 2: Add Token-Bucket Rate Limiter to Embedder

**Files:**
- Modify: `ingestion/embedder.py`
- Modify: `tests/test_embedder.py`

**Step 1: Write failing tests for rate limiter**

Add to `tests/test_embedder.py`:

```python
import time


class TestRateLimiter:
    async def test_rpm_throttling(self, embedder: JinaV4Embedder) -> None:
        """Requests beyond RPM limit are delayed."""
        # Set very low RPM to trigger throttling
        embedder._rpm_limiter = _TokenBucket(tokens_per_sec=2 / 60, burst=2)
        mock_resp = _mock_response([{"embedding": [0.1] * 2048}])

        with patch.object(
            embedder._client, "post",
            new_callable=AsyncMock, return_value=mock_resp,
        ):
            # First 2 should be instant (burst)
            t0 = time.monotonic()
            await embedder.embed_text(["a"])
            await embedder.embed_text(["b"])
            fast_elapsed = time.monotonic() - t0

            # Third should be delayed
            t1 = time.monotonic()
            await embedder.embed_text(["c"])
            slow_elapsed = time.monotonic() - t1

        assert fast_elapsed < 1.0
        assert slow_elapsed >= 0.5  # had to wait for token refill

    async def test_concurrent_requests_respect_semaphore(
        self, embedder: JinaV4Embedder,
    ) -> None:
        """Concurrency semaphore limits parallel requests."""
        import asyncio

        call_count = 0
        max_concurrent = 0

        original_post = embedder._client.post

        async def slow_post(*args, **kwargs):
            nonlocal call_count, max_concurrent
            call_count += 1
            current = call_count
            if current > max_concurrent:
                max_concurrent = current
            await asyncio.sleep(0.1)
            call_count -= 1
            return _mock_response([{"embedding": [0.1] * 2048}])

        embedder._semaphore = asyncio.Semaphore(2)
        with patch.object(
            embedder._client, "post", side_effect=slow_post,
        ):
            await asyncio.gather(
                embedder.embed_text(["a"]),
                embedder.embed_text(["b"]),
                embedder.embed_text(["c"]),
            )

        assert max_concurrent <= 2
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedder.py::TestRateLimiter -v`
Expected: FAIL — `_TokenBucket` not defined

**Step 3: Implement token-bucket rate limiter**

Add to `ingestion/embedder.py` before the `JinaV4Embedder` class:

```python
import time as _time


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, tokens_per_sec: float, burst: int) -> None:
        self._rate = tokens_per_sec
        self._max = float(burst)
        self._tokens = float(burst)
        self._last = _time.monotonic()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            now = _time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._max, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)
```

Update `JinaV4Embedder.__init__` to create rate limiters:

```python
def __init__(self, api_key: str | None = None) -> None:
    self._api_key = api_key or settings.jina_api_key
    self._embeddings_url = f"{settings.jina_api_url}/v1/embeddings"
    self._model = settings.jina_model
    self._client = httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(60.0),
    )
    self._semaphore = asyncio.Semaphore(settings.jina_max_concurrent)
    self._rpm_limiter = _TokenBucket(
        tokens_per_sec=settings.jina_rpm / 60.0,
        burst=settings.jina_max_concurrent,
    )
```

Update `_request` to acquire from the rate limiter before the semaphore:

```python
async def _request(
    self,
    payload: dict,
    timeout: float | None = None,
) -> dict:
    """Send request to Jina API with rate limiting + concurrency control."""
    await self._rpm_limiter.acquire()
    async with self._semaphore:
        return await self._request_with_retry(payload, timeout)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ingestion/embedder.py tests/test_embedder.py
git commit -m "feat(embedder): add token-bucket RPM rate limiter"
```

---

### Task 3: Replace pdf2image with PyMuPDF for Text Extraction + Rendering

**Files:**
- Modify: `ingestion/file_processor.py`
- Modify: `tests/test_file_processor.py`
- Modify: `pyproject.toml`

**Step 1: Swap dependency in pyproject.toml**

Replace `"pdf2image"` with `"pymupdf"` in the `dependencies` list. Add `"docling"` too.

**Step 2: Write failing tests for hybrid PDF extraction**

Update `tests/test_file_processor.py` — the existing `test_pdf_returns_multiple_pages` test should now expect BOTH text and image_bytes for PDFs with text layers. Add new tests:

```python
class TestFileToPages:
    # ... keep existing tests for png, jpg, txt, markdown, unknown ...

    def test_pdf_extracts_text_and_image(self) -> None:
        """PDF pages with text layer have both text and image_bytes."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        pages = file_to_pages("doc.pdf", content)

        assert len(pages) == 2
        for i, page in enumerate(pages):
            assert page.content_type == "pdf"
            assert page.page_number == i + 1
            assert len(page.image_bytes) > 0
            assert page.image_bytes[:4] == b"\x89PNG"
            # PyMuPDF should extract text from the text layer
            # (sample.pdf has text content on both pages)
            assert len(page.text) > 0

    def test_pdf_image_at_150dpi(self) -> None:
        """PDF pages rendered at 150 DPI are smaller than 300 DPI."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        pages = file_to_pages("doc.pdf", content)

        # 150 DPI images should be significantly smaller than 300 DPI
        # A 300 DPI letter page PNG is ~2MB; 150 DPI is ~500KB
        for page in pages:
            assert len(page.image_bytes) < 1_500_000  # well under 300 DPI size
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_file_processor.py::TestFileToPages::test_pdf_extracts_text_and_image -v`
Expected: FAIL — text is empty (current impl doesn't extract text)

**Step 4: Implement PyMuPDF-based PDF processing**

Replace `_pdf_to_pages` in `ingestion/file_processor.py`:

```python
import pymupdf


def _pdf_to_pages(content: bytes) -> list[Page]:
    """Convert PDF to Pages with text extraction (PyMuPDF) + image rendering (150 DPI).

    Each page gets:
    - text: extracted from PDF text layer (empty if scanned)
    - image_bytes: PNG rendered at 150 DPI for visual embeddings
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    pages: list[Page] = []

    for i, fitz_page in enumerate(doc):
        # Extract text from native text layer
        text = fitz_page.get_text("text").strip()

        # Render page to PNG at 150 DPI (default is 72, scale = 150/72)
        mat = pymupdf.Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")

        pages.append(
            Page(
                image_bytes=image_bytes,
                text=text,
                page_number=i + 1,
                content_type="pdf",
            )
        )

    doc.close()
    return pages
```

Remove the `from pdf2image import convert_from_bytes` import and the `import io` (if only used by pdf2image).

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_file_processor.py -v`
Expected: All PASS (update `test_pdf_returns_multiple_pages` assertion — text is no longer empty)

Note: The old `test_pdf_returns_multiple_pages` asserts `page.text == ""`. This must be updated — either remove it or merge it into the new `test_pdf_extracts_text_and_image`. Remove the old test.

**Step 6: Commit**

```bash
git add ingestion/file_processor.py tests/test_file_processor.py pyproject.toml
git commit -m "feat(ingestion): replace pdf2image with PyMuPDF for text extraction + 150dpi rendering"
```

---

### Task 4: Add Docling Fallback for Scanned PDFs

**Files:**
- Modify: `ingestion/file_processor.py`
- Create: `tests/test_docling_fallback.py`

**Step 1: Write failing test**

Create `tests/test_docling_fallback.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion.file_processor import _extract_text_docling


class TestDoclingFallback:
    def test_returns_text_when_docling_available(self) -> None:
        """Docling extracts text from image bytes when available."""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Extracted OCR text"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch(
            "ingestion.file_processor._get_docling_converter",
            return_value=mock_converter,
        ):
            text = _extract_text_docling(b"fake-image-bytes")

        assert text == "Extracted OCR text"

    def test_returns_empty_when_docling_unavailable(self) -> None:
        """Returns empty string when docling is not installed."""
        with patch(
            "ingestion.file_processor._get_docling_converter",
            return_value=None,
        ):
            text = _extract_text_docling(b"fake-image-bytes")

        assert text == ""
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_docling_fallback.py -v`
Expected: FAIL — `_extract_text_docling` not defined

**Step 3: Implement Docling fallback**

Add to `ingestion/file_processor.py`:

```python
_docling_converter = None
_docling_checked = False


def _get_docling_converter():
    """Lazily initialize Docling converter, or return None if not installed."""
    global _docling_converter, _docling_checked
    if _docling_checked:
        return _docling_converter
    _docling_checked = True
    try:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
        logger.info("Docling available for scanned PDF fallback")
    except ImportError:
        logger.info("Docling not installed, scanned PDF OCR disabled")
    return _docling_converter


def _extract_text_docling(image_bytes: bytes) -> str:
    """Extract text from image bytes using Docling. Returns '' if unavailable."""
    converter = _get_docling_converter()
    if converter is None:
        return ""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument
        import tempfile, os

        # Docling needs a file path — write temp PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name
        try:
            result = converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            os.unlink(tmp_path)
    except Exception:
        logger.exception("Docling extraction failed")
        return ""
```

Update `_pdf_to_pages` to use Docling fallback when text is empty:

```python
def _pdf_to_pages(content: bytes) -> list[Page]:
    doc = pymupdf.open(stream=content, filetype="pdf")
    pages: list[Page] = []

    for i, fitz_page in enumerate(doc):
        text = fitz_page.get_text("text").strip()

        mat = pymupdf.Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")

        # Docling fallback for scanned pages
        if not text:
            text = _extract_text_docling(image_bytes)

        pages.append(
            Page(
                image_bytes=image_bytes,
                text=text,
                page_number=i + 1,
                content_type="pdf",
            )
        )

    doc.close()
    return pages
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_docling_fallback.py tests/test_file_processor.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ingestion/file_processor.py tests/test_docling_fallback.py
git commit -m "feat(ingestion): add Docling OCR fallback for scanned PDFs"
```

---

### Task 5: Parallel Page Processing in Pipeline

**Files:**
- Modify: `ingestion/pipeline.py`

**Step 1: Refactor ingest_file for hybrid parallel processing**

The key change: for PDF pages with text, run BOTH `_process_text_page` and `_process_visual_page` concurrently. Process all pages in parallel using `asyncio.gather`.

Update `ingest_file` in `ingestion/pipeline.py`:

```python
@cocoindex.op.function()
def ingest_file(content: bytes, filename: str) -> str:
    """Process a single file: classify, embed, store to Qdrant + Neo4j."""
    t0 = time.monotonic()
    try:
        pages = file_to_pages(filename, content)
    except Exception:
        logger.exception("Failed to process file: %s", filename,
                         extra={"file_name": filename})
        return filename

    if not pages:
        logger.warning("No pages extracted from %s, skipping.", filename,
                       extra={"file_name": filename})
        return filename

    mime = _guess_mime(filename)
    now = datetime.now(tz=UTC).isoformat()
    qdrant = _get_qdrant_client()

    logger.info("Processing file: %s", filename, extra={
        "file_name": filename, "mime_type": mime, "page_count": len(pages),
    })

    graphiti_writer: GraphitiWriter | None = None
    try:
        has_text = any(p.text.strip() for p in pages)
        if has_text:
            graphiti_writer = GraphitiWriter()

        async def _process_all_pages() -> None:
            tasks = []
            for page in pages:
                tasks.extend(
                    _build_page_tasks(
                        page, filename, mime, now, qdrant, graphiti_writer,
                    )
                )
            if tasks:
                await asyncio.gather(*tasks)

        run_async(_process_all_pages())
    except Exception:
        logger.exception("Pipeline failed for file: %s", filename,
                         extra={"file_name": filename})
    finally:
        if graphiti_writer is not None:
            run_async(graphiti_writer.close())

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info("Finished file: %s in %dms", filename, duration_ms, extra={
        "file_name": filename, "mime_type": mime,
        "page_count": len(pages), "duration_ms": duration_ms,
    })
    return filename
```

Add the helper that builds async tasks per page:

```python
def _build_page_tasks(
    page,
    source_file: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    graphiti_writer: GraphitiWriter | None,
) -> list:
    """Return a list of async coroutines for processing one page."""
    tasks = []

    if page.content_type == "text":
        # Pure text files: text embedding + graphiti only
        tasks.append(_async_process_text_page(
            source_file, page.text, page.page_number,
            mime, now, qdrant, graphiti_writer,
        ))
    elif page.content_type == "pdf":
        # PDF pages: text + visual paths in parallel
        if page.text.strip():
            tasks.append(_async_process_text_page(
                source_file, page.text, page.page_number,
                mime, now, qdrant, graphiti_writer,
            ))
        tasks.append(_async_process_visual_page(
            source_file, page.image_bytes, page.page_number,
            "pdf_page", mime, now, qdrant,
        ))
    else:
        # Image files: visual path only
        tasks.append(_async_process_visual_page(
            source_file, page.image_bytes, page.page_number,
            "image", mime, now, qdrant,
        ))

    return tasks
```

Add async wrappers (the existing `_process_text_page` and `_process_visual_page` are sync — wrap them):

```python
async def _async_process_text_page(
    source_file: str, text: str, page_number: int,
    mime: str, now: str, qdrant: QdrantClient,
    graphiti_writer: GraphitiWriter | None,
) -> None:
    """Async wrapper for _process_text_page."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, _process_text_page,
        source_file, text, page_number, mime, now, qdrant, graphiti_writer,
    )


async def _async_process_visual_page(
    source_file: str, image_bytes: bytes, page_number: int,
    content_type: str, mime: str, now: str, qdrant: QdrantClient,
) -> None:
    """Async wrapper for _process_visual_page."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, _process_visual_page,
        source_file, image_bytes, page_number, content_type, mime, now, qdrant,
    )
```

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add ingestion/pipeline.py
git commit -m "feat(ingestion): parallel page processing with hybrid text+visual paths for PDFs"
```

---

### Task 6: Update .env.example and Clean Up

**Files:**
- Modify: `.env.example`
- Modify: `pyproject.toml`

**Step 1: Verify pdf2image is removed from pyproject.toml**

Confirm `pdf2image` is replaced by `pymupdf` and `docling` is added (done in Task 3).

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All PASS

**Step 3: Run linters**

Run: `uv run ruff check ingestion/ tests/ config/`
Run: `uv run ruff format --check ingestion/ tests/ config/`
Fix any issues.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: clean up deps and env example for hybrid PDF processing"
```

---

### Task 7: Manual Smoke Test

**Step 1: Start infrastructure**

```bash
docker compose up -d
./scripts/wait-for-services.sh
```

**Step 2: Run pipeline with sample files**

```bash
uv run python -m ingestion.pipeline
```

Expected: PDF processing completes in <30s (was >2min). Text chunks extracted from PDF pages. Both dense and multivec collections populated.

**Step 3: Verify Qdrant has text content for PDFs**

Check that PDF pages now have `text_content` populated in the dense collection (previously empty).
