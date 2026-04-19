# Chunking Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace naive paragraph splitting with Docling HybridChunker (structural) + Jina late chunking (semantic context). Two complementary upgrades applied in dependency order.

**Architecture:** Four tasks in sequence. Task 1 extends TextChunk and introduces FileProcessingResult. Task 2 adds docling_chunk() and wires it into file_processor. Task 3 adds late_chunking to the Jina embedder. Task 4 wires everything together in the pipeline with Docling chunks + late chunking + contextualized Graphiti text.

**Tech Stack:** Python 3.13, Docling (`HybridChunker`, `HuggingFaceTokenizer`), Jina v4 API (`late_chunking` param), Qdrant, Graphiti/Neo4j

**Design doc:** `plans/2026-03-05-chunking-strategy-design.md`

---

## Task 1: Extend TextChunk + Add FileProcessingResult

Add `contextualized_text` field to TextChunk and create the FileProcessingResult dataclass. Update `_classify_pdf_pages_docling` to return the Docling Document, `_pdf_to_pages` and `file_to_pages` to return FileProcessingResult. Update all existing tests.

**Files:**
- Modify: `ingestion/file_processor.py:17-59` (TextChunk, file_to_pages, _pdf_to_pages)
- Modify: `ingestion/pipeline.py:382` (unpack FileProcessingResult)
- Test: `tests/test_file_processor.py`

**Step 1: Add contextualized_text to TextChunk**

In `ingestion/file_processor.py`, change the TextChunk dataclass (line 27-30):

```python
@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_number: int
    contextualized_text: str | None = None
```

**Step 2: Add FileProcessingResult dataclass**

Add after TextChunk (line 31):

```python
@dataclass
class FileProcessingResult:
    pages: list[Page]
    docling_document: object | None = None
```

**Step 3: Update _classify_pdf_pages_docling to return Document**

Change the return type and implementation in `ingestion/file_processor.py` (line 118-158):

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

**Step 4: Update _pdf_to_pages to return FileProcessingResult**

Change `_pdf_to_pages` (line 193-228):

```python
def _pdf_to_pages(content: bytes) -> FileProcessingResult:
    """Convert PDF to Pages with Docling classification + PyMuPDF rendering.

    Runs Docling once on the whole PDF for OCR text and layout classification,
    then renders each page at 150 DPI via PyMuPDF.
    """
    visual_map, docling_text_map, docling_doc = _classify_pdf_pages_docling(content)

    doc = pymupdf.open(stream=content, filetype="pdf")
    pages: list[Page] = []

    for i, fitz_page in enumerate(doc):
        page_no = i + 1
        text = fitz_page.get_text("text").strip()
        mat = pymupdf.Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")

        # Docling text for scanned pages (replaces per-page OCR)
        if not text and page_no in docling_text_map:
            text = docling_text_map[page_no].strip()

        # Docling layout classification (fallback: PyMuPDF heuristics)
        if visual_map:
            has_visual = visual_map.get(page_no, False)
        else:
            has_visual = _page_has_visual_content_pymupdf(fitz_page)

        logger.debug("Page %d: has_visual=%s, text_len=%d", page_no, has_visual, len(text))
        pages.append(Page(
            image_bytes=image_bytes, text=text, page_number=page_no,
            content_type="pdf", has_visual_content=has_visual,
        ))

    doc.close()
    return FileProcessingResult(pages=pages, docling_document=docling_doc)
```

**Step 5: Update file_to_pages to return FileProcessingResult**

Change `file_to_pages` (line 33-59):

```python
def file_to_pages(filename: str, content: bytes) -> FileProcessingResult:
    """MIME-classify file and convert to FileProcessingResult.

    PDF -> multiple Pages with text extraction + PNG (150 DPI) + Docling Document.
    Image -> single Page with original bytes.
    Text -> single Page with text content.
    Unknown -> empty pages + log warning.
    """
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

**Step 6: Update pipeline.py to unpack FileProcessingResult**

In `ingestion/pipeline.py`, update the import (line 26-31):

```python
from ingestion.file_processor import (
    FileProcessingResult,
    TextChunk,
    file_to_pages,
    resize_image_for_embedding,
    semantic_chunk,
)
```

Update `ingest_file` (line 382-391):

```python
    try:
        result = file_to_pages(filename, content)
        pages = result.pages
    except Exception:
        logger.exception(
            "Failed to process file: %s",
            filename,
            extra={"file_name": filename},
        )
        return filename
```

Also update the `if not pages:` check on line 391 — it now uses the `pages` variable (already correct after the above change).

**Step 7: Update existing tests for FileProcessingResult**

In `tests/test_file_processor.py`, every test that calls `file_to_pages` must unpack `.pages`:

```python
from ingestion.file_processor import FileProcessingResult, file_to_pages, semantic_chunk

FIXTURES = Path(__file__).parent / "fixtures"


class TestFileToPages:
    def test_png_returns_single_image_page(self) -> None:
        content = (FIXTURES / "sample.png").read_bytes()
        result = file_to_pages("photo.png", content)
        assert isinstance(result, FileProcessingResult)
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "image"
        assert pages[0].image_bytes == content
        assert pages[0].text == ""
        assert pages[0].page_number == 1

    def test_jpg_detected_as_image(self) -> None:
        result = file_to_pages("photo.jpg", b"\xff\xd8\xff\xe0fake")
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "image"

    def test_txt_returns_text_page(self) -> None:
        content = (FIXTURES / "sample.txt").read_bytes()
        result = file_to_pages("doc.txt", content)
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "text"
        assert pages[0].text == content.decode("utf-8")
        assert pages[0].image_bytes == b""

    def test_markdown_detected_as_text(self) -> None:
        result = file_to_pages("readme.md", b"# Hello")
        pages = result.pages

        assert len(pages) == 1
        assert pages[0].content_type == "text"
        assert pages[0].text == "# Hello"

    def test_pdf_extracts_text_and_image(self) -> None:
        """PDF pages with text layer have both text and image_bytes."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        result = file_to_pages("doc.pdf", content)
        pages = result.pages

        assert len(pages) == 2
        for i, page in enumerate(pages):
            assert page.content_type == "pdf"
            assert page.page_number == i + 1
            assert len(page.image_bytes) > 0
            assert page.image_bytes[:4] == b"\x89PNG"
            assert len(page.text) > 0

    def test_pdf_image_at_150dpi(self) -> None:
        """PDF pages rendered at 150 DPI are smaller than 300 DPI."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        result = file_to_pages("doc.pdf", content)
        pages = result.pages

        for page in pages:
            assert len(page.image_bytes) < 1_500_000

    def test_unknown_type_returns_empty(self) -> None:
        result = file_to_pages("data.xyz123", b"mystery")
        assert result.pages == []

    def test_pdf_returns_docling_document(self) -> None:
        """PDF processing populates docling_document when Docling is available."""
        content = (FIXTURES / "sample.pdf").read_bytes()
        result = file_to_pages("doc.pdf", content)
        # docling_document may be None if Docling is not installed,
        # but the result should still have the field
        assert hasattr(result, "docling_document")

    def test_text_file_has_no_docling_document(self) -> None:
        """Text files don't produce a docling_document (yet)."""
        result = file_to_pages("readme.md", b"# Hello")
        assert result.docling_document is None
```

**Step 8: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_file_processor.py -v`
Expected: All pass

**Step 9: Commit**

```bash
git add ingestion/file_processor.py ingestion/pipeline.py tests/test_file_processor.py
git commit -m "refactor: introduce FileProcessingResult, extend TextChunk with contextualized_text

Prepares the data model for Docling HybridChunker + Jina late chunking.
file_to_pages now returns FileProcessingResult(pages, docling_document).
TextChunk gains optional contextualized_text for heading-prefixed text."
```

---

## Task 2: Add docling_chunk() Function

Implement the HybridChunker-based chunking function that produces TextChunks with contextualized headings. The chunker uses the Jina v4 tokenizer for exact token-budget alignment.

**Important API detail:** `HybridChunker` takes a tokenizer **object**, not a string. Use `HuggingFaceTokenizer.from_pretrained("jinaai/jina-embeddings-v4", max_tokens=256)`. The `contextualize()` method is called on the **chunker** object: `chunker.contextualize(chunk)`. Chunk metadata is at `chunk.meta.headings` (list of strings) and `chunk.meta.doc_items[0].prov[0].page_no` for page numbers.

**Files:**
- Modify: `ingestion/file_processor.py` (add `_get_hybrid_chunker`, `docling_chunk`)
- Test: `tests/test_file_processor.py`

**Step 1: Write the failing tests**

Add to `tests/test_file_processor.py`:

```python
from unittest.mock import MagicMock, patch


class TestDoclingChunk:
    def test_returns_text_chunks_with_page_numbers(self) -> None:
        """docling_chunk produces TextChunks with contextualized text."""
        from ingestion.file_processor import docling_chunk

        mock_doc = MagicMock()
        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "First heading content"
        mock_chunk_1.meta = MagicMock()
        mock_chunk_1.meta.headings = ["Introduction"]
        mock_chunk_1.meta.doc_items = [MagicMock()]
        mock_chunk_1.meta.doc_items[0].prov = [MagicMock(page_no=1)]

        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "Table content here"
        mock_chunk_2.meta = MagicMock()
        mock_chunk_2.meta.headings = ["Results", "Table 1"]
        mock_chunk_2.meta.doc_items = [MagicMock()]
        mock_chunk_2.meta.doc_items[0].prov = [MagicMock(page_no=2)]

        with patch(
            "ingestion.file_processor._get_hybrid_chunker"
        ) as mock_chunker_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = [mock_chunk_1, mock_chunk_2]
            mock_chunker.contextualize.side_effect = [
                "Introduction\nFirst heading content",
                "Results > Table 1\nTable content here",
            ]
            mock_chunker_factory.return_value = mock_chunker

            chunks = docling_chunk(mock_doc)

        assert len(chunks) == 2
        assert chunks[0].text == "First heading content"
        assert chunks[0].contextualized_text == "Introduction\nFirst heading content"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[1].text == "Table content here"
        assert chunks[1].contextualized_text == "Results > Table 1\nTable content here"
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

    def test_returns_empty_when_chunker_unavailable(self) -> None:
        """docling_chunk returns empty when HybridChunker not installed."""
        from ingestion.file_processor import docling_chunk

        mock_doc = MagicMock()
        with patch(
            "ingestion.file_processor._get_hybrid_chunker",
            return_value=None,
        ):
            result = docling_chunk(mock_doc)

        assert result == []

    def test_contextualize_failure_uses_raw_text(self) -> None:
        """If contextualize() fails, fall back to raw chunk.text."""
        from ingestion.file_processor import docling_chunk

        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Raw text content"
        mock_chunk.meta = MagicMock()
        mock_chunk.meta.doc_items = [MagicMock()]
        mock_chunk.meta.doc_items[0].prov = [MagicMock(page_no=1)]

        with patch(
            "ingestion.file_processor._get_hybrid_chunker"
        ) as mock_chunker_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = [mock_chunk]
            mock_chunker.contextualize.side_effect = Exception("contextualize failed")
            mock_chunker_factory.return_value = mock_chunker

            chunks = docling_chunk(mock_doc)

        assert len(chunks) == 1
        assert chunks[0].text == "Raw text content"
        assert chunks[0].contextualized_text is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_file_processor.py::TestDoclingChunk -v`
Expected: FAIL — `docling_chunk` doesn't exist yet

**Step 3: Implement _get_hybrid_chunker and docling_chunk**

Add after `semantic_chunk` in `ingestion/file_processor.py` (after line 92):

```python
_hybrid_chunker = None
_hybrid_chunker_checked = False


def _get_hybrid_chunker() -> object | None:
    """Lazily initialize Docling HybridChunker with Jina v4 tokenizer."""
    global _hybrid_chunker, _hybrid_chunker_checked  # noqa: PLW0603
    if _hybrid_chunker_checked:
        return _hybrid_chunker
    _hybrid_chunker_checked = True
    try:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )

        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name="jinaai/jina-embeddings-v4",
            max_tokens=256,
        )
        _hybrid_chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
        )
        logger.info("Docling HybridChunker available (jina-v4 tokenizer, 256 max_tokens)")
    except (ImportError, Exception):
        logger.info("Docling HybridChunker not available, will use paragraph chunker")
    return _hybrid_chunker


def docling_chunk(doc: object | None) -> list[TextChunk]:
    """Chunk a Docling Document using HybridChunker.

    Returns TextChunks with page numbers from chunk provenance and
    contextualized_text from chunker.contextualize() (heading-prefixed).
    Returns empty list if doc is None or chunking fails.
    """
    if doc is None:
        return []

    try:
        chunker = _get_hybrid_chunker()
        if chunker is None:
            return []

        raw_chunks = chunker.chunk(dl_doc=doc)
        result: list[TextChunk] = []
        for idx, chunk in enumerate(raw_chunks):
            # Extract page number from provenance
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

            # Get contextualized text (heading-prefixed)
            ctx_text: str | None = None
            try:
                ctx_text = chunker.contextualize(chunk)
            except Exception:
                logger.warning("contextualize() failed for chunk %d, using raw text", idx)

            result.append(
                TextChunk(
                    text=chunk.text,
                    chunk_index=idx,
                    page_number=page_no,
                    contextualized_text=ctx_text,
                )
            )
        return result
    except Exception:
        logger.exception("Docling chunking failed, returning empty")
        return []
```

**Step 4: Run test to verify it passes**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_file_processor.py::TestDoclingChunk -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_file_processor.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add ingestion/file_processor.py tests/test_file_processor.py
git commit -m "feat: add docling_chunk() with HybridChunker + heading context

Structure-aware chunking using Docling HybridChunker with Jina v4 tokenizer
(256 max_tokens). Produces TextChunks with contextualized_text from
chunker.contextualize() for heading-prefixed embedding/Graphiti ingestion."
```

---

## Task 3: Add late_chunking to Jina Embedder

Add `late_chunking` parameter to `embed_text()`. When True, the payload includes `"late_chunking": True` and all texts are sent in a single batch (no sub-batching) to preserve cross-chunk context.

**Files:**
- Modify: `ingestion/embedders/jina.py:78-117` (`embed_text`, `_embed_text_batch`)
- Modify: `ingestion/embedder.py:67-72` (protocol)
- Modify: `ingestion/embedders/voyage.py:70-75` (accept+ignore param)
- Test: `tests/test_embedder.py`

**Step 1: Write the failing tests**

Add to `tests/test_embedder.py`:

```python
class TestLateChunking:
    async def test_payload_includes_late_chunking(self, embedder: JinaV4Embedder) -> None:
        """When late_chunking=True, payload has 'late_chunking': True."""
        mock_resp = _mock_response([
            {"embedding": [0.1] * DENSE_DIM},
            {"embedding": [0.2] * DENSE_DIM},
        ])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["chunk 1", "chunk 2"], late_chunking=True)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["late_chunking"] is True

    async def test_no_late_chunking_by_default(self, embedder: JinaV4Embedder) -> None:
        """Default embed_text does NOT include late_chunking in payload."""
        mock_resp = _mock_response([{"embedding": [0.1] * DENSE_DIM}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["hello"])

        payload = mock_post.call_args.kwargs["json"]
        assert "late_chunking" not in payload

    async def test_single_batch_when_late_chunking(self, embedder: JinaV4Embedder) -> None:
        """late_chunking=True sends all texts in one API call, ignoring batch_size."""
        n_texts = 50  # well above typical batch_size
        mock_resp = _mock_response([{"embedding": [0.1] * DENSE_DIM}] * n_texts)
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            with patch("ingestion.embedders.jina.settings") as mock_settings:
                mock_settings.jina_batch_size = 10  # force small batches
                mock_settings.jina_api_url = "https://api.jina.ai"
                mock_settings.jina_model = "jina-embeddings-v4"
                mock_settings.jina_dense_dimensions = DENSE_DIM
                await embedder.embed_text(
                    [f"chunk {i}" for i in range(n_texts)],
                    late_chunking=True,
                )

        # Should be exactly 1 API call, not 5 batches of 10
        assert mock_post.call_count == 1
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["input"]) == n_texts
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py::TestLateChunking -v`
Expected: FAIL — `late_chunking` parameter doesn't exist

**Step 3: Update Embedder protocol**

In `ingestion/embedder.py`, change `embed_text` signature (line 67-72):

```python
    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]: ...
```

**Step 4: Update JinaV4Embedder.embed_text**

In `ingestion/embedders/jina.py`, change `embed_text` (line 78-97):

```python
    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors.

        When late_chunking=True, all texts are sent in a single API call
        (no sub-batching) so Jina can embed them as one concatenated sequence
        and return per-chunk contextual vectors.
        """
        if late_chunking:
            # Single batch — context must be preserved across all texts
            return await self._embed_text_batch(texts, task, dimensions, late_chunking=True)

        batch_size = settings.jina_batch_size
        if len(texts) <= batch_size:
            return await self._embed_text_batch(texts, task, dimensions)

        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(await self._embed_text_batch(batch, task, dimensions))
        return results
```

Update `_embed_text_batch` (line 99-117):

```python
    async def _embed_text_batch(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]:
        """Send a single embed_text API call for one batch."""
        jina_task = _TASK_MAP.get(task, task)
        dims = dimensions if dimensions is not None else self._dimensions
        payload: dict = {
            "model": self._model,
            "task": jina_task,
            "dimensions": dims,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": t} for t in texts],
        }
        if late_chunking:
            payload["late_chunking"] = True
        data = await self._request(payload)
        return [item["embedding"] for item in data["data"]]
```

**Step 5: Update VoyageEmbedder to accept and ignore the parameter**

In `ingestion/embedders/voyage.py`, change `embed_text` (line 70-86):

```python
    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
        late_chunking: bool = False,
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors.

        late_chunking is accepted for protocol compatibility but ignored
        (Voyage does not support late chunking).
        """
        voyage_task = _TASK_MAP.get(task, task)
        dims = dimensions if dimensions is not None else self._dimensions
        payload = {
            "model": self._text_model,
            "input": texts,
            "input_type": voyage_task,
            "output_dimensions": dims,
        }
        data = await self._request(self._text_url, payload)
        return [item["embedding"] for item in data["data"]]
```

**Step 6: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_embedder.py -v`
Expected: All pass (existing tests unchanged, new tests pass)

**Step 7: Commit**

```bash
git add ingestion/embedder.py ingestion/embedders/jina.py ingestion/embedders/voyage.py tests/test_embedder.py
git commit -m "feat: add late_chunking parameter to Jina embed_text

When late_chunking=True, Jina concatenates all inputs internally and
returns per-chunk contextual vectors. All texts sent in one API call
(no sub-batching) to preserve cross-chunk semantic context."
```

---

## Task 4: Wire Docling Chunks + Late Chunking into Pipeline

Connect everything: pipeline uses `docling_chunk()` when available, embeds page-grouped chunks with `late_chunking=True`, stores contextualized text in Qdrant payloads, and sends contextualized text to Graphiti.

**Files:**
- Modify: `ingestion/pipeline.py:26-31,178-242,350-371,374-398` (imports, _process_text_page, _ingest_to_graphiti, ingest_file)
- Test: `tests/test_pipeline_chunking.py` (new)

**Step 1: Write the failing tests**

Create `tests/test_pipeline_chunking.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_processor import TextChunk


class TestProcessTextPageWithDoclingChunks:
    async def test_uses_docling_chunks_when_provided(self) -> None:
        """_process_text_page uses pre-computed docling chunks instead of semantic_chunk."""
        from ingestion.pipeline import _process_text_page

        dl_chunks = [
            TextChunk(
                text="Revenue grew 15%",
                chunk_index=0,
                page_number=1,
                contextualized_text="Financials > Q3\nRevenue grew 15%",
            ),
            TextChunk(
                text="Expenses decreased",
                chunk_index=1,
                page_number=1,
                contextualized_text="Financials > Q3\nExpenses decreased",
            ),
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=[[0.1] * 10, [0.2] * 10])
        mock_qdrant = MagicMock()

        with patch("ingestion.pipeline.semantic_chunk") as mock_semantic:
            await _process_text_page(
                source_file="report.pdf",
                text="ignored when docling chunks provided",
                page_number=1,
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=mock_qdrant,
                embedder=mock_embedder,
                graphiti_writer=None,
                docling_chunks=dl_chunks,
            )

        # semantic_chunk should NOT have been called
        mock_semantic.assert_not_called()
        # embed_text should have been called with late_chunking=True
        mock_embedder.embed_text.assert_called_once()
        call_kwargs = mock_embedder.embed_text.call_args
        assert call_kwargs.kwargs.get("late_chunking") is True or (
            len(call_kwargs.args) > 2 and call_kwargs.args[2] is True
        )

    async def test_falls_back_to_semantic_chunk(self) -> None:
        """Without docling chunks, uses semantic_chunk (existing behavior)."""
        from ingestion.pipeline import _process_text_page

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=[[0.1] * 10])
        mock_qdrant = MagicMock()

        with patch("ingestion.pipeline.semantic_chunk") as mock_semantic:
            mock_semantic.return_value = [
                TextChunk(text="Fallback chunk", chunk_index=0, page_number=1)
            ]
            await _process_text_page(
                source_file="report.pdf",
                text="Some text content",
                page_number=1,
                mime="application/pdf",
                now="2026-03-05T00:00:00Z",
                qdrant=mock_qdrant,
                embedder=mock_embedder,
                graphiti_writer=None,
                docling_chunks=None,
            )

        mock_semantic.assert_called_once()
        # No late_chunking when using semantic_chunk fallback
        call_kwargs = mock_embedder.embed_text.call_args
        assert call_kwargs.kwargs.get("late_chunking", False) is False


class TestQdrantPayloadContextualizedText:
    async def test_stores_contextualized_text_in_payload(self) -> None:
        """Qdrant point payload includes contextualized_text when available."""
        from ingestion.pipeline import _process_text_page

        dl_chunks = [
            TextChunk(
                text="Raw text",
                chunk_index=0,
                page_number=1,
                contextualized_text="Heading > Sub\nRaw text",
            ),
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=[[0.1] * 10])
        mock_qdrant = MagicMock()

        await _process_text_page(
            source_file="doc.pdf",
            text="ignored",
            page_number=1,
            mime="application/pdf",
            now="2026-03-05T00:00:00Z",
            qdrant=mock_qdrant,
            embedder=mock_embedder,
            graphiti_writer=None,
            docling_chunks=dl_chunks,
        )

        upsert_call = mock_qdrant.upsert.call_args
        points = upsert_call.kwargs["points"]
        payload = points[0].payload
        assert payload["text_content"] == "Raw text"
        assert payload["contextualized_text"] == "Heading > Sub\nRaw text"


class TestGraphitiContextualizedText:
    async def test_graphiti_receives_contextualized_text(self) -> None:
        """Graphiti ingestion uses contextualized_text when available."""
        from ingestion.pipeline import _ingest_to_graphiti

        chunks = [
            TextChunk(
                text="Raw text",
                chunk_index=0,
                page_number=1,
                contextualized_text="Heading > Sub\nRaw text",
            ),
        ]

        mock_writer = AsyncMock()
        await _ingest_to_graphiti("doc.pdf", chunks, mock_writer)

        call_kwargs = mock_writer.ingest_chunk.call_args.kwargs
        assert call_kwargs["chunk_text"] == "Heading > Sub\nRaw text"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_chunking.py -v`
Expected: FAIL — `_process_text_page` doesn't accept `docling_chunks` parameter

**Step 3: Update imports in pipeline.py**

In `ingestion/pipeline.py`, update the import (line 26-31):

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

**Step 4: Update _process_text_page to accept docling_chunks**

Replace `_process_text_page` (line 178-242):

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
    """Process a text page: chunk, embed, store dense, ingest to Graphiti.

    When docling_chunks are provided, uses them (filtered to this page)
    with late_chunking=True for contextual embeddings.
    Falls back to semantic_chunk() without late chunking otherwise.
    """
    use_late_chunking = False
    if docling_chunks is not None:
        chunks = [c for c in docling_chunks if c.page_number == page_number]
        use_late_chunking = bool(chunks)

    if not docling_chunks or not use_late_chunking:
        chunks = semantic_chunk(text, page_number=page_number)

    if not chunks:
        return

    # Batch-embed all chunks in a single API call
    try:
        all_texts = [chunk.text for chunk in chunks]
        vectors = await embedder.embed_text(all_texts, late_chunking=use_late_chunking)
    except Exception:
        logger.exception(
            "Text embedding failed for %s page %d (%d chunks)",
            source_file,
            page_number,
            len(chunks),
        )
        return

    dense_points: list[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        chunk_id = _make_chunk_id(
            source_file,
            page_number,
            chunk.chunk_index,
        )
        point_id = _make_point_id(chunk_id)
        payload = {
            "source_file": source_file,
            "content_type": "text_chunk",
            "page_number": page_number,
            "chunk_index": chunk.chunk_index,
            "char_count": len(chunk.text),
            "text_content": chunk.text,
            "metadata": {
                "mime_type": mime,
                "ingested_at": now,
                "source_key": source_file,
            },
        }
        if chunk.contextualized_text is not None:
            payload["contextualized_text"] = chunk.contextualized_text

        dense_points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ),
        )

    if dense_points:
        qdrant.upsert(
            collection_name=DENSE_COLLECTION,
            points=dense_points,
        )

    # Graphiti episode ingestion (handles entity extraction internally)
    if graphiti_writer is not None:
        await _ingest_to_graphiti(source_file, chunks, graphiti_writer)
```

**Step 5: Update _ingest_to_graphiti to use contextualized_text**

Replace `_ingest_to_graphiti` (line 350-371):

```python
async def _ingest_to_graphiti(
    source_file: str,
    chunks: list[TextChunk],
    graphiti_writer: GraphitiWriter,
) -> None:
    """Ingest chunks as Graphiti episodes (entity extraction is automatic).

    Uses contextualized_text (heading-prefixed) when available for better
    entity resolution. Falls back to raw text.
    """
    ref_time = datetime.now(tz=UTC)
    for chunk in chunks:
        chunk_text = chunk.contextualized_text or chunk.text
        try:
            await graphiti_writer.ingest_chunk(
                chunk_text=chunk_text,
                source_key=source_file,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                reference_time=ref_time,
            )
        except Exception:
            logger.exception(
                "Graphiti ingestion failed for %s chunk %d",
                source_file,
                chunk.chunk_index,
            )
```

**Step 6: Update _build_page_tasks to pass docling_chunks**

Update `_build_page_tasks` signature and calls (line 94-175):

```python
def _build_page_tasks(
    page,  # type: ignore[no-untyped-def]
    source_file: str,
    mime: str,
    now: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    graphiti_writer: GraphitiWriter | None,
    docling_chunks: list[TextChunk] | None = None,
) -> _PageTasks:
    """Return text and image coroutines for processing one page."""
    tasks = _PageTasks()

    if page.content_type == "text":
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
                docling_chunks=docling_chunks,
            )
        )
    elif page.content_type == "pdf":
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
                    docling_chunks=docling_chunks,
                )
            )

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
    else:
        tasks.image.append(
            _process_visual_page(
                source_file,
                page.image_bytes,
                page.page_number,
                "image",
                mime,
                now,
                qdrant,
                embedder,
            )
        )

    return tasks
```

**Step 7: Update ingest_file to compute and pass Docling chunks**

In `ingest_file`, after `result = file_to_pages(...)` and before `_process_all_pages()`, add Docling chunking. Replace lines 374-453:

```python
@cocoindex.op.function()
def ingest_file(content: bytes, filename: str) -> str:
    """Process a single file: classify, embed, store to Qdrant + Neo4j.

    Returns filename as passthrough for CocoIndex lineage tracking.
    """
    t0 = time.monotonic()
    try:
        result = file_to_pages(filename, content)
        pages = result.pages
    except Exception:
        logger.exception(
            "Failed to process file: %s",
            filename,
            extra={"file_name": filename},
        )
        return filename

    if not pages:
        logger.warning(
            "No pages extracted from %s, skipping.",
            filename,
            extra={"file_name": filename},
        )
        return filename

    mime = _guess_mime(filename)
    now = datetime.now(tz=UTC).isoformat()
    qdrant = _get_qdrant_client()
    graphiti_writer: GraphitiWriter | None = None

    # Compute Docling chunks once for the whole document
    dl_chunks = docling_chunk(result.docling_document) if result.docling_document else None
    if dl_chunks:
        logger.info(
            "Using Docling HybridChunker: %d chunks for %s",
            len(dl_chunks),
            filename,
            extra={"file_name": filename, "chunk_count": len(dl_chunks)},
        )

    logger.info(
        "Processing file: %s",
        filename,
        extra={
            "file_name": filename,
            "mime_type": mime,
            "page_count": len(pages),
        },
    )

    try:
        has_text = any(p.text.strip() for p in pages)
        if has_text:
            graphiti_writer = GraphitiWriter()

        async def _process_all_pages() -> None:
            embedder = create_embedder()
            sem = asyncio.Semaphore(2)

            async def _bounded(coro):  # type: ignore[no-untyped-def]
                async with sem:
                    return await coro

            try:
                text_tasks: list = []  # type: ignore[type-arg]
                image_tasks: list = []  # type: ignore[type-arg]
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
                    )
                    text_tasks.extend(pt.text)
                    image_tasks.extend(pt.image)

                # Text: concurrent (lightweight, small TPM footprint)
                if text_tasks:
                    await asyncio.gather(*[_bounded(t) for t in text_tasks])

                # Images: sequential (heavy, TPM-sensitive)
                for task in image_tasks:
                    await task
            finally:
                await embedder.close()

        run_async(_process_all_pages())
    except Exception:
        logger.exception(
            "Pipeline failed for file: %s",
            filename,
            extra={"file_name": filename},
        )
    finally:
        if graphiti_writer is not None:
            run_async(graphiti_writer.close())

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "Finished file: %s in %dms",
        filename,
        duration_ms,
        extra={
            "file_name": filename,
            "mime_type": mime,
            "page_count": len(pages),
            "duration_ms": duration_ms,
        },
    )
    return filename
```

**Step 8: Run tests**

Run: `cd /home/rodo/Coding/spektr && python -m pytest tests/test_pipeline_chunking.py tests/test_file_processor.py tests/test_embedder.py -v`
Expected: All pass

**Step 9: Run full test suite**

Run: `cd /home/rodo/Coding/spektr && python -m pytest -v`
Expected: All pass

**Step 10: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline_chunking.py
git commit -m "feat: wire Docling chunks + late chunking into ingestion pipeline

Pipeline now uses docling_chunk() when a Docling Document is available,
embeds page-grouped chunks with late_chunking=True for cross-chunk
semantic context, stores contextualized_text in Qdrant payloads, and
sends heading-prefixed text to Graphiti for better entity resolution.
Falls back to semantic_chunk() without late chunking when Docling is unavailable."
```

---

## Summary

| Task | Files Changed | Risk | Dependencies |
|-|-|-|-|
| 1. FileProcessingResult + TextChunk | file_processor.py, pipeline.py, test_file_processor.py | Low | None |
| 2. docling_chunk() | file_processor.py, test_file_processor.py | Low | Task 1 |
| 3. Late chunking | jina.py, embedder.py, voyage.py, test_embedder.py | Low | None (parallel with 1-2) |
| 4. Pipeline wiring | pipeline.py, test_pipeline_chunking.py | Medium | Tasks 1, 2, 3 |

**Execution order:** Tasks 1→2 must be sequential. Task 3 is independent (can parallel with 1-2). Task 4 depends on all three.

**Not included in this plan (separate scope):**
- Non-PDF Docling parsing (text/markdown files) — requires additional DocumentConverter pipeline config. Can be added later as an incremental enhancement.
- Updating the retrieval/search path to use `contextualized_text` from Qdrant payloads.
