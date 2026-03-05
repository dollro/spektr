# File Processing

The `file_processor.py` module handles MIME classification, PDF-to-image conversion, and semantic text chunking. It transforms raw file bytes into structured `Page` and `TextChunk` objects consumed by the embedding and graph stages.

**Source:** `ingestion/file_processor.py`

## Data Flow

```mermaid
flowchart LR
    Bytes["File bytes\n+ filename"] --> FTP["file_to_pages()"]
    FTP --> Pages["list[Page]"]
    Pages -->|text pages| SC["semantic_chunk()"]
    SC --> Chunks["list[TextChunk]"]
```

## Core Data Structures

### `Page`

Represents a single page extracted from a file.

| Field | Type | Description |
|-|-|-|
| `image_bytes` | `bytes` | PNG bytes for image/PDF pages; empty for text |
| `text` | `str` | Text content for text pages; empty for image |
| `page_number` | `int` | 1-based page number |
| `content_type` | `str` | `"pdf"`, `"image"`, or `"text"` |
| `has_visual_content` | `bool` | Whether Docling detected figures, tables, or formulas (default `False`) |

### `TextChunk`

A chunk of text after semantic splitting.

| Field | Type | Description |
|-|-|-|
| `text` | `str` | Chunk text content |
| `chunk_index` | `int` | 0-based index within the page |
| `page_number` | `int` | Source page number |

## `file_to_pages()`

Classifies a file by MIME type and extension, then converts it to a list of `Page` objects.

```python
def file_to_pages(filename: str, content: bytes) -> list[Page]
```

**Classification logic:**

| Condition | Result |
|-|-|
| `application/pdf` or `.pdf` extension | PDF -> multiple Pages with PNG bytes (150 DPI via `pdf2image`) |
| Image MIME or extension in `.png .jpg .jpeg .gif .bmp .webp` | Single Page with original image bytes |
| Text MIME or extension in `.md .txt .csv .json .xml .html .yaml .yml` | Single Page with decoded UTF-8 text |
| Unknown | Empty list + warning log |

### PDF conversion

PDFs are processed in two stages:

1. **Docling classification** — `_classify_pdf_pages_docling()` runs Docling once on the entire PDF. This extracts OCR text for scanned pages and classifies each page's layout, flagging pages with `TableItem`, `PictureItem`, or `FormulaItem` as `has_visual_content=True`.
2. **PyMuPDF rendering** — each page is rasterized at **150 DPI** to PNG format via PyMuPDF.

If Docling is unavailable, `_page_has_visual_content_pymupdf()` uses heuristics (embedded images, drawings, table detection) as a fallback.

### `resize_image_for_embedding()`

Resizes an image so its longest side is at most `max_px` pixels, using Lanczos resampling. Returns PNG bytes. Used to reduce token cost before embedding (400px default) and for thumbnails (200px).

## `semantic_chunk()`

Splits text into chunks preserving paragraph boundaries (`\n\n` delimiters).

```python
def semantic_chunk(
    text: str,
    max_chunk_size: int = 512,
    page_number: int = 1,
) -> list[TextChunk]
```

**Algorithm:**

1. Split text on `\n\n` into paragraphs
2. Accumulate paragraphs into a chunk until adding the next paragraph would exceed `max_chunk_size` (default 512 characters)
3. When the limit is reached, emit the current chunk and start a new one
4. If a single paragraph exceeds `max_chunk_size`, pass it to `_split_long_text()`

Each chunk carries the `page_number` from its source page for downstream traceability.

## `_split_long_text()`

Fallback for paragraphs that exceed `max_chunk_size`. Splits on word boundaries:

```python
def _split_long_text(text: str, max_size: int) -> list[str]
```

Iterates over words, accumulating into parts that stay within `max_size`. Never splits mid-word.

## Integration

`file_to_pages()` and `semantic_chunk()` are called inside the [`ingest_file` CocoIndex op](cocoindex.md) in `pipeline.py`:

1. `file_to_pages(filename, content)` classifies and extracts pages
2. For text pages: `semantic_chunk(page.text, page_number=page.page_number)` produces chunks
3. Chunks are embedded via [Jina v4](embeddings.md) and ingested into [Graphiti](knowledge-graph.md)

See also: [Pipeline Overview](overview.md) | [Architecture Data Flow](../architecture/data-flow.md)
