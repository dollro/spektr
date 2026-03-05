# Hybrid PDF Processing + Jina Rate Limiting

## Problem

PDF ingestion is slow because:
1. No text extraction — PDFs only go through the image embedding path
2. Sequential processing — pages and API calls run one at a time
3. 300 DPI renders — large images sent to Jina, slow uploads + rate limits hit
4. No rate limiting — only a concurrency semaphore, no RPM/TPM enforcement

A 2-page PDF takes >2 minutes instead of seconds.

## Design

### 1. PDF Text Extraction (Hybrid)

**Primary: PyMuPDF** — extracts the native text layer from PDFs in ~0.12s. Covers 90%+ of modern PDFs (reports, articles, business docs). Also replaces `pdf2image`/poppler for page-to-image rendering.

**Fallback: Docling** — AI-powered extraction when PyMuPDF finds no text layer (scanned PDFs). Uses DocLayNet for layout analysis + TableFormer for tables. ~6s/page, GPU recommended.

Decision logic per page:
```
text = pymupdf_extract_text(page)
if text.strip():
    # Fast path: text layer found
    process_text(text)  # chunk → text embed → Graphiti
    process_image(render_page(150dpi))  # dense + multivec image embed
else:
    # Fallback: scanned page
    if docling_available:
        text = docling_extract(page_image)
        if text.strip():
            process_text(text)
    process_image(render_page(150dpi))  # always do image embeddings
```

### 2. Parallel Processing

All pages processed concurrently via `asyncio.gather()`. Within each page:
- Text embedding + Graphiti ingestion run concurrently with image embeddings
- Dense + multivec image embeddings run concurrently

For a 2-page PDF with text layers:
- Before: 4 sequential Jina calls + 2 sequential Graphiti calls = ~120s
- After: All calls concurrent, bounded by rate limiter = ~10-15s

### 3. Jina Rate Limiting

New configurable settings (defaults = free tier):
- `JINA_RPM=100` — requests per minute
- `JINA_TPM=100000` — tokens per minute

Implementation: token-bucket rate limiter in `JinaV4Embedder` that enforces RPM/TPM alongside the existing concurrency semaphore. Requests wait when bucket is empty.

### 4. Image Rendering

- Reduce DPI from 300 → 150 (4x smaller images, plenty for semantic search)
- Use PyMuPDF's built-in renderer instead of poppler/pdf2image

### 5. Dependency Changes

| Action | Package | Reason |
|-|-|-|
| Add | `pymupdf` | Text extraction + image rendering |
| Add | `docling` (optional) | Scanned PDF fallback |
| Remove | `pdf2image` | Replaced by PyMuPDF |

### 6. Settings Changes

```
# .env
JINA_RPM=100          # Requests per minute (free=100, paid=500, premium=5000)
JINA_TPM=100000       # Tokens per minute (free=100K, paid=2M, premium=50M)
JINA_MAX_CONCURRENT=5 # Max concurrent requests (existing)
```

### 7. Files Changed

- `ingestion/file_processor.py` — PyMuPDF text extraction + rendering, Docling fallback
- `ingestion/pipeline.py` — parallel page processing, hybrid text+image flow
- `ingestion/embedder.py` — RPM/TPM rate limiter
- `config/settings.py` — new rate limit settings
- `.env.example` — document new vars
- `pyproject.toml` — swap pdf2image → pymupdf, add docling
- `tests/test_file_processor.py` — new tests for hybrid extraction
