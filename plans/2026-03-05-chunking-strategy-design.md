# Chunking Strategy Design: Docling HybridChunker + Jina Late Chunking

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace naive paragraph splitting with structure-aware Docling HybridChunker and add Jina late chunking for contextual embeddings. These are complementary — HybridChunker solves *where* to split (structural), late chunking solves *how* to embed (semantic context).

**Replaces:** Task 2 from `plans/2026-03-05-retrieval-quality-improvements.md`. All other tasks (1, 3-7) remain unchanged.

**Tech Stack:** Python 3.13, Docling (HybridChunker + DocumentConverter), Jina v4 API (`late_chunking` parameter), Qdrant, Graphiti/Neo4j

---

## Architecture

### Current Flow (Naive)
```
file_to_pages → page.text → semantic_chunk (split on \n\n, 512 char limit)
  → embed_text (each chunk independently) → Qdrant + Graphiti
```

### New Flow
```
file_to_pages → FileProcessingResult(pages, docling_document)
  → docling_chunk(docling_document) → TextChunks with contextualized_text
  → group chunks by page
  → embed_text(page_chunks, late_chunking=True) → contextual vectors
  → Qdrant upsert (store both raw + contextualized text)
  → Graphiti ingest (contextualized text for better entity resolution)
```

### Key Insight: Complementary, Not Competing

| Dimension | Docling HybridChunker | Jina Late Chunking |
|-|-|-|
| What it decides | Where to split the text | How to embed each split |
| Input | DoclingDocument (structured) | Pre-chunked text list |
| Output | Text chunks with metadata | Contextual embeddings |
| Context preserved | Structural (headings, sections) | Semantic (coreference, topic flow) |
| Problem solved | "This table got split mid-row" | "This chunk says 'the company' but which one?" |

---

## Component Design

### 1. HybridChunker Integration (`file_processor.py`)

**New function:** `docling_chunk(doc: DoclingDocument) -> list[TextChunk]`

- Uses `HybridChunker(tokenizer="jinaai/jina-embeddings-v4", max_tokens=256, merge_peers=True)`
- Calls `chunk.meta.headings` or `contextualize()` to get heading-prefixed text
- Extracts page numbers from chunk provenance (`chunk.meta.doc_items[0].prov[0].page_no`)

**TextChunk extension:**
```python
@dataclass
class TextChunk:
    text: str                          # raw chunk text
    chunk_index: int
    page_number: int
    contextualized_text: str | None = None  # heading-prefixed version
```

**Text usage split:**
- Embedding: raw `text` (late chunking provides cross-chunk context)
- Graphiti: `contextualized_text` (heading prefix aids entity resolution)
- Qdrant payload: stores both for flexibility

**Lazy initialization:** `_get_hybrid_chunker()` with module-level singleton. Falls back to `None` if Docling unavailable.

### 2. FileProcessingResult (`file_processor.py`)

```python
@dataclass
class FileProcessingResult:
    pages: list[Page]
    docling_document: object | None = None
```

- `_classify_pdf_pages_docling` returns `(visual_map, text_map, docling_document)` — third element is the Docling Document
- `_pdf_to_pages` returns `FileProcessingResult(pages=..., docling_document=doc)`
- `file_to_pages` returns `FileProcessingResult` for all file types

### 3. Non-PDF Document Support (`file_processor.py`)

For markdown/text files:
- Run `DocumentConverter` with lightweight pipeline (no OCR, no layout model) to parse into `DoclingDocument`
- Feed to `docling_chunk()` for structure-aware splitting
- If parsing fails → fall back to `semantic_chunk()`

### 4. Late Chunking (`jina.py`)

**API parameter:** `"late_chunking": true` on the `/v1/embeddings` payload.

Behavior: Jina concatenates all inputs, embeds as one sequence, then splits back. Returns one vector per input item.

**Changes to `embed_text()`:**
```python
async def embed_text(
    self,
    texts: list[str],
    task: str = "passage",
    late_chunking: bool = False,
) -> list[list[float]]:
```

- When `late_chunking=True`: add `"late_chunking": True` to payload
- Batching: when `late_chunking=True`, all texts in the call go in **one batch** (they must be in the same API call for context preservation). Standard batching disabled for late-chunking calls.
- **Constraint:** `late_chunking=True` cannot combine with `return_multivector=True` (Jina API limitation). Code enforces this.

**Embedder protocol update:**
- `embed_text()` gains `late_chunking: bool = False` parameter
- `VoyageEmbedder` ignores the parameter

### 5. Pipeline Changes (`pipeline.py`)

**`ingest_file`:**
```python
result = file_to_pages(filename, content)
pages = result.pages
dl_chunks = docling_chunk(result.docling_document) if result.docling_document else None
```

**`_process_text_page`:**
- Receives pre-computed `docling_chunks` for the whole document
- Filters to chunks matching `page_number`
- Calls `embed_text(page_texts, late_chunking=True)` for all chunks on the page
- Falls back to `semantic_chunk()` + non-late-chunking if no Docling chunks

**Graphiti ingestion:**
- Uses `chunk.contextualized_text` instead of `chunk.text`
- Heading prefix helps entity resolution: "the CTO" → "Leadership > Executive Team > the CTO"

---

## Error Handling

| Failure | Fallback | Logged |
|-|-|-|
| HybridChunker init fails | `semantic_chunk()` | Warning |
| Docling text/md parsing fails | `semantic_chunk()` | Warning |
| Late chunking API error | Retry without `late_chunking=True` | Warning |
| Page exceeds 32K tokens | Embed chunks individually | Info |
| `contextualize()` fails | Use raw `chunk.text` everywhere | Warning |

---

## Testing

**Unit tests (`test_file_processor.py`):**
- `TestDoclingChunk`: mocked Document → chunks with page numbers and contextualized text
- `TestDoclingChunk.test_returns_empty_for_none`: None → empty list
- `TestDoclingChunk.test_fallback_on_error`: chunker exception → empty list
- All `TestFileToPages` tests updated for `FileProcessingResult`

**Unit tests (`test_embedder.py`):**
- `TestLateCunking.test_payload_includes_late_chunking`: verify API payload has `"late_chunking": True`
- `TestLateChunking.test_single_batch_when_late_chunking`: all texts sent in one API call
- `TestLateChunking.test_no_late_chunking_by_default`: existing behavior preserved

**Integration tests (`test_pipeline.py`):**
- Mock Jina API, verify page-grouped chunks sent with `late_chunking=True`
- Verify Graphiti receives contextualized text

---

## Constraints & Notes

- **Jina API:** `late_chunking` is text-only. Cannot combine with `return_multivector`.
- **32K window:** A single page's chunks must fit in 32K tokens. Virtually always true.
- **TPM impact:** Fewer API calls (one per page instead of one per chunk) but same total tokens. Better for free-tier RPM limits.
- **Backward compat:** `semantic_chunk()` is preserved as fallback. No breaking changes.
- **Tokenizer:** `jinaai/jina-embeddings-v4` for exact token-budget alignment with the embedding model.

---

## Sources

- [Jina Embeddings v4 Model](https://jina.ai/models/jina-embeddings-v4/) — confirms late_chunking support
- [Jina Late Chunking Paper](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — technique explanation
- [Jina API OpenAPI Spec](https://api.jina.ai/openapi.json) — `late_chunking: boolean` parameter definition
- [Docling HybridChunker](https://github.com/docling-ai/docling) — structure-aware chunking
