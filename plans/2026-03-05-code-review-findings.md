# Code Review Findings: Chunking Strategy + Retrieval Quality Plans

**Date:** 2026-03-05
**Reviewed against:**
- `plans/2026-03-05-chunking-strategy-design.md`
- `plans/2026-03-05-retrieval-quality-improvements.md`

**Overall:** Implementation is solid — all 7 tasks from the retrieval-quality plan have code and passing tests (66/66). The chunking-strategy design is partially implemented. Two high-priority gaps prevent the core chunking benefits from being active.

---

## Task Status

| Task | Status | Notes |
|-|-|-|
| 1. Image token estimation | Done | Tile-based `_estimate_image_tokens` in jina.py |
| 2. Docling HybridChunker | Partial | Code exists but unreachable for PDFs (see #1) |
| 3. Delete propagation | Done | `handle_file_delete` + backward-compat alias |
| 4. Token logging | Done | `tokens_used` property + pipeline logging |
| 5. Voyage+multivec validation | Done | `model_validator` in Settings |
| 6. VLM->Graphiti | Done | `_caption_and_ingest_visual` |
| 7. Dual-embed mixed pages | Partial | Always embeds images, no smart gating (see #2) |
| Late chunking (chunking plan) | Done | `late_chunking=True` wired in embed_text |
| Contextualized text (chunking plan) | Done | `contextualize()` + Graphiti uses it |

---

## Findings

### #1 [HIGH] `_pdf_to_pages` never sets `docling_document` — HybridChunker is dead code for PDFs

**Location:** `ingestion/file_processor.py:145-176`

`_pdf_to_pages` uses PyMuPDF directly and returns `FileProcessingResult(pages=pages)` without setting `docling_document`. The field defaults to `None`. In `ingest_file`, the check `if result.docling_document` is always falsy for PDFs, so `docling_chunk()` is never called for the primary use case.

The test `test_pdf_returns_docling_document` only asserts `hasattr(result, "docling_document")`, not that it's non-None, masking the gap.

**Impact:** The entire chunking-strategy plan's benefit (structure-aware chunking + late chunking for PDFs) is inactive. All PDFs fall back to `semantic_chunk()`.

**Fix options:**
1. Run Docling's `DocumentConverter` in `_pdf_to_pages` alongside PyMuPDF and pass the document through
2. Repurpose the existing `_classify_pdf_pages_docling` (which already runs Docling) to also return the Document object for chunking

---

### #2 [HIGH] Dual-embed doesn't gate on `has_visual_content` — every PDF page gets image-embedded

**Location:** `ingestion/pipeline.py:120-147`

The plan specifies `image_embed_strategy` ("smart" vs "all") and a `has_visual_content` flag on `Page`. Neither exists in the code. Current behavior:

```python
elif page.content_type == "pdf":
    if page.text.strip():
        tasks.text.append(...)
    tasks.image.append(...)  # unconditional — every PDF page
```

The `Page` dataclass has no `has_visual_content` field. Settings has no `image_embed_strategy`.

**Impact:** Text-only PDF pages get unnecessary image embeddings, wasting API tokens. The plan's `_store_page_thumbnail` fallback for non-visual pages is also missing.

---

### #3 [MEDIUM] `Embedder` protocol missing `late_chunking` parameter

**Location:** `ingestion/embedder.py:67-72`

The protocol's `embed_text` signature doesn't include `late_chunking: bool = False`. The chunking-design plan explicitly requires this. `VoyageEmbedder.embed_text` also lacks the parameter.

**Impact:** If `VoyageEmbedder` is used with late-chunking code paths, it raises `TypeError`. The pipeline passes `late_chunking=use_late_chunking` without checking provider.

**Fix:** Add `late_chunking: bool = False` to the `Embedder` protocol and to `VoyageEmbedder.embed_text` (ignored).

---

### #4 [MEDIUM] Graphiti edge invalidation is a no-op (in-memory only)

**Location:** `ingestion/pipeline.py:669-698`

```python
for edge in edges:
    if edge.source_description == s3_key:
        edge.expired_at = datetime.now(tz=UTC)  # sets Python attribute only
        invalidated += 1
```

Setting `expired_at` on a result object doesn't persist to Neo4j. No save/update call follows. The test only verifies `run_async` was called, not that edges are persisted.

**Impact:** Deleted files' knowledge graph edges remain active after deletion.

---

### #5 [MEDIUM] VLM caption creates a new API client per call

**Location:** `ingestion/pipeline.py:343-407`

`_caption_visual_page` instantiates `anthropic.AsyncAnthropic()` or `openai.AsyncOpenAI()` on every invocation. For multi-page PDFs with many visual pages, this creates many short-lived HTTP clients with connection overhead.

**Suggestion:** Create the client once per file processing session or use a module-level lazy singleton.

---

### #6 [LOW] `handle_file_delete` parameter still named `s3_key`

**Location:** `ingestion/pipeline.py:610`

```python
def handle_file_delete(s3_key: str) -> None:
```

The function was renamed from `handle_s3_delete` but the parameter name wasn't updated to `source_key` as the plan specifies.

---

### #7 [LOW] Test warnings: unawaited coroutines in dual-embed tests

**Location:** `tests/test_pipeline_dual_embed.py`

```
RuntimeWarning: coroutine '_process_visual_page' was never awaited
RuntimeWarning: coroutine '_process_text_page' was never awaited
```

Tests create coroutines via `_build_page_tasks` but never await them. Not harmful but noisy.

---

## Recommended Priority

1. **#1** — Wire `docling_document` through `_pdf_to_pages` (unlocks HybridChunker + late chunking)
2. **#3** — Add `late_chunking` to Embedder protocol + VoyageEmbedder (prevents TypeError)
3. **#4** — Fix Graphiti edge persistence in delete handler
4. **#2** — Add `has_visual_content` gating (or document intentional deviation)
5. **#5** — Share VLM client across calls
6. **#6, #7** — Minor cleanup
