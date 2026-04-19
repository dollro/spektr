# Implementation Plan: Medium-Priority Code Review Fixes

## Overview

Three targeted fixes from the 2026-03-05 code review. All are independent and can be implemented in parallel.

## Task Registry

| ID | Task | Depends On | Parallel With | Parallel Safe | Effort |
|-|-|-|-|-|-|
| 1.1 | Add `late_chunking` to Embedder protocol + VoyageEmbedder | -- | 1.2, 1.3 | yes, different files | S |
| 1.2 | Persist Graphiti edge invalidation to Neo4j | -- | 1.1, 1.3 | yes, different files | S |
| 1.3 | Singleton VLM API client | -- | 1.1, 1.2 | yes, different code section | S |

---

## Task 1.1: Add `late_chunking` to Embedder protocol and VoyageEmbedder

**Finding:** #3 -- `Embedder` protocol missing `late_chunking` parameter

**Files:**
- `ingestion/embedder.py` (lines 67-72)
- `ingestion/embedders/voyage.py` (lines 80-85)

**Changes:**

1. In `ingestion/embedder.py`, add `late_chunking: bool = False` to the `embed_text` protocol method:
   ```python
   async def embed_text(
       self,
       texts: list[str],
       task: str = "passage",
       dimensions: int | None = None,
       late_chunking: bool = False,
   ) -> list[list[float]]: ...
   ```

2. In `ingestion/embedders/voyage.py`, add the same parameter to `embed_text` (ignored, since Voyage does not support late chunking):
   ```python
   async def embed_text(
       self,
       texts: list[str],
       task: str = "passage",
       dimensions: int | None = None,
       late_chunking: bool = False,  # noqa: ARG002 -- Voyage does not support late chunking
   ) -> list[list[float]]:
   ```
   No behavioral change needed -- the parameter is accepted and silently ignored.

**Acceptance criteria:**
- `VoyageEmbedder` satisfies the `Embedder` protocol (no mypy errors)
- Calling `VoyageEmbedder.embed_text(texts, late_chunking=True)` does not raise TypeError
- Existing JinaV4Embedder still works (already has the param)

---

## Task 1.2: Persist Graphiti edge invalidation to Neo4j

**Finding:** #4 -- `handle_file_delete` sets `expired_at` in memory only

**File:** `ingestion/pipeline.py` (lines 669-698)

**Root cause:** The `_invalidate_graph` inner function sets `edge.expired_at` on the Python object but never calls `edge.save(driver)` to persist it to Neo4j.

**Graphiti API:** `EntityEdge` has a `save(driver)` method that persists all fields including `expired_at` (confirmed in `graphiti_core/edges.py` line 330-367). The driver is accessible via `client.driver`.

**Changes:**

Replace the loop body in `_invalidate_graph()` (pipeline.py ~line 680-683) with:

```python
for edge in edges:
    if edge.source_description == s3_key:
        edge.expired_at = datetime.now(tz=UTC)
        await edge.save(client.driver)
        invalidated += 1
```

Note: `client.search()` returns `EntityEdge` objects. The `save()` call issues a Cypher MERGE that persists the updated `expired_at` timestamp.

**Edge case:** The `search` method returns edges matching the query text, not filtering by `source_description`. The existing `if edge.source_description == s3_key` guard is correct -- only edges whose `source_description` matches the deleted file key get invalidated.

**Acceptance criteria:**
- After calling `handle_file_delete("some/file.pdf")`, matching edges in Neo4j have `expired_at` set
- Unit test mocks `edge.save()` and asserts it was called with `client.driver`

---

## Task 1.3: Singleton VLM API client

**Finding:** #5 -- `_caption_visual_page` creates a new API client per call

**File:** `ingestion/pipeline.py` (lines 343-407)

**Changes:**

Add a module-level lazy singleton pattern above `_caption_visual_page`:

```python
_vlm_client_anthropic: anthropic.AsyncAnthropic | None = None
_vlm_client_openai: openai.AsyncOpenAI | None = None


def _get_vlm_client() -> anthropic.AsyncAnthropic | openai.AsyncOpenAI:
    """Return a lazily-initialized VLM API client (singleton)."""
    provider = settings.llm_api_type.lower()
    if provider == "anthropic":
        global _vlm_client_anthropic  # noqa: PLW0603
        if _vlm_client_anthropic is None:
            import anthropic

            _vlm_client_anthropic = anthropic.AsyncAnthropic(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
            )
        return _vlm_client_anthropic
    else:
        global _vlm_client_openai  # noqa: PLW0603
        if _vlm_client_openai is None:
            import openai

            _vlm_client_openai = openai.AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
            )
        return _vlm_client_openai
```

Then update `_caption_visual_page` to call `_get_vlm_client()` instead of constructing clients inline. Remove the `import anthropic` / `import openai` and `client = ...` lines from inside the function body, replacing with `client = _get_vlm_client()`.

Since the imports are type-guarded by the provider check, add the imports at module level behind `TYPE_CHECKING` for type annotations, and keep runtime imports inside `_get_vlm_client()`.

**Acceptance criteria:**
- Processing a 10-page PDF with VLM enabled creates exactly 1 API client instance
- Existing VLM caption tests pass
- No behavioral change in caption output

---

## Testing

All three fixes should have unit tests added or updated:
- **1.1**: Add a test in `tests/test_embedder.py` that `VoyageEmbedder.embed_text(..., late_chunking=True)` does not raise
- **1.2**: Update the mock in the file-delete test to assert `edge.save(client.driver)` is awaited
- **1.3**: Add a test that calls `_caption_visual_page` twice and asserts the client constructor is called only once
