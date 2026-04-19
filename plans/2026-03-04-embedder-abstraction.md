# Embedder Abstraction Layer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hard-coded `JinaV4Embedder` with a protocol-based abstraction so embedding providers can be swapped via `EMBEDDING_PROVIDER` env var.

**Architecture:** `typing.Protocol` defines the embedder interface. A factory function in `ingestion/embedder.py` reads `settings.embedding_provider` and lazily imports the right provider from `ingestion/embedders/`. Each provider owns its own retry/rate-limit logic. Consumers use `create_embedder()` instead of `JinaV4Embedder()`.

**Tech Stack:** Python 3.13, httpx, tenacity, pydantic-settings, pytest

---

### Task 1: Create Protocol + factory in `ingestion/embedder.py`

**Files:**
- Modify: `ingestion/embedder.py`
- Create: `ingestion/embedders/__init__.py`
- Test: `tests/test_embedder.py`

**Step 1: Write the failing test**

Add to `tests/test_embedder.py`:

```python
from ingestion.embedder import Embedder, create_embedder


class TestEmbedderProtocol:
    def test_create_embedder_returns_protocol_compliant(self) -> None:
        """Factory returns an object satisfying the Embedder protocol."""
        embedder = create_embedder()
        assert hasattr(embedder, "embed_text")
        assert hasattr(embedder, "embed_text_query")
        assert hasattr(embedder, "embed_image")
        assert hasattr(embedder, "embed_multi_vector")
        assert hasattr(embedder, "embed_query_multi_vector")
        assert hasattr(embedder, "close")

    def test_create_embedder_unknown_provider_raises(self) -> None:
        """Unknown provider raises ValueError."""
        from unittest.mock import patch

        with patch("ingestion.embedder.settings") as mock_settings:
            mock_settings.embedding_provider = "nonexistent"
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                create_embedder()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedder.py::TestEmbedderProtocol -v`
Expected: FAIL — `ImportError: cannot import name 'Embedder'`

**Step 3: Rewrite `ingestion/embedder.py` — Protocol + factory only**

Replace entire file contents with:

```python
from __future__ import annotations

import asyncio
import time as _time
from typing import Protocol, runtime_checkable

from config.settings import settings


@runtime_checkable
class Embedder(Protocol):
    """Provider-agnostic embedding interface."""

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
    ) -> list[list[float]]: ...

    async def embed_text_query(
        self, query: str, dimensions: int | None = None,
    ) -> list[float]: ...

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[float]: ...

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[list[float]]: ...

    async def embed_query_multi_vector(
        self, query: str,
    ) -> list[list[float]]: ...

    async def close(self) -> None: ...


class TokenBucket:
    """Simple async token-bucket rate limiter.

    Not thread-safe — designed for use within a single asyncio event loop.
    """

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


def create_embedder() -> Embedder:
    """Factory: return the configured embedding provider."""
    provider = settings.embedding_provider
    if provider == "jina":
        from ingestion.embedders.jina import JinaV4Embedder

        return JinaV4Embedder()
    if provider == "voyage":
        from ingestion.embedders.voyage import VoyageEmbedder

        return VoyageEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")
```

Create empty `ingestion/embedders/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedder.py::TestEmbedderProtocol -v`
Expected: PASS (factory creates JinaV4Embedder from Task 2, so run after Task 2)

**Step 5: Commit**

```bash
git add ingestion/embedder.py ingestion/embedders/__init__.py tests/test_embedder.py
git commit -m "feat(embedder): add Embedder protocol and create_embedder factory"
```

**Note:** This task and Task 2 must be done together — the factory imports `JinaV4Embedder` from `ingestion/embedders/jina.py` which doesn't exist yet. Write both files, then run tests.

---

### Task 2: Move JinaV4Embedder to `ingestion/embedders/jina.py`

**Files:**
- Create: `ingestion/embedders/jina.py`
- Test: `tests/test_embedder.py` (existing tests must still pass)

**Step 1: Write the failing test**

The existing tests in `tests/test_embedder.py` import from `ingestion.embedder`. They need updating to import from the new location. Update the import at the top of the file:

```python
# Old:
from ingestion.embedder import JinaV4Embedder, _TokenBucket

# New:
from ingestion.embedder import TokenBucket, create_embedder
from ingestion.embedders.jina import JinaV4Embedder
```

Also update all `_TokenBucket` references to `TokenBucket`.

Run: `uv run pytest tests/test_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.embedders.jina'`

**Step 2: Create `ingestion/embedders/jina.py`**

Move the `JinaV4Embedder` class and `_is_retryable_status` helper from the old `embedder.py` into this file. Adapt `task` parameter to use generic names (`"passage"` / `"query"`) and map internally:

```python
from __future__ import annotations

import asyncio
import base64
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from ingestion.embedder import TokenBucket

logger = logging.getLogger(__name__)

_TASK_MAP = {
    "passage": "retrieval.passage",
    "query": "retrieval.query",
}


def _is_retryable_status(exc: BaseException) -> bool:
    """Return True for HTTP 429 and 5xx errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class JinaV4Embedder:
    """Jina v4 embedding client using a shared httpx.AsyncClient."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.jina_api_key
        self._embeddings_url = f"{settings.jina_api_url}/v1/embeddings"
        self._model = settings.jina_model
        self._dimensions = settings.jina_dense_dimensions
        self._max_concurrent = settings.jina_max_concurrent
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.jina_rpm / 60.0,
            burst=settings.jina_max_concurrent,
        )
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop_resources(self) -> None:
        """Recreate loop-bound async resources when the event loop changes."""
        loop = asyncio.get_running_loop()
        if self._bound_loop is not loop:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0),
            )
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._bound_loop = loop

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors."""
        payload = {
            "model": self._model,
            "task": _TASK_MAP.get(task, task),
            "dimensions": dimensions or self._dimensions,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": t} for t in texts],
        }
        data = await self._request(payload)
        return [item["embedding"] for item in data["data"]]

    async def embed_text_query(
        self, query: str, dimensions: int | None = None,
    ) -> list[float]:
        """Single query text -> dense vector."""
        results = await self.embed_text(
            [query], task="query", dimensions=dimensions,
        )
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[float]:
        """Single image -> dense vector."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model,
            "task": "retrieval.passage",
            "dimensions": self._dimensions,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"image": f"data:{media_type};base64,{b64}"}],
        }
        data = await self._request(payload, timeout=120.0)
        return data["data"][0]["embedding"]

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[list[float]]:
        """Single image -> ColBERT token vectors (128d each)."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model,
            "task": "retrieval.passage",
            "dimensions": 128,
            "normalized": True,
            "embedding_type": "float",
            "embedding_type_params": {"output_type": "colbert"},
            "input": [{"image": f"data:{media_type};base64,{b64}"}],
        }
        data = await self._request(payload, timeout=120.0)
        return data["data"][0]["embedding"]

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        """Single query text -> ColBERT token vectors (128d)."""
        payload = {
            "model": self._model,
            "task": "retrieval.query",
            "dimensions": 128,
            "normalized": True,
            "embedding_type": "float",
            "embedding_type_params": {"output_type": "colbert"},
            "input": [{"text": query}],
        }
        data = await self._request(payload)
        return data["data"][0]["embedding"]

    async def _request(
        self,
        payload: dict,
        timeout: float | None = None,
    ) -> dict:
        """Send request to Jina API with rate limiting + concurrency control."""
        self._ensure_loop_resources()
        await self._rpm_limiter.acquire()
        async with self._semaphore:
            return await self._request_with_retry(payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable_status),
        before_sleep=lambda rs: logger.warning(
            "Jina API retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception(),
        ),
    )
    async def _request_with_retry(
        self,
        payload: dict,
        timeout: float | None = None,
    ) -> dict:
        """Send request with retry on 429/5xx errors."""
        kwargs: dict = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self._embeddings_url, **kwargs)
        if resp.status_code != 200:
            exc = httpx.HTTPStatusError(
                f"Jina API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
            code = resp.status_code
            if code != 429 and code < 500:
                raise httpx.HTTPStatusError(
                    f"Jina API client error ({code}): {resp.text}",
                    request=resp.request,
                    response=resp,
                ) from exc
            raise exc
        return resp.json()
```

**Step 3: Run all existing embedder tests**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: PASS — all existing tests pass with updated imports.

**Step 4: Commit**

```bash
git add ingestion/embedders/jina.py tests/test_embedder.py
git commit -m "refactor(embedder): move JinaV4Embedder to ingestion/embedders/jina"
```

---

### Task 3: Add settings for provider selection + Voyage config

**Files:**
- Modify: `config/settings.py`
- Modify: `.env.example`

**Step 1: Write the failing test**

```python
# tests/test_embedder.py — add to TestEmbedderProtocol

def test_settings_has_embedding_provider(self) -> None:
    """Settings exposes embedding_provider field."""
    from config.settings import settings
    assert hasattr(settings, "embedding_provider")
    assert settings.embedding_provider in ("jina", "voyage")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedder.py::TestEmbedderProtocol::test_settings_has_embedding_provider -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'embedding_provider'`

**Step 3: Add settings fields**

In `config/settings.py`, add after the Jina section:

```python
    # Embedding provider selection
    embedding_provider: str = "jina"  # "jina" | "voyage"

    # Voyage (only needed when embedding_provider = "voyage")
    voyage_api_key: str = ""
    voyage_api_url: str = "https://api.voyageai.com"
    voyage_text_model: str = "voyage-4-large"
    voyage_multimodal_model: str = "voyage-multimodal-3.5"
    voyage_dense_dimensions: int = 1024
    voyage_rpm: int = 300
    voyage_max_concurrent: int = 10
```

In `.env.example`, add a new section after the Jina section:

```bash
# -----------------------------------------------------------------------------
# Embedding Provider
# -----------------------------------------------------------------------------
# Which embedding provider to use: "jina" or "voyage"
EMBEDDING_PROVIDER=jina                 # "jina" (default) or "voyage"

# -----------------------------------------------------------------------------
# Voyage AI Embeddings (only when EMBEDDING_PROVIDER=voyage)
# -----------------------------------------------------------------------------
# Get your API key at https://dash.voyageai.com/
VOYAGE_API_KEY=                         # [REQUIRED if voyage] Voyage API key
VOYAGE_API_URL=https://api.voyageai.com # Base URL
VOYAGE_TEXT_MODEL=voyage-4-large        # Text embedding model
VOYAGE_MULTIMODAL_MODEL=voyage-multimodal-3.5  # Image embedding model
VOYAGE_DENSE_DIMENSIONS=1024            # Output dimensions (256, 512, 1024, 2048)
VOYAGE_RPM=300                          # Requests per minute
VOYAGE_MAX_CONCURRENT=10                # Max parallel requests
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedder.py::TestEmbedderProtocol::test_settings_has_embedding_provider -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config/settings.py .env.example
git commit -m "feat(config): add embedding_provider setting and Voyage config"
```

---

### Task 4: Create VoyageEmbedder in `ingestion/embedders/voyage.py`

**Files:**
- Create: `ingestion/embedders/voyage.py`
- Test: `tests/test_voyage_embedder.py`

**Step 1: Write the failing tests**

Create `tests/test_voyage_embedder.py`:

```python
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


@pytest.fixture
def voyage_settings():
    """Patch settings to use Voyage provider."""
    with patch("ingestion.embedders.voyage.settings") as mock:
        mock.voyage_api_key = "test-voyage-key"
        mock.voyage_api_url = "https://api.voyageai.com"
        mock.voyage_text_model = "voyage-4-large"
        mock.voyage_multimodal_model = "voyage-multimodal-3.5"
        mock.voyage_dense_dimensions = 1024
        mock.voyage_rpm = 300
        mock.voyage_max_concurrent = 10
        mock.max_retries = 3
        yield mock


@pytest.fixture
async def embedder(voyage_settings):
    from ingestion.embedders.voyage import VoyageEmbedder

    e = VoyageEmbedder()
    e._ensure_loop_resources()
    return e


def _mock_response(data: list[dict]) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestVoyageEmbedText:
    async def test_correct_payload(self, embedder) -> None:
        mock_resp = _mock_response(
            [{"embedding": [0.1] * 1024}, {"embedding": [0.2] * 1024}],
        )
        with patch.object(
            embedder._client, "post",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "voyage-4-large"
        assert payload["input_type"] == "document"
        assert payload["input"] == ["hello", "world"]
        assert len(result) == 2
        assert len(result[0]) == 1024

    async def test_query_task_maps_correctly(self, embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 1024}])
        with patch.object(
            embedder._client, "post",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"], task="query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["input_type"] == "query"


class TestVoyageEmbedImage:
    async def test_correct_payload(self, embedder) -> None:
        image_bytes = b"\x89PNG\r\nfakedata"
        expected_b64 = base64.b64encode(image_bytes).decode()
        mock_resp = _mock_response([{"embedding": [0.1] * 1024}])
        with patch.object(
            embedder._client, "post",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_image(image_bytes)

        call_args = mock_post.call_args
        # Multimodal endpoint uses different URL
        assert "/v1/multimodalembeddings" in call_args.args[0]
        payload = call_args.kwargs["json"]
        assert payload["model"] == "voyage-multimodal-3.5"
        assert len(result) == 1024

    async def test_uses_multimodal_endpoint(self, embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 1024}])
        with patch.object(
            embedder._client, "post",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_image(b"fake")

        url = mock_post.call_args.args[0]
        assert url.endswith("/v1/multimodalembeddings")


class TestVoyageMultiVectorNotSupported:
    async def test_embed_multi_vector_raises(self, embedder) -> None:
        with pytest.raises(NotImplementedError):
            await embedder.embed_multi_vector(b"fake")

    async def test_embed_query_multi_vector_raises(self, embedder) -> None:
        with pytest.raises(NotImplementedError):
            await embedder.embed_query_multi_vector("query")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voyage_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.embedders.voyage'`

**Step 3: Create `ingestion/embedders/voyage.py`**

```python
from __future__ import annotations

import asyncio
import base64
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from ingestion.embedder import TokenBucket

logger = logging.getLogger(__name__)

_TASK_MAP = {
    "passage": "document",
    "query": "query",
}


def _is_retryable_status(exc: BaseException) -> bool:
    """Return True for HTTP 429 and 5xx errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class VoyageEmbedder:
    """Voyage AI embedding client using a shared httpx.AsyncClient."""

    def __init__(self) -> None:
        self._api_key = settings.voyage_api_key
        self._text_url = f"{settings.voyage_api_url}/v1/embeddings"
        self._multimodal_url = (
            f"{settings.voyage_api_url}/v1/multimodalembeddings"
        )
        self._text_model = settings.voyage_text_model
        self._multimodal_model = settings.voyage_multimodal_model
        self._dimensions = settings.voyage_dense_dimensions
        self._max_concurrent = settings.voyage_max_concurrent
        self._rpm_limiter = TokenBucket(
            tokens_per_sec=settings.voyage_rpm / 60.0,
            burst=settings.voyage_max_concurrent,
        )
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop_resources(self) -> None:
        """Recreate loop-bound async resources when the event loop changes."""
        loop = asyncio.get_running_loop()
        if self._bound_loop is not loop:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0),
            )
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._bound_loop = loop

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()

    async def embed_text(
        self,
        texts: list[str],
        task: str = "passage",
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """Batch text -> list of dense vectors."""
        payload = {
            "model": self._text_model,
            "input": texts,
            "input_type": _TASK_MAP.get(task, task),
            "output_dimensions": dimensions or self._dimensions,
        }
        data = await self._request(self._text_url, payload)
        return [item["embedding"] for item in data["data"]]

    async def embed_text_query(
        self, query: str, dimensions: int | None = None,
    ) -> list[float]:
        """Single query text -> dense vector."""
        results = await self.embed_text(
            [query], task="query", dimensions=dimensions,
        )
        return results[0]

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[float]:
        """Single image -> dense vector via multimodal endpoint."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._multimodal_model,
            "input": [[{"type": "image_base64", "image_base64": b64, "media_type": media_type}]],
            "input_type": "document",
            "output_dimensions": self._dimensions,
        }
        data = await self._request(self._multimodal_url, payload, timeout=120.0)
        return data["data"][0]["embedding"]

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[list[float]]:
        """Not supported by Voyage AI."""
        raise NotImplementedError(
            "Voyage AI does not support ColBERT multi-vector output"
        )

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        """Not supported by Voyage AI."""
        raise NotImplementedError(
            "Voyage AI does not support ColBERT multi-vector output"
        )

    async def _request(
        self,
        url: str,
        payload: dict,
        timeout: float | None = None,
    ) -> dict:
        """Send request with rate limiting + concurrency control."""
        self._ensure_loop_resources()
        await self._rpm_limiter.acquire()
        async with self._semaphore:
            return await self._request_with_retry(url, payload, timeout)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception(_is_retryable_status),
        before_sleep=lambda rs: logger.warning(
            "Voyage API retry attempt %d after %s",
            rs.attempt_number,
            rs.outcome.exception(),
        ),
    )
    async def _request_with_retry(
        self,
        url: str,
        payload: dict,
        timeout: float | None = None,
    ) -> dict:
        """Send request with retry on 429/5xx errors."""
        kwargs: dict = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(url, **kwargs)
        if resp.status_code != 200:
            exc = httpx.HTTPStatusError(
                f"Voyage API error: {resp.text}",
                request=resp.request,
                response=resp,
            )
            code = resp.status_code
            if code != 429 and code < 500:
                raise httpx.HTTPStatusError(
                    f"Voyage API client error ({code}): {resp.text}",
                    request=resp.request,
                    response=resp,
                ) from exc
            raise exc
        return resp.json()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_voyage_embedder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ingestion/embedders/voyage.py tests/test_voyage_embedder.py
git commit -m "feat(embedder): add VoyageEmbedder provider"
```

---

### Task 5: Update consumers to use `create_embedder()`

**Files:**
- Modify: `server/tools/vector_search.py`
- Modify: `server/tools/visual_search.py`
- Modify: `ingestion/pipeline.py`
- Rename: `ingestion/jina_cocoindex_ops.py` → `ingestion/cocoindex_ops.py`
- Modify: `tests/conftest.py`

**Step 1: Update `server/tools/vector_search.py`**

Change imports and type hints:

```python
# Old:
from ingestion.embedder import JinaV4Embedder
_embedder: JinaV4Embedder | None = None

def _get_embedder() -> JinaV4Embedder:
    global _embedder
    if _embedder is None:
        _embedder = JinaV4Embedder()
    return _embedder

# New:
from ingestion.embedder import Embedder, create_embedder
_embedder: Embedder | None = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder
```

**Step 2: Update `server/tools/visual_search.py`**

Same pattern — replace `JinaV4Embedder` with `Embedder` + `create_embedder`. Also update the module docstring to be provider-agnostic:

```python
# Old docstring:
"""ColBERT multi-vector visual search tool for MCP server.

Embeds a query via Jina v4 ColBERT and searches the Qdrant
multi-vector collection for visually similar document pages.
"""

# New:
"""Multi-vector visual search tool for MCP server.

Embeds a query and searches the Qdrant multi-vector collection
for visually similar document pages.
"""
```

Import changes same as vector_search.

**Step 3: Update `ingestion/pipeline.py`**

Change import and all type hints:

```python
# Old:
from ingestion.embedder import JinaV4Embedder

# In _build_page_tasks, _process_text_page, _process_visual_page:
embedder: JinaV4Embedder,

# In _process_all_pages:
embedder = JinaV4Embedder()

# New:
from ingestion.embedder import Embedder, create_embedder

# In _build_page_tasks, _process_text_page, _process_visual_page:
embedder: Embedder,

# In _process_all_pages:
embedder = create_embedder()
```

**Step 4: Rename `ingestion/jina_cocoindex_ops.py` → `ingestion/cocoindex_ops.py`**

```bash
git mv ingestion/jina_cocoindex_ops.py ingestion/cocoindex_ops.py
```

Then update its contents:

```python
# Old:
from ingestion.embedder import JinaV4Embedder

_embedder: JinaV4Embedder | None = None

def _get_embedder() -> JinaV4Embedder:
    global _embedder
    if _embedder is None:
        _embedder = JinaV4Embedder(api_key=settings.jina_api_key)
    return _embedder

@cocoindex.op.function()
def jina_embed_text(text: str) -> list[float]:

@cocoindex.op.function()
def jina_embed_image(image_bytes: bytes) -> list[float]:

@cocoindex.op.function()
def jina_embed_image_multivec(image_bytes: bytes) -> list[list[float]]:

# New:
from ingestion.embedder import Embedder, create_embedder

_embedder: Embedder | None = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder

@cocoindex.op.function()
def embed_text(text: str) -> list[float]:

@cocoindex.op.function()
def embed_image(image_bytes: bytes) -> list[float]:

@cocoindex.op.function()
def embed_image_multivec(image_bytes: bytes) -> list[list[float]]:
```

**Step 5: Update any imports of the renamed module**

Search for `jina_cocoindex_ops` and update to `cocoindex_ops`. Check `ingestion/pipeline.py` and any test files. Also rename the function references (`jina_embed_text` → `embed_text`, etc.).

**Step 6: Update `tests/conftest.py` docstring**

```python
# Old:
"""Mock JinaV4Embedder returning deterministic vectors."""

# New:
"""Mock embedder returning deterministic vectors."""
```

**Step 7: Run all tests**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: PASS

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor(embedder): update all consumers to use create_embedder factory"
```

---

### Task 6: Update test imports and verify full suite

**Files:**
- Modify: `tests/test_embedder.py` (update Jina-specific tests)
- Modify: `tests/test_tools.py` (verify mocks still work)
- Modify: `tests/test_jina_cocoindex_ops.py` (if exists — rename/update)

**Step 1: Check for any remaining `JinaV4Embedder` imports in tests**

Search all test files for `from ingestion.embedder import JinaV4Embedder` and update to `from ingestion.embedders.jina import JinaV4Embedder`.

**Step 2: Update `tests/test_embedder.py` fixture**

```python
# Old:
@pytest.fixture
async def embedder() -> JinaV4Embedder:
    e = JinaV4Embedder(api_key="test-key")
    e._ensure_loop_resources()
    return e

# New:
@pytest.fixture
async def embedder() -> JinaV4Embedder:
    from ingestion.embedders.jina import JinaV4Embedder
    e = JinaV4Embedder(api_key="test-key")
    e._ensure_loop_resources()
    return e
```

**Step 3: Check for `test_jina_cocoindex_ops.py`**

If it exists, rename to `test_cocoindex_ops.py` and update imports:

```bash
git mv tests/test_jina_cocoindex_ops.py tests/test_cocoindex_ops.py
```

Update imports from `ingestion.jina_cocoindex_ops` to `ingestion.cocoindex_ops` and function names from `jina_embed_text` to `embed_text`, etc.

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: ALL PASS

**Step 5: Run ruff to check formatting**

Run: `uv run ruff check ingestion/ server/ tests/ config/`
Run: `uv run ruff format ingestion/ server/ tests/ config/`

**Step 6: Commit**

```bash
git add -A
git commit -m "test(embedder): update all test imports for embedder abstraction"
```

---

### Task 7: Verify protocol compliance and backward compatibility

**Files:**
- Test: `tests/test_embedder.py`

**Step 1: Add protocol compliance test**

```python
class TestProtocolCompliance:
    def test_jina_satisfies_protocol(self) -> None:
        """JinaV4Embedder is a valid Embedder."""
        from ingestion.embedders.jina import JinaV4Embedder
        assert isinstance(JinaV4Embedder.__new__(JinaV4Embedder), Embedder)

    def test_voyage_satisfies_protocol(self) -> None:
        """VoyageEmbedder is a valid Embedder."""
        from ingestion.embedders.voyage import VoyageEmbedder
        assert isinstance(VoyageEmbedder.__new__(VoyageEmbedder), Embedder)

    def test_mock_embedder_satisfies_protocol(self) -> None:
        """The test mock_embedder fixture satisfies the protocol."""
        from unittest.mock import AsyncMock, MagicMock
        mock = MagicMock()
        mock.embed_text = AsyncMock()
        mock.embed_text_query = AsyncMock()
        mock.embed_image = AsyncMock()
        mock.embed_multi_vector = AsyncMock()
        mock.embed_query_multi_vector = AsyncMock()
        mock.close = AsyncMock()
        assert isinstance(mock, Embedder)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_embedder.py::TestProtocolCompliance -v`
Expected: PASS

**Step 3: Run full suite one final time**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/test_embedder.py
git commit -m "test(embedder): add protocol compliance verification"
```
