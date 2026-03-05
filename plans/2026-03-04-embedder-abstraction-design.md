# Embedder Abstraction Layer — Design

## Goal

Replace the hard-coded `JinaV4Embedder` with a provider-agnostic abstraction so
embedding providers (Jina, Voyage, future additions) can be swapped via config.

## Approach

**Protocol-based (duck typing).** A `typing.Protocol` defines the interface.
Each provider implements it independently. A factory function reads
`settings.embedding_provider` and returns the right instance.

No base class, no shared infrastructure. Retry/rate-limit logic stays in each
provider — it's ~30 lines and not worth abstracting until 3+ providers exist.

## File Structure

```
ingestion/
├── embedder.py              # Protocol + factory + _TokenBucket utility
├── embedders/
│   ├── __init__.py
│   ├── jina.py              # JinaV4Embedder (moved from embedder.py)
│   └── voyage.py            # VoyageEmbedder (new)
```

`embedder.py` remains the public import path. Consumers use
`from ingestion.embedder import Embedder, create_embedder`.

## Protocol

```python
class Embedder(Protocol):
    async def embed_text(
        self, texts: list[str], task: str = "passage", dimensions: int = ...,
    ) -> list[list[float]]: ...

    async def embed_text_query(self, query: str, dimensions: int = ...) -> list[float]: ...

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[float]: ...

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png",
    ) -> list[list[float]]: ...

    async def embed_query_multi_vector(self, query: str) -> list[list[float]]: ...

    async def close(self) -> None: ...
```

- `task` uses generic values: `"passage"` / `"query"`. Each provider maps internally
  (Jina → `"retrieval.passage"`, Voyage → `"document"`).
- `dimensions` defaults come from settings, not hardcoded.

## Factory

```python
def create_embedder() -> Embedder:
    provider = settings.embedding_provider
    if provider == "jina":
        from ingestion.embedders.jina import JinaV4Embedder
        return JinaV4Embedder()
    elif provider == "voyage":
        from ingestion.embedders.voyage import VoyageEmbedder
        return VoyageEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")
```

Lazy imports so unused providers don't require their config/deps.

## Settings

New fields in `config/settings.py`:

```python
# Embedding provider
embedding_provider: str = "jina"  # "jina" | "voyage"

# Voyage
voyage_api_key: str = ""
voyage_api_url: str = "https://api.voyageai.com"
voyage_text_model: str = "voyage-4-large"
voyage_multimodal_model: str = "voyage-multimodal-3.5"
voyage_dense_dimensions: int = 1024
voyage_rpm: int = 300
voyage_max_concurrent: int = 10
```

## Consumer Changes

4 call sites switch from `JinaV4Embedder()` to `create_embedder()`:

| File | Change |
|-|-|
| `server/tools/vector_search.py` | Type hint `Embedder`, use `create_embedder()` |
| `server/tools/visual_search.py` | Same |
| `ingestion/pipeline.py` | Same |
| `ingestion/jina_cocoindex_ops.py` | Rename to `cocoindex_ops.py`, use `create_embedder()` |

No changes to: Qdrant setup, search logic, pipeline structure, MCP server.

## Testing

Mock against the `Embedder` protocol. Existing mocks work since method
signatures are preserved.
