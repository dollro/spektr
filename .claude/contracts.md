# Integration Contracts — Spektr

These contracts define the exact interfaces between modules. Teammates MUST implement these signatures exactly. Deviation requires lead approval.

---

## Phases 1–3 Contracts (Implemented)

### Config Module (Contract 1)
```python
# config/settings.py — Settings(BaseSettings) with Jina, Qdrant, Neo4j, PostgreSQL, AWS, LLM, MCP sections
# config/constants.py — DENSE_COLLECTION, MULTIVEC_COLLECTION, DENSE_DIM=2048, MULTIVEC_DIM=128
```

### JinaV4Embedder (Contract 2)
```python
# ingestion/embedder.py
class JinaV4Embedder:
    async def embed_text(texts, task="retrieval.passage", dimensions=2048) -> list[list[float]]
    async def embed_text_query(query, dimensions=2048) -> list[float]
    async def embed_image(image_bytes, media_type="image/png") -> list[float]
    async def embed_multi_vector(image_bytes, media_type="image/png") -> list[list[float]]
    async def embed_query_multi_vector(query) -> list[list[float]]
    async def close() -> None
```

### File Processor (Contract 3)
```python
# ingestion/file_processor.py — Page, TextChunk dataclasses
# file_to_pages(filename, content) -> list[Page]
# semantic_chunk(text, max_chunk_size=512) -> list[TextChunk]
```

### Entity Extractor (Contract 4)
```python
# ingestion/entity_extractor.py
# Entity, Relationship, ExtractionResult models
# extract_entities(text, llm_client) -> ExtractionResult
# LLMClient protocol, AnthropicClient, OpenAIClient, get_llm_client()
```

### Graph Writer (Contract 5)
```python
# ingestion/graph_writer.py — GraphWriter class with upsert_* and write_extraction_result
```

### MCP Server + Tools (Contracts 9-15)
```python
# server/mcp_server.py — FastMCP("rag-knowledge-base"), 4 tools registered
# server/tools/vector_search.py — async def vector_search(query, limit=10, content_type=None, source_file=None) -> list[dict]
# server/tools/visual_search.py — async def visual_search(query, limit=5) -> list[dict]
# server/tools/graph_search.py — async def graph_search(query, search_type="entity", limit=10) -> list[dict]
# server/tools/hybrid_search.py — async def hybrid_search(query, limit=10) -> dict
# server/models.py — SearchResult, VisualSearchResult, GraphEntity, GraphPath, HybridSearchResponse
# server/providers.py — re-exports LLMClient, get_llm_client from entity_extractor
```

---

## Phases 4–5 Contracts (Current Build)

### Frozen: MCP Tool Signatures

These signatures MUST NOT change. All agents build against them.

```python
async def vector_search(query: str, limit: int = 10, content_type: str | None = None, source_file: str | None = None) -> list[dict]
# Returns: [{"score": float, "text": str, "source_file": str, "page_number": int, "content_type": str, "metadata": dict}]

async def visual_search(query: str, limit: int = 5) -> list[dict]
# Returns: [{"score": float, "source_file": str, "page_number": int, "content_type": str, "source_key": str, "metadata": dict}]

async def graph_search(query: str, search_type: str = "entity", limit: int = 10) -> list[dict]
# entity: [{"entity": str, "type": str, "description": str, "connections": list, "source_documents": list}]
# path: [{"path": list, "relationships": list, "hop_count": int}]

async def hybrid_search(query: str, limit: int = 10) -> dict
# {"vector_results": list, "graph_results": list, "query": str, "strategy": "parallel"}
```

### Contract A: Agent Module (produced by agent-dev)

```python
# agent/agent.py
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

SYSTEM_PROMPT = """You are a RAG assistant with access to search tools.
- vector_search: text/semantic questions
- visual_search: visual/layout/image questions
- graph_search: relationship/entity questions
- hybrid_search: complex multi-faceted questions"""

async def create_rag_agent() -> tuple[Agent, MCPServerHTTP]:
    """Create agent connected to MCP server. Returns (agent, server)."""

# agent/api.py — FastAPI
# POST /query {"query": str, "stream": bool} -> {"answer": str, "sources": list[dict]}
# GET /health -> {"status": "ok"}
```

### Contract B: Settings Ownership

**resilience-dev** adds to `config/settings.py` (after `# MCP` section):
```python
    # Resilience
    jina_max_concurrent: int = 5
    extraction_timeout: int = 30
    tool_timeout: int = 30
    max_retries: int = 3
    rerank_enabled: bool = False
    vlm_generation_enabled: bool = False
```

**resilience-dev** adds `"tenacity"` to `pyproject.toml` dependencies.

**observability-dev** adds to `config/settings.py` (after `# Resilience` section, AFTER resilience-dev completes):
```python
    # Observability
    log_level: str = "INFO"
    log_format: str = "json"
```

**agent-dev** does NOT modify `config/settings.py` or `pyproject.toml`.

### Contract C: Logging Interface (produced by observability-dev)

```python
# config/logging.py
import logging

def setup_logging() -> None:
    """Configure root logger from settings."""

def get_logger(name: str) -> logging.Logger:
    """Return named logger."""
```

### Contract D: Error Response Format (produced by resilience-dev)

Tools return structured errors instead of raising:
```python
# Error case — tool returns dict with "error" key
{"error": "Search failed: connection timeout", "query": "...", "partial_results": []}

# hybrid_search partial failure:
{"vector_results": [...], "graph_results": [], "query": "...", "strategy": "parallel", "errors": ["graph_search: connection refused"]}
```

### File Ownership Matrix

| File | agent-dev | resilience-dev | observability-dev |
|-|-|-|-|
| `agent/agent.py` | CREATE | - | - |
| `agent/api.py` | CREATE | - | - |
| `tests/test_agent.py` | CREATE | - | - |
| `tests/test_e2e.py` | CREATE | - | - |
| `tests/conftest.py` | MODIFY | - | - |
| `config/settings.py` | - | MODIFY (1st) | MODIFY (2nd) |
| `config/logging.py` | - | - | CREATE |
| `pyproject.toml` | - | MODIFY | - |
| `ingestion/embedder.py` | - | MODIFY | - |
| `ingestion/entity_extractor.py` | - | MODIFY | - |
| `ingestion/graph_writer.py` | - | MODIFY | - |
| `server/tools/vector_search.py` | - | MODIFY | - |
| `server/tools/visual_search.py` | - | MODIFY | - |
| `server/tools/graph_search.py` | - | MODIFY | - |
| `server/tools/hybrid_search.py` | - | MODIFY | - |
| `server/tools/reranker.py` | - | CREATE | - |
| `server/tools/vlm_generator.py` | - | CREATE | - |
| `ingestion/pipeline.py` | - | - | MODIFY |
| `server/mcp_server.py` | - | - | MODIFY |
