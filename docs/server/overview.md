# MCP Server Overview

Spektr exposes its RAG search capabilities to LLM agents through a [FastMCP](https://github.com/jlowin/fastmcp) server. The server registers six tools (search + introspection) — five always on, plus `visual_search` when `MULTIVEC_ENABLED=true` — and optionally enforces Bearer token authentication.

## Server Setup

The server is defined in `server/mcp_server.py`. A single `FastMCP` instance is created with the `BearerAuthMiddleware` attached, and each tool function is registered via `mcp.tool()`:

```python
from fastmcp import FastMCP

mcp = FastMCP(
    "rag-knowledge-base",
    middleware=[BearerAuthMiddleware()],
)
mcp.tool()(vector_search)
mcp.tool()(graph_search)
mcp.tool()(hybrid_search)
mcp.tool()(list_documents)
mcp.tool()(list_document_chunks)
# visual_search registered only when MULTIVEC_ENABLED=true
```

FastMCP inspects each function's signature and docstring to generate the MCP tool schema automatically. No manual schema definitions are needed.

## Registered Tools

| Tool | Always on? | Description |
|-|-|-|
| `vector_search` | yes | Dense semantic search via Qdrant; `session_id` aware |
| `graph_search` | yes | Engine-agnostic graph search (GLiNER2 or Graphiti) |
| `hybrid_search` | yes | Parallel vector + graph fusion with session separation |
| `list_documents` | yes | Enumerate ingested docs with chunk + page counts |
| `list_document_chunks` | yes | Exhaustive paginated enumeration of one document's chunks in `(page, chunk_index)` order |
| `visual_search` | needs `MULTIVEC_ENABLED=true` | ColBERT multi-vector search for visual content |

See [Search Tools](search-tools.md) for full parameter and return schema documentation. See [Client Setup](client-setup.md) for wiring the server into Claude Code, Cursor, or any other MCP-aware agent.

## Transport Options

The server supports three transports, controlled by the `MCP_TRANSPORT` environment variable:

| Transport | Default | Use Case |
|-|-|-|
| `http` | **Yes** | Streamable-HTTP — modern MCP default. Request/response per tool call, clean disconnects. Supported by Claude Code, Claude Desktop, Cursor, and `pydantic_ai.mcp.MCPServerStreamableHTTP`. |
| `sse` | No | Legacy Server-Sent-Events transport. Only use if a client is SSE-only. Emits noisy (but harmless) `ClosedResourceError` tracebacks on client disconnect — an upstream FastMCP/MCP-SDK quirk. |
| `stdio` | No | Subprocess transport. Host/port/path are ignored. Good for toolchains that spawn the server. |

The bind address (`MCP_HOST`, default `0.0.0.0`), port (`MCP_PORT`, default `8080`), and URL path (`MCP_PATH`, default `/mcp`) apply to `http` and `sse` transports only.

## Starting the Server

```bash
task serve
# equivalent to:
uv run python -m server.mcp_server
```

On startup the server logs the transport, endpoint URL, and list of registered tools:

```
MCP server starting on http http://0.0.0.0:8080/mcp with tools: vector_search,
graph_search, hybrid_search, list_documents, list_document_chunks
```

## Middleware Stack

The server includes one middleware by default:

- **BearerAuthMiddleware** -- Intercepts `tools/call` requests and validates the `Authorization: Bearer <key>` header against `MCP_API_KEY`. See [Authentication](authentication.md) for details.

Additional middleware can be appended to the `middleware` list in the `FastMCP` constructor.

## Client Integration

LLM agents connect to the server as an MCP tool provider. Two well-worn paths:

- **Pydantic AI** — `agent/agent.py` and `scripts/ask.py` talk to Spektr over streamable-http (`MCPServerStreamableHTTP`). For legacy SSE deployments, `MCPServerSSE` is a drop-in alternative. See [Agent Overview](../agent/overview.md).
- **Claude Code / Cursor / other MCP-aware IDEs** — drop a `.mcp.json` into the project root; Claude Code spawns or connects to the Spektr server and exposes its tools as `mcp__spektr__*`. See [Client Setup](client-setup.md) for stdio and http recipes.
