# HTTP API

The agent layer includes an optional **FastAPI** HTTP endpoint that wraps the
Pydantic AI agent, making it accessible to any HTTP client. The API supports
both synchronous responses and streaming SSE.

## Starting the Server

```bash
uv run python -m agent.api
```

The server runs on **port 8001** by default.

## App Lifecycle

The FastAPI app uses a lifespan context manager to create the RAG agent and
hold the MCP server connection open for the entire application lifetime:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _agent, _server
    _agent, _server = await create_rag_agent()
    async with _server:
        yield
    _agent, _server = None, None
```

When the app starts, it connects to the [MCP server](../server/overview.md)
over SSE. When the app shuts down, the connection is closed automatically.

## Endpoints

| Method | Path | Description |
|-|-|-|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Run a query against the RAG agent |

### `GET /health`

Returns a simple health check response.

**Response:**

```json
{"status": "ok"}
```

### `POST /query`

Runs a natural-language query through the RAG agent, which selects and calls
the appropriate [search tools](../server/search-tools.md) via MCP.

**Request body — `QueryRequest`:**

| Field | Type | Default | Description |
|-|-|-|-|
| `query` | `str` | *(required)* | Natural-language question |
| `stream` | `bool` | `false` | Enable streaming SSE response |

**Response body — `QueryResponse`** (when `stream=false`):

| Field | Type | Description |
|-|-|-|
| `answer` | `str` | Agent-generated answer |
| `sources` | `list[dict]` | Source documents referenced (may be empty) |

#### Standard Request

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What entities are related to Project Alpha?"}'
```

```json
{
  "answer": "Project Alpha is connected to ...",
  "sources": []
}
```

#### Streaming Request

When `stream` is `true`, the endpoint returns a `text/event-stream` response.
Each chunk is sent as an SSE `data:` frame, and the stream ends with a
`[DONE]` sentinel:

```bash
curl -N -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the latest report", "stream": true}'
```

```
data: Project Alpha's latest
data:  report covers three
data:  main areas...
data: [DONE]
```

## CORS

CORS middleware is enabled for **all origins** (`*`), all methods, and all
headers — suitable for development. Restrict origins in production.

## Source

:   `agent/api.py`
