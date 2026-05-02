# Agent Overview

The Spektr agent is a **Pydantic AI** agent that connects to the
[MCP server](../server/overview.md) and exposes all
[search tools](../server/search-tools.md) as callable functions for any LLM.

## Creating the Agent

`create_rag_agent()` returns an `(Agent, MCPServer)` tuple, where
`MCPServer = MCPServerStreamableHTTP | MCPServerSSE`. The caller owns the
server lifecycle.

```python
from agent.agent import create_rag_agent

agent, server = await create_rag_agent()

async with server:
    result = await agent.run("What changed in the Q4 report?")
    print(result.output)
```

## MCP Server Connection

The transport is selected by `MCP_TRANSPORT`:

|-|-|
| `MCP_TRANSPORT` | Server class |
| `http` *(default)* / `streamable-http` | `MCPServerStreamableHTTP` |
| `sse` | `MCPServerSSE` |

The server URL is built from settings as
`http://localhost:{settings.mcp_port}{settings.mcp_path}`. By default
`MCP_PATH=/mcp`, so the agent connects to `http://localhost:8080/mcp`.

### MCP Authentication

When `MCP_API_KEY` is set, the agent injects an
`Authorization: Bearer <MCP_API_KEY>` header on every MCP request. If the
key is empty, no auth header is sent.

## System Prompt

The built-in system prompt instructs the LLM to pick the right search tool
for each query type:

| Query type | Tool | When to use |
|-|-|-|
| Semantic / keyword | `vector_search` | Find documents by meaning, keyword relevance, or topic |
| Visual / layout | `visual_search` | Find documents by visual content (ColBERT multi-vector) |
| Entity / relationship | `graph_search` | Find entities and connections in the knowledge graph |
| Complex / multi-faceted | `hybrid_search` | Parallel vector + graph search with result fusion |

The prompt also instructs the model to always cite source documents and to
acknowledge when no relevant results are found rather than guessing.

## Model Configuration

By default, the agent model string is built from two settings:

```
{settings.llm_api_type}:{settings.llm_model}
```

Set these via environment variables (see `.env.example`):

```bash
LLM_API_TYPE=openai        # or anthropic
LLM_MODEL=gpt-4o           # provider-specific model name
```

### Custom base URL (OpenAI-compatible)

When `LLM_BASE_URL` is set, the agent uses an **OpenAI-compatible** client
instead of the provider-specific string. This lets you point at any
OpenAI-compatible server (OpenRouter, Ollama, vLLM, LiteLLM, etc.):

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=anthropic/claude-sonnet-4-20250514
LLM_API_KEY=sk-or-...
```

## CLI

`scripts/ask.py` provides a one-shot command-line entry point that wires
the agent to the running MCP server and prints the answer:

```bash
task ask -- "What changed in the Q4 report?"
```

This requires `task serve` to be running so the MCP server is reachable.

## Testing

Override the model and toolsets to test agent logic without real LLM calls
or an MCP server:

```python
from pydantic_ai.models.function import FunctionModel

from agent.agent import create_rag_agent

agent, server = await create_rag_agent()

async def mock_model(messages, info):
    return "mocked answer"

with agent.override(model=FunctionModel(mock_model), toolsets=[]):
    result = await agent.run("test query")
    assert result.output == "mocked answer"
```

## Source

:   `agent/agent.py`, `scripts/ask.py`
