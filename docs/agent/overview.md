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

| Tool | When to use |
|-|-|
| `multi_search` | Fused keyword + semantic search, reranked. Fast and cheap — the default choice. |
| `hybrid_search` | Same fused pipeline as `multi_search`, plus it splits multi-part questions into sub-queries and retries when results look weak. Use for complex or compound questions. Slower, costs one extra model call. |
| `vector_search` | Semantic-only search. Use when you specifically want conceptual similarity without keyword matching. |
| `visual_search` | Finds pages by visual layout — charts, diagrams, tables. |
| `graph_search` | Entity and relationship facts from the knowledge graph. |

The prompt also explains `multi_search`/`hybrid_search`'s shared response shape — `results` (ranked chunks, with `channels` showing whether a hit came from keyword matching, semantic similarity, or both), `graph_facts` (supporting entity facts, not ranked against the chunks), `live_results` (active-session chunks, when a session is set), and `degraded` (present only when part of the pipeline failed — results are still usable but less complete than normal). See [Search Tools](../server/search-tools.md) for the full schema.

The prompt also instructs the model to always cite source documents and to
acknowledge when no relevant results are found rather than guessing.

Source: `SYSTEM_PROMPT` in `agent/agent.py`.

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
LLM_MODEL=anthropic/claude-sonnet-5
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
