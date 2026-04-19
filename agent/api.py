"""FastAPI HTTP endpoint wrapping the Pydantic AI RAG agent."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.agent import create_rag_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_agent = None
_server = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage MCP server connection across app lifetime."""
    from config.observability import setup_observability

    setup_observability()
    global _agent, _server  # noqa: PLW0603
    _agent, _server = await create_rag_agent()
    async with _server:
        logger.info("MCP server connected")
        yield
    _agent, _server = None, None


app = FastAPI(title="Spektr RAG Agent", lifespan=lifespan)

# Instrument eagerly — setup_observability in lifespan covers runtime spans,
# but FastAPI-level instrumentation hooks on the app object need the app
# to exist first.
from config.observability import instrument_fastapi  # noqa: E402

instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> StreamingResponse | QueryResponse:
    if _agent is None:
        raise RuntimeError("Agent not initialized")
    logger.info("Query: %s (stream=%s)", request.query, request.stream)

    if request.stream:
        return StreamingResponse(
            _stream_response(request.query),
            media_type="text/event-stream",
        )

    result = await _agent.run(request.query)
    return QueryResponse(answer=result.output, sources=[])


async def _stream_response(query: str) -> AsyncIterator[str]:
    """Yield SSE chunks from an agent streaming run."""
    if _agent is None:
        raise RuntimeError("Agent not initialized")
    async with _agent.run_stream(query) as stream:
        async for chunk in stream.stream_text(delta=True):
            yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    from config.logging import setup_logging

    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=8001)
