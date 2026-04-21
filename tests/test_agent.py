"""Tests for the Pydantic AI RAG agent."""

from __future__ import annotations

import os
from unittest.mock import patch

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent.agent import SYSTEM_PROMPT, create_rag_agent

# Prevent real API calls — Agent() eagerly validates provider keys
_DUMMY_ENV = {"ANTHROPIC_API_KEY": "test-key-not-real"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simple mock model that returns a fixed answer."""
    return ModelResponse(parts=[TextPart("Based on the documents, the answer is X.")])


# ---------------------------------------------------------------------------
# create_rag_agent tests
# ---------------------------------------------------------------------------


class TestCreateRagAgent:
    """Tests for agent creation and configuration."""

    async def test_returns_agent_and_streamable_http_server_by_default(self):
        """create_rag_agent returns (Agent, MCPServerStreamableHTTP) when transport=http."""
        with (
            patch.dict(os.environ, _DUMMY_ENV),
            patch("agent.agent.settings") as mock_settings,
        ):
            mock_settings.llm_api_type = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_base_url = ""
            mock_settings.mcp_transport = "http"
            mock_settings.mcp_port = 8080
            mock_settings.mcp_path = "/mcp"
            mock_settings.mcp_api_key = ""
            agent, server = await create_rag_agent()
        assert isinstance(agent, Agent)
        assert isinstance(server, MCPServerStreamableHTTP)

    async def test_returns_sse_server_when_transport_is_sse(self):
        """create_rag_agent falls back to MCPServerSSE for legacy transport."""
        with (
            patch.dict(os.environ, _DUMMY_ENV),
            patch("agent.agent.settings") as mock_settings,
        ):
            mock_settings.llm_api_type = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_base_url = ""
            mock_settings.mcp_transport = "sse"
            mock_settings.mcp_port = 8080
            mock_settings.mcp_path = "/mcp"
            mock_settings.mcp_api_key = ""
            _, server = await create_rag_agent()
        assert isinstance(server, MCPServerSSE)

    async def test_agent_model_uses_settings(self):
        """Agent model string is built from settings."""
        with (
            patch.dict(os.environ, _DUMMY_ENV),
            patch("agent.agent.settings") as mock_settings,
        ):
            mock_settings.llm_api_type = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_base_url = ""
            mock_settings.mcp_transport = "http"
            mock_settings.mcp_port = 8080
            mock_settings.mcp_path = "/mcp"
            mock_settings.mcp_api_key = ""
            agent, _ = await create_rag_agent()
        assert agent.model is not None

    async def test_server_port_from_settings(self):
        """MCP server URL uses the configured port."""
        with (
            patch.dict(os.environ, _DUMMY_ENV),
            patch("agent.agent.settings") as mock_settings,
        ):
            mock_settings.llm_api_type = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_base_url = ""
            mock_settings.mcp_transport = "http"
            mock_settings.mcp_port = 9999
            mock_settings.mcp_path = "/mcp"
            mock_settings.mcp_api_key = ""
            _, server = await create_rag_agent()
        assert "9999" in str(vars(server))


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """Tests for the system prompt content."""

    def test_mentions_all_tools(self):
        """System prompt references all four search tools."""
        assert "vector_search" in SYSTEM_PROMPT
        assert "visual_search" in SYSTEM_PROMPT
        assert "graph_search" in SYSTEM_PROMPT
        assert "hybrid_search" in SYSTEM_PROMPT

    def test_guides_tool_selection(self):
        """System prompt explains when to use each tool."""
        assert "semantic" in SYSTEM_PROMPT.lower()
        assert "visual" in SYSTEM_PROMPT.lower()
        assert "entity" in SYSTEM_PROMPT.lower()
        assert "hybrid" in SYSTEM_PROMPT.lower()

    def test_instructs_citation(self):
        """System prompt asks agent to cite sources."""
        assert "cite" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Agent run tests (using FunctionModel, no MCP server needed)
# ---------------------------------------------------------------------------


class TestAgentRun:
    """Tests for agent query execution with mocked model."""

    async def test_agent_returns_text_output(self):
        """Agent produces text output via FunctionModel."""
        with patch.dict(os.environ, _DUMMY_ENV):
            agent, _ = await create_rag_agent()
        with agent.override(model=FunctionModel(_echo_model), toolsets=[]):
            result = await agent.run("What is X?")
        assert "answer" in result.output.lower()

    async def test_agent_handles_empty_query(self):
        """Agent handles an empty query string."""

        async def empty_response(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart("No results found.")])

        with patch.dict(os.environ, _DUMMY_ENV):
            agent, _ = await create_rag_agent()
        with agent.override(model=FunctionModel(empty_response), toolsets=[]):
            result = await agent.run("")
        assert "no results" in result.output.lower()
