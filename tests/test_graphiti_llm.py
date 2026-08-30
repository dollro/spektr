from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message

from ingestion.graphiti_llm import SizeAwareGenericClient


def _config() -> LLMConfig:
    return LLMConfig(api_key="k", model="big/model", base_url="http://endpoint")


def test_delegate_uses_small_model_on_the_same_endpoint() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="small/model")
    assert client.model == "big/model"
    assert client._delegate is not None
    assert client._delegate.model == "small/model"


def test_no_delegate_when_small_model_is_empty() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="")
    assert client._delegate is None


@pytest.mark.asyncio
async def test_small_calls_go_to_the_delegate() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="small/model")
    assert client._delegate is not None
    client._delegate._generate_response = AsyncMock(  # type: ignore[method-assign]
        return_value={"routed": "small"}
    )

    with patch.object(
        OpenAIGenericClient,
        "_generate_response",
        new=AsyncMock(return_value={"routed": "big"}),
    ):
        result = await client._generate_response(
            [Message(role="user", content="hi")],
            model_size=ModelSize.small,
        )

    assert result == {"routed": "small"}
    client._delegate._generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_medium_calls_go_to_the_main_model() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="small/model")
    assert client._delegate is not None
    client._delegate._generate_response = AsyncMock(  # type: ignore[method-assign]
        return_value={"routed": "small"}
    )

    with patch.object(
        OpenAIGenericClient,
        "_generate_response",
        new=AsyncMock(return_value={"routed": "big"}),
    ):
        result = await client._generate_response(
            [Message(role="user", content="hi")],
            model_size=ModelSize.medium,
        )

    assert result == {"routed": "big"}
    client._delegate._generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_small_calls_fall_back_to_main_model_when_unconfigured() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="")

    with patch.object(
        OpenAIGenericClient,
        "_generate_response",
        new=AsyncMock(return_value={"routed": "big"}),
    ):
        result = await client._generate_response(
            [Message(role="user", content="hi")],
            model_size=ModelSize.small,
        )

    assert result == {"routed": "big"}
