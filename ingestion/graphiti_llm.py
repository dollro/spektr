"""Size-aware LLM client for Graphiti.

Graphiti marks its high-volume maintenance calls -- per-node dedupe and
attribute extraction, per-edge resolution and invalidation -- with
`ModelSize.small`, meaning a cheaper model is adequate. `BaseOpenAIClient`
honours that via `_get_model_for_size`; `OpenAIGenericClient`, the only client
that talks to an arbitrary OpenAI-compatible endpoint and therefore the only
one usable with OpenRouter, does not. It records `model.size` as a tracing
attribute and then sends `self.model` regardless
(`openai_generic_client.py:124,157`).

On the 2026-08-30 baseline that meant 1,824 of 2,920 calls -- 62% -- ran on the
full model at full price.

The fix delegates to a second client rather than mutating `self.model` around a
`super()` call: `SEMAPHORE_LIMIT` is 10, so the instance is shared across
concurrent coroutines and swapping an attribute mid-flight is a data race.
"""

from __future__ import annotations

import typing

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel


class SizeAwareGenericClient(OpenAIGenericClient):
    """`OpenAIGenericClient` that honours `ModelSize.small`.

    With *small_model* empty there is no delegate and behaviour is identical to
    the base class.
    """

    def __init__(self, config: LLMConfig, small_model: str = "") -> None:
        super().__init__(config=config)
        self._delegate: OpenAIGenericClient | None = None
        if small_model:
            self._delegate = OpenAIGenericClient(
                config=LLMConfig(
                    api_key=config.api_key,
                    model=small_model,
                    base_url=config.base_url,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
            )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # The delegate's own `generate_response` is deliberately bypassed: the
        # outer client's already applied tracing and retry before reaching here.
        if model_size == ModelSize.small and self._delegate is not None:
            return await self._delegate._generate_response(
                messages,
                response_model,
                max_tokens=max_tokens,
                model_size=model_size,
            )
        return await super()._generate_response(
            messages,
            response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )
