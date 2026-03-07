from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, Protocol

from pydantic import BaseModel, field_validator

from config.constants import RELATIONSHIP_TYPES
from config.settings import settings

logger = logging.getLogger(__name__)


EntityType = Literal[
    "person",
    "organization",
    "technology",
    "concept",
    "metric",
    "location",
    "event",
]


class Entity(BaseModel):
    name: str
    type: EntityType
    description: str


class Relationship(BaseModel):
    source: str
    target: str
    relation: str
    properties: dict[str, Any] = {}

    @field_validator("relation")
    @classmethod
    def warn_unknown_relation(cls, v: str) -> str:
        if v not in RELATIONSHIP_TYPES:
            logger.warning("Unknown relationship type: %s", v)
        return v


class ExtractionResult(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> str: ...


class AnthropicClient:
    """LLM client for Anthropic Claude models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=messages,
        )
        return response.content[0].text


class OpenAIClient:
    """LLM client for OpenAI models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
    ) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    """Factory: create LLM client based on settings."""
    provider = settings.llm_api_type.lower()
    if provider == "anthropic":
        return AnthropicClient(model=settings.llm_model)
    if provider == "openai":
        return OpenAIClient(model=settings.llm_model)
    msg = f"Unsupported LLM provider: {provider}"
    raise ValueError(msg)


EXTRACTION_PROMPT = """\
Extract entities and relationships from the following text.

Return JSON with this exact structure:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", \
"relation": "...", "properties": {{}}}}
  ]
}}

Rules:
- Entity types MUST be one of: person, organization, technology, \
concept, metric, location, event
- Relationship types should be snake_case (e.g. \
created_by, uses, part_of, mentions, improves, measured_by, located_in, describes)
- Normalize entity names (strip whitespace, title case)
- Only extract clearly stated relationships, don't infer
- Keep descriptions brief (1 sentence max)

Text:
{text}
"""

MAX_RETRIES = 2


def _normalize_entity_name(name: str) -> str:
    """Strip whitespace and apply title case."""
    return name.strip().title()


def _parse_extraction(raw: str) -> ExtractionResult:
    """Parse LLM response JSON into ExtractionResult."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()

    data = json.loads(text)
    result = ExtractionResult.model_validate(data)

    # Normalize entity names
    for entity in result.entities:
        entity.name = _normalize_entity_name(entity.name)
    for rel in result.relationships:
        rel.source = _normalize_entity_name(rel.source)
        rel.target = _normalize_entity_name(rel.target)

    return result


async def extract_entities(text: str, llm_client: LLMClient) -> ExtractionResult:
    """Extract entities and relationships from text via LLM.

    Retries once on parse failure. Returns empty result after
    max retries. Applies asyncio.timeout for overall time bound.
    """
    prompt = EXTRACTION_PROMPT.format(text=text)
    messages = [{"role": "user", "content": prompt}]

    last_error: Exception | None = None
    try:
        async with asyncio.timeout(settings.extraction_timeout):
            for attempt in range(MAX_RETRIES):
                try:
                    raw = await llm_client.chat(messages)
                    return _parse_extraction(raw)
                except (
                    json.JSONDecodeError,
                    ValueError,
                    KeyError,
                ) as exc:
                    last_error = exc
                    logger.warning(
                        "Entity extraction parse failure (attempt %d): %s",
                        attempt + 1,
                        exc,
                    )
                    if attempt == 0:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": raw,
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your response was not "
                                    f"valid JSON: {exc}. "
                                    "Please return only valid "
                                    "JSON matching the schema "
                                    "above."
                                ),
                            }
                        )
    except TimeoutError:
        logger.warning(
            "Entity extraction timed out after %ds",
            settings.extraction_timeout,
        )
        return ExtractionResult()

    logger.warning(
        "Entity extraction failed after %d attempts: %s",
        MAX_RETRIES,
        last_error,
    )
    return ExtractionResult()
