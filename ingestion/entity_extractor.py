from __future__ import annotations

import json
import logging
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

EntityType = Literal[
    "PERSON",
    "ORGANIZATION",
    "PRODUCT",
    "TECHNOLOGY",
    "LOCATION",
    "CONCEPT",
    "EVENT",
]
RelationType = Literal[
    "WORKS_AT",
    "PARTNERS_WITH",
    "PRODUCES",
    "USES_TECHNOLOGY",
    "LOCATED_IN",
    "ACQUIRED",
    "COMPETES_WITH",
    "REFERENCES",
]


class Entity(BaseModel):
    name: str
    type: EntityType
    description: str


class Relationship(BaseModel):
    source: str
    target: str
    relation: RelationType
    properties: dict[str, Any] = {}


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
    """Factory: create LLM client based on settings.llm_provider."""
    provider = settings.llm_provider.lower()
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
    {{"name": "...", "type": "PERSON|ORGANIZATION|PRODUCT|TECHNOLOGY|\
LOCATION|CONCEPT|EVENT", "description": "..."}}
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", \
"relation": "WORKS_AT|PARTNERS_WITH|PRODUCES|USES_TECHNOLOGY|\
LOCATED_IN|ACQUIRED|COMPETES_WITH|REFERENCES", "properties": {{}}}}
  ]
}}

Rules:
- Normalize entity names (strip whitespace, title case)
- Only extract clearly stated relationships, don't infer
- Keep descriptions brief (1 sentence max)
- Use CONCEPT type for abstract topics, methodologies, standards

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
    max retries.
    """
    prompt = EXTRACTION_PROMPT.format(text=text)
    messages = [{"role": "user", "content": prompt}]

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = await llm_client.chat(messages)
            return _parse_extraction(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Entity extraction parse failure (attempt %d): %s",
                attempt + 1,
                exc,
            )
            # On first failure, add error feedback for retry
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your response was not valid JSON: {exc}. "
                            "Please return only valid JSON matching the "
                            "schema above."
                        ),
                    }
                )

    logger.warning(
        "Entity extraction failed after %d attempts: %s",
        MAX_RETRIES,
        last_error,
    )
    return ExtractionResult()
