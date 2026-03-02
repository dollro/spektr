from __future__ import annotations

import json

import pytest

from ingestion.entity_extractor import (
    ExtractionResult,
    _normalize_entity_name,
    _parse_extraction,
    extract_entities,
)


class MockLLMClient:
    """Mock LLM client returning pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> str:
        self.calls.append(messages)
        return self._responses.pop(0)


VALID_JSON = json.dumps(
    {
        "entities": [
            {
                "name": "Google",
                "type": "ORGANIZATION",
                "description": "Tech company.",
            },
            {
                "name": "Python",
                "type": "TECHNOLOGY",
                "description": "Programming language.",
            },
        ],
        "relationships": [
            {
                "source": "Google",
                "target": "Python",
                "relation": "USES_TECHNOLOGY",
                "properties": {},
            },
        ],
    }
)


async def test_extract_valid_json() -> None:
    """Valid JSON response is parsed into ExtractionResult."""
    client = MockLLMClient([VALID_JSON])
    result = await extract_entities("Google uses Python.", client)

    assert len(result.entities) == 2
    assert result.entities[0].name == "Google"
    assert result.entities[0].type == "ORGANIZATION"
    assert len(result.relationships) == 1
    assert result.relationships[0].relation == "USES_TECHNOLOGY"


async def test_extract_malformed_json_retries() -> None:
    """Malformed JSON triggers retry; second failure returns empty."""
    client = MockLLMClient(["not json", "still not json"])
    result = await extract_entities("Some text", client)

    assert result == ExtractionResult()
    assert len(client.calls) == 2


async def test_extract_malformed_then_valid() -> None:
    """First malformed, second valid — returns parsed result."""
    client = MockLLMClient(["bad json", VALID_JSON])
    result = await extract_entities("Google uses Python.", client)

    assert len(result.entities) == 2
    assert len(client.calls) == 2


async def test_entity_name_normalization() -> None:
    """Entity names are stripped and title-cased."""
    assert _normalize_entity_name("  google LLC ") == "Google Llc"
    assert _normalize_entity_name("APPLE") == "Apple"
    assert _normalize_entity_name("  sam altman  ") == "Sam Altman"


async def test_prompt_contains_text() -> None:
    """The extraction prompt includes the input text."""
    client = MockLLMClient([VALID_JSON])
    text = "Unique test text for prompt check."
    await extract_entities(text, client)

    sent_messages = client.calls[0]
    assert any(text in msg["content"] for msg in sent_messages)


async def test_parse_code_fenced_json() -> None:
    """JSON wrapped in markdown code fences is parsed correctly."""
    fenced = f"```json\n{VALID_JSON}\n```"
    result = _parse_extraction(fenced)
    assert len(result.entities) == 2


async def test_normalization_in_relationships() -> None:
    """Source/target names in relationships are also normalized."""
    raw = json.dumps(
        {
            "entities": [
                {
                    "name": "  google ",
                    "type": "ORGANIZATION",
                    "description": "Tech co.",
                },
                {
                    "name": " python  ",
                    "type": "TECHNOLOGY",
                    "description": "Lang.",
                },
            ],
            "relationships": [
                {
                    "source": "  google ",
                    "target": " python  ",
                    "relation": "USES_TECHNOLOGY",
                },
            ],
        }
    )
    result = _parse_extraction(raw)
    assert result.relationships[0].source == "Google"
    assert result.relationships[0].target == "Python"


@pytest.mark.integration
async def test_real_llm_extraction() -> None:
    """Integration test: real LLM call on sample text."""
    from ingestion.entity_extractor import get_llm_client

    client = get_llm_client()
    text = (
        "Apple Inc. announced a partnership with Google to develop "
        "new AI technology in San Francisco."
    )
    result = await extract_entities(text, client)

    assert len(result.entities) > 0
    entity_names = [e.name for e in result.entities]
    assert any("Apple" in n for n in entity_names)
