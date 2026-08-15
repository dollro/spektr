"""LLM-based per-document schema induction for GLiNER2.

Analyzes a text sample from each document and proposes domain-specific
entity and relationship types. Results are cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

from config.constants import ENTITY_TYPES, RELATIONSHIP_TYPES
from config.settings import settings

logger = logging.getLogger(__name__)

_MIN_TEXT_LEN = 200


@dataclass
class InducedSchema:
    """Schema proposed by the LLM for a specific document."""

    entity_types: dict[str, str] = field(default_factory=dict)
    relationship_types: dict[str, str] = field(default_factory=dict)


@dataclass
class MergedSchema:
    """Base schema + induced types merged together."""

    entity_types: dict[str, str] = field(default_factory=dict)
    relationship_types: dict[str, str] = field(default_factory=dict)


_PROMPT_TEMPLATE = """\
You are a knowledge graph schema designer. Given the following document excerpt,
propose entity types and relationship types that would capture the key information.

Return JSON with two keys:
- "entity_types": {{"type_name": "description of what this type represents"}}
- "relationship_types": {{"rel_name": "description: X [rel] Y means..."}}

Rules:
- Propose 3-8 entity types specific to this document's domain
- Propose 3-6 relationship types
- Descriptions must be clear enough for a non-expert NER model to use
- Do not duplicate these base types (they are already included): {base_types}

Document excerpt:
---
{sample_text}
---"""


async def _call_llm(prompt: str) -> str:
    """Call the schema induction LLM (cheap/fast model)."""
    from ingestion.entity_extractor import get_llm_client

    client = get_llm_client(settings.schema_induction_model)
    return await client.chat(
        [{"role": "user", "content": prompt}],
    )


class SchemaInducer:
    """Proposes domain-specific entity/relationship types for a document."""

    def __init__(self, cache_ttl: int | None = None) -> None:
        self._cache: dict[str, tuple[InducedSchema, float]] = {}
        self._ttl = cache_ttl if cache_ttl is not None else settings.schema_cache_ttl

    @staticmethod
    def _cache_key(sample_text: str) -> str:
        """Hash first 500 chars as domain signature."""
        return hashlib.sha256(sample_text[:500].encode()).hexdigest()

    async def induce(self, sample_text: str) -> InducedSchema:
        """Analyze sample text, return entity + relationship types."""
        if len(sample_text) < _MIN_TEXT_LEN:
            return InducedSchema()

        key = self._cache_key(sample_text)
        now = time.monotonic()

        # Check cache
        if key in self._cache:
            schema, cached_at = self._cache[key]
            if now - cached_at < self._ttl:
                return schema

        # Call LLM
        base_type_names = ", ".join(ENTITY_TYPES.keys())
        prompt = _PROMPT_TEMPLATE.format(
            base_types=base_type_names,
            sample_text=sample_text[:2000],
        )

        try:
            raw = await _call_llm(prompt)
            # Extract JSON from response (may be wrapped in markdown)
            json_str = raw
            if "```" in raw:
                json_str = raw.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            data = json.loads(json_str)
            schema = InducedSchema(
                entity_types=data.get("entity_types", {}),
                relationship_types=data.get("relationship_types", {}),
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            logger.warning("Schema induction returned malformed response")
            schema = InducedSchema()
        except Exception:
            logger.exception("Schema induction LLM call failed")
            schema = InducedSchema()

        self._cache[key] = (schema, now)
        return schema

    def merge_with_base(self, induced: InducedSchema) -> MergedSchema:
        """Merge induced types with base schema. Base types take priority."""
        merged_entities = dict(ENTITY_TYPES)
        for name, desc in induced.entity_types.items():
            if name not in merged_entities:
                merged_entities[name] = desc

        merged_rels = dict(RELATIONSHIP_TYPES)
        for name, desc in induced.relationship_types.items():
            if name not in merged_rels:
                merged_rels[name] = desc

        return MergedSchema(
            entity_types=merged_entities,
            relationship_types=merged_rels,
        )
